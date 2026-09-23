"""Physical SAPIEN backend for the V4 high-level training contract.

This module executes a standard first pass and subsequent primitives without
resetting the plant between lifecycle phases.  It uses hidden residual state for
the explicitly disclosed simulation-training potential reward and offline
terminal scoring; hidden truth is never routed into a policy observation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from typing import Callable

import gymnasium as gym
import numpy as np

from .decision_protocol import (
    DecisionContext,
    DecisionEncodingConfig,
    residual_excess_potential_gain,
)
from .high_level_training import (
    BackendReset,
    BackendTransition,
    HighLevelTrainingError,
)
from .information_flow import ObservationMode, SimulatedVisionConfig
from .low_level_control import ForceControllerConfig
from .modality_adapter import MatchedModalityAdapter, MatchedModalityPacket
from .native_bridge import NativeStepRecord, create_native_step_bridge
from .path_pose_control import CausalPathForcePoseController
from .primitive_pose_control import (
    CausalPrimitivePoseController,
)
from .primitives import PrimitiveProposal, compile_primitive
from .sapien_adapter import SapienNativeRateAdapter
from .scenarios import ScenarioSpec, make_scenario_path
from .scenarios import surface_normal
from .training_evidence import (
    PhysicalPhaseCode,
    ScientificEpisodeFailure,
    ScientificFailureCode,
    TrainingEpisodeEvidence,
    tracking_summary,
)

# Importing the environment module registers ForceWipeV4-v1 with Gymnasium.
from . import sapien_v4_env as _sapien_v4_env  # noqa: F401,E402


class SapienTrainingBackendError(HighLevelTrainingError):
    pass


@dataclass(frozen=True)
class SapienTrainingBackendConfig:
    maximum_primitives: int = 4
    residual_cells: int = 64
    native_rate_hz: int = 100
    maximum_episode_control_steps: int = 12_000
    maximum_first_pass_control_steps: int = 4_500
    maximum_primitive_control_steps: int = 4_500
    terminal_residual_mass_ratio: float = 0.15
    bind_primitive_force_to_scenario: bool = True
    hover_clearance_m: float = 0.035
    hover_position_tolerance_m: float = 0.004
    stable_tracking_start_progress: float = 0.05

    def validate(self) -> None:
        integer_values = (
            self.maximum_primitives,
            self.residual_cells,
            self.native_rate_hz,
            self.maximum_episode_control_steps,
            self.maximum_first_pass_control_steps,
            self.maximum_primitive_control_steps,
        )
        if not all(int(value) > 0 for value in integer_values):
            raise SapienTrainingBackendError("backend counts/rates must be positive")
        if int(self.native_rate_hz) != 100:
            raise SapienTrainingBackendError("V4 training requires the native 100 Hz audit")
        if not 0.0 < float(self.terminal_residual_mass_ratio) < 1.0:
            raise SapienTrainingBackendError("terminal residual ratio must lie in (0, 1)")
        if not 0.0 < float(self.hover_position_tolerance_m) < float(self.hover_clearance_m):
            raise SapienTrainingBackendError("hover clearance/tolerance are invalid")
        if not 0.0 <= float(self.stable_tracking_start_progress) < 1.0:
            raise SapienTrainingBackendError("stable tracking progress is invalid")


ScenarioFactory = Callable[[int], ScenarioSpec]
EvaluationIdSource = Callable[[int, int], int]
EvidenceSink = Callable[[TrainingEpisodeEvidence], None]
NormalForceControllerFactory = Callable[[], object]
FirstPassControllerFactory = Callable[[], object]


class SapienPrimitiveLifecycleBackend:
    """Execute an A1 categorical primitive policy in the real V4 simulator."""

    def __init__(
        self,
        *,
        mode: ObservationMode,
        scenario_factory: ScenarioFactory,
        primitive_count: int,
        config: SapienTrainingBackendConfig = SapienTrainingBackendConfig(),
        vision_config: SimulatedVisionConfig = SimulatedVisionConfig(),
        visual_decoder=None,
        force_config: ForceControllerConfig = ForceControllerConfig(
            kp_m_per_n=0.0008, ki_m_per_n_s=0.0012
        ),
        evaluation_id_source: EvaluationIdSource | None = None,
        evidence_sink: EvidenceSink | None = None,
        initial_episode_ordinal: int = 0,
        normal_force_controller_factory: NormalForceControllerFactory | None = None,
        first_pass_controller_factory: FirstPassControllerFactory | None = None,
        controller_diagnostics_enabled: bool = False,
    ) -> None:
        config.validate()
        self.mode = ObservationMode(mode)
        if self.mode not in {
            ObservationMode.VISION_ONLY,
            ObservationMode.FORCE_ONLY,
            ObservationMode.FUSION,
        }:
            raise SapienTrainingBackendError("training backend supports matched modalities only")
        if int(primitive_count) <= 0:
            raise SapienTrainingBackendError("primitive count must be positive")
        self.scenario_factory = scenario_factory
        self.primitive_count = int(primitive_count)
        self.config = config
        self.vision_config = vision_config
        self.visual_decoder = visual_decoder
        self.force_config = force_config
        self.normal_force_controller_factory = normal_force_controller_factory
        self.first_pass_controller_factory = first_pass_controller_factory
        self.controller_diagnostics_enabled = bool(controller_diagnostics_enabled)
        if (evaluation_id_source is None) != (evidence_sink is None):
            raise SapienTrainingBackendError(
                "evaluation identity source and evidence sink must be configured together"
            )
        if int(initial_episode_ordinal) < 0:
            raise SapienTrainingBackendError("initial episode ordinal must be nonnegative")
        self.evaluation_id_source = evaluation_id_source
        self.evidence_sink = evidence_sink
        self._next_episode_ordinal = int(initial_episode_ordinal)
        self.env = None
        self.raw = None
        self.spec: ScenarioSpec | None = None
        self.path = None
        self.bridge = None
        self.truth_capability = None
        self.adapter: SapienNativeRateAdapter | None = None
        self.modality: MatchedModalityAdapter | None = None
        self._last_record: NativeStepRecord | None = None
        self._initial_truth_mass: float | None = None
        self._control_step = 0
        self._completed_primitives = 0
        self._last_primitive_id: int | None = None
        self._force_violation_ever = False
        self._constraint_violation_ever = False
        self._moving_obstacle_contact_ever = False
        self._episode_peak_force_n = 0.0
        self.last_execution_audit: dict[str, object] | None = None
        self._closed = False
        self._episode_open = False
        self._evaluation_id: int | None = None
        self._phase_code = PhysicalPhaseCode.FIRST_PASS
        self._active_primitive_index: int | None = None
        self._current_force_target_n = 0.0
        self._current_controller_mode = "uninitialized"
        self._current_path_error_m: float | None = None
        self._current_force_command = None
        self._current_geometric_phase: str | None = None
        self._current_geometric_transition_reason: str | None = None
        self._native_rows: list[dict[str, object]] = []
        self._control_rows: list[dict[str, object]] = []
        self._controller_diagnostic_rows: list[dict[str, object]] = []
        self._primitive_rows: list[dict[str, object]] = []
        self._first_pass_success = False
        self._evidence_finalized = False

    @staticmethod
    def _bool_from_info(info, key: str) -> bool:
        if key not in info:
            raise SapienTrainingBackendError(f"simulator info lacks {key}")
        return bool(np.asarray(info[key]).reshape(-1)[0])

    def _context(self, record: NativeStepRecord) -> DecisionContext:
        elapsed = float(record.time_ns) * 1e-9
        horizon = float(self.config.maximum_episode_control_steps) / 100.0
        if elapsed > horizon + 1e-9:
            raise SapienTrainingBackendError("episode exceeded the frozen physical horizon")
        return DecisionContext(
            path_progress=float(record.progress),
            remaining_budget=max(
                0, int(self.config.maximum_primitives) - self._completed_primitives
            ),
            maximum_budget=int(self.config.maximum_primitives),
            elapsed_s=min(elapsed, horizon),
            episode_horizon_s=horizon,
            last_primitive_id=self._last_primitive_id,
            primitive_count=self.primitive_count,
        )

    @property
    def evidence_enabled(self) -> bool:
        return self.evidence_sink is not None

    @property
    def controller_diagnostic_rows(self) -> tuple[dict[str, object], ...]:
        return tuple(self._controller_diagnostic_rows)

    def _new_normal_force_controller(self):
        if self.normal_force_controller_factory is None:
            return None
        controller = self.normal_force_controller_factory()
        if not callable(getattr(controller, "reset", None)) or not callable(
            getattr(controller, "command", None)
        ):
            raise SapienTrainingBackendError(
                "normal-force controller factory returned an invalid controller"
            )
        return controller

    def _new_first_pass_controller(self):
        if self.first_pass_controller_factory is None:
            return CausalPathForcePoseController(
                force_config=self.force_config,
                normal_force_controller=self._new_normal_force_controller(),
            )
        controller = self.first_pass_controller_factory()
        if not callable(getattr(controller, "reset", None)) or not callable(
            getattr(controller, "command", None)
        ):
            raise SapienTrainingBackendError(
                "first-pass controller factory returned an invalid controller"
            )
        return controller

    def _normal_clearance_m(self) -> float | None:
        if self._last_record is None or self.spec is None or self.path is None or self.raw is None:
            return None
        relative = np.asarray(self._last_record.tcp_position_xyz, dtype=np.float64).copy()
        relative[2] -= float(self.raw.v4_surface_geometry.top_origin_z_m)
        projection = self.path.project(relative)
        normal = np.asarray(
            surface_normal(self.spec, float(projection.point_xyz[0])), dtype=np.float64
        )
        normal /= np.linalg.norm(normal)
        return float(np.dot(relative - projection.point_xyz, normal))

    def _geometric_hover_returned(self) -> tuple[bool, float | None]:
        clearance = self._normal_clearance_m()
        if clearance is None or self._last_record is None:
            return False, clearance
        required = float(self.config.hover_clearance_m) - float(
            self.config.hover_position_tolerance_m
        )
        return bool(
            float(self._last_record.audit_force_n) <= 0.2 and clearance >= required
        ), clearance

    def _native_row(self, record: NativeStepRecord) -> dict[str, object]:
        if self._evaluation_id is None:
            raise SapienTrainingBackendError("evidence-enabled episode lacks an evaluation ID")
        return {
            "evaluation_id": self._evaluation_id,
            "step": len(self._native_rows),
            "time_ns": int(record.time_ns),
            "control_step": int(
                self._control_step
                if record.control_step_index is None
                else record.control_step_index
            ),
            "substep": int(0 if record.substep_index is None else record.substep_index),
            "phase_code": int(self._phase_code),
            "primitive_index": self._active_primitive_index,
            "f_aud_n": float(record.audit_force_n),
            "f_ctrl_hold_n": (
                None if record.tracking_force_n is None else float(record.tracking_force_n)
            ),
            "f_ctrl_source_sample": record.tracking_force_measurement_sample_index,
            "f_ctrl_age_samples": record.tracking_force_age_samples,
            "progress": float(record.progress),
            "tangential_speed_m_s": float(record.tangential_speed_m_s),
            "contact": bool(record.contact),
            "near_limit": bool(record.near_limit),
            "force_violation": bool(record.force_violation),
            "tcp_x_m": float(record.tcp_position_xyz[0]),
            "tcp_y_m": float(record.tcp_position_xyz[1]),
            "tcp_z_m": float(record.tcp_position_xyz[2]),
        }

    def _control_row(self, info, audit) -> dict[str, object]:
        if self._evaluation_id is None or self._last_record is None:
            raise SapienTrainingBackendError("control evidence lacks episode identity/state")
        return {
            "evaluation_id": self._evaluation_id,
            "step": len(self._control_rows),
            "time_ns": int(self._last_record.time_ns),
            "phase_code": int(self._phase_code),
            "primitive_index": self._active_primitive_index,
            "f_ctrl_input_n": float(audit.tracking_input_force_n),
            "f_trk_n": float(audit.end_tracking_force_n),
            "force_target_n": float(self._current_force_target_n),
            "progress": float(self._last_record.progress),
            "contact": bool(float(audit.end_tracking_force_n) >= 3.0),
            "constraint_violation": self._bool_from_info(info, "constraint_violation"),
            "moving_obstacle_contact": self._bool_from_info(
                info, "moving_obstacle_contact"
            ),
            "controller_mode": str(self._current_controller_mode),
            "path_error_m": (
                float(self._last_record.path_distance_m)
                if self._current_path_error_m is None
                else float(self._current_path_error_m)
            ),
        }

    def _controller_diagnostic_row(self) -> dict[str, object]:
        if self._evaluation_id is None or self._last_record is None:
            raise SapienTrainingBackendError("controller diagnostics lack episode identity/state")
        command = self._current_force_command
        required = (
            "controller_state",
            "state_transition_reason",
            "force_error_n",
            "force_rate_n_s",
            "integral_error_n_s",
            "proportional_step_m",
            "integral_step_m",
            "derivative_step_m",
            "raw_normal_step_m",
            "clipped_normal_step_m",
            "confirmation_count",
            "tangential_motion_allowed",
        )
        if command is None:
            raise SapienTrainingBackendError("V5.3 controller command is absent")
        manual_modes = {
            "unexpected_hover_contact_lift",
            "hover_reposition",
            "primitive_return_hover",
            "primitive_complete",
        }
        decomposed = all(hasattr(command, name) for name in required)
        if not decomposed and str(getattr(command, "mode", "")) not in manual_modes:
            raise SapienTrainingBackendError(
                "V5.3 force-control phase lacks a decomposed controller command"
            )
        state = (
            str(command.controller_state)
            if decomposed
            else str(command.mode).upper()
        )
        reason = (
            str(command.state_transition_reason)
            if decomposed
            else "manual_non_force_phase"
        )
        p_step = float(command.proportional_step_m) if decomposed else 0.0
        i_step = float(command.integral_step_m) if decomposed else 0.0
        d_step = float(command.derivative_step_m) if decomposed else 0.0
        raw_step = (
            float(command.raw_normal_step_m)
            if decomposed
            else float(command.normal_step_m)
        )
        clipped_step = (
            float(command.clipped_normal_step_m)
            if decomposed
            else float(command.normal_step_m)
        )
        return {
            "evaluation_id": self._evaluation_id,
            "step": len(self._controller_diagnostic_rows),
            "time_ns": int(self._last_record.time_ns),
            "phase_code": int(self._phase_code),
            "primitive_index": self._active_primitive_index,
            "controller_state": state,
            "state_transition_reason": reason,
            "force_error_n": float(command.force_error_n),
            "force_rate_n_s": float(command.force_rate_n_s),
            "integral_error_n_s": float(command.integral_error_n_s),
            "p_step_m": p_step,
            "i_step_m": i_step,
            "d_step_m": d_step,
            "raw_normal_step_m": raw_step,
            "clipped_normal_step_m": clipped_step,
            "confirmation_count": int(command.confirmation_count) if decomposed else 0,
            "tangential_motion_allowed": (
                bool(command.tangential_motion_allowed) if decomposed else False
            ),
            "geometric_phase": (
                "LEGACY_OR_MANUAL"
                if self._current_geometric_phase is None
                else str(self._current_geometric_phase)
            ),
            "geometric_transition_reason": (
                "none"
                if self._current_geometric_transition_reason is None
                else str(self._current_geometric_transition_reason)
            ),
        }

    def _finalize_evidence(
        self,
        *,
        failure_code: ScientificFailureCode,
        physical_lifecycle_complete: bool,
        synthetic_coverage_success: bool | None,
        synthetic_residual_mass_ratio: float | None,
    ) -> TrainingEpisodeEvidence | None:
        if not self.evidence_enabled:
            return None
        if self._evidence_finalized:
            raise SapienTrainingBackendError("episode evidence was finalized twice")
        if self._evaluation_id is None or self.spec is None:
            raise SapienTrainingBackendError("cannot finalize unidentified episode evidence")
        geometric_hover, clearance = self._geometric_hover_returned()
        summary = tracking_summary(
            tuple(self._control_rows),
            target_force_n=float(self.spec.target_force_n),
            stable_progress_minimum=float(self.config.stable_tracking_start_progress),
        )
        if not self._primitive_rows:
            self._primitive_rows.append(
                {
                    "evaluation_id": self._evaluation_id,
                    "step": 0,
                    "event_kind": "first_pass_failure",
                    "action_index": -1,
                    "completed": False,
                    "native_start": 0,
                    "native_end": len(self._native_rows),
                    "duration_s": (
                        0.0
                        if self._last_record is None
                        else float(self._last_record.time_ns) * 1e-9
                    ),
                    "peak_force_n": float(self._episode_peak_force_n),
                    "force_violation": bool(self._force_violation_ever),
                    "spatial_constraint_violation": bool(
                        self._constraint_violation_ever
                        or self._moving_obstacle_contact_ever
                    ),
                    "physical_lifecycle_complete": False,
                    "synthetic_coverage_success": None,
                    "synthetic_residual_mass_ratio": None,
                }
            )
        evidence = TrainingEpisodeEvidence(
            evaluation_id=self._evaluation_id,
            native_rows=tuple(self._native_rows),
            control_rows=tuple(self._control_rows),
            primitive_rows=tuple(self._primitive_rows),
            episode_row={
                "evaluation_id": self._evaluation_id,
                "physical_lifecycle_complete": bool(physical_lifecycle_complete),
                "synthetic_coverage_success": synthetic_coverage_success,
                "synthetic_residual_mass_ratio": synthetic_residual_mass_ratio,
                "failure_code": int(failure_code),
                "first_pass_success": bool(self._first_pass_success),
                "planned_primitives": int(self.config.maximum_primitives),
                "actual_primitives": sum(
                    row["event_kind"] == "primitive" for row in self._primitive_rows
                ),
                "native_samples": len(self._native_rows),
                "control_samples": len(self._control_rows),
                "peak_force_n": float(self._episode_peak_force_n),
                **summary,
                "geometric_hover_returned": bool(geometric_hover),
                "terminal_normal_clearance_m": clearance,
            },
        )
        evidence.validate()
        self.evidence_sink(evidence)
        self._evidence_finalized = True
        return evidence

    def _scientific_failure(
        self, code: ScientificFailureCode, message: str
    ) -> ScientificEpisodeFailure:
        self._finalize_evidence(
            failure_code=code,
            physical_lifecycle_complete=False,
            synthetic_coverage_success=None,
            synthetic_residual_mass_ratio=None,
        )
        return ScientificEpisodeFailure(code, message)

    def _record_sink(self, record: NativeStepRecord) -> None:
        self._last_record = record
        self._force_violation_ever |= bool(record.force_violation)
        self._episode_peak_force_n = max(
            self._episode_peak_force_n, float(record.audit_force_n)
        )
        if self.modality is None:
            raise SapienTrainingBackendError("modality adapter is unavailable")
        self.modality.observe(record.policy_observation, self._context(record))
        if self.evidence_enabled:
            self._native_rows.append(self._native_row(record))

    def _packet(self) -> MatchedModalityPacket:
        if self._last_record is None or self.modality is None or self.bridge is None:
            raise SapienTrainingBackendError("no causal native observation is available")
        return self.modality.encode(
            self._last_record.policy_observation,
            self._context(self._last_record),
            visual_frame=self.bridge.policy_visual_frame(),
        )

    def _apply_compliance(self, stiffness: float, damping: float) -> None:
        self.raw.set_v4_tool_operational_compliance(
            normal_stiffness_n_m=float(stiffness),
            normal_damping_n_s_m=float(damping),
        )

    def _action(self, pose_action: np.ndarray) -> np.ndarray:
        action = np.zeros(self.env.action_space.shape, dtype=np.float32)
        action[:6] = np.asarray(pose_action, dtype=np.float32)
        action[-1] = -1.0
        return action

    def _step(self, action_factory):
        result, audit = self.adapter.step_with_action_factory(
            self.env,
            action_factory,
            control_step_index=self._control_step,
        )
        self._control_step += 1
        _observation, _reward, terminated, truncated, info = result
        self._constraint_violation_ever |= self._bool_from_info(
            info, "constraint_violation"
        )
        self._moving_obstacle_contact_ever |= self._bool_from_info(
            info, "moving_obstacle_contact"
        )
        if self.evidence_enabled:
            self._control_rows.append(self._control_row(info, audit))
            if self.controller_diagnostics_enabled:
                self._controller_diagnostic_rows.append(
                    self._controller_diagnostic_row()
                )
        return (
            bool(np.asarray(terminated).reshape(-1)[0]),
            bool(np.asarray(truncated).reshape(-1)[0]),
            info,
            audit,
        )

    def _close_episode(self) -> None:
        if self.adapter is not None:
            self.adapter.uninstall()
        if self.env is not None:
            self.env.close()
        self.env = None
        self.raw = None
        self.adapter = None
        self._episode_open = False

    def reset(self, *, seed: int) -> BackendReset:
        if self._closed:
            raise SapienTrainingBackendError("cannot reset a closed backend")
        if self._episode_open:
            self._close_episode()
        spec = self.scenario_factory(int(seed))
        spec.validate()
        if self.evidence_enabled:
            self._evaluation_id = int(
                self.evaluation_id_source(self._next_episode_ordinal, int(seed))
            )
            if self._evaluation_id < 0:
                raise SapienTrainingBackendError("evaluation ID must be nonnegative")
        else:
            self._evaluation_id = None
        self._next_episode_ordinal += 1
        self.spec = spec
        self.path = make_scenario_path(spec, points=801)
        self.env = gym.make(
            "ForceWipeV4-v1",
            scenario_spec=asdict(spec),
            lifecycle_mode=True,
            num_envs=1,
            obs_mode="state_dict",
            reward_mode="dense",
            control_mode="pd_ee_delta_pose",
            render_mode=None,
            sim_backend="physx_cpu",
            render_backend="none",
            sim_config=dict(control_freq=100),
            max_episode_steps=int(self.config.maximum_episode_control_steps) + 10,
        )
        self.env.reset(seed=int(seed))
        self.raw = self.env.unwrapped
        episode_vision = replace(self.vision_config, seed=int(seed))
        self.bridge, self.truth_capability = create_native_step_bridge(
            spec,
            mode=self.mode,
            residual_cells=int(self.config.residual_cells),
            vision_config=episode_vision,
            visual_decoder=self.visual_decoder,
        )
        initial_truth = self.bridge.truth_audit_snapshot(self.truth_capability)
        self._initial_truth_mass = float(initial_truth.mass)
        self.modality = MatchedModalityAdapter(
            decision_config=DecisionEncodingConfig(
                residual_cell_count=int(self.config.residual_cells)
            )
        )
        self._last_record = None
        self._control_step = 0
        self._completed_primitives = 0
        self._last_primitive_id = None
        self._force_violation_ever = False
        self._constraint_violation_ever = False
        self._moving_obstacle_contact_ever = False
        self._episode_peak_force_n = 0.0
        self.last_execution_audit = None
        self._phase_code = PhysicalPhaseCode.FIRST_PASS
        self._active_primitive_index = None
        self._current_force_target_n = float(spec.target_force_n)
        self._current_controller_mode = "first_pass_uninitialized"
        self._current_path_error_m = None
        self._current_force_command = None
        self._current_geometric_phase = None
        self._current_geometric_transition_reason = None
        self._native_rows = []
        self._control_rows = []
        self._controller_diagnostic_rows = []
        self._primitive_rows = []
        self._first_pass_success = False
        self._evidence_finalized = False
        self.adapter = SapienNativeRateAdapter(
            self.raw, self.bridge, record_sink=self._record_sink
        )
        self.adapter.install()
        self._episode_open = True
        controller = self._new_first_pass_controller()
        success = False
        terminal = False
        for _ in range(int(self.config.maximum_first_pass_control_steps)):
            holder = {}

            def action_factory(measurement):
                command = controller.command(
                    measured_force_n=measurement.value_n,
                    target_force_n=spec.target_force_n,
                    tool_position_xyz_m=self.raw.v4_tool.pose.p[0].detach().cpu().numpy(),
                    tcp_quaternion_wxyz=self.raw.agent.tcp.pose.q[0].detach().cpu().numpy(),
                    scenario=spec,
                    path=self.path,
                    surface_top_origin_z_m=self.raw.v4_surface_geometry.top_origin_z_m,
                )
                holder["command"] = command
                self._current_force_target_n = float(spec.target_force_n)
                self._current_controller_mode = str(command.force_command.mode)
                self._current_path_error_m = None
                self._current_force_command = command.force_command
                self._current_geometric_phase = getattr(
                    command, "geometric_phase", None
                )
                self._current_geometric_transition_reason = getattr(
                    command, "geometric_transition_reason", None
                )
                return self._action(command.normalized_pose_action)

            terminated, truncated, info, _audit = self._step(action_factory)
            terminal = terminated or truncated
            success = self._bool_from_info(info, "task_success")
            if success or terminal or self._force_violation_ever:
                break
        if not success:
            if self._force_violation_ever:
                code = ScientificFailureCode.FIRST_PASS_FORCE_VIOLATION
            elif terminal:
                code = ScientificFailureCode.FIRST_PASS_ENVIRONMENT_TERMINATION
            else:
                code = ScientificFailureCode.FIRST_PASS_TRACKING_TIMEOUT
            raise self._scientific_failure(
                code,
                "standard first pass failed before the high-level decision boundary",
            )
        self._first_pass_success = True
        return BackendReset(
            packet=self._packet(),
            first_pass_native_samples=int(self._last_record.sample_index + 1),
            first_pass_force_violation=bool(self._force_violation_ever),
            first_pass_peak_force_n=float(self._episode_peak_force_n),
        )

    def _terminal_score(self) -> tuple[bool, float]:
        if self._initial_truth_mass is None:
            raise SapienTrainingBackendError("initial terminal scorer state is absent")
        final = self.bridge.truth_audit_snapshot(self.truth_capability)
        ratio = float(final.mass) / max(float(self._initial_truth_mass), np.finfo(float).eps)
        success = bool(
            ratio <= float(self.config.terminal_residual_mass_ratio)
            and not self._force_violation_ever
            and not self._constraint_violation_ever
            and not self._moving_obstacle_contact_ever
        )
        return success, float(ratio)

    def execute_primitive(
        self, primitive: PrimitiveProposal, *, primitive_index: int
    ) -> BackendTransition:
        if not self._episode_open or self.spec is None:
            raise SapienTrainingBackendError("execute_primitive requires an open episode")
        if not 0 <= int(primitive_index) < self.primitive_count:
            raise SapienTrainingBackendError("primitive index is outside the frozen library")
        executed = (
            replace(primitive, target_force_n=float(self.spec.target_force_n))
            if self.config.bind_primitive_force_to_scenario
            else primitive
        )
        reference = compile_primitive(executed, self.path, native_rate_hz=100.0)
        controller = CausalPrimitivePoseController(
            reference,
            force_config=self.force_config,
            normal_force_controller=self._new_normal_force_controller(),
            compliance_parameter_sink=self._apply_compliance,
        )
        self._phase_code = PhysicalPhaseCode.PRIMITIVE
        self._active_primitive_index = int(primitive_index)
        self._current_force_target_n = float(executed.target_force_n)
        self._current_controller_mode = "primitive_uninitialized"
        self._current_path_error_m = None
        start_index = int(self._last_record.sample_index + 1)
        start_time_ns = int(self._last_record.time_ns)
        if self._initial_truth_mass is None:
            raise SapienTrainingBackendError("initial reward-potential state is absent")
        truth_before = self.bridge.truth_audit_snapshot(self.truth_capability)
        residual_ratio_before = float(truth_before.mass) / max(
            float(self._initial_truth_mass), np.finfo(float).eps
        )
        transition_peak_force_n = 0.0
        terminal = False
        completed = False
        phase_counts: dict[str, int] = {}
        last_command = None
        for _ in range(int(self.config.maximum_primitive_control_steps)):
            holder = {}

            def action_factory(measurement):
                command = controller.command(
                    measured_force_n=measurement.value_n,
                    tool_position_xyz_m=self.raw.v4_tool.pose.p[0].detach().cpu().numpy(),
                    tcp_quaternion_wxyz=self.raw.agent.tcp.pose.q[0].detach().cpu().numpy(),
                    scenario=self.spec,
                    surface_top_origin_z_m=self.raw.v4_surface_geometry.top_origin_z_m,
                )
                holder["command"] = command
                self._current_force_target_n = float(command.target_force_n)
                self._current_controller_mode = (
                    f"{command.phase.value}:{command.force_command.mode}"
                )
                self._current_path_error_m = float(command.planar_reference_error_m)
                self._current_force_command = command.force_command
                self._current_geometric_phase = None
                self._current_geometric_transition_reason = None
                return self._action(command.normalized_pose_action)

            before = int(self._last_record.sample_index)
            terminated, truncated, _info, _audit = self._step(action_factory)
            new_record = self._last_record
            if int(new_record.sample_index) <= before:
                raise SapienTrainingBackendError("primitive step produced no native sample")
            transition_peak_force_n = max(
                transition_peak_force_n, float(new_record.audit_force_n)
            )
            completed = bool(holder["command"].completed)
            last_command = holder["command"]
            phase_name = holder["command"].phase.value
            phase_counts[phase_name] = phase_counts.get(phase_name, 0) + 1
            terminal = bool(terminated or truncated)
            if completed or terminal or self._force_violation_ever:
                break
        if completed:
            self._completed_primitives += 1
            self._last_primitive_id = int(primitive_index)
        budget_terminal = self._completed_primitives >= int(self.config.maximum_primitives)
        geometric_hover, _clearance = self._geometric_hover_returned()
        physical_complete = bool(budget_terminal and completed and geometric_hover)
        synthetic_success = None
        synthetic_ratio = None
        scientific_terminal = bool(
            budget_terminal
            or terminal
            or not completed
            or self._force_violation_ever
            or self._constraint_violation_ever
            or self._moving_obstacle_contact_ever
        )
        if scientific_terminal:
            synthetic_success, synthetic_ratio = self._terminal_score()
        truth_after = self.bridge.truth_audit_snapshot(self.truth_capability)
        residual_ratio_after = float(truth_after.mass) / max(
            float(self._initial_truth_mass), np.finfo(float).eps
        )
        residual_target = float(self.config.terminal_residual_mass_ratio)
        potential_gain = residual_excess_potential_gain(
            residual_ratio_before,
            residual_ratio_after,
            target_ratio=residual_target,
        )
        if self.evidence_enabled:
            self._primitive_rows.append(
                {
                    "evaluation_id": self._evaluation_id,
                    "step": len(self._primitive_rows),
                    "event_kind": "primitive",
                    "action_index": int(primitive_index),
                    "completed": bool(completed),
                    "native_start": int(start_index),
                    "native_end": int(self._last_record.sample_index + 1),
                    "duration_s": float(self._last_record.time_ns - start_time_ns)
                    * 1e-9,
                    "peak_force_n": float(transition_peak_force_n),
                    "force_violation": bool(self._force_violation_ever),
                    "spatial_constraint_violation": bool(
                        self._constraint_violation_ever
                        or self._moving_obstacle_contact_ever
                    ),
                    "physical_lifecycle_complete": physical_complete,
                    "synthetic_coverage_success": synthetic_success,
                    "synthetic_residual_mass_ratio": synthetic_ratio,
                }
            )
        self.last_execution_audit = {
            "completed": bool(completed),
            "terminal": bool(terminal),
            "phase_counts": phase_counts,
            "last_phase": last_command.phase.value if last_command is not None else None,
            "last_reference_index": (
                int(last_command.reference_index) if last_command is not None else None
            ),
            "reference_samples": int(len(reference.progress)),
            "last_planar_reference_error_m": (
                float(last_command.planar_reference_error_m)
                if last_command is not None
                else None
            ),
            "last_audit_force_n": float(self._last_record.audit_force_n),
        }
        transition = BackendTransition(
            packet=self._packet(),
            training_residual_excess_potential_gain=potential_gain,
            duration_s=float(self._last_record.time_ns - start_time_ns) * 1e-9,
            shield_intervened=False,
            force_limit_violation=bool(self._force_violation_ever),
            spatial_constraint_violation=bool(
                self._constraint_violation_ever or self._moving_obstacle_contact_ever
            ),
            physical_lifecycle_complete=physical_complete,
            backend_terminal=bool(terminal or not completed),
            executed_primitive_index=int(primitive_index) if completed else None,
            native_samples=int(self._last_record.sample_index - start_index + 1),
            native_peak_force_n=float(transition_peak_force_n),
            synthetic_coverage_success=synthetic_success,
            synthetic_residual_mass_ratio=synthetic_ratio,
        )
        if scientific_terminal and self.evidence_enabled:
            if self._force_violation_ever:
                failure = ScientificFailureCode.FORCE_LIMIT_VIOLATION
            elif self._constraint_violation_ever or self._moving_obstacle_contact_ever:
                failure = ScientificFailureCode.SPATIAL_CONSTRAINT_VIOLATION
            elif terminal:
                failure = ScientificFailureCode.PLANT_TERMINATION
            elif not completed:
                failure = ScientificFailureCode.PRIMITIVE_TIMEOUT
            elif not physical_complete:
                failure = ScientificFailureCode.STOP_RETURN_TIMEOUT
            else:
                failure = ScientificFailureCode.NONE
            self._finalize_evidence(
                failure_code=failure,
                physical_lifecycle_complete=physical_complete,
                synthetic_coverage_success=synthetic_success,
                synthetic_residual_mass_ratio=synthetic_ratio,
            )
        return transition

    def request_stop(self) -> BackendTransition:
        if not self._episode_open or self.spec is None:
            raise SapienTrainingBackendError("request_stop requires an open episode")
        start_index = int(self._last_record.sample_index + 1)
        start_time_ns = int(self._last_record.time_ns)
        self._phase_code = PhysicalPhaseCode.STOP_RETURN
        self._active_primitive_index = None
        self._current_force_target_n = 0.0
        self._current_controller_mode = "stop_return_uninitialized"
        self._current_path_error_m = None
        # A stop is complete only after both force release and geometric hover.
        already_hovering, _clearance = self._geometric_hover_returned()
        terminal = False
        transition_peak_force_n = 0.0
        if not already_hovering:
            center = float(self._last_record.progress)
            lift_template = PrimitiveProposal(
                center_s=float(np.clip(center, 0.05, 0.95)),
                window_width_s=0.10,
                travel_length_m=0.01,
                direction=1,
                target_force_n=float(self.spec.target_force_n),
                stiffness_n_m=600.0,
                damping_n_s_m=30.0,
                tangential_speed_m_s=0.03,
            )
            controller = CausalPrimitivePoseController(
                compile_primitive(lift_template, self.path, native_rate_hz=100.0),
                force_config=self.force_config,
                normal_force_controller=self._new_normal_force_controller(),
                compliance_parameter_sink=self._apply_compliance,
            )
            for _ in range(int(self.config.maximum_primitive_control_steps)):
                holder = {}

                def action_factory(measurement):
                    command = controller.command(
                        measured_force_n=measurement.value_n,
                        tool_position_xyz_m=self.raw.v4_tool.pose.p[0].detach().cpu().numpy(),
                        tcp_quaternion_wxyz=self.raw.agent.tcp.pose.q[0].detach().cpu().numpy(),
                        scenario=self.spec,
                        surface_top_origin_z_m=self.raw.v4_surface_geometry.top_origin_z_m,
                    )
                    holder["command"] = command
                    self._current_force_target_n = 0.0
                    self._current_controller_mode = (
                        f"stop:{command.phase.value}:{command.force_command.mode}"
                    )
                    self._current_path_error_m = float(
                        command.planar_reference_error_m
                    )
                    self._current_force_command = command.force_command
                    self._current_geometric_phase = None
                    self._current_geometric_transition_reason = None
                    return self._action(command.normalized_pose_action)

                terminated, truncated, _info, _audit = self._step(action_factory)
                transition_peak_force_n = max(
                    transition_peak_force_n, float(self._last_record.audit_force_n)
                )
                terminal = bool(terminated or truncated)
                geometric_hover, _clearance = self._geometric_hover_returned()
                if geometric_hover:
                    already_hovering = True
                    break
                if terminal or self._force_violation_ever:
                    break
        synthetic_success, synthetic_ratio = self._terminal_score()
        if self.evidence_enabled:
            self._primitive_rows.append(
                {
                    "evaluation_id": self._evaluation_id,
                    "step": len(self._primitive_rows),
                    "event_kind": "stop",
                    "action_index": -1,
                    "completed": bool(already_hovering),
                    "native_start": int(start_index),
                    "native_end": int(self._last_record.sample_index + 1),
                    "duration_s": float(self._last_record.time_ns - start_time_ns)
                    * 1e-9,
                    "peak_force_n": float(transition_peak_force_n),
                    "force_violation": bool(self._force_violation_ever),
                    "spatial_constraint_violation": bool(
                        self._constraint_violation_ever
                        or self._moving_obstacle_contact_ever
                    ),
                    "physical_lifecycle_complete": bool(already_hovering),
                    "synthetic_coverage_success": synthetic_success,
                    "synthetic_residual_mass_ratio": synthetic_ratio,
                }
            )
        transition = BackendTransition(
            packet=self._packet(),
            training_residual_excess_potential_gain=0.0,
            duration_s=float(self._last_record.time_ns - start_time_ns) * 1e-9,
            shield_intervened=False,
            force_limit_violation=bool(self._force_violation_ever),
            spatial_constraint_violation=bool(
                self._constraint_violation_ever or self._moving_obstacle_contact_ever
            ),
            physical_lifecycle_complete=bool(already_hovering),
            backend_terminal=True,
            executed_primitive_index=None,
            native_samples=max(0, int(self._last_record.sample_index - start_index + 1)),
            native_peak_force_n=float(transition_peak_force_n),
            synthetic_coverage_success=synthetic_success,
            synthetic_residual_mass_ratio=synthetic_ratio,
        )
        if self.evidence_enabled:
            if self._force_violation_ever:
                failure = ScientificFailureCode.FORCE_LIMIT_VIOLATION
            elif self._constraint_violation_ever or self._moving_obstacle_contact_ever:
                failure = ScientificFailureCode.SPATIAL_CONSTRAINT_VIOLATION
            elif terminal:
                failure = ScientificFailureCode.PLANT_TERMINATION
            elif not already_hovering:
                failure = ScientificFailureCode.STOP_RETURN_TIMEOUT
            else:
                failure = ScientificFailureCode.NONE
            self._finalize_evidence(
                failure_code=failure,
                physical_lifecycle_complete=bool(already_hovering),
                synthetic_coverage_success=synthetic_success,
                synthetic_residual_mass_ratio=synthetic_ratio,
            )
        return transition

    def close(self) -> None:
        if self._episode_open:
            self._close_episode()
        self._closed = True
