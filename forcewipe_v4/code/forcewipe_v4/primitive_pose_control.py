"""Physical execution state machine for one compiled V4 wiping primitive.

The first-pass controller reports monotone task progress.  A local re-wiping
primitive has different semantics: its reference may reverse direction and is
therefore indexed directly, never ratcheted through projected path progress.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
import math

import numpy as np

from .low_level_baselines import BaselineControlOutput, SolverAudit
from .low_level_control import (
    CausalNormalForceController,
    ForceControlCommand,
    ForceControllerConfig,
    LowLevelControlError,
)
from .path_pose_control import desired_tool_rotation, root_aligned_rotation_delta_xyz
from .primitives import PrimitiveReference
from .scenarios import ScenarioSpec, surface_normal


class PrimitiveExecutionPhase(str, Enum):
    HOVER_REPOSITION = "hover_reposition"
    CONTACT_ACQUISITION = "contact_acquisition"
    TRACK = "track"
    RETURN_HOVER = "return_hover"
    COMPLETE = "complete"


@dataclass(frozen=True)
class PrimitivePoseControllerConfig:
    dt_s: float = 0.01
    hover_clearance_m: float = 0.035
    hover_position_tolerance_m: float = 0.004
    hover_position_gain: float = 0.45
    max_hover_step_m: float = 0.006
    reference_position_tolerance_m: float = 0.015
    reference_error_gain: float = 0.55
    max_tangent_step_m: float = 0.0035
    contact_force_n: float = 3.0
    contact_confirmation_steps: int = 3
    path_motion_headroom_force_n: float = 13.5
    return_lift_step_m: float = 0.004
    max_rotation_step_rad: float = 0.05
    action_position_scale_m: float = 0.1
    action_rotation_scale_rad: float = 0.1

    def validate(self) -> None:
        positive = (
            self.dt_s,
            self.hover_clearance_m,
            self.hover_position_tolerance_m,
            self.hover_position_gain,
            self.max_hover_step_m,
            self.reference_position_tolerance_m,
            self.reference_error_gain,
            self.max_tangent_step_m,
            self.contact_force_n,
            self.path_motion_headroom_force_n,
            self.return_lift_step_m,
            self.max_rotation_step_rad,
            self.action_position_scale_m,
            self.action_rotation_scale_rad,
        )
        if not all(math.isfinite(float(value)) and float(value) > 0.0 for value in positive):
            raise LowLevelControlError("primitive controller parameters must be positive")
        if int(self.contact_confirmation_steps) <= 0:
            raise LowLevelControlError("contact confirmation count must be positive")
        if self.contact_force_n >= self.path_motion_headroom_force_n:
            raise LowLevelControlError("contact threshold must remain below headroom")
        if self.max_rotation_step_rad > self.action_rotation_scale_rad:
            raise LowLevelControlError("rotation step exceeds the pose-action scale")


@dataclass(frozen=True)
class PrimitivePoseCommand:
    normalized_pose_action: np.ndarray
    force_command: ForceControlCommand
    phase: PrimitiveExecutionPhase
    reference_index: int
    reference_progress: float
    target_force_n: float
    stiffness_n_m: float
    damping_n_s_m: float
    target_point_xyz_m: np.ndarray
    tangent_xyz: np.ndarray
    orientation_tangent_xyz: np.ndarray
    outward_normal_xyz: np.ndarray
    position_step_xyz_m: np.ndarray
    tangent_step_xyz_m: np.ndarray
    rotation_delta_xyz_rad: np.ndarray
    planar_reference_error_m: float
    solver_audit: SolverAudit | None
    completed: bool


def _force_command_from_output(
    output: ForceControlCommand | BaselineControlOutput,
) -> tuple[ForceControlCommand, SolverAudit | None]:
    if isinstance(output, BaselineControlOutput):
        return output.command, output.audit
    if isinstance(output, ForceControlCommand):
        return output, None
    raise LowLevelControlError("normal force controller returned an unsupported type")


def _manual_force_command(
    *,
    normal_step_m: float,
    normal_xyz: np.ndarray,
    action_position_scale_m: float,
    mode: str,
) -> ForceControlCommand:
    delta = -np.asarray(normal_xyz, dtype=np.float64) * float(normal_step_m)
    return ForceControlCommand(
        normalized_position_action=np.clip(
            delta / float(action_position_scale_m), -1.0, 1.0
        ),
        normal_step_m=float(normal_step_m),
        force_error_n=0.0,
        force_rate_n_s=0.0,
        integral_error_n_s=0.0,
        mode=str(mode),
    )


class CausalPrimitivePoseController:
    """Execute one immutable primitive through explicit safe phases.

    Reference advancement is permitted only during confirmed contact, below
    force headroom, and near the current planar reference.  Contact loss holds
    the reference index while the low-level force loop reacquires contact.
    """

    def __init__(
        self,
        reference: PrimitiveReference,
        *,
        force_config: ForceControllerConfig = ForceControllerConfig(),
        pose_config: PrimitivePoseControllerConfig = PrimitivePoseControllerConfig(),
        normal_force_controller: object | None = None,
        compliance_parameter_sink: Callable[[float, float], None] | None = None,
    ) -> None:
        pose_config.validate()
        if not math.isclose(reference.dt_s, pose_config.dt_s, rel_tol=0.0, abs_tol=1e-12):
            raise LowLevelControlError("primitive and controller native rates differ")
        self.reference = reference
        self.pose_config = pose_config
        self.compliance_parameter_sink = compliance_parameter_sink
        self.force_controller = (
            CausalNormalForceController(force_config)
            if normal_force_controller is None
            else normal_force_controller
        )
        if not callable(getattr(self.force_controller, "command", None)) or not callable(
            getattr(self.force_controller, "reset", None)
        ):
            raise LowLevelControlError(
                "normal force controller must expose reset() and command()"
            )
        self.reset()

    def reset(self) -> None:
        self.force_controller.reset()
        self._phase = PrimitiveExecutionPhase.HOVER_REPOSITION
        self._reference_index = 0
        self._contact_confirmation_count = 0

    @property
    def phase(self) -> PrimitiveExecutionPhase:
        return self._phase

    @property
    def reference_index(self) -> int:
        return self._reference_index

    def _reference_geometry(
        self,
        *,
        scenario: ScenarioSpec,
        surface_top_origin_z_m: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
        index = self._reference_index
        point = np.asarray(self.reference.position_xyz[index], dtype=np.float64).copy()
        point[2] += float(surface_top_origin_z_m)
        tangent = np.asarray(self.reference.tangent_xyz[index], dtype=np.float64).copy()
        normal = np.asarray(
            surface_normal(scenario, self.reference.position_xyz[index, 0]),
            dtype=np.float64,
        )
        normal /= np.linalg.norm(normal)
        target_force = float(self.reference.target_force_n[index])
        stiffness = float(self.reference.stiffness_n_m[index])
        damping = float(self.reference.damping_n_s_m[index])
        return point, tangent, normal, target_force, stiffness, damping

    @staticmethod
    def _bounded_vector(vector: np.ndarray, maximum_norm: float) -> np.ndarray:
        result = np.asarray(vector, dtype=np.float64).copy()
        norm = float(np.linalg.norm(result))
        if norm > float(maximum_norm):
            result *= float(maximum_norm) / norm
        return result

    def command(
        self,
        *,
        measured_force_n: float,
        tool_position_xyz_m: np.ndarray,
        tcp_quaternion_wxyz: np.ndarray,
        scenario: ScenarioSpec,
        surface_top_origin_z_m: float,
    ) -> PrimitivePoseCommand:
        force = float(measured_force_n)
        tool_position = np.asarray(tool_position_xyz_m, dtype=np.float64)
        if not math.isfinite(force) or force < 0.0:
            raise LowLevelControlError("measured force must be finite and nonnegative")
        if tool_position.shape != (3,) or not np.all(np.isfinite(tool_position)):
            raise LowLevelControlError("tool position must be a finite three-vector")

        cfg = self.pose_config
        point, tangent, normal, target_force, stiffness, damping = self._reference_geometry(
            scenario=scenario,
            surface_top_origin_z_m=surface_top_origin_z_m,
        )
        if self.compliance_parameter_sink is not None:
            self.compliance_parameter_sink(stiffness, damping)
        planar_error = point - tool_position
        planar_error -= float(np.dot(planar_error, normal)) * normal
        planar_error_norm = float(np.linalg.norm(planar_error))
        solver_audit = None

        if self._phase is PrimitiveExecutionPhase.HOVER_REPOSITION:
            hover_target = point + normal * cfg.hover_clearance_m
            hover_error = hover_target - tool_position
            if force > 0.2:
                position_step = normal * cfg.return_lift_step_m
                force_command = _manual_force_command(
                    normal_step_m=-cfg.return_lift_step_m,
                    normal_xyz=normal,
                    action_position_scale_m=cfg.action_position_scale_m,
                    mode="unexpected_hover_contact_lift",
                )
            else:
                position_step = self._bounded_vector(
                    cfg.hover_position_gain * hover_error,
                    cfg.max_hover_step_m,
                )
                force_command = _manual_force_command(
                    normal_step_m=0.0,
                    normal_xyz=normal,
                    action_position_scale_m=cfg.action_position_scale_m,
                    mode="hover_reposition",
                )
                if float(np.linalg.norm(hover_error)) <= cfg.hover_position_tolerance_m:
                    self._phase = PrimitiveExecutionPhase.CONTACT_ACQUISITION
                    self.force_controller.reset()
            tangent_step = position_step - float(np.dot(position_step, normal)) * normal

        elif self._phase is PrimitiveExecutionPhase.CONTACT_ACQUISITION:
            force_output = self.force_controller.command(
                measured_force_n=force,
                target_force_n=target_force,
                outward_normal_xyz=normal,
            )
            force_command, solver_audit = _force_command_from_output(force_output)
            if bool(getattr(force_command, "tangential_motion_allowed", True)):
                tangent_step = self._bounded_vector(
                    cfg.reference_error_gain * planar_error,
                    cfg.max_tangent_step_m,
                )
            else:
                tangent_step = np.zeros(3, dtype=np.float64)
            position_step = tangent_step - normal * force_command.normal_step_m
            if force >= cfg.contact_force_n and planar_error_norm <= cfg.reference_position_tolerance_m:
                self._contact_confirmation_count += 1
            else:
                self._contact_confirmation_count = 0
            if self._contact_confirmation_count >= cfg.contact_confirmation_steps:
                self._phase = PrimitiveExecutionPhase.TRACK

        elif self._phase is PrimitiveExecutionPhase.TRACK:
            controller_state = getattr(self.force_controller, "state", None)
            controller_state_value = getattr(controller_state, "value", controller_state)
            conditional_track_ready = (
                controller_state is None or str(controller_state_value) == "TRACK"
            )
            can_advance = (
                cfg.contact_force_n <= force < cfg.path_motion_headroom_force_n
                and planar_error_norm <= cfg.reference_position_tolerance_m
                and conditional_track_ready
            )
            if can_advance:
                if self._reference_index == len(self.reference.progress) - 1:
                    self._phase = PrimitiveExecutionPhase.RETURN_HOVER
                    self.force_controller.reset()
                else:
                    self._reference_index += 1
                    point, tangent, normal, target_force, stiffness, damping = self._reference_geometry(
                        scenario=scenario,
                        surface_top_origin_z_m=surface_top_origin_z_m,
                    )
                    if self.compliance_parameter_sink is not None:
                        self.compliance_parameter_sink(stiffness, damping)
                    planar_error = point - tool_position
                    planar_error -= float(np.dot(planar_error, normal)) * normal
                    planar_error_norm = float(np.linalg.norm(planar_error))
            if self._phase is PrimitiveExecutionPhase.TRACK:
                force_output = self.force_controller.command(
                    measured_force_n=force,
                    target_force_n=target_force,
                    outward_normal_xyz=normal,
                )
                force_command, solver_audit = _force_command_from_output(force_output)
                if (
                    force < cfg.contact_force_n
                    or force >= cfg.path_motion_headroom_force_n
                    or force_command.mode == "headroom_lift"
                    or not bool(getattr(force_command, "tangential_motion_allowed", True))
                ):
                    tangent_step = np.zeros(3, dtype=np.float64)
                else:
                    tangent_step = self._bounded_vector(
                        cfg.reference_error_gain * planar_error,
                        cfg.max_tangent_step_m,
                    )
                    tangent_step -= float(np.dot(tangent_step, normal)) * normal
                position_step = tangent_step - normal * force_command.normal_step_m
            else:
                tangent_step = np.zeros(3, dtype=np.float64)
                position_step = normal * cfg.return_lift_step_m
                force_command = _manual_force_command(
                    normal_step_m=-cfg.return_lift_step_m,
                    normal_xyz=normal,
                    action_position_scale_m=cfg.action_position_scale_m,
                    mode="primitive_return_hover",
                )

        elif self._phase is PrimitiveExecutionPhase.RETURN_HOVER:
            hover_target = point + normal * cfg.hover_clearance_m
            hover_error = hover_target - tool_position
            position_step = self._bounded_vector(
                cfg.hover_position_gain * hover_error,
                cfg.max_hover_step_m,
            )
            tangent_step = position_step - float(np.dot(position_step, normal)) * normal
            force_command = _manual_force_command(
                normal_step_m=-max(0.0, float(np.dot(position_step, normal))),
                normal_xyz=normal,
                action_position_scale_m=cfg.action_position_scale_m,
                mode="primitive_return_hover",
            )
            if force <= 0.2 and float(np.linalg.norm(hover_error)) <= cfg.hover_position_tolerance_m:
                self._phase = PrimitiveExecutionPhase.COMPLETE

        else:
            position_step = np.zeros(3, dtype=np.float64)
            tangent_step = np.zeros(3, dtype=np.float64)
            force_command = _manual_force_command(
                normal_step_m=0.0,
                normal_xyz=normal,
                action_position_scale_m=cfg.action_position_scale_m,
                mode="primitive_complete",
            )

        orientation_tangent = tangent * float(
            self.reference.motion_sign[self._reference_index]
        )
        desired_rotation = desired_tool_rotation(orientation_tangent, normal)
        rotation_delta = root_aligned_rotation_delta_xyz(
            tcp_quaternion_wxyz,
            desired_rotation,
            max_step_rad=cfg.max_rotation_step_rad,
        )
        normalized_position = np.clip(
            position_step / cfg.action_position_scale_m,
            -1.0,
            1.0,
        )
        normalized_rotation = np.clip(
            rotation_delta / cfg.action_rotation_scale_rad,
            -1.0,
            1.0,
        )
        return PrimitivePoseCommand(
            normalized_pose_action=np.concatenate((normalized_position, normalized_rotation)),
            force_command=force_command,
            phase=self._phase,
            reference_index=self._reference_index,
            reference_progress=float(self.reference.progress[self._reference_index]),
            target_force_n=target_force,
            stiffness_n_m=stiffness,
            damping_n_s_m=damping,
            target_point_xyz_m=point,
            tangent_xyz=tangent,
            orientation_tangent_xyz=orientation_tangent,
            outward_normal_xyz=normal,
            position_step_xyz_m=position_step,
            tangent_step_xyz_m=tangent_step,
            rotation_delta_xyz_rad=rotation_delta,
            planar_reference_error_m=planar_error_norm,
            solver_audit=solver_audit,
            completed=self._phase is PrimitiveExecutionPhase.COMPLETE,
        )
