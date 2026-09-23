"""V5.4 path/force supervisor with bounded, escapable recovery phases."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .low_level_control import LowLevelControlError
from .low_level_control_v5_3 import ConditionalControllerState
from .low_level_control_v5_4 import (
    ConditionalCausalNormalForceControllerV54,
    ConditionalForceControllerConfigV54,
)
from .path_pose_control import PathPoseControllerConfig
from .path_pose_control_v5_3_r2 import (
    GeometricRecoveryConfigV53R2,
    GeometricRecoveryPhaseV53R2,
    PathForcePoseCommandV53R2,
)
from .path_pose_control_v5_3_r3 import CausalPathForcePoseControllerV53R3
from .scenarios import surface_normal


@dataclass(frozen=True)
class GeometricRecoveryConfigV54(GeometricRecoveryConfigV53R2):
    # The V5.3-r3 plant audit showed a detached-force floor near 0.8 N.  The
    # earlier 0.2-N release condition was therefore not live in many episodes.
    release_force_n: float = 1.25
    maximum_lift_samples: int = 100
    bounded_exit_contact_force_n: float = 3.0

    def validate(self) -> None:
        super().validate()
        if not isinstance(self.maximum_lift_samples, int) or isinstance(
            self.maximum_lift_samples, bool
        ) or self.maximum_lift_samples <= 0:
            raise LowLevelControlError("V5.4 maximum LIFT samples must be positive")
        if (
            not math.isfinite(float(self.bounded_exit_contact_force_n))
            or self.bounded_exit_contact_force_n <= 0.0
        ):
            raise LowLevelControlError("V5.4 bounded-exit force must be positive")
        if not self.release_force_n < self.bounded_exit_contact_force_n:
            raise LowLevelControlError("V5.4 soft release must precede bounded exit")


class CausalPathForcePoseControllerV54(CausalPathForcePoseControllerV53R3):
    """V5.4 supervisor.

    CONFIRM requires consecutive force-state confirmation but no impossible
    planar-error condition: HOVER_REPOSITION already handles geometric
    alignment, and the TRACK transition remains a zero-tangent/zero-progress
    hold.  LIFT exits normally at the soft-release criterion or, after a fixed
    duration, enters HOVER_REPOSITION whenever force is below the benchmark's
    contact threshold.  Safety states still preempt task progression.
    """

    def __init__(
        self,
        *,
        force_config: ConditionalForceControllerConfigV54 = ConditionalForceControllerConfigV54(),
        path_config: PathPoseControllerConfig = PathPoseControllerConfig(),
        recovery_config: GeometricRecoveryConfigV54 = GeometricRecoveryConfigV54(),
    ) -> None:
        force_config.validate()
        recovery_config.validate()
        super().__init__(
            force_config=force_config,
            path_config=path_config,
            recovery_config=recovery_config,
        )
        self.force_config = force_config
        self.recovery_config = recovery_config
        self.force_controller = ConditionalCausalNormalForceControllerV54(force_config)
        self.reset()

    def reset(self) -> None:
        super().reset()
        self._previous_tool_position_xyz_m: np.ndarray | None = None
        self._previous_normal_command_m = 0.0
        self._v54_confirm_count = 0
        self._last_output_phase: str | None = None
        self._phase_elapsed_samples = 0

    def _instantaneous_normal_velocity(
        self,
        *,
        tool_position_xyz_m: np.ndarray,
        scenario,
        path,
        surface_top_origin_z_m: float,
    ) -> float:
        tool = np.asarray(tool_position_xyz_m, dtype=np.float64)
        if self._previous_tool_position_xyz_m is None:
            return 0.0
        relative = tool.copy()
        relative[2] -= float(surface_top_origin_z_m)
        projection = path.project(relative)
        normal = np.asarray(
            surface_normal(scenario, float(projection.point_xyz[0])),
            dtype=np.float64,
        )
        normal /= np.linalg.norm(normal)
        return float(
            np.dot(
                (tool - self._previous_tool_position_xyz_m) / self.path_config.dt_s,
                normal,
            )
        )

    def _clearance_audit(self, **kwargs) -> tuple[float, bool, bool]:
        tool = np.asarray(kwargs["tool_position_xyz_m"], dtype=np.float64)
        target, _tangent, normal = self._reference_geometry(
            progress=float(self._committed_progress),
            scenario=kwargs["scenario"],
            path=kwargs["path"],
            surface_top_origin_z_m=float(kwargs["surface_top_origin_z_m"]),
        )
        clearance = float(np.dot(tool - target, normal))
        released = float(kwargs["measured_force_n"]) <= float(
            self.recovery_config.release_force_n
        )
        high_enough = clearance >= (
            float(self.recovery_config.hover_clearance_m)
            - float(self.recovery_config.hover_position_tolerance_m)
        )
        return clearance, bool(released), bool(high_enough)

    def command(self, **kwargs) -> PathForcePoseCommandV53R2:
        original_phase = self._phase
        progress_before = float(self._committed_progress)
        elapsed_before = (
            self._phase_elapsed_samples
            if self._last_output_phase == original_phase.value
            else 0
        )
        clearance, release_met, high_enough = self._clearance_audit(**kwargs)
        normal_velocity = self._instantaneous_normal_velocity(
            tool_position_xyz_m=kwargs["tool_position_xyz_m"],
            scenario=kwargs["scenario"],
            path=kwargs["path"],
            surface_top_origin_z_m=kwargs["surface_top_origin_z_m"],
        )
        previous_command = float(self._previous_normal_command_m)
        self.force_controller.set_kinematic_context(
            normal_velocity_m_s=normal_velocity,
            previous_normal_command_m=previous_command,
        )

        bounded_lift_exit = bool(
            original_phase is GeometricRecoveryPhaseV53R2.LIFT
            and elapsed_before >= self.recovery_config.maximum_lift_samples
            and float(kwargs["measured_force_n"])
            <= self.recovery_config.bounded_exit_contact_force_n
        )
        if bounded_lift_exit:
            self._phase = GeometricRecoveryPhaseV53R2.HOVER_REPOSITION

        command = super().command(**kwargs)
        force_state = command.force_command.controller_state

        # Replace the r2 planar-error-dependent CONFIRM latch with a strictly
        # force-confirmed latch.  Entry into TRACK remains a geometric hold.
        if original_phase is GeometricRecoveryPhaseV53R2.CONFIRM:
            if force_state == ConditionalControllerState.TRACK.value:
                self._v54_confirm_count += 1
                if self._v54_confirm_count >= self.recovery_config.track_confirm_samples:
                    command = self._zero_tangent_and_restore_progress(
                        command,
                        committed_progress=progress_before,
                        phase=GeometricRecoveryPhaseV53R2.TRACK,
                        reason="v54_force_confirm_complete_to_track_hold",
                    )
                else:
                    command = self._zero_tangent_and_restore_progress(
                        command,
                        committed_progress=progress_before,
                        phase=GeometricRecoveryPhaseV53R2.CONFIRM,
                        reason="v54_force_confirm_hold",
                    )
            else:
                self._v54_confirm_count = 0
        elif command.geometric_phase == GeometricRecoveryPhaseV53R2.CONFIRM.value:
            self._v54_confirm_count = 1
        elif command.geometric_phase != GeometricRecoveryPhaseV53R2.TRACK.value:
            self._v54_confirm_count = 0

        if bounded_lift_exit:
            command = type(command)(
                **{
                    **command.__dict__,
                    "geometric_transition_reason": (
                        "bounded_lift_exit_below_contact_threshold"
                    ),
                }
            )

        output_phase = command.geometric_phase
        if output_phase == self._last_output_phase:
            self._phase_elapsed_samples += 1
        else:
            self._phase_elapsed_samples = 1
            self._last_output_phase = output_phase

        self._previous_tool_position_xyz_m = np.asarray(
            kwargs["tool_position_xyz_m"], dtype=np.float64
        ).copy()
        self._previous_normal_command_m = float(
            command.force_command.clipped_normal_step_m
        )

        diagnostic = self._diagnostic_rows[-1]
        diagnostic.update(
            {
                "normal_velocity_m_s": float(normal_velocity),
                "previous_normal_command_m": previous_command,
                "phase_elapsed_samples": int(self._phase_elapsed_samples),
                "clearance_m": float(clearance),
                "release_criterion_met": bool(release_met),
                "high_enough_criterion_met": bool(high_enough),
                "bounded_lift_exit_triggered": bounded_lift_exit,
                "predictive_brake_triggered": bool(
                    getattr(command.force_command, "predictive_brake_triggered", False)
                ),
                "brake_condition_count": int(
                    getattr(command.force_command, "brake_condition_count", 0)
                ),
                "v54_force_confirmation_count": int(self._v54_confirm_count),
                "geometric_confirmation_count": int(self._v54_confirm_count),
                "geometric_phase_before": original_phase.value,
                "geometric_phase": command.geometric_phase,
                "geometric_reason": command.geometric_transition_reason,
                "committed_progress": float(command.committed_progress),
                "instantaneous_progress": float(command.instantaneous_progress),
                "commanded_progress": float(command.commanded_progress),
                "tangent_step_norm_m": float(
                    np.linalg.norm(command.tangent_step_xyz_m)
                ),
                "tangential_motion_permitted": bool(
                    diagnostic["tangential_motion_permitted"]
                    and command.geometric_phase
                    == GeometricRecoveryPhaseV53R2.TRACK.value
                    and command.force_command.controller_state
                    == ConditionalControllerState.TRACK.value
                ),
            }
        )
        return command
