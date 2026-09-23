"""V5.5 bounded workspace-retreat and contact-reacquisition supervisor."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np

from .low_level_control import LowLevelControlError
from .low_level_control_v5_3 import ConditionalControllerState
from .low_level_control_v5_5 import (
    ConditionalCausalNormalForceControllerV55,
    ConditionalForceControllerConfigV55,
)
from .path_pose_control import PathPoseControllerConfig
from .path_pose_control_v5_3_r2 import (
    GeometricRecoveryPhaseV53R2,
    PathForcePoseCommandV53R2,
)
from .path_pose_control_v5_4 import (
    CausalPathForcePoseControllerV54,
    GeometricRecoveryConfigV54,
)


SAFE_HOLD_PHASE = "SAFE_HOLD"


@dataclass(frozen=True)
class GeometricRecoveryConfigV55(GeometricRecoveryConfigV54):
    recovery_retreat_distance_m: float = 0.03
    maximum_hover_reposition_samples: int = 100
    maximum_contact_acquire_samples: int = 250
    maximum_recovery_cycles: int = 3

    def validate(self) -> None:
        super().validate()
        counts = (
            self.maximum_hover_reposition_samples,
            self.maximum_contact_acquire_samples,
            self.maximum_recovery_cycles,
        )
        if not all(
            isinstance(value, int) and not isinstance(value, bool) and value > 0
            for value in counts
        ):
            raise LowLevelControlError("V5.5 recovery bounds must be positive integers")
        if (
            not math.isfinite(float(self.recovery_retreat_distance_m))
            or self.recovery_retreat_distance_m <= 0.0
        ):
            raise LowLevelControlError("V5.5 retreat distance must be positive")


class CausalPathForcePoseControllerV55(CausalPathForcePoseControllerV54):
    """Bound each recovery phase and preserve TRACK during predictive braking.

    Each geometric recovery cycle retreats the reference by 3 cm before lift
    and hover repositioning.  CONTACT_ACQUIRE, LIFT, and HOVER_REPOSITION have
    explicit per-attempt bounds.  After three unsuccessful cycles the
    controller enters an auditable SAFE_HOLD: zero Cartesian authority below
    headroom and inherited outward HEADROOM authority otherwise.
    """

    def __init__(
        self,
        *,
        force_config: ConditionalForceControllerConfigV55 = ConditionalForceControllerConfigV55(),
        path_config: PathPoseControllerConfig = PathPoseControllerConfig(),
        recovery_config: GeometricRecoveryConfigV55 = GeometricRecoveryConfigV55(),
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
        self.force_controller = ConditionalCausalNormalForceControllerV55(force_config)
        self.reset()

    def reset(self) -> None:
        super().reset()
        self._recovery_cycle_count = 0
        self._recovery_anchor_progress: float | None = None
        self._pre_recovery_progress: float | None = None
        self._safe_hold_active = False
        self._safe_hold_samples = 0

    def _start_recovery_cycle(self, *, path) -> None:
        self._recovery_cycle_count += 1
        self._pre_recovery_progress = float(self._committed_progress)
        retreat_fraction = float(self.recovery_config.recovery_retreat_distance_m) / float(
            path.total_length
        )
        self._recovery_anchor_progress = max(
            0.0, float(self._committed_progress) - retreat_fraction
        )
        self._committed_progress = float(self._recovery_anchor_progress)

    def _rewrite_diagnostic(
        self,
        command: PathForcePoseCommandV53R2,
        *,
        phase_before: str,
        forced_reason: str | None,
        bounded_hover_exit: bool,
        bounded_acquire_exit: bool,
    ) -> None:
        row = self._diagnostic_rows[-1]
        row.update(
            {
                "geometric_phase_before": phase_before,
                "geometric_phase": command.geometric_phase,
                "geometric_reason": (
                    command.geometric_transition_reason
                    if forced_reason is None
                    else forced_reason
                ),
                "committed_progress": float(command.committed_progress),
                "commanded_progress": float(command.commanded_progress),
                "tangent_step_norm_m": float(
                    np.linalg.norm(command.tangent_step_xyz_m)
                ),
                "cartesian_position_action_norm": float(
                    np.linalg.norm(command.normalized_pose_action[:3])
                ),
                "rotation_action_norm": float(
                    np.linalg.norm(command.normalized_pose_action[3:])
                ),
                "tangential_motion_permitted": bool(
                    command.geometric_phase == GeometricRecoveryPhaseV53R2.TRACK.value
                    and command.force_command.controller_state
                    == ConditionalControllerState.TRACK.value
                    and not getattr(
                        command.force_command, "track_preserving_brake", False
                    )
                ),
                "track_preserving_brake": bool(
                    getattr(command.force_command, "track_preserving_brake", False)
                ),
                "cancelled_previous_downpress_m": float(
                    getattr(
                        command.force_command,
                        "cancelled_previous_downpress_m",
                        0.0,
                    )
                ),
                "recovery_cycle_count": int(self._recovery_cycle_count),
                "recovery_anchor_progress": self._recovery_anchor_progress,
                "pre_recovery_progress": self._pre_recovery_progress,
                "bounded_hover_exit_triggered": bool(bounded_hover_exit),
                "bounded_contact_acquire_exit_triggered": bool(
                    bounded_acquire_exit
                ),
                "recovery_exhausted_safe_hold": bool(self._safe_hold_active),
                "safe_hold_samples": int(self._safe_hold_samples),
                "phase_elapsed_samples": int(self._phase_elapsed_samples),
            }
        )

    def _replace_with_safe_hold_output(
        self,
        command: PathForcePoseCommandV53R2,
        *,
        progress_before: float,
    ) -> PathForcePoseCommandV53R2:
        """Replace the current sample before it can reach the plant.

        This helper is also used on the exact transition sample that exhausts
        the recovery budget.  Consequently, a below-headroom LIFT command
        cannot leak through merely because SAFE_HOLD was entered after the
        inherited supervisor had already constructed that command.
        """
        force = command.force_command
        preserve_outward_headroom = (
            force.controller_state == ConditionalControllerState.HEADROOM.value
            and float(force.normal_step_m) < 0.0
        )
        if preserve_outward_headroom:
            position = command.normalized_pose_action[:3]
        else:
            position = np.zeros(3, dtype=np.float64)
            force = replace(
                force,
                normalized_position_action=np.zeros(3, dtype=np.float64),
                normal_step_m=0.0,
                proportional_step_m=0.0,
                integral_step_m=0.0,
                derivative_step_m=0.0,
                raw_normal_step_m=0.0,
                clipped_normal_step_m=0.0,
                tangential_motion_allowed=False,
            )
        self._committed_progress = progress_before
        self._phase = GeometricRecoveryPhaseV53R2.HOVER_REPOSITION
        self._last_output_phase = SAFE_HOLD_PHASE
        return replace(
            command,
            normalized_pose_action=np.concatenate(
                (np.asarray(position, dtype=np.float64), np.zeros(3, dtype=np.float64))
            ),
            force_command=force,
            progress=progress_before,
            commanded_progress=progress_before,
            tangent_step_xyz_m=np.zeros(3, dtype=np.float64),
            rotation_delta_xyz_rad=np.zeros(3, dtype=np.float64),
            geometric_phase=SAFE_HOLD_PHASE,
            geometric_transition_reason="recovery_attempt_budget_exhausted_safe_hold",
            committed_progress=progress_before,
        )

    def _safe_hold_command(self, *, phase_before: str, kwargs) -> PathForcePoseCommandV53R2:
        progress_before = float(self._committed_progress)
        # Use the inherited path solely to retain global HEADROOM preemption.
        # Its Cartesian result is replaced below before it reaches the plant.
        command = super().command(**kwargs)
        self._safe_hold_samples += 1
        self._phase_elapsed_samples = self._safe_hold_samples
        command = self._replace_with_safe_hold_output(
            command,
            progress_before=progress_before,
        )
        self._rewrite_diagnostic(
            command,
            phase_before=phase_before,
            forced_reason="recovery_attempt_budget_exhausted_safe_hold",
            bounded_hover_exit=False,
            bounded_acquire_exit=False,
        )
        return command

    def command(self, **kwargs) -> PathForcePoseCommandV53R2:
        phase_before = self._phase.value
        progress_before = float(self._committed_progress)
        if self._safe_hold_active:
            return self._safe_hold_command(phase_before=SAFE_HOLD_PHASE, kwargs=kwargs)

        elapsed = (
            self._phase_elapsed_samples
            if self._last_output_phase == phase_before
            else 0
        )
        bounded_hover_exit = False
        bounded_acquire_exit = False
        forced_reason: str | None = None
        cycle_started_before_super = False

        if (
            self._phase is GeometricRecoveryPhaseV53R2.HOVER_REPOSITION
            and elapsed >= self.recovery_config.maximum_hover_reposition_samples
        ):
            self._phase = GeometricRecoveryPhaseV53R2.CONTACT_ACQUIRE
            self.force_controller.reset()
            bounded_hover_exit = True
            forced_reason = "bounded_hover_timeout_to_contact_acquire"
        elif (
            self._phase is GeometricRecoveryPhaseV53R2.CONTACT_ACQUIRE
            and elapsed >= self.recovery_config.maximum_contact_acquire_samples
        ):
            if self._recovery_cycle_count >= self.recovery_config.maximum_recovery_cycles:
                self._safe_hold_active = True
                return self._safe_hold_command(
                    phase_before=phase_before,
                    kwargs=kwargs,
                )
            self._start_recovery_cycle(path=kwargs["path"])
            self._phase = GeometricRecoveryPhaseV53R2.LIFT
            self.force_controller.reset()
            cycle_started_before_super = True
            bounded_acquire_exit = True
            forced_reason = "bounded_contact_acquire_timeout_to_lift"

        command = super().command(**kwargs)
        entered_lift = bool(
            command.geometric_phase == GeometricRecoveryPhaseV53R2.LIFT.value
            and phase_before != GeometricRecoveryPhaseV53R2.LIFT.value
        )
        if entered_lift and not cycle_started_before_super:
            if self._recovery_cycle_count >= self.recovery_config.maximum_recovery_cycles:
                self._safe_hold_active = True
                self._safe_hold_samples = 1
                self._phase_elapsed_samples = 1
                command = self._replace_with_safe_hold_output(
                    command,
                    progress_before=progress_before,
                )
                forced_reason = "recovery_attempt_budget_exhausted_safe_hold"
            else:
                self._start_recovery_cycle(path=kwargs["path"])

        if entered_lift and not self._safe_hold_active:
            command = replace(
                command,
                progress=float(self._committed_progress),
                commanded_progress=float(self._committed_progress),
                committed_progress=float(self._committed_progress),
            )

        if getattr(command.force_command, "track_preserving_brake", False):
            command = self._zero_tangent_and_restore_progress(
                command,
                committed_progress=progress_before,
                phase=GeometricRecoveryPhaseV53R2.TRACK,
                reason="track_preserving_predictive_brake_hold",
            )
            forced_reason = "track_preserving_predictive_brake_hold"

        if forced_reason is not None:
            command = replace(command, geometric_transition_reason=forced_reason)

        self._rewrite_diagnostic(
            command,
            phase_before=phase_before,
            forced_reason=forced_reason,
            bounded_hover_exit=bounded_hover_exit,
            bounded_acquire_exit=bounded_acquire_exit,
        )
        return command
