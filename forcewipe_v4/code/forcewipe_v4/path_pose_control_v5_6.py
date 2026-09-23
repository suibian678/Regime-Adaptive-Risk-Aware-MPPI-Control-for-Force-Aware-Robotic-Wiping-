"""V5.6 recovery-stabilization and TRACK-compensation supervisor.

The two mechanisms are state-mutually-exclusive:

* recovery stabilization is active only while geometric state is non-TRACK;
* low-force compensation is eligible only when geometric and force states are
  both TRACK.

The inherited three-sample contact confirmation, HEADROOM thresholds,
predictive brake, Cartesian limits, and three-cycle recovery cap are unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .low_level_control import LowLevelControlError
from .low_level_control_v5_3 import ConditionalControllerState
from .low_level_control_v5_6 import (
    ConditionalCausalNormalForceControllerV56,
    ConditionalForceControllerConfigV56,
)
from .path_pose_control import PathPoseControllerConfig
from .path_pose_control_v5_3_r2 import (
    GeometricRecoveryPhaseV53R2,
    PathForcePoseCommandV53R2,
)
from .path_pose_control_v5_5 import GeometricRecoveryConfigV55
from .path_pose_control_v5_5_r2 import CausalPathForcePoseControllerV55R2


@dataclass(frozen=True)
class GeometricRecoveryConfigV56(GeometricRecoveryConfigV55):
    """Require a fixed non-tangential dwell after recovered contact confirms."""

    recovery_stabilization_samples: int = 20

    def validate(self) -> None:
        super().validate()
        value = self.recovery_stabilization_samples
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise LowLevelControlError(
                "V5.6 recovery stabilization samples must be a positive integer"
            )


class CausalPathForcePoseControllerV56(CausalPathForcePoseControllerV55R2):
    """One predeclared V5.6 controller for DEV-only evaluation."""

    def __init__(
        self,
        *,
        force_config: ConditionalForceControllerConfigV56 = ConditionalForceControllerConfigV56(),
        path_config: PathPoseControllerConfig = PathPoseControllerConfig(),
        recovery_config: GeometricRecoveryConfigV56 = GeometricRecoveryConfigV56(),
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
        self.force_controller = ConditionalCausalNormalForceControllerV56(force_config)
        self.reset()

    def reset(self) -> None:
        super().reset()
        self._v56_recovery_stabilization_count = 0

    def _annotate_v56(
        self,
        command: PathForcePoseCommandV53R2,
        *,
        recovery_active: bool,
    ) -> None:
        force = command.force_command
        row = self._diagnostic_rows[-1]
        tracking_applied = bool(
            getattr(force, "tracking_compensation_applied", False)
        )
        row.update(
            {
                "v56_recovery_stabilization_active": bool(recovery_active),
                "v56_recovery_stabilization_count": int(
                    self._v56_recovery_stabilization_count
                ),
                "v56_recovery_stabilization_required": int(
                    self.recovery_config.recovery_stabilization_samples
                ),
                "v56_tracking_compensation_eligible": bool(
                    getattr(force, "tracking_compensation_eligible", False)
                ),
                "v56_tracking_compensation_latched": bool(
                    getattr(force, "tracking_compensation_latched", False)
                ),
                "v56_tracking_compensation_applied": tracking_applied,
                "v56_tracking_compensation_step_m": float(
                    getattr(force, "tracking_compensation_step_m", 0.0)
                ),
                "v56_mechanism_overlap_violation": bool(
                    recovery_active and tracking_applied
                ),
            }
        )

    def command(self, **kwargs) -> PathForcePoseCommandV53R2:
        phase_before = self._phase
        progress_before = float(self._committed_progress)
        recovery_cycle_before = int(self._recovery_cycle_count)
        self.force_controller.set_geometric_track_context(
            geometric_track=bool(
                phase_before is GeometricRecoveryPhaseV53R2.TRACK
                and not self._safe_hold_active
            )
        )
        command = super().command(**kwargs)

        recovered_transition = bool(
            recovery_cycle_before > 0
            and phase_before is GeometricRecoveryPhaseV53R2.CONFIRM
            and command.geometric_phase == GeometricRecoveryPhaseV53R2.TRACK.value
            and command.force_command.controller_state
            == ConditionalControllerState.TRACK.value
        )
        recovery_active = False
        if recovered_transition:
            self._v56_recovery_stabilization_count += 1
            required = int(self.recovery_config.recovery_stabilization_samples)
            if self._v56_recovery_stabilization_count < required:
                recovery_active = True
                command = self._zero_tangent_and_restore_progress(
                    command,
                    committed_progress=progress_before,
                    phase=GeometricRecoveryPhaseV53R2.CONFIRM,
                    reason="v56_recovery_stabilization_hold",
                )
                self._v54_confirm_count = int(self.force_config.contact_confirm_samples)
                self._track_confirmation_count = int(
                    self.force_config.contact_confirm_samples
                )
                self._last_output_phase = GeometricRecoveryPhaseV53R2.CONFIRM.value
                self._phase_elapsed_samples += 1
                row = self._diagnostic_rows[-1]
                row.update(
                    {
                        "geometric_phase": command.geometric_phase,
                        "geometric_reason": command.geometric_transition_reason,
                        "committed_progress": float(command.committed_progress),
                        "commanded_progress": float(command.commanded_progress),
                        "tangent_step_norm_m": 0.0,
                        "tangential_motion_permitted": False,
                        "v54_force_confirmation_count": int(
                            self._v54_confirm_count
                        ),
                        "geometric_confirmation_count": int(
                            self._v54_confirm_count
                        ),
                        "phase_elapsed_samples": int(self._phase_elapsed_samples),
                    }
                )
            else:
                # The inherited transition hold remains in force.  Tangential
                # authority can begin only on the following stable TRACK sample.
                self._v56_recovery_stabilization_count = required
        elif command.geometric_phase != GeometricRecoveryPhaseV53R2.CONFIRM.value:
            self._v56_recovery_stabilization_count = 0
        elif command.force_command.controller_state != ConditionalControllerState.TRACK.value:
            self._v56_recovery_stabilization_count = 0

        self._annotate_v56(command, recovery_active=recovery_active)
        if self._diagnostic_rows[-1]["v56_mechanism_overlap_violation"]:
            raise LowLevelControlError("V5.6 recovery and TRACK mechanisms overlapped")
        return replace(command, geometric_transition_reason=command.geometric_transition_reason)
