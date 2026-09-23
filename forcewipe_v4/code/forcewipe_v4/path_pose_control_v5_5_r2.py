"""V5.5-r2 geometric brake gate.

V5.5-r1 is preserved as a superseded, pre-execution definition.  This revision
changes only the geometric handling of a low-level track-preserving brake: a
brake may preserve an already active geometric TRACK phase, but it may not
promote CONTACT_ACQUIRE or CONFIRM into TRACK.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .low_level_control import LowLevelControlError
from .path_pose_control_v5_3_r2 import (
    GeometricRecoveryPhaseV53R2,
    PathForcePoseCommandV53R2,
)
from .path_pose_control_v5_5 import CausalPathForcePoseControllerV55


class CausalPathForcePoseControllerV55R2(CausalPathForcePoseControllerV55):
    """Prevent a force-state brake from bypassing geometric confirmation."""

    def _rewrite_diagnostic(self, *args, **kwargs) -> None:
        # The inherited V5.5 implementation recomputed permission from only the
        # final state.  Preserve the stricter causal permission produced by the
        # r3 layer so a CONFIRM->TRACK transition hold is never logged as a
        # tangent-authorized sample.
        inherited_permission = bool(
            self._diagnostic_rows[-1].get("tangential_motion_permitted", False)
        )
        super()._rewrite_diagnostic(*args, **kwargs)
        row = self._diagnostic_rows[-1]
        row["tangential_motion_permitted"] = bool(
            inherited_permission and not row.get("track_preserving_brake", False)
        )

    def command(self, **kwargs) -> PathForcePoseCommandV53R2:
        phase_before = self._phase
        progress_before = float(self._committed_progress)
        v54_confirm_before = int(self._v54_confirm_count)
        legacy_confirm_before = int(self._track_confirmation_count)
        last_output_before = self._last_output_phase
        elapsed_before = int(self._phase_elapsed_samples)

        command = super().command(**kwargs)
        braking = bool(
            getattr(command.force_command, "track_preserving_brake", False)
        )
        if not braking:
            return command

        if phase_before is GeometricRecoveryPhaseV53R2.TRACK:
            # V5.5-r1 already implements the desired TRACK hold.
            return command
        if phase_before not in {
            GeometricRecoveryPhaseV53R2.CONFIRM,
            GeometricRecoveryPhaseV53R2.CONTACT_ACQUIRE,
        }:
            raise LowLevelControlError(
                "track-preserving brake reached an ineligible geometric phase"
            )

        # The inherited call may have temporarily incremented confirmation and
        # promoted the phase to TRACK.  Restore the exact pre-call confirmation
        # state: a braking sample is a hold, not a valid confirmation sample.
        self._v54_confirm_count = v54_confirm_before
        self._track_confirmation_count = legacy_confirm_before
        reason = f"track_brake_hold_in_{phase_before.value.lower()}"
        command = self._zero_tangent_and_restore_progress(
            command,
            committed_progress=progress_before,
            phase=phase_before,
            reason=reason,
        )
        self._last_output_phase = phase_before.value
        self._phase_elapsed_samples = (
            elapsed_before + 1
            if last_output_before == phase_before.value
            else 1
        )

        row = self._diagnostic_rows[-1]
        row.update(
            {
                "geometric_phase_before": phase_before.value,
                "geometric_phase": command.geometric_phase,
                "geometric_reason": reason,
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
                "tangential_motion_permitted": False,
                "v54_force_confirmation_count": int(self._v54_confirm_count),
                "geometric_confirmation_count": int(self._v54_confirm_count),
                "phase_elapsed_samples": int(self._phase_elapsed_samples),
            }
        )
        return replace(command, geometric_transition_reason=reason)
