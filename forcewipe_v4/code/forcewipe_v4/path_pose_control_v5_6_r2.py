"""V5.6-preauth-r2 supervisor with an explicit stabilization phase."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .low_level_control_v5_3 import ConditionalControllerState
from .low_level_control_v5_6 import ConditionalForceControllerConfigV56
from .low_level_control_v5_6_r2 import ConditionalCausalNormalForceControllerV56R2
from .path_pose_control import PathPoseControllerConfig
from .path_pose_control_v5_3_r2 import (
    GeometricRecoveryPhaseV53R2,
    PathForcePoseCommandV53R2,
)
from .path_pose_control_v5_6 import GeometricRecoveryConfigV56
from .path_pose_control_v5_5_r2 import CausalPathForcePoseControllerV55R2


RECOVERY_STABILIZE_PHASE = "RECOVERY_STABILIZE"


class CausalPathForcePoseControllerV56R2(CausalPathForcePoseControllerV55R2):
    """Separate recovery stabilization from CONFIRM and stable TRACK control."""

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
        self.force_controller = ConditionalCausalNormalForceControllerV56R2(force_config)
        self.reset()

    def reset(self) -> None:
        super().reset()
        self._v56_stabilizing = False
        self._v56_recovery_stabilization_count = 0

    def _annotate(self, command: PathForcePoseCommandV53R2) -> None:
        force = command.force_command
        recovery_active = bool(
            command.geometric_phase == RECOVERY_STABILIZE_PHASE
        )
        tracking_applied = bool(
            getattr(force, "tracking_compensation_applied", False)
        )
        row = self._diagnostic_rows[-1]
        row.update(
            {
                "v56_recovery_stabilization_active": recovery_active,
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
                # This is evidence, not an exception: the one-shot finalizer
                # classifies any nonzero count as completed_scientific_fail.
                "v56_mechanism_overlap_violation": bool(
                    recovery_active and tracking_applied
                ),
            }
        )

    def _stabilization_hold(
        self,
        command: PathForcePoseCommandV53R2,
        *,
        progress_before: float,
        phase_before_label: str,
    ) -> PathForcePoseCommandV53R2:
        command = self._zero_tangent_and_restore_progress(
            command,
            committed_progress=progress_before,
            phase=GeometricRecoveryPhaseV53R2.CONFIRM,
            reason="v56_recovery_stabilization_hold",
        )
        command = replace(
            command,
            geometric_phase=RECOVERY_STABILIZE_PHASE,
            geometric_transition_reason="v56_recovery_stabilization_hold",
        )
        self._v54_confirm_count = int(self.force_config.contact_confirm_samples)
        self._track_confirmation_count = int(self.force_config.contact_confirm_samples)
        self._last_output_phase = RECOVERY_STABILIZE_PHASE
        self._phase_elapsed_samples = int(self._v56_recovery_stabilization_count)
        row = self._diagnostic_rows[-1]
        row.update(
            {
                "geometric_phase_before": phase_before_label,
                "geometric_phase": RECOVERY_STABILIZE_PHASE,
                "geometric_reason": "v56_recovery_stabilization_hold",
                "committed_progress": float(command.committed_progress),
                "commanded_progress": float(command.commanded_progress),
                "tangent_step_norm_m": 0.0,
                "tangential_motion_permitted": False,
                "v54_force_confirmation_count": int(self._v54_confirm_count),
                "geometric_confirmation_count": int(self._v54_confirm_count),
                "phase_elapsed_samples": int(self._phase_elapsed_samples),
            }
        )
        return command

    def command(self, **kwargs) -> PathForcePoseCommandV53R2:
        internal_phase_before = self._phase
        phase_before_label = (
            RECOVERY_STABILIZE_PHASE
            if self._v56_stabilizing
            else internal_phase_before.value
        )
        progress_before = float(self._committed_progress)
        recovery_cycle_before = int(self._recovery_cycle_count)
        self.force_controller.set_geometric_track_context(
            geometric_track=bool(
                not self._v56_stabilizing
                and internal_phase_before is GeometricRecoveryPhaseV53R2.TRACK
                and not self._safe_hold_active
            )
        )
        command = super().command(**kwargs)

        stable_force = bool(
            command.force_command.controller_state
            == ConditionalControllerState.TRACK.value
        )
        transition_candidate = bool(
            recovery_cycle_before > 0
            and internal_phase_before is GeometricRecoveryPhaseV53R2.CONFIRM
            and command.geometric_phase == GeometricRecoveryPhaseV53R2.TRACK.value
            and stable_force
        )
        required = int(self.recovery_config.recovery_stabilization_samples)

        if transition_candidate:
            self._v56_stabilizing = True
            self._v56_recovery_stabilization_count += 1
            if self._v56_recovery_stabilization_count < required:
                command = self._stabilization_hold(
                    command,
                    progress_before=progress_before,
                    phase_before_label=phase_before_label,
                )
            else:
                # The twentieth sample is the inherited zero-progress TRACK
                # transition hold; normal tangential TRACK starts next sample.
                self._v56_stabilizing = False
                row = self._diagnostic_rows[-1]
                row["geometric_phase_before"] = RECOVERY_STABILIZE_PHASE
                row["geometric_phase"] = command.geometric_phase
                row["geometric_reason"] = "v56_recovery_stabilization_complete_to_track_hold"
                row["tangential_motion_permitted"] = False
                row["tangent_step_norm_m"] = 0.0
                command = replace(
                    command,
                    geometric_transition_reason=(
                        "v56_recovery_stabilization_complete_to_track_hold"
                    ),
                )
        elif self._v56_stabilizing:
            # A force loss or safety transition terminates stabilization and
            # returns to the inherited recovery chain without shortening the
            # three-sample confirmation requirement.
            self._v56_stabilizing = False
            self._v56_recovery_stabilization_count = 0
            self._diagnostic_rows[-1]["geometric_phase_before"] = (
                RECOVERY_STABILIZE_PHASE
            )
        elif command.geometric_phase != GeometricRecoveryPhaseV53R2.CONFIRM.value:
            self._v56_recovery_stabilization_count = 0

        self._annotate(command)
        return command
