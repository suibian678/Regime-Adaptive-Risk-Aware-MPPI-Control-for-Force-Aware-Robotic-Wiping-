"""V5.7 verified-return recovery-budget semantics.

The inherited three-cycle bound is retained for consecutive unsuccessful
recovery attempts.  A recovery episode is considered closed only after 50
consecutive ordinary joint-TRACK samples and at least 0.02 committed path
progress.  At that point the consecutive-attempt counter is reset; total
recovery cycles remain observable and are never erased.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from .low_level_control_v5_3 import ConditionalControllerState
from .low_level_control_v5_7 import (
    ConditionalCausalNormalForceControllerV57,
    ConditionalForceControllerConfigV57,
)
from .path_pose_control import PathPoseControllerConfig
from .path_pose_control_v5_3_r2 import PathForcePoseCommandV53R2
from .path_pose_control_v5_6 import GeometricRecoveryConfigV56
from .path_pose_control_v5_6_r2 import CausalPathForcePoseControllerV56R2


@dataclass(frozen=True)
class GeometricRecoveryConfigV57(GeometricRecoveryConfigV56):
    verified_return_samples: int = 50
    verified_return_min_progress: float = 0.02

    def validate(self) -> None:
        super().validate()
        if (
            not isinstance(self.verified_return_samples, int)
            or isinstance(self.verified_return_samples, bool)
            or self.verified_return_samples != 50
        ):
            raise ValueError("V5.7 verified return is frozen at 50 samples")
        if (
            not math.isfinite(float(self.verified_return_min_progress))
            or self.verified_return_min_progress != 0.02
        ):
            raise ValueError("V5.7 verified return progress is frozen at 0.02")


class CausalPathForcePoseControllerV57(CausalPathForcePoseControllerV56R2):
    """Reset only the consecutive failed-attempt budget after verified TRACK."""

    def __init__(
        self,
        *,
        force_config: ConditionalForceControllerConfigV57 = ConditionalForceControllerConfigV57(),
        path_config: PathPoseControllerConfig = PathPoseControllerConfig(),
        recovery_config: GeometricRecoveryConfigV57 = GeometricRecoveryConfigV57(),
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
        self.force_controller = ConditionalCausalNormalForceControllerV57(
            force_config
        )
        self.reset()

    def reset(self) -> None:
        super().reset()
        self._v57_total_recovery_cycles = 0
        self._v57_verified_return_count = 0
        self._v57_verified_return_start_progress: float | None = None
        self._v57_recovery_budget_reset_count = 0

    def command(self, **kwargs) -> PathForcePoseCommandV53R2:
        cycles_before = int(self._recovery_cycle_count)
        command = super().command(**kwargs)
        cycles_after = int(self._recovery_cycle_count)
        if cycles_after > cycles_before:
            self._v57_total_recovery_cycles += cycles_after - cycles_before

        row = self._diagnostic_rows[-1]
        stable_return_sample = bool(
            command.geometric_phase == "TRACK"
            and command.force_command.controller_state
            == ConditionalControllerState.TRACK.value
            and bool(row.get("tangential_motion_permitted", False))
            and not bool(row.get("track_preserving_brake", False))
            and not bool(
                getattr(
                    command.force_command,
                    "predictive_headroom_triggered",
                    False,
                )
            )
        )
        reset_triggered = False
        progress_gain = 0.0
        if stable_return_sample and self._recovery_cycle_count > 0:
            if self._v57_verified_return_count == 0:
                self._v57_verified_return_start_progress = float(
                    command.committed_progress
                )
            self._v57_verified_return_count += 1
            progress_gain = float(command.committed_progress) - float(
                self._v57_verified_return_start_progress
            )
            if (
                self._v57_verified_return_count
                >= self.recovery_config.verified_return_samples
                and progress_gain
                >= self.recovery_config.verified_return_min_progress
            ):
                self._recovery_cycle_count = 0
                self._v57_recovery_budget_reset_count += 1
                reset_triggered = True
                self._v57_verified_return_count = 0
                self._v57_verified_return_start_progress = None
        else:
            self._v57_verified_return_count = 0
            self._v57_verified_return_start_progress = None

        force = command.force_command
        row.update(
            {
                "recovery_cycle_count": int(self._recovery_cycle_count),
                "v57_consecutive_recovery_cycles": int(
                    self._recovery_cycle_count
                ),
                "v57_total_recovery_cycles": int(
                    self._v57_total_recovery_cycles
                ),
                "v57_verified_return_sample": stable_return_sample,
                "v57_verified_return_count": int(
                    self._v57_verified_return_count
                ),
                "v57_verified_return_progress_gain": float(progress_gain),
                "v57_recovery_budget_reset": reset_triggered,
                "v57_recovery_budget_reset_count": int(
                    self._v57_recovery_budget_reset_count
                ),
                "v57_tracking_reference_gain": float(
                    getattr(force, "tracking_reference_gain", 1.0)
                ),
                "v57_tracking_reference_n": float(
                    getattr(force, "tracking_reference_n", kwargs["target_force_n"])
                ),
                "v57_tracking_reference_bias_step_m": float(
                    getattr(force, "tracking_reference_bias_step_m", 0.0)
                ),
                "v57_predictive_headroom_force_n": float(
                    getattr(force, "predictive_headroom_force_n", kwargs["measured_force_n"])
                ),
                "v57_predictive_headroom_triggered": bool(
                    getattr(force, "predictive_headroom_triggered", False)
                ),
                "v57_predictive_headroom_horizon_samples": int(
                    getattr(force, "predictive_headroom_horizon_samples", 0)
                ),
                "v57_reference_headroom_overlap_violation": bool(
                    float(getattr(force, "tracking_reference_bias_step_m", 0.0))
                    > 0.0
                    and bool(
                        getattr(force, "predictive_headroom_triggered", False)
                    )
                ),
            }
        )
        return command
