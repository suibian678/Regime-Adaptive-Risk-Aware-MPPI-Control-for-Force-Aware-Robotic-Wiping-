"""V5.6-preauth-r2 stable-TRACK compensation gate.

The V5.6-preauth-r1 definition is preserved and was never executed.  This
revision excludes every force-state transition or confirmation-hold sample
from low-force compensation.  The numerical controller parameters are
unchanged.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .low_level_control_v5_3 import ConditionalControllerState
from .low_level_control_v5_5 import ConditionalCausalNormalForceControllerV55
from .low_level_control_v5_6 import (
    ConditionalForceControlCommandV56,
    ConditionalForceControllerConfigV56,
)


class ConditionalCausalNormalForceControllerV56R2(
    ConditionalCausalNormalForceControllerV55
):
    """Compensate only an uninterrupted, ordinary TRACK regulation sample."""

    def __init__(
        self,
        config: ConditionalForceControllerConfigV56 = ConditionalForceControllerConfigV56(),
    ) -> None:
        config.validate()
        self.config = config
        self.reset()

    def reset(self) -> None:
        super().reset()
        self._pending_geometric_track = False
        self._tracking_compensation_latched = False

    def set_geometric_track_context(self, *, geometric_track: bool) -> None:
        if not isinstance(geometric_track, bool):
            raise TypeError("V5.6-r2 geometric TRACK context must be boolean")
        self._pending_geometric_track = geometric_track

    def command(self, **kwargs) -> ConditionalForceControlCommandV56:
        geometric_track = bool(self._pending_geometric_track)
        self._pending_geometric_track = False
        state_before = self.state
        base = super().command(**kwargs)
        force = float(kwargs["measured_force_n"])
        target = float(kwargs["target_force_n"])
        stable_force_track = bool(
            state_before is ConditionalControllerState.TRACK
            and base.controller_state == ConditionalControllerState.TRACK.value
            and base.state_transition_reason == "track_one_sided_derivative"
            and not base.track_preserving_brake
            and not base.predictive_brake_triggered
        )
        eligible = bool(geometric_track and stable_force_track)

        if not eligible:
            self._tracking_compensation_latched = False
        elif force >= self.config.tracking_compensation_exit_target_fraction * target:
            self._tracking_compensation_latched = False
        elif force <= self.config.tracking_compensation_enter_target_fraction * target:
            self._tracking_compensation_latched = True

        applied = bool(eligible and self._tracking_compensation_latched)
        step = float(self.config.tracking_compensation_step_m if applied else 0.0)
        command = ConditionalForceControlCommandV56(
            **base.__dict__,
            tracking_compensation_eligible=eligible,
            tracking_compensation_latched=bool(self._tracking_compensation_latched),
            tracking_compensation_applied=applied,
            tracking_compensation_step_m=step,
        )
        if not applied:
            return command

        raw = float(base.raw_normal_step_m) + step
        clipped = float(
            np.clip(
                raw,
                -self.config.max_lift_step_m,
                self.config.track_max_press_step_m,
            )
        )
        normal = self._unit_normal(kwargs["outward_normal_xyz"])
        normalized = np.clip(
            (-normal * clipped) / self.config.action_position_scale_m,
            -1.0,
            1.0,
        )
        return replace(
            command,
            normalized_position_action=normalized,
            normal_step_m=clipped,
            raw_normal_step_m=raw,
            clipped_normal_step_m=clipped,
            state_transition_reason="stable_track_bounded_low_force_compensation",
        )
