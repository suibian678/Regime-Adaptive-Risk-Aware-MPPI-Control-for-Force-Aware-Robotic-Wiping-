"""V5.6 mutually exclusive TRACK compensation.

V5.5-r3 is an immutable scientific failure.  This module defines one new
development controller and makes no closed-loop or safety claim.  The only
TRACK-side change is a bounded, hysteretic low-force bias.  It is enabled by
the path supervisor only when both the geometric and force state machines are
already in TRACK.  Acquisition, recovery, predictive braking, and HEADROOM
therefore cannot receive this additional downward authority.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np

from .low_level_control import LowLevelControlError
from .low_level_control_v5_3 import ConditionalControllerState
from .low_level_control_v5_5 import (
    ConditionalCausalNormalForceControllerV55,
    ConditionalForceControlCommandV55,
    ConditionalForceControllerConfigV55,
)


@dataclass(frozen=True)
class ConditionalForceControllerConfigV56(ConditionalForceControllerConfigV55):
    """One predeclared bounded low-force compensation rule."""

    tracking_compensation_enter_target_fraction: float = 0.90
    tracking_compensation_exit_target_fraction: float = 0.95
    tracking_compensation_step_m: float = 0.0005

    def validate(self) -> None:
        super().validate()
        enter = float(self.tracking_compensation_enter_target_fraction)
        exit_ = float(self.tracking_compensation_exit_target_fraction)
        step = float(self.tracking_compensation_step_m)
        if not all(math.isfinite(value) for value in (enter, exit_, step)):
            raise LowLevelControlError("V5.6 tracking compensation must be finite")
        if not 0.0 < enter < exit_ < 1.0:
            raise LowLevelControlError(
                "V5.6 tracking compensation fractions must be ordered in (0,1)"
            )
        if not 0.0 < step <= self.track_max_press_step_m:
            raise LowLevelControlError(
                "V5.6 tracking compensation step exceeds TRACK authority"
            )


@dataclass(frozen=True)
class ConditionalForceControlCommandV56(ConditionalForceControlCommandV55):
    tracking_compensation_eligible: bool
    tracking_compensation_latched: bool
    tracking_compensation_applied: bool
    tracking_compensation_step_m: float


class ConditionalCausalNormalForceControllerV56(
    ConditionalCausalNormalForceControllerV55
):
    """Apply a bounded bias only under joint force/geometric TRACK authority."""

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
            raise LowLevelControlError("V5.6 geometric TRACK context must be boolean")
        self._pending_geometric_track = geometric_track

    def command(self, **kwargs) -> ConditionalForceControlCommandV56:
        geometric_track = bool(self._pending_geometric_track)
        self._pending_geometric_track = False
        base = super().command(**kwargs)
        force = float(kwargs["measured_force_n"])
        target = float(kwargs["target_force_n"])
        eligible = bool(
            geometric_track
            and base.controller_state == ConditionalControllerState.TRACK.value
            and not base.track_preserving_brake
            and not base.predictive_brake_triggered
        )

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
            state_transition_reason="track_bounded_low_force_compensation",
        )
