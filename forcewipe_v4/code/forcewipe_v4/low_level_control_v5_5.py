"""V5.5 TRACK-preserving velocity/history brake.

V5.4 is an immutable scientific failure.  This module defines a new bounded
development controller; it contains no physical-result claim.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np

from .low_level_control import LowLevelControlError
from .low_level_control_v5_3 import ConditionalControllerState
from .low_level_control_v5_4 import (
    ConditionalCausalNormalForceControllerV54,
    ConditionalForceControlCommandV54,
    ConditionalForceControllerConfigV54,
)


@dataclass(frozen=True)
class ConditionalForceControllerConfigV55(ConditionalForceControllerConfigV54):
    """Cancel, but do not exceed, the preceding bounded downpress request."""

    brake_cancel_fraction: float = 1.0
    brake_cancel_max_step_m: float = 0.002

    def validate(self) -> None:
        super().validate()
        if (
            not math.isfinite(float(self.brake_cancel_fraction))
            or not 0.0 < self.brake_cancel_fraction <= 1.0
        ):
            raise LowLevelControlError("V5.5 brake cancellation fraction must lie in (0,1]")
        if (
            not math.isfinite(float(self.brake_cancel_max_step_m))
            or self.brake_cancel_max_step_m <= 0.0
            or self.brake_cancel_max_step_m > self.max_lift_step_m
        ):
            raise LowLevelControlError("V5.5 brake cancellation cap is invalid")


@dataclass(frozen=True)
class ConditionalForceControlCommandV55(ConditionalForceControlCommandV54):
    track_preserving_brake: bool
    cancelled_previous_downpress_m: float


class ConditionalCausalNormalForceControllerV55(
    ConditionalCausalNormalForceControllerV54
):
    """Use the V5.4 trigger without forcing geometric contact reacquisition.

    A triggered sample clears integral memory and revokes tangential authority,
    but it remains in the force TRACK state.  Its outward command cancels at
    most the preceding downpress request and is capped at 2 mm.  Subsequent
    force loss still enters RECOVER through the inherited causal thresholds.
    """

    def __init__(
        self,
        config: ConditionalForceControllerConfigV55 = ConditionalForceControllerConfigV55(),
    ) -> None:
        config.validate()
        self.config = config
        self.reset()

    def command(self, **kwargs) -> ConditionalForceControlCommandV55:
        base = super().command(**kwargs)
        command = ConditionalForceControlCommandV55(
            **base.__dict__,
            track_preserving_brake=False,
            cancelled_previous_downpress_m=0.0,
        )
        if not base.predictive_brake_triggered:
            return command

        cfg = self.config
        cancelled = min(
            float(cfg.brake_cancel_max_step_m),
            float(cfg.brake_cancel_fraction)
            * max(0.0, float(base.previous_normal_command_m)),
        )
        if cancelled <= 0.0:
            raise LowLevelControlError("V5.5 triggered brake has no cancellable downpress")
        self._enter(ConditionalControllerState.TRACK)
        self._integral_error_n_s = 0.0
        normal = self._unit_normal(kwargs["outward_normal_xyz"])
        normal_step = -cancelled
        normalized = np.clip(
            (-normal * normal_step) / cfg.action_position_scale_m,
            -1.0,
            1.0,
        )
        return replace(
            command,
            normalized_position_action=normalized,
            normal_step_m=normal_step,
            integral_error_n_s=0.0,
            mode="track_brake_hold",
            controller_state=ConditionalControllerState.TRACK.value,
            state_transition_reason="track_preserving_history_brake",
            proportional_step_m=0.0,
            integral_step_m=0.0,
            derivative_step_m=0.0,
            raw_normal_step_m=normal_step,
            clipped_normal_step_m=normal_step,
            confirmation_count=0,
            tangential_motion_allowed=False,
            track_preserving_brake=True,
            cancelled_previous_downpress_m=cancelled,
        )
