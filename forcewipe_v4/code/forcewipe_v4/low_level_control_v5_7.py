"""V5.7 causal TRACK reference bias and predictive headroom guard.

V5.6-r2 remains an immutable development failure.  This module defines one
new development controller.  It does not contain a closed-loop result claim.

The TRACK reference bias uses the already deployed 1.10 force-fraction bound
as a single force-tier-independent calibration.  It is implemented as the
proportional term that would result from replacing ``F*`` by ``1.10 F*``;
state thresholds continue to use the physical target ``F*``.  A two-sample
causal force forecast preempts this bias and enters HEADROOM before the
sampled force reaches the ordinary 13.5-N entry threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np

from .low_level_control import LowLevelControlError
from .low_level_control_v5_3 import ConditionalControllerState
from .low_level_control_v5_5 import ConditionalCausalNormalForceControllerV55
from .low_level_control_v5_6 import (
    ConditionalForceControlCommandV56,
    ConditionalForceControllerConfigV56,
)


@dataclass(frozen=True)
class ConditionalForceControllerConfigV57(ConditionalForceControllerConfigV56):
    """Single predeclared V5.7 TRACK/headroom configuration."""

    tracking_reference_gain: float = 1.10
    tracking_reference_bias_max_step_m: float = 0.0010
    predictive_headroom_horizon_samples: int = 2

    def validate(self) -> None:
        super().validate()
        if (
            not math.isfinite(float(self.tracking_reference_gain))
            or not 1.0 < self.tracking_reference_gain <= 1.10
        ):
            raise LowLevelControlError(
                "V5.7 TRACK reference gain must lie in (1, 1.10]"
            )
        if (
            not math.isfinite(float(self.tracking_reference_bias_max_step_m))
            or self.tracking_reference_bias_max_step_m <= 0.0
            or self.tracking_reference_bias_max_step_m
            > self.track_max_press_step_m
        ):
            raise LowLevelControlError("V5.7 TRACK reference-bias cap is invalid")
        if (
            not isinstance(self.predictive_headroom_horizon_samples, int)
            or isinstance(self.predictive_headroom_horizon_samples, bool)
            or self.predictive_headroom_horizon_samples != 2
        ):
            raise LowLevelControlError(
                "V5.7 predictive headroom horizon is frozen at two samples"
            )


@dataclass(frozen=True)
class ConditionalForceControlCommandV57(ConditionalForceControlCommandV56):
    tracking_reference_gain: float
    tracking_reference_n: float
    tracking_reference_bias_step_m: float
    predictive_headroom_force_n: float
    predictive_headroom_triggered: bool
    predictive_headroom_horizon_samples: int


class ConditionalCausalNormalForceControllerV57(
    ConditionalCausalNormalForceControllerV55
):
    """Apply mutually exclusive reference-bias and headroom mechanisms."""

    def __init__(
        self,
        config: ConditionalForceControllerConfigV57 = ConditionalForceControllerConfigV57(),
    ) -> None:
        config.validate()
        self.config = config
        self.reset()

    def reset(self) -> None:
        super().reset()
        self._pending_geometric_track = False
        self._tracking_reference_latched = False

    def set_geometric_track_context(self, *, geometric_track: bool) -> None:
        if not isinstance(geometric_track, bool):
            raise TypeError("V5.7 geometric TRACK context must be boolean")
        self._pending_geometric_track = geometric_track

    def command(self, **kwargs) -> ConditionalForceControlCommandV57:
        geometric_track = bool(self._pending_geometric_track)
        self._pending_geometric_track = False
        state_before = self.state

        # Deliberately bypass the V5.6 fixed 0.5-mm output bias.  V5.7 retains
        # the V5.5 force-state machine and replaces that bias with one
        # target-scaled reference term below.
        base = ConditionalCausalNormalForceControllerV55.command(self, **kwargs)
        cfg = self.config
        force = float(kwargs["measured_force_n"])
        target = float(kwargs["target_force_n"])
        forecast = force + (
            float(cfg.predictive_headroom_horizon_samples)
            * float(cfg.dt_s)
            * max(float(base.force_rate_n_s), 0.0)
        )
        predictive_headroom = bool(
            base.controller_state == ConditionalControllerState.TRACK.value
            and not base.track_preserving_brake
            and forecast >= float(cfg.headroom_enter_n)
        )

        eligible = bool(
            geometric_track
            and state_before is ConditionalControllerState.TRACK
            and base.controller_state == ConditionalControllerState.TRACK.value
            and base.state_transition_reason == "track_one_sided_derivative"
            and not base.track_preserving_brake
            and not predictive_headroom
        )
        if not eligible:
            self._tracking_reference_latched = False
        elif force >= cfg.tracking_compensation_exit_target_fraction * target:
            self._tracking_reference_latched = False
        elif force <= cfg.tracking_compensation_enter_target_fraction * target:
            self._tracking_reference_latched = True

        reference = float(cfg.tracking_reference_gain * target)
        reference_bias = min(
            float(cfg.tracking_reference_bias_max_step_m),
            float(cfg.kp_m_per_n)
            * (float(cfg.tracking_reference_gain) - 1.0)
            * target,
        )
        reference_applied = bool(eligible and self._tracking_reference_latched)
        applied_bias = float(reference_bias if reference_applied else 0.0)
        command = ConditionalForceControlCommandV57(
            **base.__dict__,
            # Compatibility aliases keep the V5.6 diagnostic schema usable.
            tracking_compensation_eligible=eligible,
            tracking_compensation_latched=bool(self._tracking_reference_latched),
            tracking_compensation_applied=reference_applied,
            tracking_compensation_step_m=applied_bias,
            tracking_reference_gain=float(cfg.tracking_reference_gain),
            tracking_reference_n=reference,
            tracking_reference_bias_step_m=applied_bias,
            predictive_headroom_force_n=float(forecast),
            predictive_headroom_triggered=predictive_headroom,
            predictive_headroom_horizon_samples=int(
                cfg.predictive_headroom_horizon_samples
            ),
        )

        # Safety authority has strict priority over reference shaping.
        if predictive_headroom:
            self._enter(ConditionalControllerState.HEADROOM)
            self._integral_error_n_s = 0.0
            self._tracking_reference_latched = False
            normal_step = -float(cfg.max_lift_step_m)
            normal = self._unit_normal(kwargs["outward_normal_xyz"])
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
                mode=ConditionalControllerState.HEADROOM.value.lower(),
                controller_state=ConditionalControllerState.HEADROOM.value,
                state_transition_reason="two_sample_predictive_headroom",
                proportional_step_m=0.0,
                integral_step_m=0.0,
                derivative_step_m=0.0,
                raw_normal_step_m=normal_step,
                clipped_normal_step_m=normal_step,
                confirmation_count=0,
                tangential_motion_allowed=False,
                tracking_compensation_eligible=False,
                tracking_compensation_latched=False,
                tracking_compensation_applied=False,
                tracking_compensation_step_m=0.0,
                tracking_reference_bias_step_m=0.0,
            )

        if not reference_applied:
            return command
        raw = float(base.raw_normal_step_m) + applied_bias
        clipped = float(
            np.clip(raw, -cfg.max_lift_step_m, cfg.track_max_press_step_m)
        )
        normal = self._unit_normal(kwargs["outward_normal_xyz"])
        normalized = np.clip(
            (-normal * clipped) / cfg.action_position_scale_m,
            -1.0,
            1.0,
        )
        return replace(
            command,
            normalized_position_action=normalized,
            normal_step_m=clipped,
            raw_normal_step_m=raw,
            clipped_normal_step_m=clipped,
            state_transition_reason="stable_track_target_scaled_reference_bias",
        )
