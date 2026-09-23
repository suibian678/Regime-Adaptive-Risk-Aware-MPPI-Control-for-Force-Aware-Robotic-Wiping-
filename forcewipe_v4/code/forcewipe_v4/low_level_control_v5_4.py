"""V5.4 velocity/history-aware conditional normal-force controller.

V5.4 is a new development method.  V5.3-r3 remains an immutable scientific
failure and none of its CAL observations are reclassified as held-out data.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np

from .low_level_control import LowLevelControlError
from .low_level_control_v5_3 import ConditionalControllerState
from .low_level_control_v5_3_r2 import (
    ConditionalCausalNormalForceControllerV53R2,
    ConditionalForceControlCommandV53R2,
    ConditionalForceControllerConfigV53R2,
)


@dataclass(frozen=True)
class ConditionalForceControllerConfigV54(ConditionalForceControllerConfigV53R2):
    """Single frozen-shape V5.4 predictive-braking rule."""

    brake_min_previous_downpress_m: float = 0.0015
    brake_max_force_rate_n_s: float = -80.0
    brake_max_outward_normal_velocity_m_s: float = -0.0002
    brake_max_force_target_fraction: float = 0.85
    brake_outward_step_m: float = 0.0030

    def validate(self) -> None:
        super().validate()
        positive = (
            self.brake_min_previous_downpress_m,
            self.brake_max_force_target_fraction,
            self.brake_outward_step_m,
        )
        if not all(math.isfinite(float(value)) and float(value) > 0.0 for value in positive):
            raise LowLevelControlError("V5.4 braking magnitudes must be positive")
        if not math.isfinite(float(self.brake_max_force_rate_n_s)) or self.brake_max_force_rate_n_s >= 0.0:
            raise LowLevelControlError("V5.4 force-rate threshold must be negative")
        if (
            not math.isfinite(float(self.brake_max_outward_normal_velocity_m_s))
            or self.brake_max_outward_normal_velocity_m_s >= 0.0
        ):
            raise LowLevelControlError("V5.4 inward-velocity threshold must be negative")
        if not 0.0 < self.brake_max_force_target_fraction < 1.0:
            raise LowLevelControlError("V5.4 braking force fraction must lie in (0,1)")
        if self.brake_outward_step_m > self.max_lift_step_m:
            raise LowLevelControlError("V5.4 braking step exceeds outward authority")


@dataclass(frozen=True)
class ConditionalForceControlCommandV54(ConditionalForceControlCommandV53R2):
    normal_velocity_m_s: float
    previous_normal_command_m: float
    predictive_brake_triggered: bool
    brake_condition_count: int


class ConditionalCausalNormalForceControllerV54(
    ConditionalCausalNormalForceControllerV53R2
):
    """Add one predeclared outward-braking rule to the r2 force state machine.

    The rule triggers only while the inherited controller would remain in
    TRACK and all four observations agree: the preceding action requested
    material downpress, force is now falling rapidly, the tool is still moving
    inward along the surface normal, and measured force remains below a fixed
    fraction of target.  Triggering clears integral memory, enters RECOVER,
    revokes tangential authority, and commands one bounded outward step.
    """

    def __init__(
        self,
        config: ConditionalForceControllerConfigV54 = ConditionalForceControllerConfigV54(),
    ) -> None:
        config.validate()
        self.config = config
        self.reset()

    def reset(self) -> None:
        super().reset()
        self._pending_normal_velocity_m_s: float | None = None
        self._pending_previous_normal_command_m: float | None = None

    def set_kinematic_context(
        self,
        *,
        normal_velocity_m_s: float,
        previous_normal_command_m: float,
    ) -> None:
        velocity = float(normal_velocity_m_s)
        previous = float(previous_normal_command_m)
        if not math.isfinite(velocity) or not math.isfinite(previous):
            raise LowLevelControlError("V5.4 kinematic context must be finite")
        self._pending_normal_velocity_m_s = velocity
        self._pending_previous_normal_command_m = previous

    def command(
        self,
        *,
        measured_force_n: float,
        target_force_n: float,
        outward_normal_xyz: np.ndarray,
    ) -> ConditionalForceControlCommandV54:
        velocity = float(
            0.0
            if self._pending_normal_velocity_m_s is None
            else self._pending_normal_velocity_m_s
        )
        previous_command = float(
            0.0
            if self._pending_previous_normal_command_m is None
            else self._pending_previous_normal_command_m
        )
        self._pending_normal_velocity_m_s = None
        self._pending_previous_normal_command_m = None

        base = super().command(
            measured_force_n=measured_force_n,
            target_force_n=target_force_n,
            outward_normal_xyz=outward_normal_xyz,
        )
        cfg = self.config
        conditions = (
            previous_command >= cfg.brake_min_previous_downpress_m,
            base.force_rate_n_s <= cfg.brake_max_force_rate_n_s,
            velocity <= cfg.brake_max_outward_normal_velocity_m_s,
            float(measured_force_n)
            <= cfg.brake_max_force_target_fraction * float(target_force_n),
        )
        triggered = bool(
            base.controller_state == ConditionalControllerState.TRACK.value
            and all(conditions)
        )
        command = ConditionalForceControlCommandV54(
            **base.__dict__,
            normal_velocity_m_s=velocity,
            previous_normal_command_m=previous_command,
            predictive_brake_triggered=False,
            brake_condition_count=sum(bool(value) for value in conditions),
        )
        if not triggered:
            return command

        self._enter(ConditionalControllerState.RECOVER)
        self._integral_error_n_s = 0.0
        normal = self._unit_normal(outward_normal_xyz)
        normal_step = -float(cfg.brake_outward_step_m)
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
            mode=ConditionalControllerState.RECOVER.value.lower(),
            controller_state=ConditionalControllerState.RECOVER.value,
            state_transition_reason="velocity_history_outward_brake",
            proportional_step_m=0.0,
            integral_step_m=0.0,
            derivative_step_m=0.0,
            raw_normal_step_m=normal_step,
            clipped_normal_step_m=normal_step,
            confirmation_count=0,
            tangential_motion_allowed=False,
            predictive_brake_triggered=True,
        )
