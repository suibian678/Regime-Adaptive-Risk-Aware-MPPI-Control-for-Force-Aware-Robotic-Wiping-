"""Revised V5.3-r2 conditional normal-force controller.

The r2 revision adds an immediate severe-force-drop transition and a
target-dependent recovery-exit threshold.  It remains a pre-execution method
definition; no physical claim follows from this module or its unit tests.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .low_level_control import LowLevelControlError
from .low_level_control_v5_3 import (
    ConditionalCausalNormalForceControllerV53,
    ConditionalControllerState,
    ConditionalForceControlCommandV53,
    ConditionalForceControllerConfigV53,
)


@dataclass(frozen=True)
class ConditionalForceControllerConfigV53R2(ConditionalForceControllerConfigV53):
    severe_drop_floor_n: float = 3.0
    severe_drop_target_fraction: float = 0.5
    recover_exit_floor_n: float = 3.0
    recover_exit_target_fraction: float = 0.65

    def validate(self) -> None:
        super().validate()
        extra = (
            self.severe_drop_floor_n,
            self.severe_drop_target_fraction,
            self.recover_exit_floor_n,
            self.recover_exit_target_fraction,
        )
        if not all(math.isfinite(float(value)) and float(value) > 0.0 for value in extra):
            raise LowLevelControlError("V5.3-r2 drop/recovery parameters must be positive")
        if not 0.0 < self.severe_drop_target_fraction < self.recover_exit_target_fraction < 1.0:
            raise LowLevelControlError("V5.3-r2 target fractions must be ordered in (0,1)")
        if self.severe_drop_floor_n > self.recover_exit_floor_n:
            raise LowLevelControlError("severe-drop floor exceeds recovery-exit floor")
        if self.recover_exit_floor_n > self.headroom_exit_n:
            raise LowLevelControlError("recovery-exit floor reaches force headroom")


@dataclass(frozen=True)
class ConditionalForceControlCommandV53R2(ConditionalForceControlCommandV53):
    severe_drop_threshold_n: float
    recovery_exit_threshold_n: float


class ConditionalCausalNormalForceControllerV53R2(
    ConditionalCausalNormalForceControllerV53
):
    """Conditional controller with immediate severe-drop recovery."""

    def __init__(
        self,
        config: ConditionalForceControllerConfigV53R2 = ConditionalForceControllerConfigV53R2(),
    ) -> None:
        config.validate()
        self.config = config
        self.reset()

    @staticmethod
    def _thresholds(
        config: ConditionalForceControllerConfigV53R2,
        target_force_n: float,
    ) -> tuple[float, float]:
        severe = max(
            float(config.severe_drop_floor_n),
            float(config.severe_drop_target_fraction) * float(target_force_n),
        )
        recovery_exit = max(
            float(config.recover_exit_floor_n),
            float(config.recover_exit_target_fraction) * float(target_force_n),
        )
        if not severe < recovery_exit < config.headroom_enter_n:
            raise LowLevelControlError(
                "target-dependent recovery thresholds are not strictly ordered"
            )
        return severe, recovery_exit

    def command(
        self,
        *,
        measured_force_n: float,
        target_force_n: float,
        outward_normal_xyz: np.ndarray,
    ) -> ConditionalForceControlCommandV53R2:
        force = float(measured_force_n)
        target = float(target_force_n)
        if not math.isfinite(force) or force < 0.0:
            raise LowLevelControlError("measured force must be finite and nonnegative")
        if not math.isfinite(target) or target <= 0.0:
            raise LowLevelControlError("target force must be finite and positive")
        normal = self._unit_normal(outward_normal_xyz)
        cfg = self.config
        severe_threshold, recovery_exit_threshold = self._thresholds(cfg, target)
        force_rate = 0.0 if self._previous_force_n is None else (
            force - self._previous_force_n
        ) / cfg.dt_s
        force_rate = float(
            np.clip(force_rate, -cfg.force_rate_clip_n_s, cfg.force_rate_clip_n_s)
        )
        error = target - force
        reason = "state_hold"
        hold_for_confirmation = False

        # Priority 1: HEADROOM preempts all acquisition and tracking logic.
        if force >= cfg.headroom_enter_n:
            if self._state is not ConditionalControllerState.HEADROOM:
                self._enter(ConditionalControllerState.HEADROOM)
                reason = "headroom_enter_threshold"
            else:
                reason = "headroom_active"
            self._integral_error_n_s = min(self._integral_error_n_s, 0.0)
        elif self._state is ConditionalControllerState.HEADROOM:
            if force <= cfg.headroom_exit_n:
                self._headroom_exit_confirm_count += 1
                reason = "headroom_exit_confirmation"
                if self._headroom_exit_confirm_count >= cfg.headroom_exit_confirm_samples:
                    next_state = (
                        ConditionalControllerState.TRACK
                        if self._contact_established and force >= recovery_exit_threshold
                        else ConditionalControllerState.RECOVER
                        if self._contact_established
                        else ConditionalControllerState.ACQUIRE
                    )
                    self._enter(next_state)
                    reason = f"headroom_exit_confirmed_to_{next_state.value.lower()}"
                    hold_for_confirmation = True
            else:
                self._headroom_exit_confirm_count = 0
                reason = "headroom_exit_hysteresis_hold"

        # Priority 2: a severe force drop causes immediate RECOVER entry.  No
        # two-sample debounce is used at this threshold, and the transition
        # command is a zero normal hold after clearing the integral state.
        if (
            self._state is ConditionalControllerState.TRACK
            and force <= severe_threshold
        ):
            self._enter(ConditionalControllerState.RECOVER)
            self._integral_error_n_s = 0.0
            reason = "severe_force_drop_to_recover"
            hold_for_confirmation = True

        if self._state is ConditionalControllerState.ACQUIRE:
            self._integral_error_n_s = 0.0
            if force >= recovery_exit_threshold - 1e-12:
                self._contact_confirm_count += 1
                hold_for_confirmation = True
                reason = "acquire_exit_confirmation"
                if self._contact_confirm_count >= cfg.contact_confirm_samples:
                    self._contact_established = True
                    self._enter(ConditionalControllerState.TRACK)
                    reason = "acquire_exit_confirmed_to_track"
            else:
                self._contact_confirm_count = 0
        elif self._state is ConditionalControllerState.RECOVER:
            self._integral_error_n_s = 0.0
            if force >= recovery_exit_threshold - 1e-12:
                self._contact_confirm_count += 1
                hold_for_confirmation = True
                reason = "recover_exit_confirmation"
                if self._contact_confirm_count >= cfg.contact_confirm_samples:
                    self._contact_established = True
                    self._enter(ConditionalControllerState.TRACK)
                    reason = "recover_exit_confirmed_to_track"
            else:
                self._contact_confirm_count = 0
        elif self._state is ConditionalControllerState.TRACK:
            # Moderate loss still uses a two-sample confirmation.  Severe loss
            # has already been handled above and cannot reach this branch.
            if force <= cfg.track_exit_n:
                self._loss_confirm_count += 1
                hold_for_confirmation = True
                reason = "track_loss_confirmation"
                if self._loss_confirm_count >= cfg.loss_confirm_samples:
                    self._enter(ConditionalControllerState.RECOVER)
                    reason = "track_loss_confirmed_to_recover"
            else:
                self._loss_confirm_count = 0

        p_step = 0.0
        i_step = 0.0
        d_step = 0.0
        if self._state is ConditionalControllerState.HEADROOM:
            raw_step = -cfg.max_lift_step_m
            normal_step = raw_step
        elif hold_for_confirmation:
            raw_step = 0.0
            normal_step = 0.0
        elif self._state is ConditionalControllerState.ACQUIRE:
            raw_step = min(
                cfg.acquire_max_press_step_m,
                cfg.acquire_initial_step_m + cfg.acquire_increment_m * self._state_steps,
            )
            normal_step = raw_step
            reason = "acquire_bounded_ramp"
        elif self._state is ConditionalControllerState.RECOVER:
            raw_step = min(
                cfg.recover_max_press_step_m,
                cfg.recover_initial_step_m + cfg.recover_increment_m * self._state_steps,
            )
            normal_step = raw_step
            reason = "recover_bounded_ramp"
        else:
            candidate_integral = float(
                np.clip(
                    self._integral_error_n_s + error * cfg.dt_s,
                    -cfg.integral_clip_n_s,
                    cfg.integral_clip_n_s,
                )
            )
            p_step = cfg.kp_m_per_n * error
            i_step = cfg.ki_m_per_n_s * candidate_integral
            d_step = -cfg.kd_m_s_per_n * max(force_rate, 0.0)
            raw_step = p_step + i_step + d_step
            normal_step = float(
                np.clip(raw_step, -cfg.max_lift_step_m, cfg.track_max_press_step_m)
            )
            saturated_high = raw_step > cfg.track_max_press_step_m and error > 0.0
            saturated_low = raw_step < -cfg.max_lift_step_m and error < 0.0
            if not (saturated_high or saturated_low):
                self._integral_error_n_s = candidate_integral
            reason = "track_one_sided_derivative"

        normalized = np.clip(
            (-normal * normal_step) / cfg.action_position_scale_m,
            -1.0,
            1.0,
        )
        confirmation_count = max(
            self._contact_confirm_count,
            self._loss_confirm_count,
            self._headroom_exit_confirm_count,
        )
        state = self._state
        if not hold_for_confirmation:
            self._state_steps += 1
        self._previous_force_n = force
        return ConditionalForceControlCommandV53R2(
            normalized_position_action=normalized,
            normal_step_m=float(normal_step),
            force_error_n=float(error),
            force_rate_n_s=float(force_rate),
            integral_error_n_s=float(self._integral_error_n_s),
            mode=state.value.lower(),
            controller_state=state.value,
            state_transition_reason=reason,
            proportional_step_m=float(p_step),
            integral_step_m=float(i_step),
            derivative_step_m=float(d_step),
            raw_normal_step_m=float(raw_step),
            clipped_normal_step_m=float(normal_step),
            confirmation_count=int(confirmation_count),
            tangential_motion_allowed=state is ConditionalControllerState.TRACK,
            severe_drop_threshold_n=float(severe_threshold),
            recovery_exit_threshold_n=float(recovery_exit_threshold),
        )
