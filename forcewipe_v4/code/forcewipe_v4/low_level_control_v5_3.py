"""Conditional V5.3 normal-force controller definition.

This module is intentionally separate from :mod:`low_level_control`.  The
V5.1/V5.2 negative diagnostic remains tied to the earlier controller and must
not be reinterpreted after this new control law is introduced.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

import numpy as np

from .low_level_control import ForceControlCommand, LowLevelControlError


class ConditionalControllerState(str, Enum):
    """Mutually exclusive states in descending safety priority."""

    HEADROOM = "HEADROOM"
    ACQUIRE = "ACQUIRE"
    RECOVER = "RECOVER"
    TRACK = "TRACK"


@dataclass(frozen=True)
class ConditionalForceControllerConfigV53:
    """Frozen-shape configuration for the single V5.3 controller.

    Numerical values are implementation defaults only until a V5.3 execution
    protocol is independently authorized.  No runtime sweep is supported.
    """

    dt_s: float = 0.01
    force_limit_n: float = 15.0
    headroom_enter_n: float = 13.5
    headroom_exit_n: float = 12.8
    track_enter_n: float = 3.0
    track_exit_n: float = 2.0
    contact_confirm_samples: int = 3
    loss_confirm_samples: int = 2
    headroom_exit_confirm_samples: int = 3
    acquire_initial_step_m: float = 0.0025
    acquire_increment_m: float = 0.0005
    acquire_max_press_step_m: float = 0.0060
    recover_initial_step_m: float = 0.0010
    recover_increment_m: float = 0.00025
    recover_max_press_step_m: float = 0.0040
    track_max_press_step_m: float = 0.0100
    max_lift_step_m: float = 0.0035
    kp_m_per_n: float = 0.0008
    ki_m_per_n_s: float = 0.0012
    kd_m_s_per_n: float = 0.000006
    integral_clip_n_s: float = 15.0
    force_rate_clip_n_s: float = 300.0
    action_position_scale_m: float = 0.1

    def validate(self) -> None:
        positive_reals = (
            self.dt_s,
            self.force_limit_n,
            self.headroom_enter_n,
            self.headroom_exit_n,
            self.track_enter_n,
            self.track_exit_n,
            self.acquire_initial_step_m,
            self.acquire_increment_m,
            self.acquire_max_press_step_m,
            self.recover_initial_step_m,
            self.recover_increment_m,
            self.recover_max_press_step_m,
            self.track_max_press_step_m,
            self.max_lift_step_m,
            self.kp_m_per_n,
            self.ki_m_per_n_s,
            self.kd_m_s_per_n,
            self.integral_clip_n_s,
            self.force_rate_clip_n_s,
            self.action_position_scale_m,
        )
        if not all(math.isfinite(float(value)) and float(value) > 0.0 for value in positive_reals):
            raise LowLevelControlError("all V5.3 controller real-valued parameters must be positive")
        counts = (
            self.contact_confirm_samples,
            self.loss_confirm_samples,
            self.headroom_exit_confirm_samples,
        )
        if not all(isinstance(value, int) and not isinstance(value, bool) and value > 0 for value in counts):
            raise LowLevelControlError("all V5.3 confirmation counts must be positive integers")
        if not 0.0 < self.track_exit_n < self.track_enter_n:
            raise LowLevelControlError("tracking hysteresis thresholds are not strictly ordered")
        if not self.track_enter_n < self.headroom_exit_n < self.headroom_enter_n < self.force_limit_n:
            raise LowLevelControlError("tracking, headroom, and force-limit thresholds are not strictly ordered")
        if self.acquire_initial_step_m > self.acquire_max_press_step_m:
            raise LowLevelControlError("acquisition ramp starts above its press limit")
        if self.recover_initial_step_m > self.recover_max_press_step_m:
            raise LowLevelControlError("recovery ramp starts above its press limit")
        if self.acquire_max_press_step_m > self.track_max_press_step_m:
            raise LowLevelControlError("acquisition authority exceeds TRACK press authority")
        if self.recover_max_press_step_m > self.track_max_press_step_m:
            raise LowLevelControlError("recovery authority exceeds TRACK press authority")


@dataclass(frozen=True)
class ConditionalForceControlCommandV53(ForceControlCommand):
    """Base-compatible force command with auditable controller decomposition."""

    controller_state: str
    state_transition_reason: str
    proportional_step_m: float
    integral_step_m: float
    derivative_step_m: float
    raw_normal_step_m: float
    clipped_normal_step_m: float
    confirmation_count: int
    tangential_motion_allowed: bool


class ConditionalCausalNormalForceControllerV53:
    """HEADROOM > ACQUIRE/RECOVER > TRACK conditional force controller.

    A negative measured force rate never adds downward authority in TRACK:
    ``u_D = -K_d max(dF/dt, 0)``.  ACQUIRE and RECOVER use separate bounded
    ramps, clear the integral state, and do not share the saturated TRACK PI
    request.  Hysteresis transitions require consecutive samples, and the
    controller holds normal position while confirming entry to TRACK or loss
    of TRACK.
    """

    def __init__(
        self,
        config: ConditionalForceControllerConfigV53 = ConditionalForceControllerConfigV53(),
    ) -> None:
        config.validate()
        self.config = config
        self.reset()

    @property
    def state(self) -> ConditionalControllerState:
        return self._state

    def reset(self) -> None:
        self._state = ConditionalControllerState.ACQUIRE
        self._integral_error_n_s = 0.0
        self._previous_force_n: float | None = None
        self._contact_established = False
        self._state_steps = 0
        self._contact_confirm_count = 0
        self._loss_confirm_count = 0
        self._headroom_exit_confirm_count = 0

    def _enter(self, state: ConditionalControllerState) -> None:
        self._state = state
        self._state_steps = 0
        self._contact_confirm_count = 0
        self._loss_confirm_count = 0
        self._headroom_exit_confirm_count = 0
        if state in {
            ConditionalControllerState.HEADROOM,
            ConditionalControllerState.ACQUIRE,
            ConditionalControllerState.RECOVER,
        }:
            self._integral_error_n_s = min(self._integral_error_n_s, 0.0)
        if state in {
            ConditionalControllerState.ACQUIRE,
            ConditionalControllerState.RECOVER,
        }:
            self._integral_error_n_s = 0.0

    @staticmethod
    def _unit_normal(outward_normal_xyz: np.ndarray) -> np.ndarray:
        normal = np.asarray(outward_normal_xyz, dtype=np.float64)
        if normal.shape != (3,) or not np.all(np.isfinite(normal)):
            raise LowLevelControlError("surface normal must be a finite three-vector")
        norm = float(np.linalg.norm(normal))
        if norm <= 0.0:
            raise LowLevelControlError("surface normal must be nonzero")
        return normal / norm

    def command(
        self,
        *,
        measured_force_n: float,
        target_force_n: float,
        outward_normal_xyz: np.ndarray,
    ) -> ConditionalForceControlCommandV53:
        force = float(measured_force_n)
        target = float(target_force_n)
        if not math.isfinite(force) or force < 0.0:
            raise LowLevelControlError("measured force must be finite and nonnegative")
        if not math.isfinite(target) or target <= 0.0:
            raise LowLevelControlError("target force must be finite and positive")
        normal = self._unit_normal(outward_normal_xyz)
        cfg = self.config
        force_rate = 0.0 if self._previous_force_n is None else (
            force - self._previous_force_n
        ) / cfg.dt_s
        force_rate = float(
            np.clip(force_rate, -cfg.force_rate_clip_n_s, cfg.force_rate_clip_n_s)
        )
        error = target - force
        reason = "state_hold"
        hold_for_confirmation = False

        # Priority 1: the headroom guard preempts every other state.
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
                        if self._contact_established and force >= cfg.track_exit_n
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

        # Priority 2: acquisition and recovery have independent bounded ramps.
        if self._state is ConditionalControllerState.ACQUIRE:
            self._integral_error_n_s = 0.0
            if force >= cfg.track_enter_n:
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
            if force >= cfg.track_enter_n:
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
            # No increasing downward request while a hysteresis exit/loss is
            # being confirmed, including the transition sample itself.
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

        delta_position = -normal * normal_step
        normalized = np.clip(
            delta_position / cfg.action_position_scale_m,
            -1.0,
            1.0,
        )
        confirmation_count = max(
            self._contact_confirm_count,
            self._loss_confirm_count,
            self._headroom_exit_confirm_count,
        )
        state = self._state
        # Confirmation/transition holds do not consume one step of the new
        # state's bounded ramp.  The first subsequent ACQUIRE/RECOVER command
        # therefore uses the declared initial step exactly.
        if not hold_for_confirmation:
            self._state_steps += 1
        self._previous_force_n = force
        return ConditionalForceControlCommandV53(
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
        )


def controller_diagnostic_row_v5_3(
    command: ConditionalForceControlCommandV53,
) -> dict[str, object]:
    """Return the mandatory V5.3 per-control-step audit fields."""

    if not isinstance(command, ConditionalForceControlCommandV53):
        raise LowLevelControlError("V5.3 diagnostics require a conditional command")
    return {
        "controller_state": command.controller_state,
        "state_transition_reason": command.state_transition_reason,
        "force_error_n": command.force_error_n,
        "force_rate_n_s": command.force_rate_n_s,
        "integral_error_n_s": command.integral_error_n_s,
        "p_step_m": command.proportional_step_m,
        "i_step_m": command.integral_step_m,
        "d_step_m": command.derivative_step_m,
        "raw_normal_step_m": command.raw_normal_step_m,
        "clipped_normal_step_m": command.clipped_normal_step_m,
        "confirmation_count": command.confirmation_count,
        "tangential_motion_allowed": command.tangential_motion_allowed,
    }
