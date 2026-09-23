"""Causal 100 Hz low-level normal-force controller for V4 method pilots."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


class LowLevelControlError(ValueError):
    pass


@dataclass(frozen=True)
class ForceControllerConfig:
    dt_s: float = 0.01
    contact_threshold_n: float = 0.2
    force_limit_n: float = 15.0
    headroom_threshold_n: float = 13.5
    approach_step_m: float = 0.0100
    max_press_step_m: float = 0.0100
    max_lift_step_m: float = 0.0035
    kp_m_per_n: float = 0.0008
    ki_m_per_n_s: float = 0.0012
    kd_m_s_per_n: float = 0.000006
    integral_clip_n_s: float = 15.0
    force_rate_clip_n_s: float = 300.0
    action_position_scale_m: float = 0.1
    preload_ramp_enabled: bool = True
    impact_brake_enabled: bool = True
    preload_initial_step_m: float = 0.0025
    preload_ramp_increment_m: float = 0.0005

    def validate(self) -> None:
        values = tuple(
            float(getattr(self, name))
            for name in (
                "dt_s",
                "contact_threshold_n",
                "force_limit_n",
                "headroom_threshold_n",
                "approach_step_m",
                "max_press_step_m",
                "max_lift_step_m",
                "kp_m_per_n",
                "ki_m_per_n_s",
                "kd_m_s_per_n",
                "integral_clip_n_s",
                "force_rate_clip_n_s",
                "action_position_scale_m",
                "preload_initial_step_m",
                "preload_ramp_increment_m",
            )
        )
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise LowLevelControlError("all force-controller parameters must be positive")
        if not self.contact_threshold_n < self.headroom_threshold_n < self.force_limit_n:
            raise LowLevelControlError("force thresholds must be strictly ordered")
        if self.preload_initial_step_m > self.approach_step_m:
            raise LowLevelControlError("preload ramp cannot start above the approach step")


@dataclass(frozen=True)
class ForceControlCommand:
    normalized_position_action: np.ndarray
    normal_step_m: float
    force_error_n: float
    force_rate_n_s: float
    integral_error_n_s: float
    mode: str


class CausalNormalForceController:
    def __init__(self, config: ForceControllerConfig = ForceControllerConfig()):
        config.validate()
        self.config = config
        self.reset()

    def reset(self) -> None:
        self._integral_error_n_s = 0.0
        self._previous_force_n: float | None = None
        self._contact_acquisition_steps = 0

    def command(
        self,
        *,
        measured_force_n: float,
        target_force_n: float,
        outward_normal_xyz: np.ndarray,
    ) -> ForceControlCommand:
        force = float(measured_force_n)
        target = float(target_force_n)
        normal = np.asarray(outward_normal_xyz, dtype=np.float64)
        if not math.isfinite(force) or force < 0.0 or not math.isfinite(target) or target <= 0.0:
            raise LowLevelControlError("force inputs must be finite and physically valid")
        if normal.shape != (3,) or not np.all(np.isfinite(normal)):
            raise LowLevelControlError("surface normal must be a finite three-vector")
        norm = float(np.linalg.norm(normal))
        if norm <= 0.0:
            raise LowLevelControlError("surface normal must be nonzero")
        normal = normal / norm
        cfg = self.config
        force_rate = 0.0 if self._previous_force_n is None else (
            force - self._previous_force_n
        ) / cfg.dt_s
        force_rate = float(np.clip(force_rate, -cfg.force_rate_clip_n_s, cfg.force_rate_clip_n_s))
        error = target - force

        if force >= cfg.headroom_threshold_n:
            normal_step = -cfg.max_lift_step_m
            mode = "headroom_lift"
            self._integral_error_n_s = min(self._integral_error_n_s, 0.0)
        elif force < cfg.contact_threshold_n:
            if cfg.preload_ramp_enabled:
                normal_step = min(
                    cfg.approach_step_m,
                    cfg.preload_initial_step_m
                    + cfg.preload_ramp_increment_m * self._contact_acquisition_steps,
                )
            else:
                normal_step = cfg.approach_step_m
            mode = "contact_acquisition"
            self._integral_error_n_s = 0.0
            self._contact_acquisition_steps += 1
        else:
            self._contact_acquisition_steps = 0
            candidate_integral = float(
                np.clip(
                    self._integral_error_n_s + error * cfg.dt_s,
                    -cfg.integral_clip_n_s,
                    cfg.integral_clip_n_s,
                )
            )
            raw_step = (
                cfg.kp_m_per_n * error
                + cfg.ki_m_per_n_s * candidate_integral
                - (
                    cfg.kd_m_s_per_n * force_rate
                    if cfg.impact_brake_enabled
                    else 0.0
                )
            )
            normal_step = float(
                np.clip(raw_step, -cfg.max_lift_step_m, cfg.max_press_step_m)
            )
            saturated_high = raw_step > cfg.max_press_step_m and error > 0.0
            saturated_low = raw_step < -cfg.max_lift_step_m and error < 0.0
            if not (saturated_high or saturated_low):
                self._integral_error_n_s = candidate_integral
            mode = "force_tracking"

        delta_position = -normal * normal_step
        normalized = np.clip(
            delta_position / cfg.action_position_scale_m,
            -1.0,
            1.0,
        )
        self._previous_force_n = force
        return ForceControlCommand(
            normalized_position_action=normalized,
            normal_step_m=normal_step,
            force_error_n=error,
            force_rate_n_s=force_rate,
            integral_error_n_s=self._integral_error_n_s,
            mode=mode,
        )
