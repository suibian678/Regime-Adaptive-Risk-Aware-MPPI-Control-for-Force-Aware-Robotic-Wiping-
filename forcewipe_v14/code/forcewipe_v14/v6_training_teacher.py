"""Training-only causal teacher for the V6 direct Cartesian interface.

The teacher exists only to populate the initial replay buffer.  It is not
available to the learned policy or planner during evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class V6TrainingTeacherConfig:
    force_limit_n: float = 15.0
    contact_threshold_n: float = 3.0
    control_period_s: float = 0.01
    tangential_track_action: float = 0.50
    tangential_regulation_action: float = 0.12
    proportional_action_per_n: float = 0.105
    integral_action_per_n_s: float = 0.055
    derivative_action_per_n_s: float = 0.0018
    integral_limit_n_s: float = 5.0
    acquisition_action: float = 1.0
    reacquisition_floor_action: float = 0.30
    prediction_horizon_s: float = 0.02
    headroom_n: float = 13.5
    target_overshoot_guard_n: float = 1.5
    cross_track_full_action_m: float = 0.002

    def validate(self) -> None:
        values = tuple(float(value) for value in self.__dict__.values())
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("teacher configuration values must be finite and positive")
        if not self.contact_threshold_n < self.headroom_n < self.force_limit_n:
            raise ValueError("teacher force thresholds are inconsistent")


class V6TrainingTeacher:
    """Causal PI-D teacher expressed entirely in the learner's action space."""

    def __init__(self, config: V6TrainingTeacherConfig | None = None) -> None:
        self.config = config or V6TrainingTeacherConfig()
        self.config.validate()
        self._integral_n_s = 0.0

    def reset(self) -> None:
        self._integral_n_s = 0.0

    @staticmethod
    def _latest(observation: object) -> np.ndarray:
        array = np.asarray(observation, dtype=np.float64).reshape(-1)
        if array.size < 16 or array.size % 16:
            raise ValueError("observation must contain one or more causal 16-D frames")
        if not np.all(np.isfinite(array)):
            raise ValueError("observation must be finite")
        return array[-16:]

    def act(self, observation: object) -> np.ndarray:
        frame = self._latest(observation)
        cfg = self.config
        force_n = float(frame[0] * cfg.force_limit_n)
        target_n = float(frame[1] * cfg.force_limit_n)
        force_rate_n_s = float(frame[2] * 300.0)
        progress = float(np.clip(frame[3], 0.0, 1.0))
        cross_track_m = float(frame[4] * 0.02)

        cross = float(np.clip(-cross_track_m / cfg.cross_track_full_action_m, -1.0, 1.0))
        positive_rate = max(force_rate_n_s, 0.0)
        predicted_force = force_n + cfg.prediction_horizon_s * positive_rate
        high_gate = min(cfg.headroom_n, target_n + cfg.target_overshoot_guard_n)

        # Measured or predicted high force globally preempts inward authority.
        if force_n >= cfg.headroom_n or predicted_force >= high_gate:
            self._integral_n_s = min(self._integral_n_s, 0.0)
            excess = max(force_n - target_n, predicted_force - high_gate, 0.0)
            inward = -float(np.clip(0.30 + 0.18 * excess, 0.30, 1.0))
            return np.asarray([0.0, cross, inward], dtype=np.float32)

        # Initial acquisition is purely normal; no path progress is requested.
        if force_n < 0.5:
            self._integral_n_s = 0.0
            return np.asarray([0.0, cross, cfg.acquisition_action], dtype=np.float32)

        error_n = target_n - force_n
        if force_n < cfg.contact_threshold_n:
            self._integral_n_s = 0.0
            inward = float(np.clip(
                cfg.reacquisition_floor_action + cfg.proportional_action_per_n * error_n,
                cfg.reacquisition_floor_action,
                1.0,
            ))
            return np.asarray([0.0, cross, inward], dtype=np.float32)

        candidate_integral = float(np.clip(
            self._integral_n_s + error_n * cfg.control_period_s,
            -cfg.integral_limit_n_s,
            cfg.integral_limit_n_s,
        ))
        raw_inward = (
            cfg.proportional_action_per_n * error_n
            + cfg.integral_action_per_n_s * candidate_integral
            - cfg.derivative_action_per_n_s * max(force_rate_n_s, 0.0)
        )
        inward = float(np.clip(raw_inward, -1.0, 1.0))
        # Final-command anti-windup.
        if abs(raw_inward - inward) < 1e-12 or raw_inward * error_n <= 0.0:
            self._integral_n_s = candidate_integral

        tracking_ready = abs(error_n) <= max(0.75, 0.10 * target_n) and abs(force_rate_n_s) <= 80.0
        tangent = cfg.tangential_track_action if tracking_ready else cfg.tangential_regulation_action
        if progress >= 0.995:
            tangent = 0.0
        return np.asarray([tangent, cross, inward], dtype=np.float32)

