"""Continuous-time-equivalent scaling of the passing target-pose force law."""

from __future__ import annotations

from dataclasses import dataclass
import math


class RateScaledForceError(ValueError):
    pass


@dataclass(frozen=True)
class RateScaledForceConfig:
    control_hz: float = 500.0
    reference_hz: float = 100.0
    force_error_gain_m_per_n_at_reference_step: float = 0.00005
    positive_rate_damping_m_per_n_s_at_reference_step: float = 0.000001
    maximum_inward_increment_m_at_reference_step: float = 0.0008
    maximum_outward_increment_m_at_reference_step: float = 0.0008
    approach_increment_m_at_reference_step: float = 0.0005
    maximum_target_lead_m: float = 0.01
    contact_threshold_n: float = 0.2
    headroom_enter_n: float = 13.5
    projected_bound_n: float = 14.5
    rate_horizon_s: float = 0.02
    candidate_gain_n_per_m: float = 500.0
    envelope_margin_n: float = 0.3
    force_rate_clip_n_s: float = 300.0

    def validate(self) -> None:
        positive = (
            self.control_hz, self.reference_hz,
            self.force_error_gain_m_per_n_at_reference_step,
            self.positive_rate_damping_m_per_n_s_at_reference_step,
            self.maximum_inward_increment_m_at_reference_step,
            self.maximum_outward_increment_m_at_reference_step,
            self.approach_increment_m_at_reference_step,
            self.maximum_target_lead_m, self.contact_threshold_n,
            self.projected_bound_n, self.rate_horizon_s,
            self.candidate_gain_n_per_m, self.envelope_margin_n,
            self.force_rate_clip_n_s,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in positive):
            raise RateScaledForceError("rate-scaled force parameters must be positive and finite")
        if not math.isfinite(self.headroom_enter_n) or self.headroom_enter_n >= self.projected_bound_n:
            raise RateScaledForceError("headroom threshold must precede the projected bound")

    @property
    def step_scale(self) -> float:
        return self.reference_hz / self.control_hz

    @property
    def dt_s(self) -> float:
        return 1.0 / self.control_hz


def rate_scaled_normal_increment(
    *,
    target_force_n: float,
    measured_force_n: float,
    previous_force_n: float,
    target_lead_down_m: float,
    config: RateScaledForceConfig = RateScaledForceConfig(),
) -> tuple[float, dict]:
    config.validate()
    target, measured, previous, lead = map(
        float, (target_force_n, measured_force_n, previous_force_n, target_lead_down_m)
    )
    if not all(math.isfinite(value) for value in (target, measured, previous, lead)):
        raise RateScaledForceError("rate-scaled force inputs must be finite")
    if target <= 0.0 or measured < 0.0 or previous < 0.0 or lead < 0.0:
        raise RateScaledForceError("rate-scaled force inputs are invalid")
    scale = config.step_scale
    maximum_inward = config.maximum_inward_increment_m_at_reference_step * scale
    maximum_outward = config.maximum_outward_increment_m_at_reference_step * scale
    approach = config.approach_increment_m_at_reference_step * scale
    rate = max(
        -config.force_rate_clip_n_s,
        min(config.force_rate_clip_n_s, (measured - previous) / config.dt_s),
    )
    if measured < config.contact_threshold_n:
        increment = 0.0 if lead >= config.maximum_target_lead_m else approach
        return increment, {
            "mode": "APPROACH",
            "force_rate_n_s": rate,
            "force_error_n": target - measured,
            "raw_increment_m": approach,
            "projected_increment_m": increment,
            "projection_active": increment == 0.0,
            "control_hz": config.control_hz,
        }
    if measured >= config.headroom_enter_n:
        increment = -min(
            maximum_outward,
            scale * (0.0003 + 0.0001 * (measured - config.headroom_enter_n)),
        )
        return increment, {
            "mode": "HEADROOM",
            "force_rate_n_s": rate,
            "force_error_n": target - measured,
            "raw_increment_m": increment,
            "projected_increment_m": increment,
            "projection_active": True,
            "control_hz": config.control_hz,
        }
    error = target - measured
    raw_increment = scale * (
        config.force_error_gain_m_per_n_at_reference_step * error
        - config.positive_rate_damping_m_per_n_s_at_reference_step * max(rate, 0.0)
    )
    actuator_limited = max(-maximum_outward, min(maximum_inward, raw_increment))
    envelope_base = measured + max(rate, 0.0) * config.rate_horizon_s + config.envelope_margin_n
    projected = actuator_limited
    if actuator_limited > 0.0:
        allowed = max(0.0, (config.projected_bound_n - envelope_base) / config.candidate_gain_n_per_m)
        projected = min(projected, allowed)
    return projected, {
        "mode": "TRACK",
        "force_rate_n_s": rate,
        "force_error_n": error,
        "raw_increment_m": raw_increment,
        "actuator_limited_increment_m": actuator_limited,
        "projected_increment_m": projected,
        "projection_active": abs(projected - raw_increment) > 1.0e-15,
        "envelope_base_n": envelope_base,
        "control_hz": config.control_hz,
    }
