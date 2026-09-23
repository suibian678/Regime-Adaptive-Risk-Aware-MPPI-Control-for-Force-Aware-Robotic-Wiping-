"""Target-pose normal-force controller with smooth low-force P-gain scheduling."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ScheduledPConfig:
    dt_s: float = 0.01
    base_gain_m_per_n_step: float = 0.00005
    additional_low_force_gain_m_per_n_step: float = 0.00005
    gain_schedule_start_error_n: float = 0.5
    gain_schedule_full_error_n: float = 1.5
    positive_rate_damping_m_per_n_per_s: float = 0.000001
    maximum_inward_increment_m: float = 0.0008
    maximum_outward_increment_m: float = 0.0008
    approach_increment_m: float = 0.0005
    maximum_target_lead_m: float = 0.01
    contact_threshold_n: float = 0.2
    headroom_enter_n: float = 13.5
    projected_bound_n: float = 14.5
    force_limit_n: float = 15.0
    rate_horizon_s: float = 0.02
    candidate_gain_n_per_m: float = 500.0
    envelope_margin_n: float = 0.3

    def validate(self) -> None:
        if self.dt_s <= 0.0:
            raise ValueError("dt_s must be positive")
        if self.gain_schedule_start_error_n < 0.0:
            raise ValueError("gain schedule start must be nonnegative")
        if self.gain_schedule_full_error_n <= self.gain_schedule_start_error_n:
            raise ValueError("gain schedule full point must exceed start point")
        if self.base_gain_m_per_n_step <= 0.0 or self.additional_low_force_gain_m_per_n_step < 0.0:
            raise ValueError("invalid force-error gains")
        if not 0.0 < self.contact_threshold_n < self.headroom_enter_n:
            raise ValueError("invalid contact/headroom thresholds")
        if not self.headroom_enter_n < self.projected_bound_n < self.force_limit_n:
            raise ValueError("invalid projected/force limits")


@dataclass(frozen=True)
class ScheduledPDecision:
    mode: str
    normal_increment_m: float
    force_rate_n_s: float
    force_error_n: float
    gain_schedule_scale: float
    effective_proportional_gain_m_per_n_step: float
    raw_normal_increment_m: float
    actuator_limited_increment_m: float
    executed_normal_increment_m: float
    envelope_base_n: float
    projection_active: bool
    actuator_saturation_active: bool

    def to_dict(self) -> dict:
        return asdict(self)


class ScheduledPController:
    def __init__(self, config: ScheduledPConfig | None = None):
        self.config = config or ScheduledPConfig()
        self.config.validate()
        self.previous_force_n = 0.0

    def reset(self) -> None:
        self.previous_force_n = 0.0

    def _gain_scale(self, error_n: float) -> float:
        config = self.config
        if error_n <= config.gain_schedule_start_error_n:
            return 0.0
        if error_n >= config.gain_schedule_full_error_n:
            return 1.0
        return (
            (error_n - config.gain_schedule_start_error_n)
            / (config.gain_schedule_full_error_n - config.gain_schedule_start_error_n)
        )

    def step(
        self,
        *,
        target_force_n: float,
        measured_force_n: float,
        target_lead_down_m: float,
    ) -> ScheduledPDecision:
        config = self.config
        if target_force_n <= 0.0 or measured_force_n < 0.0 or target_lead_down_m < 0.0:
            raise ValueError("invalid causal controller input")
        rate = max(
            -300.0,
            min(300.0, (measured_force_n - self.previous_force_n) / config.dt_s),
        )
        self.previous_force_n = measured_force_n
        error = target_force_n - measured_force_n
        envelope_base = measured_force_n + max(rate, 0.0) * config.rate_horizon_s + config.envelope_margin_n

        if measured_force_n < config.contact_threshold_n:
            executed = 0.0 if target_lead_down_m >= config.maximum_target_lead_m else config.approach_increment_m
            return ScheduledPDecision(
                mode="APPROACH",
                normal_increment_m=executed,
                force_rate_n_s=rate,
                force_error_n=error,
                gain_schedule_scale=0.0,
                effective_proportional_gain_m_per_n_step=config.base_gain_m_per_n_step,
                raw_normal_increment_m=config.approach_increment_m,
                actuator_limited_increment_m=executed,
                executed_normal_increment_m=executed,
                envelope_base_n=envelope_base,
                projection_active=executed == 0.0,
                actuator_saturation_active=False,
            )

        if measured_force_n >= config.headroom_enter_n:
            executed = -min(
                config.maximum_outward_increment_m,
                0.0003 + 0.0001 * (measured_force_n - config.headroom_enter_n),
            )
            return ScheduledPDecision(
                mode="HEADROOM",
                normal_increment_m=executed,
                force_rate_n_s=rate,
                force_error_n=error,
                gain_schedule_scale=0.0,
                effective_proportional_gain_m_per_n_step=config.base_gain_m_per_n_step,
                raw_normal_increment_m=executed,
                actuator_limited_increment_m=executed,
                executed_normal_increment_m=executed,
                envelope_base_n=envelope_base,
                projection_active=True,
                actuator_saturation_active=False,
            )

        schedule_scale = self._gain_scale(error)
        effective_gain = (
            config.base_gain_m_per_n_step
            + schedule_scale * config.additional_low_force_gain_m_per_n_step
        )
        raw = effective_gain * error - config.positive_rate_damping_m_per_n_per_s * max(rate, 0.0)
        actuator_limited = max(
            -config.maximum_outward_increment_m,
            min(config.maximum_inward_increment_m, raw),
        )
        executed = actuator_limited
        if actuator_limited > 0.0:
            allowed = max(
                0.0,
                (config.projected_bound_n - envelope_base) / config.candidate_gain_n_per_m,
            )
            executed = min(executed, allowed)
        return ScheduledPDecision(
            mode="TRACK",
            normal_increment_m=executed,
            force_rate_n_s=rate,
            force_error_n=error,
            gain_schedule_scale=schedule_scale,
            effective_proportional_gain_m_per_n_step=effective_gain,
            raw_normal_increment_m=raw,
            actuator_limited_increment_m=actuator_limited,
            executed_normal_increment_m=executed,
            envelope_base_n=envelope_base,
            projection_active=abs(executed - actuator_limited) > 1e-15,
            actuator_saturation_active=abs(actuator_limited - raw) > 1e-15,
        )
