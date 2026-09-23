"""Stateful normal-force loop and force-first tangential governor.

The module is simulator-independent. It produces a normal target-pose
increment and a [0, 1] tangential advancement scale from causal force inputs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class HybridForceFirstConfig:
    dt_s: float = 0.01
    proportional_increment_m_per_n: float = 0.00005
    integral_increment_m_per_n_step: float = 0.000001
    positive_rate_damping_m_per_n_per_s: float = 0.000001
    integral_bias_limit_m: float = 0.0012
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
    integral_enable_fraction: float = 0.8
    integral_enable_maximum_error_n: float = 2.0
    governor_full_speed_error_n: float = 0.25
    governor_stop_error_n: float = 1.0

    def validate(self) -> None:
        if self.dt_s <= 0.0:
            raise ValueError("dt_s must be positive")
        if not 0.0 < self.contact_threshold_n < self.headroom_enter_n:
            raise ValueError("invalid contact/headroom thresholds")
        if not self.headroom_enter_n < self.projected_bound_n < self.force_limit_n:
            raise ValueError("invalid projected/force limits")
        if self.governor_full_speed_error_n < 0.0:
            raise ValueError("governor full-speed error must be nonnegative")
        if self.governor_stop_error_n <= self.governor_full_speed_error_n:
            raise ValueError("governor stop error must exceed full-speed error")
        if self.integral_bias_limit_m < 0.0:
            raise ValueError("integral bias limit must be nonnegative")


@dataclass
class HybridForceFirstState:
    previous_force_n: float = 0.0
    integral_bias_m: float = 0.0


@dataclass(frozen=True)
class HybridForceFirstDecision:
    mode: str
    normal_increment_m: float
    tangential_scale: float
    measured_force_n: float
    force_rate_n_s: float
    force_error_n: float
    integral_bias_before_m: float
    integral_bias_after_m: float
    raw_normal_increment_m: float
    actuator_limited_increment_m: float
    executed_normal_increment_m: float
    envelope_base_n: float
    projection_active: bool
    actuator_saturation_active: bool
    integrator_update_applied: bool
    tangential_governor_active: bool

    def to_dict(self) -> dict:
        return asdict(self)


class HybridForceFirstController:
    def __init__(self, config: HybridForceFirstConfig | None = None):
        self.config = config or HybridForceFirstConfig()
        self.config.validate()
        self.state = HybridForceFirstState()

    def reset(self) -> None:
        self.state = HybridForceFirstState()

    def _governor_scale(self, error_n: float) -> float:
        config = self.config
        if error_n <= config.governor_full_speed_error_n:
            return 1.0
        if error_n >= config.governor_stop_error_n:
            return 0.0
        span = config.governor_stop_error_n - config.governor_full_speed_error_n
        return (config.governor_stop_error_n - error_n) / span

    def step(
        self,
        *,
        target_force_n: float,
        measured_force_n: float,
        target_lead_down_m: float,
    ) -> HybridForceFirstDecision:
        config = self.config
        if target_force_n <= 0.0 or measured_force_n < 0.0 or target_lead_down_m < 0.0:
            raise ValueError("force targets, measurements, and target lead must be valid")
        force_rate = max(
            -300.0,
            min(300.0, (measured_force_n - self.state.previous_force_n) / config.dt_s),
        )
        self.state.previous_force_n = measured_force_n
        error = target_force_n - measured_force_n
        integral_before = self.state.integral_bias_m
        envelope_base = (
            measured_force_n
            + max(force_rate, 0.0) * config.rate_horizon_s
            + config.envelope_margin_n
        )

        if measured_force_n < config.contact_threshold_n:
            executed = (
                0.0
                if target_lead_down_m >= config.maximum_target_lead_m
                else config.approach_increment_m
            )
            self.state.integral_bias_m = 0.0
            return HybridForceFirstDecision(
                mode="APPROACH",
                normal_increment_m=executed,
                tangential_scale=0.0,
                measured_force_n=measured_force_n,
                force_rate_n_s=force_rate,
                force_error_n=error,
                integral_bias_before_m=integral_before,
                integral_bias_after_m=0.0,
                raw_normal_increment_m=config.approach_increment_m,
                actuator_limited_increment_m=executed,
                executed_normal_increment_m=executed,
                envelope_base_n=envelope_base,
                projection_active=executed == 0.0,
                actuator_saturation_active=False,
                integrator_update_applied=False,
                tangential_governor_active=True,
            )

        if measured_force_n >= config.headroom_enter_n or envelope_base >= config.projected_bound_n:
            excess = max(
                0.0,
                measured_force_n - config.headroom_enter_n,
                envelope_base - config.projected_bound_n,
            )
            executed = -min(
                config.maximum_outward_increment_m,
                0.0003 + 0.0001 * excess,
            )
            self.state.integral_bias_m = 0.0
            return HybridForceFirstDecision(
                mode="HEADROOM",
                normal_increment_m=executed,
                tangential_scale=0.0,
                measured_force_n=measured_force_n,
                force_rate_n_s=force_rate,
                force_error_n=error,
                integral_bias_before_m=integral_before,
                integral_bias_after_m=0.0,
                raw_normal_increment_m=executed,
                actuator_limited_increment_m=executed,
                executed_normal_increment_m=executed,
                envelope_base_n=envelope_base,
                projection_active=True,
                actuator_saturation_active=False,
                integrator_update_applied=False,
                tangential_governor_active=True,
            )

        raw = (
            config.proportional_increment_m_per_n * error
            + integral_before
            - config.positive_rate_damping_m_per_n_per_s * max(force_rate, 0.0)
        )
        actuator_limited = max(
            -config.maximum_outward_increment_m,
            min(config.maximum_inward_increment_m, raw),
        )
        executed = actuator_limited
        if actuator_limited > 0.0:
            allowed = max(
                0.0,
                (config.projected_bound_n - envelope_base)
                / config.candidate_gain_n_per_m,
            )
            executed = min(executed, allowed)
        actuator_saturation = abs(actuator_limited - raw) > 1e-15
        projection_active = abs(executed - actuator_limited) > 1e-15

        integrator_eligible = bool(
            measured_force_n >= config.integral_enable_fraction * target_force_n
            and abs(error) <= config.integral_enable_maximum_error_n
            and not actuator_saturation
            and not projection_active
        )
        integral_after = integral_before
        if integrator_eligible:
            proposed_integral = (
                integral_before
                + config.integral_increment_m_per_n_step * error
            )
            integral_after = max(
                -config.integral_bias_limit_m,
                min(config.integral_bias_limit_m, proposed_integral),
            )
        self.state.integral_bias_m = integral_after

        tangential_scale = self._governor_scale(error)
        return HybridForceFirstDecision(
            mode="TRACK",
            normal_increment_m=executed,
            tangential_scale=tangential_scale,
            measured_force_n=measured_force_n,
            force_rate_n_s=force_rate,
            force_error_n=error,
            integral_bias_before_m=integral_before,
            integral_bias_after_m=integral_after,
            raw_normal_increment_m=raw,
            actuator_limited_increment_m=actuator_limited,
            executed_normal_increment_m=executed,
            envelope_base_n=envelope_base,
            projection_active=projection_active,
            actuator_saturation_active=actuator_saturation,
            integrator_update_applied=integrator_eligible,
            tangential_governor_active=tangential_scale < 1.0,
        )
