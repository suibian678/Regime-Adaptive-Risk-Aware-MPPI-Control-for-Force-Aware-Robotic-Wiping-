"""Causal guard for rapid force collapse with queued inward target demand."""

from __future__ import annotations

from dataclasses import dataclass
import math


class QueuedDemandGuardError(ValueError):
    pass


@dataclass(frozen=True)
class QueuedDemandGuardConfig:
    maximum_force_fraction: float = 0.8
    maximum_force_rate_n_s: float = -100.0
    target_lead_threshold_m: float = 0.003
    maximum_release_m: float = 0.0003

    def validate(self) -> None:
        values = (
            self.maximum_force_fraction,
            self.maximum_force_rate_n_s,
            self.target_lead_threshold_m,
            self.maximum_release_m,
        )
        if not all(math.isfinite(value) for value in values):
            raise QueuedDemandGuardError("queued-demand guard values must be finite")
        if not 0.0 < self.maximum_force_fraction < 1.0:
            raise QueuedDemandGuardError("force fraction must lie in (0, 1)")
        if self.maximum_force_rate_n_s >= 0.0:
            raise QueuedDemandGuardError("force-rate trigger must be negative")
        if self.target_lead_threshold_m < 0.0 or self.maximum_release_m <= 0.0:
            raise QueuedDemandGuardError("lead threshold/release bound is invalid")


def apply_queued_inward_demand_guard(
    *,
    target_force_n: float,
    measured_force_n: float,
    force_rate_n_s: float,
    target_lead_inward_m: float,
    proposed_normal_increment_m: float,
    config: QueuedDemandGuardConfig = QueuedDemandGuardConfig(),
) -> tuple[float, dict]:
    """Return a bounded outward release when all causal triggers are present."""

    config.validate()
    target, measured, rate, lead, proposed = (
        float(target_force_n),
        float(measured_force_n),
        float(force_rate_n_s),
        float(target_lead_inward_m),
        float(proposed_normal_increment_m),
    )
    if not all(math.isfinite(value) for value in (target, measured, rate, lead, proposed)):
        raise QueuedDemandGuardError("queued-demand guard inputs must be finite")
    if target <= 0.0 or measured < 0.0 or lead < 0.0:
        raise QueuedDemandGuardError("queued-demand guard force/lead inputs are invalid")
    active = bool(
        proposed > 0.0
        and measured <= config.maximum_force_fraction * target
        and rate <= config.maximum_force_rate_n_s
        and lead > config.target_lead_threshold_m
    )
    release = (
        min(config.maximum_release_m, lead - config.target_lead_threshold_m)
        if active
        else 0.0
    )
    executed = -release if active else proposed
    return executed, {
        "queued_demand_guard_active": active,
        "queued_demand_guard_release_m": release,
        "normal_increment_before_queued_demand_guard_m": proposed,
        "queued_demand_force_threshold_n": config.maximum_force_fraction * target,
        "queued_demand_rate_threshold_n_s": config.maximum_force_rate_n_s,
        "queued_demand_lead_threshold_m": config.target_lead_threshold_m,
    }
