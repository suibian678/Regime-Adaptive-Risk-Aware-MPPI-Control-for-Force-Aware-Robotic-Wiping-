"""Causal force-regime conditioning for V19 risk-aware MPPI."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RiskConditionerConfig:
    contact_floor_n: float = 3.0
    recovery_ratio: float = 0.65
    approach_ratio: float = 0.90
    tracking_band_fraction: float = 0.15
    rising_rate_n_s: float = 80.0
    soft_force_cap_n: float = 14.5
    hard_force_cap_n: float = 15.0
    sigmoid_gain: float = 10.0
    base_contact_coef: float = 8.0
    base_tracking_coef: float = 6.0
    base_transient_coef: float = 100.0
    base_anchor_coef: float = 1.0


@dataclass(frozen=True)
class RegimeMembership:
    acquisition: float
    approach: float
    tracking: float
    recovery: float
    transient: float

    def phase_sum(self) -> float:
        return self.acquisition + self.approach + self.tracking + self.recovery


@dataclass(frozen=True)
class RiskWeights:
    contact: float
    tracking: float
    transient: float
    actor_anchor: float
    risk_score: float


@dataclass(frozen=True)
class PlannerBudget:
    regime: str
    num_samples: int
    num_elites: int
    iterations: int
    max_std: float


def _sigmoid(value: float) -> float:
    value = max(min(float(value), 60.0), -60.0)
    return 1.0 / (1.0 + math.exp(-value))


def _clip01(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


def compute_regime_membership(
    *,
    force_n: float,
    target_force_n: float,
    force_rate_n_s: float,
    predicted_upper_force_n: float,
    had_contact: bool,
    cfg: RiskConditionerConfig = RiskConditionerConfig(),
) -> RegimeMembership:
    """Compute smooth causal memberships for the four control phases.

    The memberships use only the current measured force, current force rate,
    requested force, the model's calibrated upper prediction, and an
    episode-level memory of whether contact has previously been established.
    """

    values = (force_n, target_force_n, force_rate_n_s, predicted_upper_force_n)
    if any(not math.isfinite(float(value)) for value in values):
        raise ValueError("risk-conditioning inputs must be finite")
    if target_force_n <= 0.0:
        raise ValueError("target_force_n must be positive")

    ratio = max(float(force_n), 0.0) / float(target_force_n)
    gain = float(cfg.sigmoid_gain)
    no_contact = _sigmoid(gain * (cfg.contact_floor_n - force_n) / target_force_n)
    acquisition_raw = no_contact if not had_contact else 0.0
    recovery_raw = (
        _sigmoid(gain * (cfg.recovery_ratio - ratio)) if had_contact else 0.0
    )
    approach_raw = _sigmoid(gain * (cfg.approach_ratio - ratio))
    tracking_raw = math.exp(
        -0.5
        * ((ratio - 1.0) / max(float(cfg.tracking_band_fraction), 1e-6)) ** 2
    )

    phase_raw = [acquisition_raw, approach_raw, tracking_raw, recovery_raw]
    total = sum(phase_raw)
    if total <= 0.0:
        phase_raw = [0.0, 0.0, 1.0, 0.0]
        total = 1.0
    acquisition, approach, tracking, recovery = [value / total for value in phase_raw]

    force_margin = (
        float(predicted_upper_force_n) - float(cfg.soft_force_cap_n)
    ) / max(float(cfg.hard_force_cap_n - cfg.soft_force_cap_n), 1e-6)
    rate_margin = max(float(force_rate_n_s), 0.0) / max(
        float(cfg.rising_rate_n_s), 1e-6
    )
    transient = 1.0 - (
        1.0 - _sigmoid(gain * force_margin)
    ) * (
        1.0 - _sigmoid(gain * (rate_margin - 1.0))
    )
    return RegimeMembership(
        acquisition=_clip01(acquisition),
        approach=_clip01(approach),
        tracking=_clip01(tracking),
        recovery=_clip01(recovery),
        transient=_clip01(transient),
    )


def compute_risk_weights(
    membership: RegimeMembership,
    *,
    cfg: RiskConditionerConfig = RiskConditionerConfig(),
) -> RiskWeights:
    """Map phase memberships to continuous MPPI objective weights."""

    if abs(membership.phase_sum() - 1.0) > 1e-6:
        raise ValueError("phase memberships must sum to one")
    risk_score = _clip01(
        membership.transient
        + 0.35 * membership.recovery
        + 0.15 * membership.acquisition
    )
    contact = cfg.base_contact_coef * (
        membership.acquisition + membership.recovery + 0.25 * membership.approach
    )
    tracking = cfg.base_tracking_coef * (
        0.20 + 0.80 * membership.tracking + 0.45 * membership.approach
    ) * (1.0 - 0.70 * membership.transient)
    transient = cfg.base_transient_coef * (0.10 + 1.90 * risk_score)
    actor_anchor = cfg.base_anchor_coef * (1.0 + 1.50 * risk_score)
    return RiskWeights(
        contact=float(contact),
        tracking=float(tracking),
        transient=float(transient),
        actor_anchor=float(actor_anchor),
        risk_score=float(risk_score),
    )


def choose_planner_budget(membership: RegimeMembership) -> PlannerBudget:
    """Select an interpretable planning budget without changing action semantics."""

    if membership.transient >= 0.50:
        return PlannerBudget("transient", 256, 32, 4, 0.10)
    if membership.recovery >= max(membership.acquisition, 0.35):
        return PlannerBudget("recovery", 160, 20, 3, 0.20)
    if membership.acquisition >= 0.35:
        return PlannerBudget("acquisition", 160, 20, 3, 0.20)
    if membership.approach > membership.tracking:
        return PlannerBudget("approach", 128, 16, 3, 0.16)
    return PlannerBudget("tracking", 96, 12, 2, 0.12)
