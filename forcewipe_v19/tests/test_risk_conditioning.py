import math

from forcewipe_v19.risk_conditioning import (
    choose_planner_budget,
    compute_regime_membership,
    compute_risk_weights,
)


def test_acquisition_dominates_before_first_contact():
    membership = compute_regime_membership(
        force_n=0.1,
        target_force_n=8.0,
        force_rate_n_s=0.0,
        predicted_upper_force_n=5.0,
        had_contact=False,
    )
    assert membership.acquisition > membership.recovery
    assert membership.acquisition > membership.tracking
    assert math.isclose(membership.phase_sum(), 1.0, abs_tol=1e-9)
    assert choose_planner_budget(membership).regime == "acquisition"


def test_recovery_is_distinct_from_initial_acquisition():
    membership = compute_regime_membership(
        force_n=1.0,
        target_force_n=8.0,
        force_rate_n_s=-40.0,
        predicted_upper_force_n=4.0,
        had_contact=True,
    )
    assert membership.recovery > membership.acquisition
    assert choose_planner_budget(membership).regime == "recovery"


def test_tracking_is_low_compute_near_target():
    membership = compute_regime_membership(
        force_n=8.0,
        target_force_n=8.0,
        force_rate_n_s=0.0,
        predicted_upper_force_n=8.5,
        had_contact=True,
    )
    budget = choose_planner_budget(membership)
    weights = compute_risk_weights(membership)
    assert membership.tracking > membership.approach
    assert membership.transient < 0.1
    assert budget.regime == "tracking"
    assert budget.num_samples == 96
    assert weights.tracking > weights.contact


def test_transient_increases_compute_and_tightens_exploration():
    safe = compute_regime_membership(
        force_n=8.0,
        target_force_n=8.0,
        force_rate_n_s=0.0,
        predicted_upper_force_n=8.5,
        had_contact=True,
    )
    risky = compute_regime_membership(
        force_n=11.5,
        target_force_n=12.0,
        force_rate_n_s=140.0,
        predicted_upper_force_n=14.7,
        had_contact=True,
    )
    safe_budget = choose_planner_budget(safe)
    risky_budget = choose_planner_budget(risky)
    safe_weights = compute_risk_weights(safe)
    risky_weights = compute_risk_weights(risky)
    assert risky_budget.regime == "transient"
    assert risky_budget.num_samples > safe_budget.num_samples
    assert risky_budget.max_std < safe_budget.max_std
    assert risky_weights.transient > safe_weights.transient
    assert risky_weights.tracking < safe_weights.tracking
