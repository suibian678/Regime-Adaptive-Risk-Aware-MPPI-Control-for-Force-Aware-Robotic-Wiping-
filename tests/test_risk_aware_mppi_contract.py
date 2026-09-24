from types import SimpleNamespace

import pytest

from forcewipe.risk_aware_mppi import (
    compose_upper_prediction,
    configure_regime_adaptive_risk_mppi,
)


def _cfg():
    return SimpleNamespace(
        force_obs_idx=0,
        force_plan_target_obs_idx=1,
    )


def test_config_removes_fixed_force_penalty_and_freezes_calibration():
    cfg = configure_regime_adaptive_risk_mppi(
        _cfg(),
        calibrated_radii_n={5.0: 0.4, 8.0: 0.6, 12.0: 0.9},
    )
    assert cfg.mpc is True
    assert cfg.force_plan_coef == 0.0
    assert cfg.ra_uncertainty_calibrated is True
    assert cfg.ra_uncertainty_radii_n == [0.4, 0.6, 0.9]
    assert cfg.num_samples == 128
    assert cfg.num_elites == 16
    assert cfg.ra_adaptive_objective is True
    assert cfg.ra_adaptive_compute is True
    assert cfg.ra_base_tracking_coef == 6.0


def test_matched_ablation_switches_are_explicit():
    cfg = configure_regime_adaptive_risk_mppi(
        _cfg(),
        calibrated_radii_n={5.0: 0.4, 8.0: 0.6, 12.0: 0.9},
        adaptive_objective=False,
        adaptive_compute=False,
    )
    assert cfg.ra_adaptive_objective is False
    assert cfg.ra_adaptive_compute is False
    assert cfg.ra_fixed_num_samples == 128
    assert cfg.ra_fixed_iterations == 4


def test_config_rejects_missing_target_calibration():
    with pytest.raises(ValueError, match="no unique calibrated radius"):
        configure_regime_adaptive_risk_mppi(
            _cfg(),
            calibrated_radii_n={5.0: 0.4, 8.0: 0.6},
        )


def test_config_rejects_unknown_envelope_mode():
    with pytest.raises(ValueError, match="unknown risk_envelope_mode"):
        configure_regime_adaptive_risk_mppi(
            _cfg(),
            calibrated_radii_n={5.0: 0.4, 8.0: 0.6, 12.0: 0.9},
            risk_envelope_mode="unknown",
        )


def test_transient_risk_representation_ladder():
    force = 0.60
    envelope = 0.70
    radius = 0.05
    assert compose_upper_prediction(
        force, envelope, radius, mode="force_only"
    ) == pytest.approx(0.60)
    assert compose_upper_prediction(
        force, envelope, radius, mode="point_envelope"
    ) == pytest.approx(0.70)
    assert compose_upper_prediction(
        force, envelope, radius, mode="calibrated"
    ) == pytest.approx(0.75)
