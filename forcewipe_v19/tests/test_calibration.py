import pytest

from forcewipe_v19.calibration import (
    conformal_upper_radius,
    fit_target_residual_radii,
    select_target_radius,
)


def test_conformal_radius_uses_conservative_finite_sample_rank():
    assert conformal_upper_radius([0.1, 0.2, 0.3, 0.4], coverage=0.8) == 0.4


def test_radii_are_fitted_per_force_target():
    records = [
        {"target_force_n": 5.0, "absolute_residual_n": 0.1},
        {"target_force_n": 5.0, "absolute_residual_n": 0.3},
        {"target_force_n": 8.0, "absolute_residual_n": 0.7},
        {"target_force_n": 8.0, "absolute_residual_n": 1.1},
    ]
    radii = fit_target_residual_radii(records, coverage=0.8)
    assert radii == {5.0: 0.3, 8.0: 1.1}
    assert select_target_radius(8.0, radii) == 1.1


def test_missing_target_is_rejected_instead_of_using_evaluation_data():
    with pytest.raises(ValueError, match="no unique calibrated radius"):
        select_target_radius(12.0, {5.0: 0.4, 8.0: 0.6})


def test_invalid_residual_is_rejected():
    with pytest.raises(ValueError, match="finite and non-negative"):
        conformal_upper_radius([0.1, -0.2], coverage=0.9)
