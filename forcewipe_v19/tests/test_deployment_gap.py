import numpy as np
import pytest

from forcewipe_v19.deployment_gap import (
    DeploymentGapConfig,
    LearnerObservationPerturbation,
    deployment_gap_conditions,
    rotated_task_frame,
)


def observation(force=6.0):
    value = np.zeros(16, dtype=np.float32)
    value[0] = force / 15.0
    value[1] = 8.0 / 15.0
    value[13] = float(force >= 3.0)
    return value


def test_one_factor_contract_rejects_confounded_stress():
    with pytest.raises(ValueError, match="one-factor"):
        DeploymentGapConfig(force_bias_n=0.8, observation_delay_samples=2).validate()


def test_nominal_filter_is_bitwise_identity():
    native = observation()
    filt = LearnerObservationPerturbation(DeploymentGapConfig(random_seed=1))
    visible, log = filt.transform(native, native_force_n=6.0, step=0, initial=True)
    assert np.array_equal(visible, native)
    assert log["native_observation"] == log["learner_visible_observation"]


def test_force_bias_changes_only_force_force_rate_and_contact_channels():
    native = observation(2.5)
    filt = LearnerObservationPerturbation(
        DeploymentGapConfig(force_bias_n=0.8, random_seed=2)
    )
    visible, _ = filt.transform(native, native_force_n=2.5, step=0, initial=True)
    changed = set(np.flatnonzero(visible != native).tolist())
    assert changed <= {0, 2, 13}
    assert visible[0] == pytest.approx(3.3 / 15.0)
    assert visible[13] == 1.0


def test_one_sample_delay_returns_previous_observation():
    filt = LearnerObservationPerturbation(
        DeploymentGapConfig(observation_delay_samples=1, random_seed=3)
    )
    initial = observation(0.0)
    filt.transform(initial, native_force_n=0.0, step=0, initial=True)
    current = observation(6.0)
    visible, log = filt.transform(current, native_force_n=6.0, step=1)
    assert np.array_equal(visible, initial)
    assert log["perturbed_current_observation"] == current.tolist()


def test_tcp_error_has_exact_frozen_magnitude_in_cross_normal_plane():
    magnitude = 0.0001
    filt = LearnerObservationPerturbation(
        DeploymentGapConfig(tcp_pose_error_magnitude_m=magnitude, random_seed=4)
    )
    visible, log = filt.transform(observation(), native_force_n=6.0, step=0, initial=True)
    assert np.linalg.norm(log["tcp_bias_cross_normal_m"]) == pytest.approx(magnitude)
    assert visible[4] != 0.0 or visible[5] != 0.0


def test_registered_frame_rotation_is_orthonormal_and_exact_magnitude():
    tangent, outward, signed_deg = rotated_task_frame(
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        magnitude_deg=5.0,
        seed=5,
    )
    assert np.linalg.norm(tangent) == pytest.approx(1.0)
    assert np.linalg.norm(outward) == pytest.approx(1.0)
    assert float(np.dot(tangent, outward)) == pytest.approx(0.0, abs=1e-12)
    assert abs(signed_deg) == pytest.approx(5.0)


def test_frozen_roster_has_375_new_evaluations_and_no_nominal_duplicates():
    per_block = {
        block: deployment_gap_conditions(block, random_seed=201)
        for block in ("B0", "S1", "S2")
    }
    assert len(per_block["B0"]) == 13
    assert len(per_block["S1"]) == len(per_block["S2"]) == 6
    evaluations = sum(len(rows) * 5 * 3 for rows in per_block.values())
    assert evaluations == 375
    assert all(row.config.factor != "nominal" for rows in per_block.values() for row in rows)
