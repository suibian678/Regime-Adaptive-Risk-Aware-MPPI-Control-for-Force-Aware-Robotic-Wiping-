import numpy as np

from forcewipe_v19.crosssim_calibration import (
    MOTIONS,
    aggregate_motion_metrics,
    comparison_metrics,
)


def test_calibration_motions_are_policy_independent_finite_action_sequences():
    assert {motion.role for motion in MOTIONS} == {"fit", "validation"}
    assert len({motion.motion_id for motion in MOTIONS}) == len(MOTIONS)
    for motion in MOTIONS:
        actions = motion.actions()
        assert actions.shape[1] == 3
        assert actions.shape[0] >= 275
        assert np.isfinite(actions).all()
        assert np.max(np.abs(actions)) <= 1.0


def test_identical_traces_have_zero_error_and_pass():
    trace = {
        "normal_force_n": [0.0, 0.2, 1.0],
        "tool_position_world_m": [[0.0, 0.0, 0.1], [0.0, 0.0, 0.099], [0.0, 0.0, 0.098]],
    }
    metrics = comparison_metrics(trace, trace)
    assert metrics["maximum_tolerance_ratio"] == 0.0
    assert metrics["within_all_tolerances"] is True
    assert aggregate_motion_metrics([metrics])["within_all_tolerances"] is True


def test_missing_candidate_contact_fails_validation():
    reference = {
        "normal_force_n": [0.0, 1.0, 2.0],
        "tool_position_world_m": [[0.0, 0.0, 0.1]] * 3,
    }
    candidate = {
        "normal_force_n": [0.0, 0.0, 0.0],
        "tool_position_world_m": [[0.0, 0.0, 0.1]] * 3,
    }
    metrics = comparison_metrics(reference, candidate)
    assert metrics["within_all_tolerances"] is False
    assert metrics["candidate_contact_onset_sample"] is None
