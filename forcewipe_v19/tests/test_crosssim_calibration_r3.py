import numpy as np

from forcewipe_v19.crosssim_calibration_r3 import (
    ACTUATOR_FIT_MOTIONS,
    HELD_OUT_MOTIONS,
)


def test_r3_fit_and_held_out_motion_names_and_roles_are_disjoint():
    fit_ids = {motion.motion_id for motion in ACTUATOR_FIT_MOTIONS}
    held_ids = {motion.motion_id for motion in HELD_OUT_MOTIONS}
    assert len(fit_ids) == len(ACTUATOR_FIT_MOTIONS)
    assert len(held_ids) == len(HELD_OUT_MOTIONS)
    assert fit_ids.isdisjoint(held_ids)
    assert {motion.role for motion in ACTUATOR_FIT_MOTIONS} == {"actuator_fit"}
    assert {motion.role for motion in HELD_OUT_MOTIONS} == {"validation"}


def test_r3_motion_actions_are_finite_and_inside_the_deployment_box():
    for motion in (*ACTUATOR_FIT_MOTIONS, *HELD_OUT_MOTIONS):
        actions = motion.actions()
        assert actions.ndim == 2 and actions.shape[1] == 3
        assert np.isfinite(actions).all()
        assert np.max(np.abs(actions)) <= 1.0


def test_actuator_fit_motions_do_not_command_inward_normal_motion_from_reset():
    for motion in ACTUATOR_FIT_MOTIONS:
        actions = motion.actions()
        cumulative_inward = np.cumsum(actions[:, 2])
        assert np.max(cumulative_inward) <= 0.0
