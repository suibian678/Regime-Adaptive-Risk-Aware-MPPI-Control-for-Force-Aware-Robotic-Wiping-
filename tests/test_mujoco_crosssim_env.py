from dataclasses import replace

import numpy as np
import pytest

from forcewipe.factor_separated_scenarios import factor_separated_scenario
from forcewipe.mujoco_crosssim_env import (
    MuJoCoContactConfig,
    MuJoCoDirectFirstPassEnv,
)


def test_contact_configuration_requires_integral_substeps_and_safe_solref():
    MuJoCoContactConfig().validate()
    assert MuJoCoContactConfig().substeps == 10
    with pytest.raises(ValueError, match="integer number"):
        MuJoCoContactConfig(physics_timestep_s=0.003).validate()
    with pytest.raises(ValueError, match="refsafety"):
        MuJoCoContactConfig(contact_time_constant_s=0.001).validate()
    with pytest.raises(ValueError, match="actuation_gain"):
        MuJoCoContactConfig(actuation_gain=1.01).validate()
    with pytest.raises(ValueError, match="actuation_time_constant_s"):
        MuJoCoContactConfig(actuation_time_constant_s=-0.01).validate()
    with pytest.raises(ValueError, match="actuation_delay_steps"):
        MuJoCoContactConfig(actuation_delay_steps=-1).validate()
    with pytest.raises(ValueError, match="contact_impedance_min"):
        MuJoCoContactConfig(contact_impedance_min=0.0).validate()
    with pytest.raises(ValueError, match="contact_impedance_max"):
        MuJoCoContactConfig(
            contact_impedance_min=0.8,
            contact_impedance_max=0.7,
        ).validate()
    with pytest.raises(ValueError, match="contact_impedance_width_m"):
        MuJoCoContactConfig(contact_impedance_width_m=0.0).validate()
    with pytest.raises(ValueError, match="drive implementation scales"):
        MuJoCoContactConfig(tool_drive_scale=0.0).validate()
    with pytest.raises(ValueError, match="drive implementation scales"):
        MuJoCoContactConfig(support_drive_scale=0.0).validate()
    with pytest.raises(ValueError, match="tool_tangential_drive_scale"):
        MuJoCoContactConfig(tool_tangential_drive_scale=0.0).validate()
    with pytest.raises(ValueError, match="friction_scale"):
        MuJoCoContactConfig(friction_scale=0.0).validate()


def test_null_tangential_scale_inherits_normal_drive_implementation_scale():
    contact = MuJoCoContactConfig(tool_drive_scale=0.8)
    assert contact.effective_tangential_drive_scale == 0.8
    explicit = MuJoCoContactConfig(
        tool_drive_scale=0.8,
        tool_tangential_drive_scale=0.3,
    )
    assert explicit.effective_tangential_drive_scale == 0.3


@pytest.mark.parametrize("block_id", ["B0", "S1", "S2"])
def test_flat_incline_and_mesh_models_reset_with_finite_task_observation(block_id):
    env = MuJoCoDirectFirstPassEnv(
        scenario_spec=factor_separated_scenario(block_id, 8.0)
    )
    observation, info = env.reset(seed=17)
    assert observation.shape == (16,)
    assert np.isfinite(observation).all()
    assert info["simulator"] == "MuJoCo"
    assert info["direct_tdmpc2_action_authority"] is True
    assert info["force_dependent_action_projection"] is False
    assert info["rewiping_enabled"] is False


def test_unmodelled_disturbance_path_is_rejected_instead_of_silently_rebound():
    spec = replace(
        factor_separated_scenario("B0", 8.0),
        disturbance_kind="normal_impulse",
        disturbance_scale=1.0,
    )
    with pytest.raises(ValueError, match="disturbance equivalence is not calibrated"):
        MuJoCoDirectFirstPassEnv(scenario_spec=spec)


def test_normalised_inward_action_builds_contact_and_preserves_authority_fields():
    env = MuJoCoDirectFirstPassEnv(
        scenario_spec=factor_separated_scenario("B0", 8.0)
    )
    observation, _ = env.reset(seed=23)
    contacted = False
    for _ in range(300):
        observation, reward, terminated, truncated, info = env.step(
            np.array([0.0, 0.0, 1.0], dtype=np.float32)
        )
        assert np.isfinite(observation).all()
        assert np.isfinite(reward)
        assert info["direct_tdmpc2_action_authority"] is True
        assert info["force_dependent_action_projection"] is False
        assert info["rewiping_enabled"] is False
        contacted |= info["normal_force_n"] > 0.0
        if terminated or truncated:
            break
    assert contacted
    assert info["peak_force_n"] > 0.0


def test_invalid_action_fails_before_a_physics_step():
    env = MuJoCoDirectFirstPassEnv(
        scenario_spec=factor_separated_scenario("B0", 5.0)
    )
    env.reset(seed=29)
    elapsed = env._elapsed_steps
    with pytest.raises(Exception, match="invalid shape/value"):
        env.step(np.array([0.0, np.nan, 0.0], dtype=np.float32))
    assert env._elapsed_steps == elapsed


def test_calibrated_actuation_gain_scales_command_without_changing_policy_action():
    env = MuJoCoDirectFirstPassEnv(
        scenario_spec=factor_separated_scenario("B0", 5.0),
        contact_config=MuJoCoContactConfig(actuation_gain=0.8),
    )
    env.reset(seed=31)
    _, _, _, _, info = env.step(np.array([0.0, 0.0, 1.0], dtype=np.float32))
    requested = np.asarray(info["requested_world_delta_m"])
    issued = np.asarray(info["issued_world_delta_m"])
    assert np.allclose(issued, 0.8 * requested)
    assert info["tdmpc2_action"] == (0.0, 0.0, 1.0)


def test_causal_actuation_lag_filters_command_without_changing_policy_action():
    contact = MuJoCoContactConfig(actuation_time_constant_s=0.09)
    env = MuJoCoDirectFirstPassEnv(
        scenario_spec=factor_separated_scenario("B0", 5.0),
        contact_config=contact,
    )
    env.reset(seed=37)
    _, _, _, _, info = env.step(np.array([0.0, 0.0, 1.0], dtype=np.float32))
    requested = np.asarray(info["requested_world_delta_m"])
    commanded = np.asarray(info["commanded_world_delta_m"])
    issued = np.asarray(info["issued_world_delta_m"])
    assert np.allclose(commanded, requested)
    assert np.allclose(issued, contact.actuation_response_fraction * requested)
    assert np.linalg.norm(issued) < np.linalg.norm(commanded)
    assert info["tdmpc2_action"] == (0.0, 0.0, 1.0)


def test_causal_integer_delay_defers_the_servo_target_without_losing_commands():
    contact = MuJoCoContactConfig(actuation_delay_steps=2)
    env = MuJoCoDirectFirstPassEnv(
        scenario_spec=factor_separated_scenario("B0", 5.0),
        contact_config=contact,
    )
    env.reset(seed=41)
    action = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    _, _, _, _, first = env.step(action)
    _, _, _, _, second = env.step(action)
    _, _, _, _, third = env.step(action)
    assert np.allclose(first["issued_world_delta_m"], 0.0)
    assert np.allclose(second["issued_world_delta_m"], 0.0)
    assert np.allclose(third["issued_world_delta_m"], first["requested_world_delta_m"])
    assert third["calibrated_actuation_delay_steps"] == 2
