from types import SimpleNamespace

import pytest
import torch

from forcewipe_v19.force_conditioned_world_model import (
    ForceConditionedWorldModel,
    configure_force_conditioned_world_model,
)
from forcewipe_v16.force_calibrated_strong_bc_training import (
    build_force_calibrated_strong_bc_config,
)
from forcewipe_v19.training import build_v19_training_config


def _cfg(**updates):
    values = {
        "latent_dim": 128,
        "simnorm_dim": 8,
        "multitask": False,
        "obs": "state",
        "obs_shape": {"state": (16,)},
    }
    values.update(updates)
    return SimpleNamespace(**values)


def test_force_condition_partition_is_explicit_and_valid():
    cfg = configure_force_conditioned_world_model(_cfg(), condition_dim=16)
    assert cfg.force_conditioned_dynamics is True
    assert cfg.force_condition_dim == 16
    assert cfg.force_condition_target_obs_idx == 1


@pytest.mark.parametrize("condition_dim", [0, 7, 128])
def test_invalid_condition_partition_is_rejected(condition_dim):
    with pytest.raises(ValueError):
        configure_force_conditioned_world_model(_cfg(), condition_dim=condition_dim)


def test_multitask_is_not_silently_accepted():
    with pytest.raises(ValueError, match="single-task"):
        configure_force_conditioned_world_model(_cfg(multitask=True))


def test_requested_force_context_is_preserved_across_imagined_step(tmp_path):
    cfg = configure_force_conditioned_world_model(
        build_force_calibrated_strong_bc_config(seed=172, work_dir=tmp_path)
    )
    model = ForceConditionedWorldModel(cfg)
    obs = torch.zeros(3, 16)
    obs[:, 1] = torch.tensor([5.0, 8.0, 12.0]) / 15.0
    action = torch.zeros(3, 3)
    z = model.encode(obs, task=None)
    next_z = model.next(z, action, task=None)
    assert z.shape == (3, 128)
    assert next_z.shape == (3, 128)
    assert torch.equal(model.force_condition(z), model.force_condition(next_z))
    assert not torch.equal(model.force_condition(z)[0], model.force_condition(z)[2])


@pytest.mark.parametrize("coefficient", [0.0, 0.5, 2.0])
def test_bc_ablation_configuration_is_matched(tmp_path, coefficient):
    cfg = build_v19_training_config(
        seed=201,
        work_dir=tmp_path,
        bc_coefficient=coefficient,
    )
    assert cfg.demo_bc_online_coef == coefficient
    assert cfg.force_conditioned_dynamics is True
