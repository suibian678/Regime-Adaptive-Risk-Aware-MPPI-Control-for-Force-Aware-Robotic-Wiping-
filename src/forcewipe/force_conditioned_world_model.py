"""Force-conditioned latent dynamics for the V19 method study.

The requested force is encoded by a dedicated branch and copied unchanged
through imagined rollouts.  The learned dynamics therefore predict only the
remaining latent state while conditioning explicitly on the requested force.
"""

from __future__ import annotations

from copy import deepcopy

import torch

from common import init, layers
from common.world_model import WorldModel
from tdmpc2.tdmpc2 import TDMPC2


def configure_force_conditioned_world_model(
    cfg,
    *,
    condition_dim: int = 16,
    target_observation_index: int = 1,
):
    condition_dim = int(condition_dim)
    latent_dim = int(cfg.latent_dim)
    simnorm_dim = int(cfg.simnorm_dim)
    if bool(cfg.multitask) or str(cfg.obs) != "state":
        raise ValueError("V19 force-conditioned dynamics currently require single-task state input")
    if condition_dim <= 0 or condition_dim >= latent_dim:
        raise ValueError("condition_dim must lie inside the latent dimension")
    if condition_dim % simnorm_dim or (latent_dim - condition_dim) % simnorm_dim:
        raise ValueError("both latent partitions must be divisible by simnorm_dim")
    if not 0 <= int(target_observation_index) < int(cfg.obs_shape["state"][0]):
        raise ValueError("target_observation_index is outside the state observation")
    cfg.force_conditioned_dynamics = True
    cfg.force_condition_dim = condition_dim
    cfg.force_condition_target_obs_idx = int(target_observation_index)
    return cfg


class ForceConditionedWorldModel(WorldModel):
    """TD-MPC2 world model with an explicit persistent target-force context."""

    def __init__(self, cfg):
        if not bool(getattr(cfg, "force_conditioned_dynamics", False)):
            raise ValueError("force-conditioned dynamics were not configured")
        super().__init__(cfg)
        self._condition_dim = int(cfg.force_condition_dim)
        self._base_latent_dim = int(cfg.latent_dim) - self._condition_dim
        base_cfg = deepcopy(cfg)
        base_cfg.latent_dim = self._base_latent_dim
        condition_cfg = deepcopy(cfg)
        condition_cfg.latent_dim = self._condition_dim

        # Replace only the encoder and latent dynamics. Reward, value, force,
        # envelope and policy heads retain the original total latent width.
        self._encoder = layers.enc(base_cfg)
        self._force_condition_encoder = layers.mlp(
            1,
            [max(32, int(cfg.enc_dim) // 4)],
            self._condition_dim,
            act=layers.SimNorm(condition_cfg),
        )
        self._dynamics = layers.mlp(
            self._base_latent_dim + self._condition_dim + int(cfg.action_dim),
            2 * [int(cfg.mlp_dim)],
            self._base_latent_dim,
            act=layers.SimNorm(base_cfg),
        )
        self._encoder.apply(init.weight_init)
        self._force_condition_encoder.apply(init.weight_init)
        self._dynamics.apply(init.weight_init)

    def encode(self, obs, task):
        if task is not None:
            raise ValueError("V19 force-conditioned model is single-task")
        base = self._encoder["state"](obs)
        index = int(self.cfg.force_condition_target_obs_idx)
        target = obs[..., index : index + 1]
        condition = self._force_condition_encoder(target)
        return torch.cat([base, condition], dim=-1)

    def next(self, z, a, task):
        if task is not None:
            raise ValueError("V19 force-conditioned model is single-task")
        base, condition = torch.split(
            z,
            [self._base_latent_dim, self._condition_dim],
            dim=-1,
        )
        predicted_base = self._dynamics(torch.cat([base, condition, a], dim=-1))
        # F* is an episode request, not a state to be predicted.  Copying its
        # embedding prevents long rollouts from forgetting the force command.
        return torch.cat([predicted_base, condition], dim=-1)

    def force_condition(self, z: torch.Tensor) -> torch.Tensor:
        return z[..., -self._condition_dim :]


def _install_force_conditioned_model(agent: TDMPC2) -> None:
    """Replace the base model and rebuild optimisers before any training."""

    cfg = agent.cfg
    agent.model = ForceConditionedWorldModel(cfg).to(agent.device)
    encoder_parameters = list(agent.model._encoder.parameters()) + list(
        agent.model._force_condition_encoder.parameters()
    )
    agent.optim = torch.optim.Adam(
        [
            {"params": encoder_parameters, "lr": cfg.lr * cfg.enc_lr_scale},
            {"params": agent.model._dynamics.parameters()},
            {"params": agent.model._reward.parameters()},
            {"params": agent.model._force.parameters() if cfg.force_pred else []},
            {
                "params": agent.model._force_regime.parameters()
                if cfg.force_regime_pred
                else []
            },
            {
                "params": agent.model._envelope.parameters()
                if getattr(cfg, "envelope_pred", False)
                else []
            },
            {"params": agent.model._phase.parameters() if cfg.phase_policy else []},
            {
                "params": agent.model._target_gate.parameters()
                if cfg.target_policy
                else []
            },
            {
                "params": agent.model._termination.parameters()
                if cfg.episodic
                else []
            },
            {"params": agent.model._Qs.parameters()},
            {
                "params": agent.model._task_emb.parameters()
                if cfg.multitask
                else []
            },
        ],
        lr=cfg.lr,
        capturable=True,
    )
    agent.pi_optim = torch.optim.Adam(
        agent.model._pi.parameters(),
        lr=cfg.lr * float(getattr(cfg, "pi_lr_scale", 1.0)),
        eps=1e-5,
        capturable=True,
    )
    agent.model.eval()
    agent._bc_teacher = None
    agent._eval_ema_model = None
    agent._update_counter = 0


class ForceConditionedTDMPC2(TDMPC2):
    """Training agent for the force-conditioned world model."""

    def __init__(self, cfg):
        super().__init__(cfg)
        _install_force_conditioned_model(self)


__all__ = [
    "ForceConditionedTDMPC2",
    "ForceConditionedWorldModel",
    "configure_force_conditioned_world_model",
    "_install_force_conditioned_model",
]
