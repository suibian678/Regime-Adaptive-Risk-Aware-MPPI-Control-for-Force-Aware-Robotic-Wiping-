"""Actor-centered trust-region MPPI for the frozen V14.1 world model."""

from __future__ import annotations

import torch

from tdmpc2.tdmpc2 import TDMPC2


TRUST_RADIUS = (0.25, 0.20, 0.12)


def configure_actor_trust_mppi(cfg):
    cfg.mpc = True
    cfg.horizon = 10
    cfg.iterations = 4
    cfg.num_samples = 128
    cfg.num_elites = 16
    cfg.num_pi_trajs = 24
    cfg.min_std = 0.03
    cfg.max_std = 0.25
    cfg.mppi_actor_trust_radius = list(TRUST_RADIUS)
    cfg.mppi_actor_deviation_coef = 2.0
    return cfg


def clip_to_actor_trust(actions: torch.Tensor, centers: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
    """Clip an action or trajectory batch around its actor-mean center."""
    if actions.ndim == 1:
        center = centers[0]
    elif actions.ndim == 3:
        center = centers.unsqueeze(1)
    else:
        raise ValueError("trust-region actions must be rank 1 or rank 3")
    return torch.maximum(torch.minimum(actions, center + radius), center - radius).clamp(-1.0, 1.0)


class ActorTrustRegionTDMPC2(TDMPC2):
    """Native TD-MPC2 planner with a general actor-distribution trust region."""

    def __init__(self, cfg):
        super().__init__(cfg)
        radius = torch.as_tensor(cfg.mppi_actor_trust_radius, dtype=torch.float32, device=self.device)
        if radius.shape != (cfg.action_dim,) or torch.any(radius <= 0) or torch.any(radius > 1):
            raise ValueError("invalid MPPI actor trust radius")
        self._actor_trust_radius = radius
        self._trust_actor_centers = None
        self._last_selected_pretrust_action = None
        self._last_selected_posttrust_action = None
        self._last_selected_actor_center = None

    @torch.no_grad()
    def _actor_centers(self, obs, task):
        model = self._eval_ema_model if self._eval_ema_model is not None else self.model
        latent = model.encode(obs, task)
        centers = torch.empty(self.cfg.horizon, self.cfg.action_dim, device=self.device)
        for step in range(self.cfg.horizon):
            _, info = model.pi(latent, task)
            centers[step] = info["mean"][0]
            if step + 1 < self.cfg.horizon:
                latent = model.next(latent, centers[step].unsqueeze(0), task)
        return centers

    @torch.no_grad()
    def _plan(self, obs, t0=False, eval_mode=False, task=None):
        self._trust_actor_centers = None
        self._last_selected_pretrust_action = None
        self._last_selected_posttrust_action = None
        self._last_selected_actor_center = None
        return super()._plan(obs, t0=t0, eval_mode=eval_mode, task=task)

    @torch.no_grad()
    def _apply_mppi_action_safety(self, actions, obs=None):
        actions = super()._apply_mppi_action_safety(actions, obs)
        if obs is None:
            return actions
        if self._trust_actor_centers is None:
            self._trust_actor_centers = self._actor_centers(obs, task=None)
        clipped = clip_to_actor_trust(actions, self._trust_actor_centers, self._actor_trust_radius)
        if actions.ndim == 1:
            self._last_selected_pretrust_action = actions.detach().clone()
            self._last_selected_posttrust_action = clipped.detach().clone()
            self._last_selected_actor_center = self._trust_actor_centers[0].detach().clone()
        return clipped

    @torch.no_grad()
    def _estimate_value(self, z, actions, task, force_target=None, model=None):
        value = super()._estimate_value(z, actions, task, force_target=force_target, model=model)
        if self._trust_actor_centers is None:
            return value
        deviation = actions - self._trust_actor_centers.unsqueeze(1)
        time_weight = torch.pow(
            torch.as_tensor(float(self.discount), device=actions.device),
            torch.arange(self.cfg.horizon, device=actions.device, dtype=actions.dtype),
        ).view(-1, 1)
        penalty = (deviation.square().sum(dim=-1) * time_weight).sum(dim=0, keepdim=False).unsqueeze(-1)
        return value - float(self.cfg.mppi_actor_deviation_coef) * penalty
