"""Asymmetric actor-anchored MPPI for the frozen V14.1 world model."""

from __future__ import annotations

import torch

from tdmpc2.tdmpc2 import TDMPC2


# Along-path MPPI may slow or reverse relative to the actor, but may not outrun
# it by more than 0.02. Cross-track and normal search remain symmetric.
LOWER_DEVIATION = (-0.25, -0.10, -0.06)
UPPER_DEVIATION = (0.02, 0.10, 0.06)


def configure_actor_anchored_mppi(cfg):
    cfg.mpc = True
    cfg.horizon = 10
    cfg.iterations = 4
    cfg.num_samples = 128
    cfg.num_elites = 16
    cfg.num_pi_trajs = 24
    cfg.min_std = 0.03
    cfg.max_std = 0.25
    cfg.mppi_actor_lower_deviation = list(LOWER_DEVIATION)
    cfg.mppi_actor_upper_deviation = list(UPPER_DEVIATION)
    cfg.mppi_actor_deviation_coef = 4.0
    return cfg


def clip_to_actor_anchor(
    actions: torch.Tensor,
    centers: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> torch.Tensor:
    """Apply componentwise asymmetric bounds around actor trajectory means."""
    if actions.ndim == 1:
        center = centers[0]
    elif actions.ndim == 3:
        center = centers.unsqueeze(1)
    else:
        raise ValueError("actor-anchor actions must be rank 1 or rank 3")
    return torch.maximum(torch.minimum(actions, center + upper), center + lower).clamp(-1.0, 1.0)


class ActorAnchoredTDMPC2(TDMPC2):
    """Full MPPI constrained to a conservative asymmetric actor neighborhood."""

    def __init__(self, cfg):
        super().__init__(cfg)
        lower = torch.as_tensor(
            cfg.mppi_actor_lower_deviation,
            dtype=torch.float32,
            device=self.device,
        )
        upper = torch.as_tensor(
            cfg.mppi_actor_upper_deviation,
            dtype=torch.float32,
            device=self.device,
        )
        if (
            lower.shape != (cfg.action_dim,)
            or upper.shape != (cfg.action_dim,)
            or torch.any(lower >= 0)
            or torch.any(upper <= 0)
            or torch.any(lower < -1)
            or torch.any(upper > 1)
        ):
            raise ValueError("invalid MPPI actor-anchor bounds")
        self._actor_lower_deviation = lower
        self._actor_upper_deviation = upper
        self._anchor_actor_centers = None
        self._last_selected_preanchor_action = None
        self._last_selected_postanchor_action = None
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
        self._anchor_actor_centers = None
        self._last_selected_preanchor_action = None
        self._last_selected_postanchor_action = None
        self._last_selected_actor_center = None
        return super()._plan(obs, t0=t0, eval_mode=eval_mode, task=task)

    @torch.no_grad()
    def _apply_mppi_action_safety(self, actions, obs=None):
        actions = super()._apply_mppi_action_safety(actions, obs)
        if obs is None:
            return actions
        if self._anchor_actor_centers is None:
            self._anchor_actor_centers = self._actor_centers(obs, task=None)
        anchored = clip_to_actor_anchor(
            actions,
            self._anchor_actor_centers,
            self._actor_lower_deviation,
            self._actor_upper_deviation,
        )
        if actions.ndim == 1:
            self._last_selected_preanchor_action = actions.detach().clone()
            self._last_selected_postanchor_action = anchored.detach().clone()
            self._last_selected_actor_center = self._anchor_actor_centers[0].detach().clone()
        return anchored

    @torch.no_grad()
    def _estimate_value(self, z, actions, task, force_target=None, model=None):
        value = super()._estimate_value(z, actions, task, force_target=force_target, model=model)
        if self._anchor_actor_centers is None:
            return value
        deviation = actions - self._anchor_actor_centers.unsqueeze(1)
        time_weight = torch.pow(
            torch.as_tensor(float(self.discount), device=actions.device),
            torch.arange(self.cfg.horizon, device=actions.device, dtype=actions.dtype),
        ).view(-1, 1)
        penalty = (deviation.square().sum(dim=-1) * time_weight).sum(dim=0).unsqueeze(-1)
        return value - float(self.cfg.mppi_actor_deviation_coef) * penalty
