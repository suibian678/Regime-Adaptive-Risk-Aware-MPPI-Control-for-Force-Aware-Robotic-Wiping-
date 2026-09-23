"""Deterministic-evaluation actor-anchored TD-MPC2 MPPI."""

from __future__ import annotations

import torch

from common import math
from forcewipe_v15.actor_anchored_mppi import ActorAnchoredTDMPC2


def select_mppi_action(
    *,
    elite_actions: torch.Tensor,
    score: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    eval_mode: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Use the deterministic elite mean for evaluation; retain native sampling for training."""
    first_std = std[0]
    if eval_mode:
        return mean[0], first_std
    rand_idx = math.gumbel_softmax_sample(score.squeeze(1))
    trajectory = torch.index_select(elite_actions, 1, rand_idx).squeeze(1)
    action = trajectory[0] + first_std * torch.randn(first_std.shape, device=first_std.device)
    return action, first_std


class DeterministicActorAnchoredTDMPC2(ActorAnchoredTDMPC2):
    """Actor-anchored MPPI whose evaluation action is the weighted elite mean."""

    @torch.no_grad()
    def _plan(self, obs, t0=False, eval_mode=False, task=None):
        self._anchor_actor_centers = None
        self._last_selected_preanchor_action = None
        self._last_selected_postanchor_action = None
        self._last_selected_actor_center = None

        model = self._eval_ema_model if eval_mode and self.cfg.eval_ema and self._eval_ema_model is not None else self.model
        z = model.encode(obs, task)
        force_target = None
        if self.cfg.force_plan and self.cfg.force_pred:
            force_target = obs[:, self.cfg.force_plan_target_obs_idx].repeat(self.cfg.num_samples, 1)
        actor_mean_actions = None
        if bool(getattr(self.cfg, "mppi_actor_mean_init", False)):
            actor_mean_actions = torch.empty(self.cfg.horizon, self.cfg.action_dim, device=self.device)
            actor_latent = z
            for step in range(self.cfg.horizon - 1):
                _, actor_info = model.pi(actor_latent, task)
                actor_mean_actions[step] = actor_info["mean"][0]
                actor_latent = model.next(actor_latent, actor_mean_actions[step].unsqueeze(0), task)
            _, actor_info = model.pi(actor_latent, task)
            actor_mean_actions[-1] = actor_info["mean"][0]

        if self.cfg.num_pi_trajs > 0:
            pi_actions = torch.empty(
                self.cfg.horizon,
                self.cfg.num_pi_trajs,
                self.cfg.action_dim,
                device=self.device,
            )
            pi_latent = z.repeat(self.cfg.num_pi_trajs, 1)
            for step in range(self.cfg.horizon - 1):
                pi_actions[step], _ = model.pi(pi_latent, task)
                pi_latent = model.next(pi_latent, pi_actions[step], task)
            pi_actions[-1], _ = model.pi(pi_latent, task)
        demo_actions = self._sample_demo_plan_actions(obs)
        num_demo_trajs = 0 if demo_actions is None else demo_actions.shape[1]

        expanded_z = z.repeat(self.cfg.num_samples, 1)
        mean = torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.device)
        std = torch.full(
            (self.cfg.horizon, self.cfg.action_dim),
            self.cfg.max_std,
            dtype=torch.float,
            device=self.device,
        )
        if actor_mean_actions is not None:
            mean.copy_(actor_mean_actions)
        elif not t0:
            mean[:-1] = self._prev_mean[1:]
        actions = torch.empty(
            self.cfg.horizon,
            self.cfg.num_samples,
            self.cfg.action_dim,
            device=self.device,
        )
        fixed_trajs = 0
        if self.cfg.num_pi_trajs > 0:
            actions[:, :self.cfg.num_pi_trajs] = pi_actions
            fixed_trajs += self.cfg.num_pi_trajs
        if num_demo_trajs > 0:
            actions[:, fixed_trajs:fixed_trajs + num_demo_trajs] = demo_actions
            fixed_trajs += num_demo_trajs

        for _ in range(self.cfg.iterations):
            num_random_trajs = self.cfg.num_samples - fixed_trajs
            if num_random_trajs > 0:
                noise = torch.randn(
                    self.cfg.horizon,
                    num_random_trajs,
                    self.cfg.action_dim,
                    device=std.device,
                )
                sampled = mean.unsqueeze(1) + std.unsqueeze(1) * noise
                actions[:, fixed_trajs:] = sampled.clamp(-1, 1)
            actions = self._apply_mppi_action_safety(actions, obs)
            if self.cfg.multitask:
                actions = actions * model._action_masks[task]
            value = self._estimate_value(
                expanded_z,
                actions,
                task,
                force_target=force_target,
                model=model,
            ).nan_to_num(0)
            elite_idxs = torch.topk(value.squeeze(1), self.cfg.num_elites, dim=0).indices
            elite_value = value[elite_idxs]
            elite_actions = actions[:, elite_idxs]
            maximum = elite_value.max(0).values
            score = torch.exp(self.cfg.temperature * (elite_value - maximum))
            score = score / score.sum(0)
            mean = (score.unsqueeze(0) * elite_actions).sum(dim=1) / (score.sum(0) + 1e-9)
            std = (
                (score.unsqueeze(0) * (elite_actions - mean.unsqueeze(1)).square()).sum(dim=1)
                / (score.sum(0) + 1e-9)
            ).sqrt().clamp(self.cfg.min_std, self.cfg.max_std)
            if self.cfg.multitask:
                mean = mean * model._action_masks[task]
                std = std * model._action_masks[task]

        action, _ = select_mppi_action(
            elite_actions=elite_actions,
            score=score,
            mean=mean,
            std=std,
            eval_mode=eval_mode,
        )
        action = self._apply_mppi_reactive_z(action, obs)
        self._prev_mean.copy_(mean)
        return self._apply_mppi_action_safety(action, obs).clamp(-1, 1)
