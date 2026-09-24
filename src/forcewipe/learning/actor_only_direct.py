"""Actor-only deployment ablation for the frozen direct TD-MPC2 checkpoints.

This module deliberately changes only the deployment-time action selector.
It uses the same checkpoint, encoder, EMA policy head, observation, action
space, and environment contract as the complete method, but never invokes
MPPI.  It therefore isolates the contribution of world-model planning at
deployment without retraining the learned model.
"""

from __future__ import annotations

import torch

from forcewipe.learning.unified_tracking_gated_actor_to_mppi import (
    UnifiedTrackingGatedActorToMPPITDMPC2,
    configure_unified_tracking_gated_actor_to_mppi,
)


ACTOR_ONLY_CONTROL_MODE = "ema_actor_only"


def configure_actor_only_direct(cfg):
    """Retain the complete method's frozen model and observation settings."""

    cfg = configure_unified_tracking_gated_actor_to_mppi(cfg)
    cfg.deployment_action_selector = ACTOR_ONLY_CONTROL_MODE
    return cfg


class ActorOnlyDirectTDMPC2(UnifiedTrackingGatedActorToMPPITDMPC2):
    """Use the checkpoint's EMA actor at every deployment control step."""

    @torch.no_grad()
    def act(self, obs, t0=False, eval_mode=False, task=None):
        del t0  # The actor is memoryless; the argument remains API-compatible.
        obs_batch = obs.to(self.device, non_blocking=True).unsqueeze(0)
        task_tensor = (
            torch.tensor([task], device=self.device) if task is not None else None
        )
        model = (
            self._eval_ema_model
            if eval_mode and self.cfg.eval_ema and self._eval_ema_model is not None
            else self.model
        )
        z = model.encode(obs_batch, task_tensor)
        action, info = model.pi(z, task_tensor)
        if eval_mode:
            action = info["mean"]
        selected = action[0]
        self._last_selected_actor_center = info["mean"][0].detach().clone()
        self._last_selected_preanchor_action = selected.detach().clone()
        self._last_selected_postanchor_action = selected.detach().clone()
        self._last_tdmpc2_control_mode = ACTOR_ONLY_CONTROL_MODE
        return selected.cpu()
