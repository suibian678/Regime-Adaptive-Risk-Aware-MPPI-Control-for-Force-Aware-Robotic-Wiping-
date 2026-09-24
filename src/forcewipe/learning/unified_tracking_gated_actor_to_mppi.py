"""Unified force-tracking gate for recovery-aware direct TD-MPC2.

The learned EMA actor owns contact establishment and recovery.  Once the
measured force reaches 85% of the commanded tier, the learned world model and
MPPI own tracking.  Both branches are parts of the same jointly trained
TD-MPC2 checkpoint; no classical force controller or action shield is used.
"""

from __future__ import annotations

import torch

from forcewipe.learning.actor_to_mppi_contact_mode import (
    ActorToMPPIContactModeTDMPC2,
    configure_actor_to_mppi_contact_mode,
)
from forcewipe.learning.contact_gated_anisotropic_mppi import ContactGatedAnisotropicRiskTDMPC2


TRACKING_ACTIVATION_FRACTION = 0.85


def use_actor_until_tracking(
    current_force_fraction: float,
    target_force_fraction: float,
) -> bool:
    """Return whether the actor still owns contact establishment/recovery."""

    if target_force_fraction <= 0.0:
        raise ValueError("target_force_fraction must be positive")
    return bool(
        current_force_fraction
        < TRACKING_ACTIVATION_FRACTION * target_force_fraction
    )


def configure_unified_tracking_gated_actor_to_mppi(cfg):
    cfg = configure_actor_to_mppi_contact_mode(cfg)
    cfg.mppi_tracking_activation_fraction = TRACKING_ACTIVATION_FRACTION
    return cfg


class UnifiedTrackingGatedActorToMPPITDMPC2(ActorToMPPIContactModeTDMPC2):
    """Recovery-aware actor below the tracking regime, MPPI within it."""

    @torch.no_grad()
    def act(self, obs, t0=False, eval_mode=False, task=None):
        flat = obs.reshape(-1)
        current_force = float(flat[int(self.cfg.force_obs_idx)].detach().cpu())
        target_force = float(
            flat[int(self.cfg.force_plan_target_obs_idx)].detach().cpu()
        )
        if use_actor_until_tracking(current_force, target_force):
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
            self._last_tdmpc2_control_mode = "recovery_aware_ema_actor"
            return selected.cpu()

        self._last_tdmpc2_control_mode = "recovery_aware_world_model_mppi"
        return ContactGatedAnisotropicRiskTDMPC2.act(
            self, obs, t0=t0, eval_mode=eval_mode, task=task
        )
