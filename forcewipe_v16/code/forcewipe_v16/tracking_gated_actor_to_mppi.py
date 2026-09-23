"""Tracking-regime-gated actor-to-MPPI policy for the 12 N tier."""

from __future__ import annotations

import torch

from forcewipe_v16.actor_to_mppi_contact_mode import (
    ActorToMPPIContactModeTDMPC2,
    configure_actor_to_mppi_contact_mode,
)
from forcewipe_v16.contact_gated_anisotropic_mppi import ContactGatedAnisotropicRiskTDMPC2


MPPI_TRACKING_ACTIVATION_FORCE_N = 10.2


def use_actor_until_tracking(current_force_fraction: float, target_force_fraction: float) -> bool:
    return bool(
        target_force_fraction >= 0.70
        and current_force_fraction < MPPI_TRACKING_ACTIVATION_FORCE_N / 15.0
    )


def configure_tracking_gated_actor_to_mppi(cfg):
    cfg = configure_actor_to_mppi_contact_mode(cfg)
    cfg.mppi_contact_mode_activation_force = MPPI_TRACKING_ACTIVATION_FORCE_N / 15.0
    return cfg


class TrackingGatedActorToMPPITDMPC2(ActorToMPPIContactModeTDMPC2):
    """Let the learned actor establish the accepted force regime before planning."""

    @torch.no_grad()
    def act(self, obs, t0=False, eval_mode=False, task=None):
        flat = obs.reshape(-1)
        current_force = float(flat[int(self.cfg.force_obs_idx)].detach().cpu())
        target_force = float(flat[int(self.cfg.force_plan_target_obs_idx)].detach().cpu())
        if use_actor_until_tracking(current_force, target_force):
            obs_batch = obs.to(self.device, non_blocking=True).unsqueeze(0)
            task_tensor = torch.tensor([task], device=self.device) if task is not None else None
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
            self._last_tdmpc2_control_mode = "ema_actor_force_establishment"
            return selected.cpu()
        self._last_tdmpc2_control_mode = "world_model_mppi_tracking"
        return ContactGatedAnisotropicRiskTDMPC2.act(
            self, obs, t0=t0, eval_mode=eval_mode, task=task
        )

