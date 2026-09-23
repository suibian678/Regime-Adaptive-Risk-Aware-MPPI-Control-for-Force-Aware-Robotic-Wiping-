"""Tier-aware contact establishment for direct multi-force TD-MPC2."""

from __future__ import annotations

import torch

from forcewipe_v16.actor_to_mppi_contact_mode import ActorToMPPIContactModeTDMPC2
from forcewipe_v16.contact_gated_anisotropic_mppi import ContactGatedAnisotropicRiskTDMPC2
from forcewipe_v16.unified_tracking_gated_actor_to_mppi import (
    TRACKING_ACTIVATION_FRACTION,
    configure_unified_tracking_gated_actor_to_mppi,
)


CONTACT_FORCE_N = 3.0
HIGH_TARGET_THRESHOLD_FRACTION = 0.70


def actor_authority_required(
    current_force_fraction: float,
    target_force_fraction: float,
) -> bool:
    if target_force_fraction <= 0.0:
        raise ValueError("target_force_fraction must be positive")
    threshold = (
        TRACKING_ACTIVATION_FRACTION * target_force_fraction
        if target_force_fraction >= HIGH_TARGET_THRESHOLD_FRACTION
        else CONTACT_FORCE_N / 15.0
    )
    return bool(current_force_fraction + 1e-9 < threshold)


def configure_tiered_actor_to_mppi(cfg):
    cfg = configure_unified_tracking_gated_actor_to_mppi(cfg)
    cfg.mppi_low_mid_contact_activation_force = CONTACT_FORCE_N / 15.0
    cfg.mppi_high_tracking_activation_fraction = TRACKING_ACTIVATION_FRACTION
    return cfg


class TieredActorToMPPITDMPC2(ActorToMPPIContactModeTDMPC2):
    """Actor establishes contact at 5/8 N and tracking force at 12 N."""

    @torch.no_grad()
    def act(self, obs, t0=False, eval_mode=False, task=None):
        flat = obs.reshape(-1)
        current_force = float(flat[int(self.cfg.force_obs_idx)].detach().cpu())
        target_force = float(
            flat[int(self.cfg.force_plan_target_obs_idx)].detach().cpu()
        )
        if actor_authority_required(current_force, target_force):
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
            self._last_tdmpc2_control_mode = "tiered_ema_actor_establishment"
            return selected.cpu()
        self._last_tdmpc2_control_mode = "tiered_world_model_mppi_tracking"
        return ContactGatedAnisotropicRiskTDMPC2.act(
            self, obs, t0=t0, eval_mode=eval_mode, task=task
        )
