"""Latched actor-to-MPPI mode transition for direct TD-MPC2.

The EMA actor establishes the tracking regime.  MPPI then remains authoritative
through ordinary tracking deviations and relinquishes control only after a
physical contact loss.  This hysteresis avoids threshold chattering while both
branches remain learned components of the same TD-MPC2 checkpoint.
"""

from __future__ import annotations

import torch

from forcewipe_v16.contact_gated_anisotropic_mppi import ContactGatedAnisotropicRiskTDMPC2
from forcewipe_v16.unified_tracking_gated_actor_to_mppi import (
    TRACKING_ACTIVATION_FRACTION,
    UnifiedTrackingGatedActorToMPPITDMPC2,
    configure_unified_tracking_gated_actor_to_mppi,
)


CONTACT_LOSS_FORCE_N = 3.0


def update_tracking_latch(
    *,
    tracking_latched: bool,
    current_force_fraction: float,
    target_force_fraction: float,
) -> bool:
    if target_force_fraction <= 0.0:
        raise ValueError("target_force_fraction must be positive")
    if current_force_fraction < CONTACT_LOSS_FORCE_N / 15.0:
        return False
    if current_force_fraction >= TRACKING_ACTIVATION_FRACTION * target_force_fraction:
        return True
    return bool(tracking_latched)


def configure_latched_tracking_actor_to_mppi(cfg):
    cfg = configure_unified_tracking_gated_actor_to_mppi(cfg)
    cfg.mppi_contact_loss_force = CONTACT_LOSS_FORCE_N / 15.0
    return cfg


class LatchedTrackingActorToMPPITDMPC2(UnifiedTrackingGatedActorToMPPITDMPC2):
    """Stateful contact-mode hysteresis around actor-to-MPPI transfer."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self._tracking_latched = False

    @torch.no_grad()
    def act(self, obs, t0=False, eval_mode=False, task=None):
        if t0:
            self._tracking_latched = False
        flat = obs.reshape(-1)
        current_force = float(flat[int(self.cfg.force_obs_idx)].detach().cpu())
        target_force = float(
            flat[int(self.cfg.force_plan_target_obs_idx)].detach().cpu()
        )
        self._tracking_latched = update_tracking_latch(
            tracking_latched=self._tracking_latched,
            current_force_fraction=current_force,
            target_force_fraction=target_force,
        )
        if not self._tracking_latched:
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

        self._last_tdmpc2_control_mode = "latched_recovery_aware_world_model_mppi"
        return ContactGatedAnisotropicRiskTDMPC2.act(
            self, obs, t0=t0, eval_mode=eval_mode, task=task
        )
