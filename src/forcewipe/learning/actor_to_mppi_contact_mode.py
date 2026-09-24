"""Causal TD-MPC2 actor-to-MPPI contact-mode policy for high-force wiping."""

from __future__ import annotations

import torch

from forcewipe.learning.contact_gated_anisotropic_mppi import (
    ContactGatedAnisotropicRiskTDMPC2,
    configure_contact_gated_anisotropic_mppi,
)


MPPI_ACTIVATION_FORCE_N = 9.0


def use_actor_acquisition(current_force_fraction: float, target_force_fraction: float) -> bool:
    return bool(target_force_fraction >= 0.70 and current_force_fraction < MPPI_ACTIVATION_FORCE_N / 15.0)


def configure_actor_to_mppi_contact_mode(cfg):
    cfg = configure_contact_gated_anisotropic_mppi(cfg)
    cfg.mppi_contact_mode_activation_force = MPPI_ACTIVATION_FORCE_N / 15.0
    return cfg


class ActorToMPPIContactModeTDMPC2(ContactGatedAnisotropicRiskTDMPC2):
    """Use the learned actor for acquisition and learned-world-model MPPI for force contact."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self._last_tdmpc2_control_mode = "uninitialized"

    @torch.no_grad()
    def act(self, obs, t0=False, eval_mode=False, task=None):
        flat = obs.reshape(-1)
        current_force = float(flat[int(self.cfg.force_obs_idx)].detach().cpu())
        target_force = float(flat[int(self.cfg.force_plan_target_obs_idx)].detach().cpu())
        if use_actor_acquisition(current_force, target_force):
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
            self._last_tdmpc2_control_mode = "ema_actor_acquisition"
            return selected.cpu()
        self._last_tdmpc2_control_mode = "world_model_mppi_contact"
        return super().act(obs, t0=t0, eval_mode=eval_mode, task=task)

