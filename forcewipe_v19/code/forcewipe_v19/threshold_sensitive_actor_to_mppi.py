"""Matched sensitivity implementation of the historical actor--MPPI gate.

This module exists only to vary the activation fraction used by the V16
controller.  It does not alter the checkpoint, actor, world model, MPPI
objective, action map, or environment.
"""

from __future__ import annotations

import torch

from forcewipe_v16.actor_to_mppi_contact_mode import (
    ActorToMPPIContactModeTDMPC2,
    configure_actor_to_mppi_contact_mode,
)
from forcewipe_v16.contact_gated_anisotropic_mppi import (
    ContactGatedAnisotropicRiskTDMPC2,
)


def configure_threshold_sensitive_actor_to_mppi(cfg, *, activation_fraction: float):
    fraction = float(activation_fraction)
    if not 0.0 < fraction < 1.0:
        raise ValueError("activation_fraction must lie strictly between zero and one")
    cfg = configure_actor_to_mppi_contact_mode(cfg)
    cfg.mppi_tracking_activation_fraction = fraction
    return cfg


class ThresholdSensitiveActorToMPPITDMPC2(ActorToMPPIContactModeTDMPC2):
    """Historical controller with one explicit, configurable switch value."""

    @torch.no_grad()
    def act(self, obs, t0=False, eval_mode=False, task=None):
        flat = obs.reshape(-1)
        current_force = float(flat[int(self.cfg.force_obs_idx)].detach().cpu())
        target_force = float(
            flat[int(self.cfg.force_plan_target_obs_idx)].detach().cpu()
        )
        threshold = float(self.cfg.mppi_tracking_activation_fraction) * target_force
        if current_force < threshold:
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
            self._last_tdmpc2_control_mode = "threshold_sensitivity_ema_actor"
            return selected.cpu()

        self._last_tdmpc2_control_mode = "threshold_sensitivity_world_model_mppi"
        return ContactGatedAnisotropicRiskTDMPC2.act(
            self, obs, t0=t0, eval_mode=eval_mode, task=task
        )
