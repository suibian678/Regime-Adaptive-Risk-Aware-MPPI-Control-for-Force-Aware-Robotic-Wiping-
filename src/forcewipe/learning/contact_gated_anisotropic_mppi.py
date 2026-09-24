"""Contact-gated envelope risk for OOD-aware direct TD-MPC2 MPPI."""

from __future__ import annotations

import torch

from tdmpc2.tdmpc2 import TDMPC2
from forcewipe.planning.actor_anchored_mppi import ActorAnchoredTDMPC2
from forcewipe.learning.anisotropic_target_gated_risk_mppi import (
    AnisotropicTargetGatedRiskTDMPC2,
    configure_anisotropic_target_gated_risk_mppi,
)


ENVELOPE_ACTIVATION_FORCE_N = 9.0


def configure_contact_gated_anisotropic_mppi(cfg):
    cfg = configure_anisotropic_target_gated_risk_mppi(cfg)
    cfg.mppi_envelope_activation_force = ENVELOPE_ACTIVATION_FORCE_N / 15.0
    return cfg


def envelope_risk_active(current_force_fraction: float, target_force_fraction: float) -> bool:
    return bool(target_force_fraction >= 0.70 and current_force_fraction >= ENVELOPE_ACTIVATION_FORCE_N / 15.0)


class ContactGatedAnisotropicRiskTDMPC2(AnisotropicTargetGatedRiskTDMPC2):
    """Activate the transient-envelope term only after high-force contact is established."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self._envelope_risk_active_for_plan = False

    @torch.no_grad()
    def _plan(self, obs, t0=False, eval_mode=False, task=None):
        self._envelope_risk_active_for_plan = envelope_risk_active(
            float(obs[0, int(self.cfg.force_obs_idx)].detach().cpu()),
            float(obs[0, int(self.cfg.force_plan_target_obs_idx)].detach().cpu()),
        )
        return super()._plan(obs, t0=t0, eval_mode=eval_mode, task=task)

    @torch.no_grad()
    def _estimate_value(self, z, actions, task, force_target=None, model=None):
        high_target = bool(
            force_target is not None
            and float(force_target[0, 0].detach().cpu()) >= float(self.cfg.mppi_high_target_threshold)
        )
        if not high_target:
            return ActorAnchoredTDMPC2._estimate_value(
                self, z, actions, task, force_target=force_target, model=model
            )

        model = self.model if model is None else model
        value = TDMPC2._estimate_value(
            self, z, actions, task, force_target=force_target, model=model
        )
        if self._anchor_actor_centers is not None:
            deviation = actions - self._anchor_actor_centers.unsqueeze(1)
            component_weights = torch.as_tensor(
                self.cfg.mppi_high_target_actor_deviation_weights,
                dtype=actions.dtype,
                device=actions.device,
            ).view(1, 1, -1)
            time_weight = torch.pow(
                torch.as_tensor(float(self.discount), device=actions.device),
                torch.arange(self.cfg.horizon, device=actions.device, dtype=actions.dtype),
            ).view(-1, 1)
            anchor_cost = (
                (deviation.square() * component_weights).sum(dim=-1) * time_weight
            ).sum(dim=0).unsqueeze(-1)
            value = value - anchor_cost

        if not self._envelope_risk_active_for_plan:
            return value
        unsafe = torch.zeros(actions.shape[1], 1, dtype=torch.bool, device=z.device)
        soft_cost = torch.zeros(actions.shape[1], 1, dtype=z.dtype, device=z.device)
        rollout_z = z
        for step in range(actions.shape[0]):
            predicted_envelope = model.envelope(rollout_z, actions[step], task)
            unsafe = unsafe | (predicted_envelope >= float(self.cfg.mppi_envelope_hard_cap))
            soft_cost = soft_cost + torch.relu(
                predicted_envelope - float(self.cfg.mppi_envelope_soft_cap)
            ).square()
            rollout_z = model.next(rollout_z, actions[step], task)
        return (
            value
            - float(self.cfg.mppi_envelope_soft_coef) * soft_cost
            - float(self.cfg.mppi_envelope_unsafe_penalty) * unsafe.float()
        )

