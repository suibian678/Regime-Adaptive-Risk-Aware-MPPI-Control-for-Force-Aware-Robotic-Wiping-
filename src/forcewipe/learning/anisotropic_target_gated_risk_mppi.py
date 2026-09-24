"""Anisotropic target-gated MPPI for V16 direct TD-MPC2."""

from __future__ import annotations

import torch

from tdmpc2.tdmpc2 import TDMPC2
from forcewipe.planning.actor_anchored_mppi import ActorAnchoredTDMPC2
from forcewipe.learning.target_gated_risk_mppi import (
    TargetGatedRiskCorrectiveTDMPC2,
    configure_target_gated_risk_mppi,
)


HIGH_TARGET_ACTOR_DEVIATION_WEIGHTS = (4.0, 4.0, 1.0)


def configure_anisotropic_target_gated_risk_mppi(cfg):
    cfg = configure_target_gated_risk_mppi(cfg)
    # Preserve V16.4 along-path/cross-track regularization while allowing the
    # learned envelope objective to use the expanded outward normal search.
    cfg.mppi_high_target_actor_deviation_weights = list(HIGH_TARGET_ACTOR_DEVIATION_WEIGHTS)
    return cfg


class AnisotropicTargetGatedRiskTDMPC2(TargetGatedRiskCorrectiveTDMPC2):
    """Apply relaxed actor regularization only to the normal action."""

    @torch.no_grad()
    def _estimate_value(self, z, actions, task, force_target=None, model=None):
        high_target = bool(
            force_target is not None
            and float(force_target[0, 0].detach().cpu()) >= float(self.cfg.mppi_high_target_threshold)
        )
        if not high_target:
            # Exact V16.4 low-target objective.
            return ActorAnchoredTDMPC2._estimate_value(
                self,
                z,
                actions,
                task,
                force_target=force_target,
                model=model,
            )

        model = self.model if model is None else model
        value = TDMPC2._estimate_value(
            self,
            z,
            actions,
            task,
            force_target=force_target,
            model=model,
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
