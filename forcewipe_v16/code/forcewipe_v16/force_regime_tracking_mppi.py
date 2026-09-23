"""Use TD-MPC2's learned force-regime head during low/mid-tier MPPI."""

from __future__ import annotations

import torch

from forcewipe_v16.tracking_gated_actor_to_mppi import (
    TrackingGatedActorToMPPITDMPC2,
    configure_tracking_gated_actor_to_mppi,
)


LOW_FORCE_REGIME_COEF = 1.0


def configure_force_regime_tracking_mppi(cfg):
    cfg = configure_tracking_gated_actor_to_mppi(cfg)
    cfg.mppi_low_force_regime_coef = LOW_FORCE_REGIME_COEF
    return cfg


def predicted_low_regime_cost(logits: torch.Tensor) -> torch.Tensor:
    """Return the learned probability of the below-target force regime."""
    if logits.shape[-1] != 3:
        raise ValueError("force-regime logits must have three classes")
    return torch.softmax(logits, dim=-1)[..., :1]


class ForceRegimeTrackingMPPITDMPC2(TrackingGatedActorToMPPITDMPC2):
    """Penalize predicted below-target force in 5/8 N MPPI rollouts.

    The head is part of the jointly trained TD-MPC2 world model.  The 12 N
    actor-to-MPPI path is unchanged because its paired OOD gate already passed.
    """

    @torch.no_grad()
    def _estimate_value(self, z, actions, task, force_target=None, model=None):
        value = super()._estimate_value(
            z, actions, task, force_target=force_target, model=model
        )
        if (
            force_target is None
            or float(force_target[0, 0].detach().cpu())
            >= float(self.cfg.mppi_high_target_threshold)
        ):
            return value

        model = self.model if model is None else model
        rollout_z = z
        cost = torch.zeros(actions.shape[1], 1, dtype=z.dtype, device=z.device)
        discount = torch.as_tensor(float(self.discount), dtype=z.dtype, device=z.device)
        weight = torch.ones((), dtype=z.dtype, device=z.device)
        for step in range(actions.shape[0]):
            logits = model.force_regime(rollout_z, actions[step], task)
            cost = cost + weight * predicted_low_regime_cost(logits)
            rollout_z = model.next(rollout_z, actions[step], task)
            weight = weight * discount
        return value - float(self.cfg.mppi_low_force_regime_coef) * cost

