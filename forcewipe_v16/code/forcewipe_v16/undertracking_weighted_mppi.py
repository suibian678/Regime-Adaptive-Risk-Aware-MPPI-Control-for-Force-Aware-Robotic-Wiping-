"""Asymmetric learned-force tracking cost for direct TD-MPC2 MPPI."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from forcewipe_v16.unified_tracking_gated_actor_to_mppi import (
    UnifiedTrackingGatedActorToMPPITDMPC2,
    configure_unified_tracking_gated_actor_to_mppi,
)


UNDERTRACKING_EXTRA_COEF = 12.0


def discounted_undertracking_penalty(
    predicted_forces: torch.Tensor,
    target_force: torch.Tensor,
    *,
    deadband: float,
    discount: float,
) -> torch.Tensor:
    """Return a per-trajectory smooth-L1 cost for force deficits only."""

    if predicted_forces.ndim != 3:
        raise ValueError("predicted_forces must have shape [horizon, samples, 1]")
    under = torch.relu(target_force.unsqueeze(0) - predicted_forces - float(deadband))
    cost = F.smooth_l1_loss(under, torch.zeros_like(under), reduction="none")
    weights = torch.pow(
        predicted_forces.new_tensor(float(discount)),
        torch.arange(predicted_forces.shape[0], device=predicted_forces.device),
    ).view(-1, 1, 1)
    return (cost * weights).sum(dim=0)


def configure_undertracking_weighted_mppi(cfg):
    cfg = configure_unified_tracking_gated_actor_to_mppi(cfg)
    cfg.mppi_force_undertracking_extra_coef = UNDERTRACKING_EXTRA_COEF
    return cfg


class UndertrackingWeightedTrackingGatedTDMPC2(UnifiedTrackingGatedActorToMPPITDMPC2):
    """Keep symmetric tracking cost and add weight only to force deficits."""

    @torch.no_grad()
    def _estimate_value(self, z, actions, task, force_target=None, model=None):
        value = super()._estimate_value(
            z, actions, task, force_target=force_target, model=model
        )
        if force_target is None or not bool(getattr(self.cfg, "force_pred", False)):
            return value
        model = self.model if model is None else model
        rollout_z = z
        predictions = []
        for step in range(actions.shape[0]):
            predictions.append(model.force(rollout_z, actions[step], task))
            rollout_z = model.next(rollout_z, actions[step], task)
        penalty = discounted_undertracking_penalty(
            torch.stack(predictions),
            force_target,
            deadband=float(self.cfg.force_plan_deadband),
            discount=float(self.discount),
        )
        return value - float(self.cfg.mppi_force_undertracking_extra_coef) * penalty
