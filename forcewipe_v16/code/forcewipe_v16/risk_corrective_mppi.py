"""Asymmetric risk-corrective MPPI for the V16 direct TD-MPC2 model."""

from __future__ import annotations

import torch

from forcewipe_v15.actor_anchored_mppi import configure_actor_anchored_mppi
from forcewipe_v15.deterministic_actor_anchored_mppi import DeterministicActorAnchoredTDMPC2


# Positive normalized z presses inward.  The revision increases only outward
# corrective authority; inward authority remains at the V15.2/V16.4 bound.
LOWER_DEVIATION = (-0.25, -0.10, -0.40)
UPPER_DEVIATION = (0.02, 0.10, 0.06)


def configure_risk_corrective_mppi(cfg):
    cfg = configure_actor_anchored_mppi(cfg)
    cfg.mppi_actor_lower_deviation = list(LOWER_DEVIATION)
    cfg.mppi_actor_upper_deviation = list(UPPER_DEVIATION)
    cfg.mppi_actor_deviation_coef = 1.0
    # The learned envelope underpredicted the single 15.051 N DEV transient by
    # 0.62 N.  A 14.1 N predicted cap leaves approximately 0.9 N to the sampled
    # 15 N limit while retaining the original 12 N tracking objective.
    cfg.mppi_envelope_soft_cap = 13.5 / 15.0
    cfg.mppi_envelope_hard_cap = 14.1 / 15.0
    cfg.mppi_envelope_soft_coef = 100.0
    cfg.mppi_envelope_unsafe_penalty = 1_000.0
    return cfg


class RiskCorrectiveDeterministicTDMPC2(DeterministicActorAnchoredTDMPC2):
    """TD-MPC2 MPPI that scores the already-trained transient-envelope head."""

    @torch.no_grad()
    def _estimate_value(self, z, actions, task, force_target=None, model=None):
        model = self.model if model is None else model
        value = super()._estimate_value(
            z,
            actions,
            task,
            force_target=force_target,
            model=model,
        )
        if not bool(getattr(self.cfg, "envelope_pred", False)):
            return value

        hard_cap = float(self.cfg.mppi_envelope_hard_cap)
        soft_cap = float(self.cfg.mppi_envelope_soft_cap)
        unsafe = torch.zeros(actions.shape[1], 1, dtype=torch.bool, device=z.device)
        soft_cost = torch.zeros(actions.shape[1], 1, dtype=z.dtype, device=z.device)
        rollout_z = z
        for step in range(actions.shape[0]):
            predicted_envelope = model.envelope(rollout_z, actions[step], task)
            unsafe = unsafe | (predicted_envelope >= hard_cap)
            soft_cost = soft_cost + torch.relu(predicted_envelope - soft_cap).square()
            rollout_z = model.next(rollout_z, actions[step], task)

        return (
            value
            - float(self.cfg.mppi_envelope_soft_coef) * soft_cost
            - float(self.cfg.mppi_envelope_unsafe_penalty) * unsafe.float()
        )
