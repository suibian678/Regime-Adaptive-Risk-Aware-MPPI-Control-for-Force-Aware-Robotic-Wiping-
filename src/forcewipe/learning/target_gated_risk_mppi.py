"""Target-gated risk-corrective MPPI for direct multi-force TD-MPC2."""

from __future__ import annotations

import torch

from tdmpc2.tdmpc2 import TDMPC2
from forcewipe.planning.actor_anchored_mppi import (
    clip_to_actor_anchor,
    configure_actor_anchored_mppi,
)
from forcewipe.planning.deterministic_actor_anchored_mppi import DeterministicActorAnchoredTDMPC2


HIGH_TARGET_THRESHOLD = 0.70
HIGH_TARGET_LOWER_DEVIATION = (-0.25, -0.10, -0.40)
HIGH_TARGET_UPPER_DEVIATION = (0.02, 0.10, 0.06)


def configure_target_gated_risk_mppi(cfg):
    # Retain the V16.4 planner exactly for 5/8 N.
    cfg = configure_actor_anchored_mppi(cfg)
    cfg.mppi_high_target_threshold = HIGH_TARGET_THRESHOLD
    cfg.mppi_high_target_lower_deviation = list(HIGH_TARGET_LOWER_DEVIATION)
    cfg.mppi_high_target_upper_deviation = list(HIGH_TARGET_UPPER_DEVIATION)
    cfg.mppi_high_target_actor_deviation_coef = 1.0
    cfg.mppi_envelope_soft_cap = 13.5 / 15.0
    cfg.mppi_envelope_hard_cap = 14.1 / 15.0
    cfg.mppi_envelope_soft_coef = 100.0
    cfg.mppi_envelope_unsafe_penalty = 1_000.0
    return cfg


class TargetGatedRiskCorrectiveTDMPC2(DeterministicActorAnchoredTDMPC2):
    """Use envelope-risk correction only for the 12 N target tier."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self._high_target_lower_deviation = torch.as_tensor(
            cfg.mppi_high_target_lower_deviation,
            dtype=torch.float32,
            device=self.device,
        )
        self._high_target_upper_deviation = torch.as_tensor(
            cfg.mppi_high_target_upper_deviation,
            dtype=torch.float32,
            device=self.device,
        )

    def _high_target_from_obs(self, obs) -> bool:
        if obs is None:
            return False
        index = int(self.cfg.force_plan_target_obs_idx)
        return bool(float(obs[0, index].detach().cpu()) >= float(self.cfg.mppi_high_target_threshold))

    @torch.no_grad()
    def _apply_mppi_action_safety(self, actions, obs=None):
        # Invoke only the native planner bounds, then apply exactly one target-
        # selected actor neighborhood.  Calling the parent here would apply the
        # low-target anchor before the high-target correction.
        actions = TDMPC2._apply_mppi_action_safety(self, actions, obs)
        if obs is None:
            return actions
        if self._anchor_actor_centers is None:
            self._anchor_actor_centers = self._actor_centers(obs, task=None)
        if self._high_target_from_obs(obs):
            lower = self._high_target_lower_deviation
            upper = self._high_target_upper_deviation
        else:
            lower = self._actor_lower_deviation
            upper = self._actor_upper_deviation
        anchored = clip_to_actor_anchor(actions, self._anchor_actor_centers, lower, upper)
        if actions.ndim == 1:
            self._last_selected_preanchor_action = actions.detach().clone()
            self._last_selected_postanchor_action = anchored.detach().clone()
            self._last_selected_actor_center = self._anchor_actor_centers[0].detach().clone()
        return anchored

    @torch.no_grad()
    def _estimate_value(self, z, actions, task, force_target=None, model=None):
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
            time_weight = torch.pow(
                torch.as_tensor(float(self.discount), device=actions.device),
                torch.arange(self.cfg.horizon, device=actions.device, dtype=actions.dtype),
            ).view(-1, 1)
            anchor_cost = (deviation.square().sum(dim=-1) * time_weight).sum(dim=0).unsqueeze(-1)
        else:
            anchor_cost = torch.zeros_like(value)

        high_target = bool(
            force_target is not None
            and float(force_target[0, 0].detach().cpu()) >= float(self.cfg.mppi_high_target_threshold)
        )
        anchor_coef = (
            float(self.cfg.mppi_high_target_actor_deviation_coef)
            if high_target
            else float(self.cfg.mppi_actor_deviation_coef)
        )
        value = value - anchor_coef * anchor_cost
        if not high_target or not bool(getattr(self.cfg, "envelope_pred", False)):
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
