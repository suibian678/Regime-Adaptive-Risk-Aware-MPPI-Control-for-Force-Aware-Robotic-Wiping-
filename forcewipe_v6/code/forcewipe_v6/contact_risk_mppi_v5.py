"""Force-tracking objective for intervention-risk-gated direct TD-MPC2 MPPI."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from forcewipe_v6.contact_risk_mppi import ContactRiskMPPIConfig
from forcewipe_v6.contact_risk_mppi_v4 import (
    ActorAcquireInterventionRiskTDMPC2,
    InterventionRiskMPPIEvaluator,
)


@dataclass(frozen=True)
class TrackingRiskMPPIConfig(ContactRiskMPPIConfig):
    target_force_n: float = 12.0
    tracking_penalty_per_n: float = 1_000.0

    def validate(self) -> None:
        super().validate()
        if self.target_force_n <= 0 or self.tracking_penalty_per_n <= 0:
            raise ValueError("tracking target and coefficient must be positive")


class TrackingInterventionRiskMPPIEvaluator(InterventionRiskMPPIEvaluator):
    """Rank risk-feasible candidates by predicted continuous force error."""

    @torch.no_grad()
    def evaluate_candidates(self, observation, actions_nha):
        actions = torch.as_tensor(actions_nha, dtype=torch.float32, device=self.device)
        if actions.ndim != 3 or actions.shape[-1] != self.model.config.action_dim:
            raise ValueError("candidate actions must be N-by-H-by-action_dim")
        if actions.shape[1] < 1:
            raise ValueError("candidate horizon must be nonempty")
        obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        if obs.ndim != 2 or obs.shape[-1] != self.model.config.observation_dim:
            raise ValueError("observation dimension does not match the contact model")
        if obs.shape[0] == 1:
            obs = obs.expand(actions.shape[0], -1)
        elif obs.shape[0] != actions.shape[0]:
            raise ValueError("observation batch must be one or match the candidate count")
        if not torch.isfinite(obs).all() or not torch.isfinite(actions).all():
            raise ValueError("candidate model inputs must be finite")
        prediction = self.model.rollout(obs, actions)
        risk = prediction["window_risk_probability"]
        maximum_upper_n = prediction["force_upper"].max(dim=1).values * self.model.config.force_scale_n
        mean_force_n = prediction["force_mean"] * self.model.config.force_scale_n
        tracking_error_n = torch.mean(
            torch.abs(mean_force_n - float(self.config.target_force_n)), dim=1
        )
        return {
            "unsafe": risk >= self.risk_threshold,
            "risk_probability": risk,
            "maximum_force_upper_n": maximum_upper_n,
            "mean_tracking_error_n": tracking_error_n,
            "unsafe_severity": risk,
        }

    def penalize_values(self, values, evaluation):
        adjusted = super().penalize_values(values, evaluation)
        tracking_penalty = (
            evaluation["mean_tracking_error_n"].to(values.device)
            * float(self.config.tracking_penalty_per_n)
        )
        return adjusted - (
            tracking_penalty.unsqueeze(1) if values.ndim == 2 else tracking_penalty
        )


class ActorAcquireTrackingRiskTDMPC2(ActorAcquireInterventionRiskTDMPC2):
    """All-learned acquisition plus safe, force-tracking TD-MPC2 contact planning."""

