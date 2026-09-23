"""Intervention-calibrated plan-risk gating for direct TD-MPC2 MPPI."""

from __future__ import annotations

from pathlib import Path

import torch

from forcewipe_v6.contact_risk_mppi import (
    ContactRiskMPPIConfig,
    ContactRiskMPPIEvaluator,
)
from forcewipe_v6.contact_risk_mppi_v3 import ActorAcquireContactRiskTDMPC2
from forcewipe_v6.contact_sequence_risk_world_model import ContactSequenceRiskWorldModelEnsemble
from forcewipe_v6.contact_sequence_world_model import ContactSequenceWorldModelConfig


class InterventionRiskMPPIEvaluator(ContactRiskMPPIEvaluator):
    """Use calibrated plan risk as the veto; retain force upper as a diagnostic."""

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path | str,
        *,
        threshold_scale: float,
        config: ContactRiskMPPIConfig,
        device,
    ):
        if not 0.0 < threshold_scale <= 1.0:
            raise ValueError("threshold scale must lie in (0, 1]")
        checkpoint = torch.load(Path(checkpoint_path), map_location=device, weights_only=False)
        model_config = ContactSequenceWorldModelConfig(**checkpoint["model_config"])
        model = ContactSequenceRiskWorldModelEnsemble(
            model_config, members=int(checkpoint["ensemble_members"])
        )
        model.load_state_dict(checkpoint["model"])
        return cls(
            model,
            risk_threshold=float(checkpoint["window_risk_threshold"]) * threshold_scale,
            config=config,
            device=device,
        )

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
        unsafe = risk >= self.risk_threshold
        return {
            "unsafe": unsafe,
            "risk_probability": risk,
            "maximum_force_upper_n": maximum_upper_n,
            "unsafe_severity": risk,
        }


class ActorAcquireInterventionRiskTDMPC2(ActorAcquireContactRiskTDMPC2):
    """All-learned actor acquisition and intervention-calibrated contact MPPI."""

