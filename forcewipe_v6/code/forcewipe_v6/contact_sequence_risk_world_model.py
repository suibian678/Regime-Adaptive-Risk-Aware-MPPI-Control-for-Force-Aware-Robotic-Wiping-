"""Contact-sequence model with a plan-level force-violation risk head."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from forcewipe_v6.contact_sequence_world_model import (
    ContactSequenceWorldModel,
    ContactSequenceWorldModelConfig,
)


class ContactSequenceRiskWorldModel(ContactSequenceWorldModel):
    """Predict force trajectories and the risk of a violation within a plan."""

    def __init__(self, config: ContactSequenceWorldModelConfig):
        super().__init__(config)
        self.window_risk_head = nn.Sequential(
            nn.Linear(2 * config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, 1),
        )

    def rollout(self, observation: torch.Tensor, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        if observation.ndim != 2 or observation.shape[-1] != self.config.observation_dim:
            raise ValueError("observation must be B-by-observation_dim")
        if actions.ndim != 3 or actions.shape[0] != observation.shape[0]:
            raise ValueError("actions must be B-by-H-by-action_dim")
        if actions.shape[-1] != self.config.action_dim:
            raise ValueError("action dimension mismatch")
        hidden = self.observation_encoder(observation)
        means, uppers, modes, states = [], [], [], []
        for step in range(actions.shape[1]):
            hidden = self.transition(self.action_encoder(actions[:, step]), hidden)
            mean = F.softplus(self.force_mean_head(hidden)).squeeze(-1)
            upper = mean + F.softplus(self.force_upper_gap_head(hidden)).squeeze(-1)
            means.append(mean)
            uppers.append(upper)
            modes.append(self.contact_mode_head(hidden))
            states.append(hidden)
        state_sequence = torch.stack(states, dim=1)
        risk_features = torch.cat(
            (state_sequence[:, -1], state_sequence.max(dim=1).values),
            dim=-1,
        )
        return {
            "force_mean": torch.stack(means, dim=1),
            "force_upper": torch.stack(uppers, dim=1),
            "mode_logits": torch.stack(modes, dim=1),
            "window_risk_logit": self.window_risk_head(risk_features).squeeze(-1),
        }


def contact_window_risk_loss(
    logit: torch.Tensor,
    unsafe_target: torch.Tensor,
    *,
    positive_weight: float,
) -> torch.Tensor:
    if logit.shape != unsafe_target.shape or logit.ndim != 1:
        raise ValueError("risk logit and target must share one-dimensional shape")
    if positive_weight < 1.0:
        raise ValueError("positive risk weight must be at least one")
    return F.binary_cross_entropy_with_logits(
        logit,
        unsafe_target.float(),
        pos_weight=torch.as_tensor(positive_weight, device=logit.device, dtype=logit.dtype),
    )


class ContactSequenceRiskWorldModelEnsemble(nn.Module):
    def __init__(self, config: ContactSequenceWorldModelConfig, members: int = 3):
        super().__init__()
        if members < 2:
            raise ValueError("at least two ensemble members are required")
        self.config = config
        self.members = nn.ModuleList(
            ContactSequenceRiskWorldModel(config) for _ in range(members)
        )

    def rollout(self, observation: torch.Tensor, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        rows = [member.rollout(observation, actions) for member in self.members]
        means = torch.stack([row["force_mean"] for row in rows], dim=0)
        uppers = torch.stack([row["force_upper"] for row in rows], dim=0)
        modes = torch.stack([row["mode_logits"] for row in rows], dim=0)
        risk_logits = torch.stack([row["window_risk_logit"] for row in rows], dim=0)
        risk_probabilities = torch.sigmoid(risk_logits)
        return {
            "member_force_mean": means,
            "member_force_upper": uppers,
            "member_mode_logits": modes,
            "member_window_risk_logit": risk_logits,
            "force_mean": means.mean(dim=0),
            "force_upper": uppers.max(dim=0).values,
            "mode_logits": modes.mean(dim=0),
            "window_risk_probability": risk_probabilities.max(dim=0).values,
        }
