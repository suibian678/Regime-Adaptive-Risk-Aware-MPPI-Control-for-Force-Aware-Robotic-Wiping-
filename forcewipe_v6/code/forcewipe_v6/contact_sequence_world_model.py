"""Mode-aware action-sequence world model for direct contact-force planning."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ContactSequenceWorldModelConfig:
    observation_dim: int
    action_dim: int = 3
    hidden_dim: int = 256
    force_scale_n: float = 15.0
    stable_contact_min_n: float = 3.0
    no_contact_max_n: float = 0.5
    transition_delta_n: float = 2.0
    upper_quantile: float = 0.975

    def validate(self) -> None:
        if self.observation_dim < 1 or self.action_dim < 1 or self.hidden_dim < 8:
            raise ValueError("invalid model dimensions")
        if self.force_scale_n <= 0.0:
            raise ValueError("force scale must be positive")
        if not 0.0 <= self.no_contact_max_n < self.stable_contact_min_n:
            raise ValueError("invalid contact thresholds")
        if self.transition_delta_n <= 0.0:
            raise ValueError("transition threshold must be positive")
        if not 0.5 < self.upper_quantile < 1.0:
            raise ValueError("upper quantile must lie in (0.5, 1)")


def contact_mode_targets(
    pre_force_n: torch.Tensor,
    post_force_n: torch.Tensor,
    config: ContactSequenceWorldModelConfig,
) -> torch.Tensor:
    """Return 0=no contact, 1=contact transition, and 2=stable contact."""

    config.validate()
    if pre_force_n.shape != post_force_n.shape:
        raise ValueError("pre- and post-force tensors must have the same shape")
    no_contact = (
        (pre_force_n <= config.no_contact_max_n)
        & (post_force_n <= config.no_contact_max_n)
    )
    stable = (
        (pre_force_n >= config.stable_contact_min_n)
        & (post_force_n >= config.stable_contact_min_n)
        & ((post_force_n - pre_force_n).abs() <= config.transition_delta_n)
    )
    result = torch.ones_like(pre_force_n, dtype=torch.long)
    result = torch.where(no_contact, torch.zeros_like(result), result)
    result = torch.where(stable, torch.full_like(result, 2), result)
    return result


class ContactSequenceWorldModel(nn.Module):
    """Predict a causal force trajectory from one observation and an action sequence."""

    def __init__(self, config: ContactSequenceWorldModelConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.observation_encoder = nn.Sequential(
            nn.Linear(config.observation_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.SiLU(),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(config.action_dim, config.hidden_dim),
            nn.SiLU(),
        )
        self.transition = nn.GRUCell(config.hidden_dim, config.hidden_dim)
        self.force_mean_head = nn.Linear(config.hidden_dim, 1)
        self.force_upper_gap_head = nn.Linear(config.hidden_dim, 1)
        self.contact_mode_head = nn.Linear(config.hidden_dim, 3)

    def rollout(self, observation: torch.Tensor, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        if observation.ndim != 2 or observation.shape[-1] != self.config.observation_dim:
            raise ValueError("observation must be B-by-observation_dim")
        if actions.ndim != 3 or actions.shape[0] != observation.shape[0]:
            raise ValueError("actions must be B-by-H-by-action_dim")
        if actions.shape[-1] != self.config.action_dim:
            raise ValueError("action dimension mismatch")
        hidden = self.observation_encoder(observation)
        means, uppers, modes = [], [], []
        for step in range(actions.shape[1]):
            hidden = self.transition(self.action_encoder(actions[:, step]), hidden)
            mean = F.softplus(self.force_mean_head(hidden)).squeeze(-1)
            upper = mean + F.softplus(self.force_upper_gap_head(hidden)).squeeze(-1)
            means.append(mean)
            uppers.append(upper)
            modes.append(self.contact_mode_head(hidden))
        return {
            "force_mean": torch.stack(means, dim=1),
            "force_upper": torch.stack(uppers, dim=1),
            "mode_logits": torch.stack(modes, dim=1),
        }


def quantile_pinball_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    quantile: float,
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError("quantile prediction and target shapes must match")
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must lie in (0, 1)")
    error = target - prediction
    return torch.maximum(float(quantile) * error, (float(quantile) - 1.0) * error)


def apply_horizon_linear_margin(
    force_upper_n: torch.Tensor,
    slope_n_per_step: float,
) -> torch.Tensor:
    """Widen a force envelope linearly with the known planning horizon."""

    if force_upper_n.ndim != 2:
        raise ValueError("force upper tensor must be B-by-H")
    if slope_n_per_step < 0.0:
        raise ValueError("horizon margin slope must be nonnegative")
    horizon = torch.arange(
        1,
        force_upper_n.shape[1] + 1,
        device=force_upper_n.device,
        dtype=force_upper_n.dtype,
    )
    return force_upper_n + float(slope_n_per_step) * horizon.unsqueeze(0)


def calibrate_horizon_linear_margin(
    force_upper_n: torch.Tensor,
    actual_force_n: torch.Tensor,
    mask: torch.Tensor,
    *,
    actual_violation_threshold_n: float = 15.0,
    planning_flag_threshold_n: float = 14.5,
    epsilon_n_per_step: float = 1e-4,
) -> float:
    """Fit the smallest linear margin covering every unsafe calibration window."""

    if not (
        force_upper_n.shape == actual_force_n.shape == mask.shape
        and force_upper_n.ndim == 2
    ):
        raise ValueError("upper, actual, and mask tensors must share B-by-H shape")
    if epsilon_n_per_step <= 0.0:
        raise ValueError("calibration epsilon must be positive")
    valid = mask.bool()
    actual_max = torch.where(
        valid,
        actual_force_n,
        torch.full_like(actual_force_n, -torch.inf),
    ).max(dim=1).values
    unsafe = actual_max > float(actual_violation_threshold_n)
    if not bool(unsafe.any()):
        raise ValueError("calibration set contains no unsafe windows")
    horizon = torch.arange(
        1,
        force_upper_n.shape[1] + 1,
        device=force_upper_n.device,
        dtype=force_upper_n.dtype,
    ).unsqueeze(0)
    required = (
        (float(planning_flag_threshold_n) - force_upper_n).clamp_min(0.0)
        / horizon
    )
    required = torch.where(valid, required, torch.full_like(required, torch.inf))
    per_window = required.min(dim=1).values
    return float(per_window[unsafe].max().item() + epsilon_n_per_step)


def contact_sequence_world_model_loss(
    prediction: dict[str, torch.Tensor],
    target_force_normalized: torch.Tensor,
    target_mode: torch.Tensor,
    mask: torch.Tensor,
    sample_weight: torch.Tensor,
    config: ContactSequenceWorldModelConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return a masked contact-weighted multi-step training objective."""

    config.validate()
    mean = prediction["force_mean"]
    upper = prediction["force_upper"]
    logits = prediction["mode_logits"]
    if not (
        mean.shape
        == upper.shape
        == target_force_normalized.shape
        == target_mode.shape
        == mask.shape
        == sample_weight.shape
    ):
        raise ValueError("force, mode, mask, and weight shapes must match")
    active_weight = mask.float() * sample_weight
    denominator = active_weight.sum().clamp_min(1.0)
    mean_loss = (
        F.smooth_l1_loss(mean, target_force_normalized, reduction="none")
        * active_weight
    ).sum() / denominator
    upper_loss = (
        quantile_pinball_loss(upper, target_force_normalized, config.upper_quantile)
        * active_weight
    ).sum() / denominator
    mode_loss_raw = F.cross_entropy(
        logits.reshape(-1, 3),
        target_mode.reshape(-1),
        reduction="none",
    ).reshape_as(target_mode)
    mode_loss = (mode_loss_raw * active_weight).sum() / denominator
    total = mean_loss + upper_loss + 0.2 * mode_loss
    return total, {
        "mean_loss": mean_loss,
        "upper_loss": upper_loss,
        "mode_loss": mode_loss,
    }


class ContactSequenceWorldModelEnsemble(nn.Module):
    def __init__(self, config: ContactSequenceWorldModelConfig, members: int = 3):
        super().__init__()
        if members < 2:
            raise ValueError("at least two ensemble members are required")
        self.config = config
        self.members = nn.ModuleList(
            ContactSequenceWorldModel(config) for _ in range(members)
        )

    def rollout(self, observation: torch.Tensor, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        rows = [member.rollout(observation, actions) for member in self.members]
        means = torch.stack([row["force_mean"] for row in rows], dim=0)
        uppers = torch.stack([row["force_upper"] for row in rows], dim=0)
        modes = torch.stack([row["mode_logits"] for row in rows], dim=0)
        return {
            "member_force_mean": means,
            "member_force_upper": uppers,
            "member_mode_logits": modes,
            "force_mean": means.mean(dim=0),
            "force_upper": uppers.max(dim=0).values,
            "mode_logits": modes.mean(dim=0),
        }
