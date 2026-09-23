"""Small actor-only utilities for training-oracle sequence distillation."""

from __future__ import annotations

import torch


def sequence_distillation_loss(
    oracle_prediction: torch.Tensor,
    oracle_action: torch.Tensor,
    demo_prediction: torch.Tensor,
    demo_action: torch.Tensor,
    *,
    demo_coefficient: float,
    oracle_dimension_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return total, oracle, and retention losses for bounded policy tuning."""

    if oracle_prediction.shape != oracle_action.shape:
        raise ValueError("oracle prediction and action shapes must match")
    if demo_prediction.shape != demo_action.shape:
        raise ValueError("demo prediction and action shapes must match")
    if oracle_prediction.ndim != 2 or oracle_prediction.shape[1] != 3:
        raise ValueError("oracle actions must be a batch of three-vectors")
    if demo_prediction.ndim != 2 or demo_prediction.shape[1] != 3:
        raise ValueError("demo actions must be a batch of three-vectors")
    if demo_coefficient < 0:
        raise ValueError("demo coefficient cannot be negative")
    oracle_squared = (oracle_prediction - oracle_action).square()
    if oracle_dimension_weights is not None:
        weights = oracle_dimension_weights.to(
            device=oracle_squared.device,
            dtype=oracle_squared.dtype,
        )
        if weights.shape == (3,):
            weights = weights.reshape(1, 3)
        elif weights.shape != oracle_squared.shape:
            raise ValueError(
                "oracle dimension weights must be three values or match the oracle batch"
            )
        if torch.any(weights <= 0):
            raise ValueError("oracle dimension weights must be positive")
        oracle_squared = oracle_squared * weights
    oracle_loss = oracle_squared.mean()
    demo_loss = (demo_prediction - demo_action).square().mean()
    total = oracle_loss + float(demo_coefficient) * demo_loss
    return total, oracle_loss, demo_loss
