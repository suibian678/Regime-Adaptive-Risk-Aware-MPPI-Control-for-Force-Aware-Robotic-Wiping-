"""V14.1 data split and tail-weighted policy training utilities."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from forcewipe_v14.native_tdmpc2_v6_training import (  # re-exported API
    audit_model,
    build_v14_config,
    config_dict,
    load_episode,
    populate_buffer,
)


def concatenate(rows: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {key: torch.cat([row[key] for row in rows], dim=0) for key in rows[0]}


def load_collection(collection_root: Path):
    result = json.loads((collection_root / "RESULT.json").read_text(encoding="utf-8"))
    if not result.get("joint_training_permitted"):
        raise ValueError("V14.1 collection gate does not permit joint training")
    train_tds, validation_tds = [], []
    train_flat, validation_flat = [], []
    for summary in result["evaluations"]:
        td, flat = load_episode(collection_root / summary["trace"])
        if int(summary["replicate"]) == 9:
            validation_tds.append(td)
            validation_flat.append(flat)
        else:
            train_tds.append(td)
            train_flat.append(flat)
    if len(train_tds) != 27 or len(validation_tds) != 3:
        raise ValueError("V14.1 episode split must be 27 TRAIN and 3 TRAIN-validation episodes")
    validation_targets = sorted(float(td["obs"][0, 1]) * 15.0 for td in validation_tds)
    if any(abs(actual - expected) > 1e-4 for actual, expected in zip(validation_targets, (5.0, 8.0, 12.0))):
        raise ValueError("V14.1 validation split must contain one episode per force tier")
    return train_tds, validation_tds, concatenate(train_flat), concatenate(validation_flat), result


def behavior_clone_step(agent, optimizer, observation, target_action) -> dict[str, float]:
    agent.model.train()
    z = agent.model.encode(observation, task=None)
    _, info = agent.model.pi(z, task=None)
    mean = info["mean"]
    per_row = (mean - target_action).square().mean(dim=-1)
    force = observation[:, 0]
    target = observation[:, 1]
    positive_rate = observation[:, 2].clamp_min(0.0)
    contact = force >= 0.20
    high_target = target >= 0.70
    tail_approach = high_target & ((force >= 0.45) | (positive_rate >= 0.10))
    near_headroom = high_target & (force >= 0.72)
    weight = (
        1.0
        + 2.0 * contact.float()
        + 3.0 * high_target.float()
        + 8.0 * tail_approach.float()
        + 12.0 * near_headroom.float()
    )
    weighted = (per_row * weight).sum() / weight.sum().clamp_min(1.0)
    mse = per_row.mean()
    loss = 0.15 * mse + 0.85 * weighted
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        list(agent.model._encoder.parameters()) + list(agent.model._pi.parameters()), 10.0
    )
    optimizer.step()
    agent.model.eval()
    return {
        "bc_loss": float(loss.detach().cpu()),
        "bc_mse": float(mse.detach().cpu()),
        "tail_fraction": float(tail_approach.float().mean().detach().cpu()),
    }
