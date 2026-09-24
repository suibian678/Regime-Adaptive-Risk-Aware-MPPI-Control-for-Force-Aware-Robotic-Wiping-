"""Target-conditioned actor-expert training for direct multi-force TD-MPC2."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from forcewipe.training_data.native_tdmpc2 import build_v14_config
from forcewipe.training_data.batch_training import behavior_clone_step as _unused


def build_target_expert_config(*, seed: int, work_dir: Path, batch_size: int = 64):
    cfg = build_v14_config(seed=seed, work_dir=work_dir, batch_size=batch_size)
    cfg.target_policy = True
    cfg.target_policy_independent_experts = True
    cfg.target_policy_coef = 1.0
    cfg.target_policy_temperature = 0.5
    cfg.target_policy_hard_gate = True
    cfg.target_policy_target_obs_idx = 1
    # Observations normalize force by the 15 N limit.  Midpoints between the
    # 5/8/12 N tiers define three unambiguous supervision labels.
    cfg.target_policy_low_threshold_n = 6.5 / 15.0
    cfg.target_policy_high_threshold_n = 10.0 / 15.0
    cfg.target_policy_balance = True
    cfg.target_policy_weight_clip = 10.0
    cfg.exp_name = "v16-v6-target-conditioned-experts"
    return cfg


def gate_pretrain_step(agent, optimizer, observation) -> dict[str, float]:
    agent.model.train()
    z = agent.model.encode(observation, task=None)
    logits = agent.model.target_gate(z, task=None)
    labels = agent.target_policy_labels_from_obs(observation)
    loss = F.cross_entropy(logits, labels)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        list(agent.model._encoder.parameters()) + list(agent.model._target_gate.parameters()),
        10.0,
    )
    optimizer.step()
    agent.model.eval()
    return {
        "target_gate_loss": float(loss.detach().cpu()),
        "target_gate_accuracy": float((logits.argmax(-1) == labels).float().mean().detach().cpu()),
    }


def target_expert_bc_step(agent, optimizer, observation, target_action) -> dict[str, float]:
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
    actor_loss = 0.15 * per_row.mean() + 0.85 * (
        (per_row * weight).sum() / weight.sum().clamp_min(1.0)
    )
    logits = agent.model.target_gate(z, task=None)
    labels = agent.target_policy_labels_from_obs(observation)
    gate_loss = F.cross_entropy(logits, labels)
    loss = actor_loss + 0.5 * gate_loss
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        list(agent.model._encoder.parameters())
        + list(agent.model._pi.parameters())
        + list(agent.model._target_gate.parameters()),
        10.0,
    )
    optimizer.step()
    agent.model.eval()
    return {
        "bc_loss": float(actor_loss.detach().cpu()),
        "bc_mse": float(per_row.mean().detach().cpu()),
        "target_gate_loss": float(gate_loss.detach().cpu()),
        "target_gate_accuracy": float((logits.argmax(-1) == labels).float().mean().detach().cpu()),
    }


@torch.no_grad()
def target_gate_audit(agent, observation: torch.Tensor) -> dict[str, object]:
    obs = observation.to(agent.device)
    z = agent.model.encode(obs, task=None)
    logits = agent.model.target_gate(z, task=None)
    labels = agent.target_policy_labels_from_obs(obs)
    predicted = logits.argmax(-1)
    confusion = torch.zeros(3, 3, dtype=torch.int64, device=agent.device)
    for truth, guess in zip(labels, predicted):
        confusion[truth, guess] += 1
    return {
        "target_gate_accuracy": float((predicted == labels).float().mean().cpu()),
        "target_gate_confusion": confusion.cpu().tolist(),
    }

