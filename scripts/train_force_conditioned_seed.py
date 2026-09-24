#!/usr/bin/env python3
"""Train one V19 force-conditioned world-model seed without physics rollout."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
TEACHER_ROOT = ROOT.parent
for source in (ROOT / "src", ROOT / "scripts", ROOT / "vendor/tdmpc2", ROOT / "vendor"):
    sys.path.insert(0, str(source))

from forcewipe.training_data.native_tdmpc2 import (  # noqa: E402
    audit_model,
    config_dict,
    populate_buffer,
)
from forcewipe.learning.recovery_aware_training import (  # noqa: E402
    load_recovery_aware_collection,
)
from forcewipe.learning.target_expert_training import (  # noqa: E402
    gate_pretrain_step,
    target_expert_bc_step,
    target_gate_audit,
)
from forcewipe.force_conditioned_world_model import (  # noqa: E402
    ForceConditionedTDMPC2,
)
from forcewipe.training import build_v19_training_config  # noqa: E402


OUTPUT_ROOT = ROOT / "results/train"


def utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


@torch.no_grad()
def force_condition_audit(agent, observations: torch.Tensor) -> dict:
    target_values = torch.tensor([5.0, 8.0, 12.0]) / 15.0
    selected = []
    for target in target_values:
        index = torch.where(torch.isclose(observations[:, 1], target, atol=1e-5))[0]
        if not len(index):
            raise RuntimeError(f"validation rows missing target {float(target * 15):g} N")
        selected.append(observations[index[0]])
    obs = torch.stack(selected).to(agent.device)
    action = torch.zeros(len(obs), agent.cfg.action_dim, device=agent.device)
    z = agent.model.encode(obs, task=None)
    next_z = agent.model.next(z, action, task=None)
    before = agent.model.force_condition(z)
    after = agent.model.force_condition(next_z)
    return {
        "condition_dimension": int(before.shape[-1]),
        "condition_persistence_max_abs_error": float((before - after).abs().max().cpu()),
        "pairwise_condition_l2": [
            float(torch.linalg.vector_norm(before[i] - before[j]).cpu())
            for i, j in ((0, 1), (0, 2), (1, 2))
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--bc-coef", type=float, required=True, choices=(0.0, 0.5, 2.0))
    args = parser.parse_args()
    seed = int(args.seed)
    bc_coefficient = float(args.bc_coef)
    run_id = f"v19_force_conditioned_seed{seed}_bc{bc_coefficient:g}_u4000"
    final = OUTPUT_ROOT / run_id
    staging = OUTPUT_ROOT / f".{run_id}.creating"
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if final.exists() or staging.exists():
        raise SystemExit(f"run id already exists: {run_id}")
    staging.mkdir()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")
    train_tds, _, train_flat, validation_flat, collection = (
        load_recovery_aware_collection(Path("unused"))
    )
    cfg = build_v19_training_config(
        seed=seed,
        work_dir=staging,
        bc_coefficient=bc_coefficient,
    )
    buffer = populate_buffer(cfg, train_tds)
    agent = ForceConditionedTDMPC2(cfg)
    bc_parameters = (
        list(agent.model._encoder.parameters())
        + list(agent.model._force_condition_encoder.parameters())
        + list(agent.model._pi.parameters())
        + list(agent.model._target_gate.parameters())
    )
    bc_optimizer = torch.optim.AdamW(bc_parameters, lr=3e-4, weight_decay=1e-5)
    generator = torch.Generator().manual_seed(seed + 1000)
    gate_updates, bc_updates, joint_updates = 200, 2000, 1800
    atomic_json(
        staging / "RUN_DEFINITION.json",
        {
            "format": "forcewipe_v19_force_conditioned_training_v1",
            "run_id": run_id,
            "started_utc": utc(),
            "seed": seed,
            "bc_coefficient": bc_coefficient,
            "gate_updates": gate_updates,
            "bc_updates": bc_updates,
            "joint_updates": joint_updates,
            "train_episodes": len(train_tds),
            "training_demo_rows": len(train_flat["observation"]),
            "validation_transitions": len(validation_flat["observation"]),
            "force_conditioned_dynamics": True,
            "condition_persistence": "target-force embedding is copied across imagined steps",
            "collection_metadata": collection,
            "config": config_dict(cfg),
            "simulator_environment_created": False,
        },
    )
    train_obs = train_flat["observation"]
    train_teacher = train_flat["action"]
    with (staging / "TRAINING_METRICS.jsonl").open(
        "w", encoding="utf-8", buffering=1
    ) as stream:
        for update in range(1, gate_updates + 1):
            index = torch.randint(len(train_obs), (cfg.batch_size,), generator=generator)
            info = gate_pretrain_step(agent, bc_optimizer, train_obs[index].cuda())
            if update == 1 or update % 50 == 0 or update == gate_updates:
                row = {"stage": "target_gate", "update": update, **info}
                stream.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
        for update in range(1, bc_updates + 1):
            index = torch.randint(len(train_obs), (cfg.batch_size,), generator=generator)
            info = target_expert_bc_step(
                agent,
                bc_optimizer,
                train_obs[index].cuda(),
                train_teacher[index].cuda(),
            )
            if update == 1 or update % 100 == 0 or update == bc_updates:
                row = {"stage": "bc", "update": update, **info}
                stream.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
        agent.capture_bc_teacher()
        agent.capture_eval_ema()
        for update in range(1, joint_updates + 1):
            index = torch.randint(len(train_obs), (cfg.batch_size,), generator=generator)
            demo_batch = (train_obs[index].cuda(), train_teacher[index].cuda())
            info = agent.update(buffer, demo_batch=demo_batch)
            if update == 1 or update % 100 == 0 or update == joint_updates:
                row = {"stage": "joint", "update": update}
                row.update({key: float(value.detach().cpu()) for key, value in info.items()})
                stream.write(json.dumps(row) + "\n")
                print(
                    json.dumps(
                        {
                            key: row[key]
                            for key in row
                            if key
                            in {
                                "stage",
                                "update",
                                "total_loss",
                                "force_loss",
                                "pi_loss",
                                "demo_bc_online_loss",
                                "target_policy_loss",
                                "target_policy_accuracy",
                            }
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    audit = audit_model(agent, validation_flat)
    audit.update(target_gate_audit(agent, validation_flat["observation"]))
    audit.update(force_condition_audit(agent, validation_flat["observation"]))
    gates = {
        "one_step_force_rmse_n": audit["one_step_force_rmse_n"] <= 1.5,
        "actor_teacher_action_rmse": audit["actor_teacher_action_rmse"] <= 0.25,
        "latent_consistency": audit["latent_consistency_mse"] <= 0.20,
        "planner_finite": audit["planner_actions_finite"],
        "planner_action_box": audit["planner_action_abs_max"] <= 1.000001,
        "target_gate_accuracy": audit["target_gate_accuracy"] == 1.0,
        "force_condition_persistence": audit["condition_persistence_max_abs_error"] == 0.0,
        "force_targets_have_distinct_conditions": min(audit["pairwise_condition_l2"]) > 1e-6,
    }
    checkpoint = staging / "v19_force_conditioned_tdmpc2.pt"
    torch.save(
        {
            "model": agent.model.state_dict(),
            "eval_ema_model": agent._eval_ema_model.state_dict(),
            "config": config_dict(cfg),
            "seed": seed,
            "bc_coefficient": bc_coefficient,
        },
        checkpoint,
    )
    passed = all(gates.values())
    atomic_json(
        staging / "RESULT.json",
        {
            "format": "forcewipe_v19_force_conditioned_training_result_v1",
            "run_id": run_id,
            "completed_utc": utc(),
            "audit": audit,
            "gate_results": gates,
            "offline_training_gate_passed": passed,
            "checkpoint": checkpoint.name,
            "checkpoint_sha256": sha256(checkpoint),
            "claim_boundary": (
                "TRAIN and TRAIN-validation only; no environment evaluation, "
                "method superiority, or hardware claim."
            ),
        },
    )
    atomic_json(
        staging / "RUN_STATE.json",
        {"status": "completed_pass" if passed else "completed_scientific_fail"},
    )
    os.replace(staging, final)
    print(json.dumps({"run_id": run_id, "passed": passed, "audit": audit}, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
