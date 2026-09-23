#!/usr/bin/env python3
"""Train one prespecified PPO sensitivity profile and seed.

The default profile is trained once to 4x and checkpointed at 1x/2x/4x.
Every non-default profile is trained to 2x.  Historical V16 files and
checkpoints are read-only inputs and are never modified.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import traceback

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
TEACHER_ROOT = ROOT.parent
V16_ROOT = TEACHER_ROOT / "forcewipe_v16"
for source in (
    ROOT / "code",
    V16_ROOT / "code",
    V16_ROOT / "scripts",
    TEACHER_ROOT / "forcewipe_v6" / "code",
    TEACHER_ROOT / "forcewipe_v4" / "code",
    ROOT / "release" / "runtime_source" / "tdmpc2",
    ROOT / "release" / "runtime_source",
):
    sys.path.insert(0, str(source))

import forcewipe_v6.tdmpc2_direct_sapien_env as direct_env_module  # noqa: E402
from forcewipe_v16.direct_ppo_baseline import (  # noqa: E402
    DirectPPOActorCritic,
    DirectPPOOptimizer,
    generalized_advantage_estimate,
)
from forcewipe_v16.v16p30_ppo_scenarios import v16p30_ppo_scenario  # noqa: E402
from forcewipe_v19.ppo_budget_sensitivity import (  # noqa: E402
    BASE_TRANSITIONS,
    PROFILE_BY_ID,
    TRAINING_SEEDS,
    checkpoint_payload,
    config_for_profile,
)
from train_v16p30_direct_ppo import OnPolicyCollector  # noqa: E402


TRAIN_ROOT = ROOT / "results" / "train" / "ppo_budget_sensitivity"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def historical_checkpoint(seed: int) -> Path:
    run = f"v16p30_direct_ppo_seed{seed}_transitions82774_20260830_r2"
    return (
        V16_ROOT
        / "results" / "train" / "v16p30_direct_ppo" / run
        / f"direct_ppo_seed{seed}.pt"
    )


def state_dict_exactly_equal(left: dict, right: dict) -> bool:
    return left.keys() == right.keys() and all(
        torch.equal(left[name].detach().cpu(), right[name].detach().cpu())
        for name in left
    )


def run(profile_id: str, seed: int) -> Path:
    if profile_id not in PROFILE_BY_ID:
        raise ValueError(f"unknown profile {profile_id}")
    if int(seed) not in TRAINING_SEEDS:
        raise ValueError(f"seed must be one of {TRAINING_SEEDS}")
    profile = PROFILE_BY_ID[profile_id]
    run_id = f"ppo_sensitivity_{profile_id}_seed{seed}"
    final = TRAIN_ROOT / run_id
    stage = TRAIN_ROOT / f".{run_id}.creating"
    if final.exists() or stage.exists():
        raise RuntimeError(f"training run already exists: {run_id}")
    TRAIN_ROOT.mkdir(parents=True, exist_ok=True)
    stage.mkdir()

    direct_env_module.nominal_direct_scenario = v16p30_ppo_scenario
    if direct_env_module.nominal_direct_scenario is not v16p30_ppo_scenario:
        raise RuntimeError("failed to bind the frozen PPO TRAIN scenario loader")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = config_for_profile(profile_id)
    model = DirectPPOActorCritic(config)
    optimizer = DirectPPOOptimizer(model, device=device)
    action_generator = torch.Generator(device=device.type).manual_seed(716_330_000 + seed)
    update_generator = torch.Generator().manual_seed(816_330_000 + seed)
    collector = OnPolicyCollector(model, device=device, action_generator=action_generator)
    endpoints = profile.checkpoint_transitions
    next_endpoint_index = 0
    transitions = 0
    optimizer_updates = 0
    metrics: list[dict] = []
    checkpoint_records: list[dict] = []

    atomic_json(stage / "RUN_DEFINITION.json", {
        "format": "forcewipe_v19_ppo_sensitivity_training_definition_v1",
        "run_id": run_id,
        "profile_id": profile_id,
        "seed": seed,
        "started_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "device": str(device),
        "config": config.__dict__,
        "checkpoint_transitions": list(endpoints),
        "training_scenario_family": "unchanged V16.30 PPO TRAIN family",
        "historical_files_modified": False,
    })

    try:
        while transitions < profile.maximum_transitions:
            next_endpoint = endpoints[next_endpoint_index]
            count = min(config.rollout_transitions, next_endpoint - transitions)
            batch = collector.collect(count)
            update = optimizer.update(batch, generator=update_generator)
            transitions += count
            optimizer_updates += int(update["optimizer_updates"])
            row = {
                "rollout": len(metrics),
                "transitions": transitions,
                "optimizer_updates": optimizer_updates,
                "episodes_completed": len(collector.episodes),
                "task_successes": collector.task_successes,
                "force_violation_episodes": collector.force_violations,
                "peak_force_n": collector.peak_force_n,
                **update,
            }
            metrics.append(row)
            print(json.dumps({"profile": profile_id, "seed": seed, **row}, sort_keys=True), flush=True)

            if transitions == next_endpoint:
                multiplier = transitions // BASE_TRANSITIONS
                checkpoint = stage / f"ppo_{profile_id}_seed{seed}_{multiplier}x.pt"
                torch.save(checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    profile_id=profile_id,
                    seed=seed,
                    transitions=transitions,
                ), checkpoint)
                record = {
                    "budget_multiplier": multiplier,
                    "environment_transitions": transitions,
                    "checkpoint": checkpoint.name,
                    "sha256": sha256(checkpoint),
                }
                if profile_id == "default" and multiplier == 1:
                    historical = torch.load(
                        historical_checkpoint(seed), map_location="cpu", weights_only=False
                    )
                    record["historical_model_state_exact_match"] = state_dict_exactly_equal(
                        model.state_dict(), historical["model"]
                    )
                    record["historical_checkpoint_sha256"] = sha256(historical_checkpoint(seed))
                checkpoint_records.append(record)
                next_endpoint_index += 1

        collector.close()
        with (stage / "TRAINING_METRICS.jsonl").open("x", encoding="utf-8") as stream:
            for row in metrics:
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        with (stage / "EPISODE_SUMMARIES.jsonl").open("x", encoding="utf-8") as stream:
            for row in collector.episodes:
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        atomic_json(stage / "RESULT.json", {
            "format": "forcewipe_v19_ppo_sensitivity_training_result_v1",
            "status": "completed",
            "run_id": run_id,
            "profile_id": profile_id,
            "seed": seed,
            "environment_transitions": transitions,
            "optimizer_updates": optimizer_updates,
            "episodes_completed": len(collector.episodes),
            "task_successes_during_training": collector.task_successes,
            "force_violation_episodes_during_training": collector.force_violations,
            "peak_force_n_during_training": collector.peak_force_n,
            "checkpoints": checkpoint_records,
            "development_or_final_evaluation_used": False,
        })
        files = sorted(path for path in stage.iterdir() if path.is_file())
        atomic_json(stage / "RUN_FILE_MANIFEST.json", {"files": [
            {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in files
        ]})
        os.replace(stage, final)
        return final
    except BaseException as exc:
        collector.close()
        atomic_json(stage / "INFRASTRUCTURE_FAILURE.json", {
            "status": "aborted_infrastructure",
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
            "transitions_completed": transitions,
        })
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True, choices=tuple(PROFILE_BY_ID))
    parser.add_argument("--seed", required=True, type=int, choices=TRAINING_SEEDS)
    args = parser.parse_args()
    print(run(args.profile, args.seed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
