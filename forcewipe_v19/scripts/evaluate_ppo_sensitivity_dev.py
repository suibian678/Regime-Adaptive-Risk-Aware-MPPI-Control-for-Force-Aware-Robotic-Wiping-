#!/usr/bin/env python3
"""Evaluate one frozen PPO sensitivity checkpoint on all six PPO DEV blocks."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
TEACHER_ROOT = ROOT.parent
V16_ROOT = TEACHER_ROOT / "forcewipe_v16"
for source in (
    ROOT / "code",
    V16_ROOT / "code",
    TEACHER_ROOT / "forcewipe_v6" / "code",
    TEACHER_ROOT / "forcewipe_v4" / "code",
    ROOT / "release" / "runtime_source",
):
    sys.path.insert(0, str(source))

import forcewipe_v6.tdmpc2_direct_sapien_env as direct_env_module  # noqa: E402
from forcewipe_v6.tdmpc2_direct_firstpass import DirectFirstPassConfig  # noqa: E402
from forcewipe_v6.tdmpc2_direct_sapien_env import V6DirectFirstPassEnv  # noqa: E402
from forcewipe_v16.direct_ppo_baseline import DirectPPOActorCritic  # noqa: E402
from forcewipe_v16.v16p30_final_analysis import analyze_evaluation  # noqa: E402
from forcewipe_v16.v16p30_ppo_scenarios import (  # noqa: E402
    DEV_SCENARIO_BLOCKS,
    TARGETS_N,
    scenario_digest,
    scenario_identity,
    v16p30_ppo_scenario,
)
from forcewipe_v19.ppo_budget_sensitivity import (  # noqa: E402
    BASE_TRANSITIONS,
    PROFILE_BY_ID,
    TRAINING_SEEDS,
    ExtendedDirectPPOConfig,
)


TRAIN_ROOT = ROOT / "results" / "train" / "ppo_budget_sensitivity"
DEV_ROOT = ROOT / "results" / "dev" / "ppo_budget_sensitivity"


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def checkpoint_path(profile_id: str, seed: int, multiplier: int) -> Path:
    return (
        TRAIN_ROOT / f"ppo_sensitivity_{profile_id}_seed{seed}"
        / f"ppo_{profile_id}_seed{seed}_{multiplier}x.pt"
    )


def load_agent(profile_id: str, seed: int, multiplier: int):
    path = checkpoint_path(profile_id, seed, multiplier)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "forcewipe_v19_ppo_budget_sensitivity_checkpoint_v1":
        raise RuntimeError("unexpected PPO sensitivity checkpoint format")
    identity = (
        payload.get("profile_id") == profile_id
        and int(payload.get("seed", -1)) == seed
        and int(payload.get("budget_multiplier", -1)) == multiplier
        and int(payload.get("environment_transitions", -1)) == multiplier * BASE_TRANSITIONS
    )
    if not identity:
        raise RuntimeError("PPO sensitivity checkpoint identity mismatch")
    config = ExtendedDirectPPOConfig(**payload["config"])
    model = DirectPPOActorCritic(config)
    model.load_state_dict(payload["model"])
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model


def seed_step(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate_cell(*, agent, profile_id: str, seed: int, multiplier: int, block_index: int, target: float):
    scenario_id, scenario_seed = scenario_identity(
        role="DEV", block_index=block_index, target_force_n=target
    )
    spec = v16p30_ppo_scenario(target, scenario_seed, scenario_id)
    env = V6DirectFirstPassEnv(
        target_force_n=target,
        scenario_seed=scenario_seed,
        scenario_id=scenario_id,
        config=DirectFirstPassConfig(),
    )
    observation, reset_info = env.reset(seed=scenario_seed)
    rows = []
    episode_return = 0.0
    profile_index = tuple(PROFILE_BY_ID).index(profile_id)
    target_index = TARGETS_N.index(float(target))
    seed_index = TRAINING_SEEDS.index(seed)
    planner_root = (
        919_200_000
        + profile_index * 10_000_000
        + multiplier * 1_000_000
        + seed_index * 100_000
        + block_index * 10_000
        + target_index * 1_500
    )
    started = time.perf_counter()
    try:
        for step in range(env.config.maximum_steps):
            planner_seed = planner_root + step
            seed_step(planner_seed)
            planning_started = time.perf_counter()
            action = agent.act(observation, deterministic=True)
            planning_ms = (time.perf_counter() - planning_started) * 1000.0
            next_observation, reward, terminated, truncated, info = env.step(action)
            episode_return += float(reward)
            row = {
                "control_step": step,
                "profile_id": profile_id,
                "budget_multiplier": multiplier,
                "method_seed": seed,
                "block_id": DEV_SCENARIO_BLOCKS[block_index]["block_id"],
                "target_force_n": target,
                "planner_step_seed": planner_seed,
                "observation": np.asarray(observation).tolist(),
                "action": np.asarray(action).tolist(),
                "next_observation": np.asarray(next_observation).tolist(),
                "reward": float(reward),
                "planning_ms": planning_ms,
                "control_mode": "direct_ppo",
                "method_action_authority": bool(info.get("direct_tdmpc2_action_authority", False)),
                **info,
            }
            json.dumps(row, allow_nan=False)
            rows.append(row)
            observation = next_observation
            if terminated or truncated:
                break
    finally:
        env.close()
    metrics = analyze_evaluation(rows, target_force_n=target)
    last = rows[-1] if rows else {}
    summary = {
        "profile_id": profile_id,
        "budget_multiplier": multiplier,
        "method_seed": seed,
        "block_id": DEV_SCENARIO_BLOCKS[block_index]["block_id"],
        "block_index": block_index,
        "target_force_n": target,
        "scenario_id": scenario_id,
        "scenario_seed": scenario_seed,
        "scenario_digest": scenario_digest(spec),
        "episode_return": episode_return,
        "completed_dose_bins": int(last.get("completed_dose_bins", 0)),
        "progress": float(last.get("progress", 0.0)),
        "wall_time_s": time.perf_counter() - started,
        **metrics,
    }
    return rows, reset_info, summary


def run(profile_id: str, seed: int, multiplier: int) -> Path:
    profile = PROFILE_BY_ID[profile_id]
    if seed not in TRAINING_SEEDS:
        raise ValueError("unexpected seed")
    if multiplier not in profile.checkpoint_multipliers:
        raise ValueError("profile/budget endpoint was not prespecified")
    run_id = f"dev_{profile_id}_{multiplier}x_seed{seed}"
    final = DEV_ROOT / run_id
    stage = DEV_ROOT / f".{run_id}.creating"
    if final.exists() or stage.exists():
        raise RuntimeError(f"DEV run already exists: {run_id}")
    DEV_ROOT.mkdir(parents=True, exist_ok=True)
    stage.mkdir()
    direct_env_module.nominal_direct_scenario = v16p30_ppo_scenario
    agent = load_agent(profile_id, seed, multiplier)
    evaluations = []
    for block_index in range(len(DEV_SCENARIO_BLOCKS)):
        for target in TARGETS_N:
            evaluation_id = f"{DEV_SCENARIO_BLOCKS[block_index]['block_id']}_{int(target)}N"
            directory = stage / evaluation_id
            directory.mkdir()
            rows, reset_info, summary = evaluate_cell(
                agent=agent,
                profile_id=profile_id,
                seed=seed,
                multiplier=multiplier,
                block_index=block_index,
                target=target,
            )
            with (directory / "TRACE.jsonl").open("x", encoding="utf-8") as stream:
                for row in rows:
                    stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            atomic_json(directory / "EVALUATION_RECORD.json", {
                "reset_info": reset_info,
                **summary,
            })
            evaluations.append(summary)
            print(json.dumps({
                "profile": profile_id,
                "budget": multiplier,
                "seed": seed,
                "block": summary["block_id"],
                "target": target,
                "task": summary["task_success"],
                "compound": bool(
                    summary["task_success"] and summary["tracking_pass"]
                    and summary["safety_pass"] and summary["authority_pass"]
                ),
            }, sort_keys=True), flush=True)
    atomic_json(stage / "RESULT.json", {
        "format": "forcewipe_v19_ppo_sensitivity_dev_result_v1",
        "status": "completed",
        "selection_only": True,
        "profile_id": profile_id,
        "budget_multiplier": multiplier,
        "seed": seed,
        "evaluations": evaluations,
    })
    os.replace(stage, final)
    return final


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True, choices=tuple(PROFILE_BY_ID))
    parser.add_argument("--budget", required=True, type=int, choices=(1, 2, 4))
    parser.add_argument("--seed", required=True, type=int, choices=TRAINING_SEEDS)
    args = parser.parse_args()
    print(run(args.profile, args.seed, args.budget))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
