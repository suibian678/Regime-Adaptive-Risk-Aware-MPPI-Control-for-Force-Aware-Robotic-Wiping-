#!/usr/bin/env python3
"""Matched development sensitivity of the historical actor--MPPI threshold."""

from __future__ import annotations

from datetime import datetime, timezone
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
V16_ROOT = ROOT / "archive/direct_study"
for source in (ROOT / "src", ROOT / "scripts", ROOT / "vendor/tdmpc2", ROOT / "vendor"):
    sys.path.insert(0, str(source))

from forcewipe.direct.tdmpc2_direct_firstpass import DirectFirstPassConfig  # noqa: E402
import forcewipe.direct.tdmpc2_direct_sapien_env as direct_env_module  # noqa: E402
from forcewipe.direct.tdmpc2_direct_sapien_env import V6DirectFirstPassEnv  # noqa: E402
from forcewipe.learning.force_calibrated_strong_bc_training import (  # noqa: E402
    build_force_calibrated_strong_bc_config,
)
from forcewipe.learning.ood_scenarios import v16_ood_scenario  # noqa: E402
from forcewipe.learning.matched_methods import tdmpc_checkpoint  # noqa: E402
from forcewipe.threshold_sensitive_actor_to_mppi import (  # noqa: E402
    ThresholdSensitiveActorToMPPITDMPC2,
    configure_threshold_sensitive_actor_to_mppi,
)
from evaluation_common import atomic_json, sha256, summarize  # noqa: E402


SEEDS = (172, 173, 174, 175, 176)
TARGETS_N = (5.0, 8.0, 12.0)
THRESHOLDS = (0.70, 0.75, 0.80, 0.85, 0.90, 0.95)
RUN_ID = "v19_historical_gate_threshold_sensitivity_dev90_20260920_r2"
OUTPUT_ROOT = ROOT / "results" / "sensitivity"


def scenario_identity(target: float) -> tuple[int, int]:
    index = TARGETS_N.index(float(target))
    return 88_100_000 + index, 8_168_000 + index


def load_agent(*, seed: int, threshold: float, work_dir: Path):
    cfg = configure_threshold_sensitive_actor_to_mppi(
        build_force_calibrated_strong_bc_config(seed=seed, work_dir=work_dir),
        activation_fraction=threshold,
    )
    agent = ThresholdSensitiveActorToMPPITDMPC2(cfg)
    path = tdmpc_checkpoint(V16_ROOT, seed)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    agent.model.load_state_dict(payload["model"])
    agent.capture_eval_ema()
    agent._eval_ema_model.load_state_dict(payload["eval_ema_model"])
    agent.model.eval(); agent._eval_ema_model.eval()
    return agent, path


def run_episode(*, agent, seed: int, threshold: float, target: float, config):
    scenario_seed, scenario_id = scenario_identity(target)
    env = V6DirectFirstPassEnv(
        target_force_n=target, scenario_seed=scenario_seed,
        scenario_id=scenario_id, config=config,
    )
    observation, reset_info = env.reset(seed=scenario_seed)
    agent._prev_mean.zero_()
    rows, episode_return = [], 0.0
    try:
        while len(rows) < config.maximum_steps:
            step_seed = 9_200_000_000 + seed * 10_000_000 + scenario_id * 2_000 + len(rows)
            random.seed(step_seed); np.random.seed(step_seed % (2**32))
            torch.manual_seed(step_seed); torch.cuda.manual_seed_all(step_seed)
            started = time.perf_counter()
            action_tensor = agent.act(
                torch.from_numpy(observation), t0=len(rows) == 0, eval_mode=True
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            planning_ms = (time.perf_counter() - started) * 1_000.0
            action = action_tensor.detach().cpu().numpy().astype(np.float32)
            actor_center = (
                agent._last_selected_actor_center.detach().cpu().numpy().astype(np.float32)
            )
            next_observation, reward, terminated, truncated, info = env.step(action)
            episode_return += float(reward)
            rows.append({
                "control_step": len(rows), "planner_seed": step_seed,
                "observation": np.asarray(observation, dtype=np.float32).tolist(),
                "action": action.tolist(),
                "actor_center": actor_center.tolist(),
                "absolute_actor_deviation": np.abs(action - actor_center).tolist(),
                "next_observation": np.asarray(next_observation, dtype=np.float32).tolist(),
                "reward": float(reward), "planning_ms": planning_ms,
                "control_mode": str(agent._last_tdmpc2_control_mode),
                **info,
            })
            observation = next_observation
            if terminated or truncated:
                break
    finally:
        env.close()
    return rows, reset_info, episode_return


def main() -> int:
    final = OUTPUT_ROOT / RUN_ID
    staging = OUTPUT_ROOT / f".{RUN_ID}.creating"
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if final.exists() or staging.exists():
        raise SystemExit(f"run id already exists: {RUN_ID}")
    staging.mkdir()
    direct_env_module.nominal_direct_scenario = v16_ood_scenario
    config = DirectFirstPassConfig()
    checkpoint_identity = {
        str(seed): {
            "path": str(tdmpc_checkpoint(V16_ROOT, seed)),
            "sha256": sha256(tdmpc_checkpoint(V16_ROOT, seed)),
        }
        for seed in SEEDS
    }
    atomic_json(staging / "RUN_DEFINITION.json", {
        "format": "forcewipe_v19_historical_threshold_sensitivity_definition_v1",
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "role": "DEVELOPMENT_SENSITIVITY",
        "thresholds": list(THRESHOLDS), "training_seeds": list(SEEDS),
        "targets_n": list(TARGETS_N), "evaluations": 90,
        "checkpoints": checkpoint_identity,
        "shared_scenarios_and_step_random_streams": True,
        "only_varied_quantity": "historical actor-to-MPPI activation fraction",
        "new_method_uses_this_threshold": False,
        "claim_boundary": "Historical-controller development sensitivity; not new-method tuning or final evidence.",
    })
    evaluations = []
    for seed in SEEDS:
        for threshold in THRESHOLDS:
            agent, _ = load_agent(seed=seed, threshold=threshold, work_dir=staging)
            for target in TARGETS_N:
                rows, reset_info, episode_return = run_episode(
                    agent=agent, seed=seed, threshold=threshold,
                    target=target, config=config,
                )
                trace = staging / f"seed_{seed}_alpha_{threshold:.2f}_target_{int(target)}n.jsonl"
                with trace.open("x", encoding="utf-8", buffering=1) as stream:
                    for row in rows:
                        stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                summary = summarize(rows, target=target, episode_return=episode_return)
                mean_force = summary["contact_mean_force_n"]
                rmse = summary["contact_rmse_n"]
                modes = [row["control_mode"] for row in rows]
                summary.update({
                    "training_seed": seed, "activation_fraction": threshold,
                    "scenario_seed": scenario_identity(target)[0],
                    "scenario_id": scenario_identity(target)[1],
                    "reset_info": reset_info, "trace": trace.name,
                    "trace_sha256": sha256(trace),
                    "mean_relative_error": (
                        abs(float(mean_force) - target) / target if mean_force is not None else None
                    ),
                    "target_normalized_rmse": (
                        float(rmse) / target if rmse is not None else None
                    ),
                    "actor_steps": sum(mode == "threshold_sensitivity_ema_actor" for mode in modes),
                    "mppi_steps": sum(mode == "threshold_sensitivity_world_model_mppi" for mode in modes),
                })
                evaluations.append(summary)
                print(json.dumps({
                    "seed": seed, "alpha": threshold, "target": target,
                    "task": summary["success"],
                    "mre": summary["mean_relative_error"],
                    "nrmse": summary["target_normalized_rmse"],
                    "peak": summary["peak_force_n"],
                    "actor_steps": summary["actor_steps"],
                    "mppi_steps": summary["mppi_steps"],
                }, sort_keys=True), flush=True)
            del agent
            torch.cuda.empty_cache()

    summaries = []
    for threshold in THRESHOLDS:
        cell = [row for row in evaluations if row["activation_fraction"] == threshold]
        contact_rows = [row for row in cell if row["mean_relative_error"] is not None]
        summaries.append({
            "activation_fraction": threshold, "evaluations": len(cell),
            "task_successes": sum(bool(row["success"]) for row in cell),
            "tracking_passes": sum(
                row["mean_relative_error"] is not None
                and row["target_normalized_rmse"] is not None
                and row["mean_relative_error"] <= 0.15
                and row["target_normalized_rmse"] <= 0.20
                for row in cell
            ),
            "no_contact_evaluations": len(cell) - len(contact_rows),
            "mean_relative_error": float(np.mean([row["mean_relative_error"] for row in contact_rows])),
            "mean_target_normalized_rmse": float(np.mean([row["target_normalized_rmse"] for row in contact_rows])),
            "maximum_peak_force_n": max(float(row["peak_force_n"]) for row in cell),
            "force_limit_violation_samples": sum(int(row["force_limit_violation_samples"]) for row in cell),
            "actor_step_fraction": (
                sum(int(row["actor_steps"]) for row in cell)
                / sum(int(row["native_samples"]) for row in cell)
            ),
        })
    atomic_json(staging / "RESULT.json", {
        "format": "forcewipe_v19_historical_threshold_sensitivity_result_v1",
        "status": "completed", "evaluations": evaluations,
        "threshold_summaries": summaries,
        "claim_boundary": "Historical-controller development sensitivity; the V19 method has no hard actor--MPPI routing threshold.",
    })
    atomic_json(staging / "RUN_STATE.json", {"status": "completed"})
    os.replace(staging, final)
    print(json.dumps({"threshold_summaries": summaries}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
