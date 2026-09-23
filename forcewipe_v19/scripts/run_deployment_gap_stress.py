#!/usr/bin/env python3
"""Run the frozen 375-evaluation one-factor deployment-gap stress panel."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
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
    ROOT / "code", ROOT / "scripts", V16_ROOT / "code", V16_ROOT / "scripts",
    TEACHER_ROOT / "forcewipe_v15" / "code",
    TEACHER_ROOT / "forcewipe_v15" / "scripts",
    TEACHER_ROOT / "forcewipe_v14" / "code",
    TEACHER_ROOT / "forcewipe_v6" / "code",
    TEACHER_ROOT / "forcewipe_v4" / "code",
    ROOT / "release" / "runtime_source" / "tdmpc2",
    ROOT / "release" / "runtime_source",
):
    sys.path.insert(0, str(source))

import forcewipe_v6.tdmpc2_direct_sapien_env as direct_env_module  # noqa: E402
from forcewipe_v6.tdmpc2_direct_firstpass import DirectFirstPassConfig  # noqa: E402
from forcewipe_v19.deployment_gap import deployment_gap_conditions  # noqa: E402
from forcewipe_v19.deployment_gap_sapien_env import V19DeploymentGapEnv  # noqa: E402
from forcewipe_v19.factor_separated_scenarios import (  # noqa: E402
    SCENARIO_SEED,
    TARGETS_N,
    factor_separated_scenario,
    scenario_digest,
    v19_factor_separated_scenario,
)
from run_v15_trust_mppi_fresh_dev3 import atomic_json, sha256, summarize  # noqa: E402
from run_v19_seed201_method_dev12 import MAIN_SEEDS, load_agent  # noqa: E402


PROTOCOL = ROOT / "config/DEPLOYMENT_GAP_STRESS_PROTOCOL_2026-09-20.json"
BASELINE_RESULT = ROOT / "results/final/v19_factor_separated_m1_m4_evaluation_20260920_r1/RESULT.json"
RUN_ID = "v19_deployment_gap_stress_20260920_r1"
OUTPUT_ROOT = ROOT / "results/final"
BLOCK_IDS = ("B0", "S1", "S2")
ARM = "M4_full_adaptive"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_episode(*, agent, training_seed: int, spec, gap_config, config):
    env = V19DeploymentGapEnv(
        gap_config=gap_config,
        target_force_n=spec.target_force_n,
        scenario_seed=spec.scenario_seed,
        scenario_id=spec.scenario_id,
        config=config,
    )
    observation, reset_info = env.reset(seed=spec.scenario_seed)
    agent._prev_mean.zero_()
    rows, episode_return = [], 0.0
    try:
        while len(rows) < config.maximum_steps:
            step_seed = (
                9_190_000_000
                + training_seed * 10_000_000
                + spec.scenario_id * 2_000
                + len(rows)
            )
            random.seed(step_seed)
            np.random.seed(step_seed % (2**32))
            torch.manual_seed(step_seed)
            torch.cuda.manual_seed_all(step_seed)
            started = time.perf_counter()
            action_tensor = agent.act(
                torch.from_numpy(observation),
                t0=len(rows) == 0,
                eval_mode=True,
            )
            torch.cuda.synchronize()
            planning_ms = (time.perf_counter() - started) * 1_000.0
            action = action_tensor.detach().cpu().numpy().astype(np.float32)
            actor_center = (
                agent._last_selected_actor_center.detach().cpu().numpy().astype(np.float32)
            )
            diagnostics = agent.risk_diagnostics()
            next_observation, reward, terminated, truncated, info = env.step(action)
            episode_return += float(reward)
            rows.append({
                "control_step": len(rows),
                "planner_seed": step_seed,
                "observation": np.asarray(observation, dtype=np.float32).tolist(),
                "action": action.tolist(),
                "actor_center": actor_center.tolist(),
                "absolute_actor_deviation": np.abs(action - actor_center).tolist(),
                "next_observation": np.asarray(next_observation, dtype=np.float32).tolist(),
                "reward": float(reward),
                "planning_ms": planning_ms,
                "risk_diagnostics": diagnostics,
                **info,
            })
            observation = next_observation
            if terminated or truncated:
                break
    finally:
        env.close()
    return rows, reset_info, episode_return


def main() -> int:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol["status"] != "CODE_ONLY_BINDING_PASS_READY_TO_EVALUATE":
        raise SystemExit("deployment-gap protocol is not executable")
    if not BASELINE_RESULT.is_file():
        raise SystemExit("frozen nominal baseline result is missing")
    direct_env_module.nominal_direct_scenario = v19_factor_separated_scenario
    final = OUTPUT_ROOT / RUN_ID
    staging = OUTPUT_ROOT / f".{RUN_ID}.creating"
    if final.exists() or staging.exists():
        raise SystemExit(f"deployment-gap run id already exists: {RUN_ID}")
    staging.mkdir(parents=True)
    definitions = []
    for block_id in BLOCK_IDS:
        for seed in MAIN_SEEDS:
            gap_seed = 7_190_000_000 + seed * 10_000_000
            for condition in deployment_gap_conditions(block_id, random_seed=gap_seed):
                for target in TARGETS_N:
                    spec = factor_separated_scenario(block_id, target)
                    definitions.append({
                        "training_seed": seed,
                        "block_id": block_id,
                        "target_force_n": target,
                        "condition_id": condition.condition_id,
                        "gap_config": asdict(condition.config),
                        "scenario_id": spec.scenario_id,
                        "scenario_seed": spec.scenario_seed,
                        "scenario_digest": scenario_digest(spec),
                    })
    if len(definitions) != 375 or len({
        (row["training_seed"], row["block_id"], row["target_force_n"], row["condition_id"])
        for row in definitions
    }) != 375:
        raise RuntimeError("deployment-gap evaluation roster is not the frozen 375-cell product")
    atomic_json(staging / "RUN_DEFINITION.json", {
        "format": "forcewipe_v19_deployment_gap_run_definition_v1",
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "protocol_sha256": digest(PROTOCOL),
        "baseline_result_sha256": digest(BASELINE_RESULT),
        "new_evaluations": 375,
        "existing_nominal_evaluations_reused": 45,
        "definitions": definitions,
        "method": ARM,
        "classical_force_controller": False,
        "action_shield": False,
        "rewiping": False,
    })
    evaluations = []
    config = DirectFirstPassConfig()
    for seed in MAIN_SEEDS:
        agent = load_agent(seed, ARM, staging, 2.0, "force_only")
        try:
            for block_id in BLOCK_IDS:
                gap_seed = 7_190_000_000 + seed * 10_000_000
                for condition in deployment_gap_conditions(block_id, random_seed=gap_seed):
                    for target in TARGETS_N:
                        spec = factor_separated_scenario(block_id, target)
                        rows, reset_info, episode_return = run_episode(
                            agent=agent,
                            training_seed=seed,
                            spec=spec,
                            gap_config=condition.config,
                            config=config,
                        )
                        trace = staging / (
                            f"seed_{seed}_{block_id}_{condition.condition_id}_"
                            f"target_{int(target)}n.jsonl"
                        )
                        with trace.open("x", encoding="utf-8", buffering=1) as stream:
                            for row in rows:
                                stream.write(json.dumps(
                                    row, sort_keys=True, allow_nan=False
                                ) + "\n")
                        summary = summarize(
                            rows, target=target, episode_return=episode_return
                        )
                        mean_force = summary["contact_mean_force_n"]
                        rmse = summary["contact_rmse_n"]
                        tracking = (
                            mean_force is not None
                            and abs(float(mean_force) - target) / target <= 0.15
                            and float(rmse) / target <= 0.20
                        )
                        summary.update({
                            "training_seed": seed,
                            "arm": ARM,
                            "block_id": block_id,
                            "condition_id": condition.condition_id,
                            "gap_config": asdict(condition.config),
                            "scenario_id": spec.scenario_id,
                            "scenario_seed": spec.scenario_seed,
                            "scenario_digest": scenario_digest(spec),
                            "tracking_pass": bool(tracking),
                            "compound_pass": bool(
                                summary["success"]
                                and tracking
                                and summary["force_limit_violation_samples"] == 0
                            ),
                            "reset_info": reset_info,
                            "trace": trace.name,
                            "trace_sha256": sha256(trace),
                        })
                        evaluations.append(summary)
                        print(json.dumps({
                            "seed": seed,
                            "block": block_id,
                            "condition": condition.condition_id,
                            "target": target,
                            "compound": summary["compound_pass"],
                            "peak": summary["peak_force_n"],
                        }, sort_keys=True), flush=True)
        finally:
            del agent
            torch.cuda.empty_cache()
    atomic_json(staging / "RESULT.json", {
        "format": "forcewipe_v19_deployment_gap_result_v1",
        "status": "completed",
        "new_evaluations": len(evaluations),
        "evaluations": evaluations,
        "total_native_samples": sum(row["native_samples"] for row in evaluations),
        "task_successes": sum(bool(row["success"]) for row in evaluations),
        "tracking_passes": sum(bool(row["tracking_pass"]) for row in evaluations),
        "compound_passes": sum(bool(row["compound_pass"]) for row in evaluations),
        "force_limit_violation_samples": sum(
            row["force_limit_violation_samples"] for row in evaluations
        ),
        "maximum_peak_force_n": max(row["peak_force_n"] for row in evaluations),
    })
    atomic_json(staging / "RUN_STATE.json", {"status": "completed"})
    os.replace(staging, final)
    print(f"COMPLETED {len(evaluations)} evaluations: {final}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
