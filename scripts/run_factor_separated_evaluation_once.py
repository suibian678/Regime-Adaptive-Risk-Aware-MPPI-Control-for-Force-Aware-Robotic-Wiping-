#!/usr/bin/env python3
"""Frozen 270-evaluation M1--M4 factor-separated method comparison."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
TEACHER_ROOT = ROOT.parent
V16_ROOT = ROOT / "archive/direct_study"
for source in (ROOT / "src", ROOT / "scripts", ROOT / "vendor/tdmpc2", ROOT / "vendor"):
    sys.path.insert(0, str(source))

import forcewipe.direct.tdmpc2_direct_sapien_env as direct_env_module  # noqa: E402
from forcewipe.direct.tdmpc2_direct_firstpass import DirectFirstPassConfig  # noqa: E402
from forcewipe.factor_separated_scenarios import (  # noqa: E402
    BLOCKS, SCENARIO_SEED, TARGETS_N, factor_separated_scenario,
    scenario_digest, v19_factor_separated_scenario,
)
from evaluation_common import atomic_json, sha256, summarize  # noqa: E402
from evaluate_method_dev import (  # noqa: E402
    MAIN_SEEDS, load_agent, run_episode,
)


PROTOCOL = ROOT / "config/V19_FACTOR_SEPARATED_EVALUATION_PROTOCOL_DRAFT.json"
RUN_ID = "v19_factor_separated_m1_m4_evaluation_20260920_r1"
OUTPUT_ROOT = ROOT / "results/final"
ARMS = ("M1_fixed_objective_fixed_compute", "M4_full_adaptive")


def main() -> int:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol.get("status") != "METHOD_FROZEN_READY_TO_EXECUTE":
        raise SystemExit("method is not frozen for factor-separated execution")
    final = OUTPUT_ROOT / RUN_ID
    staging = OUTPUT_ROOT / f".{RUN_ID}.creating"
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if final.exists() or staging.exists():
        raise SystemExit(f"run id already exists: {RUN_ID}")
    staging.mkdir()
    direct_env_module.nominal_direct_scenario = v19_factor_separated_scenario
    config = DirectFirstPassConfig()
    definitions = []
    for block in BLOCKS:
        for target in TARGETS_N:
            spec = factor_separated_scenario(block["block_id"], target)
            definitions.append({
                "block_id": block["block_id"], "factor": block["factor"],
                "level": block["level"], "target_force_n": target,
                "scenario_id": spec.scenario_id, "scenario_seed": spec.scenario_seed,
                "scenario_digest": scenario_digest(spec),
            })
    atomic_json(staging / "RUN_DEFINITION.json", {
        "format": "forcewipe_v19_factor_separated_run_definition_v1",
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "protocol_sha256": sha256(PROTOCOL), "evaluations": 270,
        "training_seeds": list(MAIN_SEEDS), "arms": list(ARMS),
        "bc_coefficient": 2.0, "risk_envelope_mode": "force_only",
        "scenario_definitions": definitions,
        "common_random_numbers": True,
        "classical_force_controller": False, "action_shield": False,
        "rewiping": False,
    })
    evaluations = []
    for seed in MAIN_SEEDS:
        for arm in ARMS:
            agent = load_agent(seed, arm, staging, 2.0, "force_only")
            for block in BLOCKS:
                for target in TARGETS_N:
                    spec = factor_separated_scenario(block["block_id"], target)
                    rows, reset_info, episode_return = run_episode(
                        agent=agent, training_seed=seed, target=target,
                        scenario_seed=spec.scenario_seed,
                        scenario_id=spec.scenario_id, config=config,
                    )
                    trace = staging / (
                        f"seed_{seed}_{arm}_{block['block_id']}_"
                        f"target_{int(target)}n.jsonl"
                    )
                    with trace.open("x", encoding="utf-8", buffering=1) as stream:
                        for row in rows:
                            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                    summary = summarize(rows, target=target, episode_return=episode_return)
                    mean_force = summary["contact_mean_force_n"]
                    rmse = summary["contact_rmse_n"]
                    summary.update({
                        "training_seed": seed, "arm": arm,
                        "block_id": block["block_id"], "factor": block["factor"],
                        "level": block["level"], "scenario_id": spec.scenario_id,
                        "scenario_seed": spec.scenario_seed,
                        "scenario_digest": scenario_digest(spec),
                        "reset_info": reset_info, "trace": trace.name,
                        "trace_sha256": sha256(trace),
                        "mean_relative_error": (
                            abs(float(mean_force) - target) / target
                            if mean_force is not None else None
                        ),
                        "signed_relative_bias": (
                            (float(mean_force) - target) / target
                            if mean_force is not None else None
                        ),
                        "target_normalized_rmse": (
                            float(rmse) / target if rmse is not None else None
                        ),
                        "mean_planning_samples": float(np.mean([
                            row["risk_diagnostics"]["budget"]["num_samples"]
                            for row in rows
                        ])),
                    })
                    evaluations.append(summary)
                    print(json.dumps({
                        "seed": seed, "arm": arm, "block": block["block_id"],
                        "target": target, "task": summary["success"],
                        "mre": summary["mean_relative_error"],
                        "nrmse": summary["target_normalized_rmse"],
                        "peak": summary["peak_force_n"],
                    }, sort_keys=True), flush=True)
            del agent
            torch.cuda.empty_cache()

    arm_summaries = []
    for arm in ARMS:
        cell = [row for row in evaluations if row["arm"] == arm]
        joint = [
            bool(row["success"])
            and row["mean_relative_error"] is not None
            and row["mean_relative_error"] <= 0.15
            and row["target_normalized_rmse"] <= 0.20
            and row["force_limit_violation_samples"] == 0
            for row in cell
        ]
        arm_summaries.append({
            "arm": arm, "evaluations": len(cell),
            "joint_passes": int(sum(joint)),
            "task_successes": int(sum(bool(row["success"]) for row in cell)),
            "tracking_passes": int(sum(
                row["mean_relative_error"] is not None
                and row["mean_relative_error"] <= 0.15
                and row["target_normalized_rmse"] <= 0.20
                for row in cell
            )),
            "force_limit_violation_samples": int(sum(
                row["force_limit_violation_samples"] for row in cell
            )),
            "maximum_peak_force_n": float(max(row["peak_force_n"] for row in cell)),
        })
    atomic_json(staging / "RESULT.json", {
        "format": "forcewipe_v19_factor_separated_result_v1",
        "status": "completed", "evaluations": evaluations,
        "arm_summaries": arm_summaries,
    })
    atomic_json(staging / "RUN_STATE.json", {"status": "completed"})
    os.replace(staging, final)
    print(json.dumps({"arm_summaries": arm_summaries}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
