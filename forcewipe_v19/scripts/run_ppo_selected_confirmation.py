#!/usr/bin/env python3
"""Confirm the single DEV-selected PPO endpoint on two frozen evaluation suites."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
TEACHER_ROOT = ROOT.parent
V16_ROOT = TEACHER_ROOT / "forcewipe_v16"
for source in (
    ROOT / "code",
    ROOT / "scripts",
    V16_ROOT / "code",
    TEACHER_ROOT / "forcewipe_v6" / "code",
    TEACHER_ROOT / "forcewipe_v4" / "code",
    ROOT / "release" / "runtime_source",
):
    sys.path.insert(0, str(source))

import forcewipe_v6.tdmpc2_direct_sapien_env as direct_env_module  # noqa: E402
from forcewipe_v6.tdmpc2_direct_firstpass import DirectFirstPassConfig  # noqa: E402
from forcewipe_v6.tdmpc2_direct_sapien_env import V6DirectFirstPassEnv  # noqa: E402
from forcewipe_v16.v16p30_final_analysis import analyze_evaluation  # noqa: E402
from forcewipe_v16.v16p30_final_scenarios import (  # noqa: E402
    FINAL_BLOCKS,
    SCENARIO_ID_BASE as FINAL_SCENARIO_ID_BASE,
    TARGETS_N,
    scenario_digest as final_scenario_digest,
    v16p30_final_scenario,
)
from forcewipe_v19.factor_separated_scenarios import (  # noqa: E402
    BLOCKS as FACTOR_BLOCKS,
    factor_separated_scenario,
    scenario_digest as factor_scenario_digest,
    v19_factor_separated_scenario,
)
from forcewipe_v19.ppo_budget_sensitivity import TRAINING_SEEDS  # noqa: E402
from evaluate_ppo_sensitivity_dev import atomic_json, load_agent, seed_step  # noqa: E402


SELECTION = ROOT / "results/analysis/ppo_budget_sensitivity/DEV_SELECTION_RESULT.json"
OUTPUT_ROOT = ROOT / "results/final"
RUN_ID = "ppo_budget_sensitivity_selected_confirmation_20260920_r1"
EXPECTED_PROFILE = "clip_0p1"
EXPECTED_BUDGET = 2


def selected_endpoint() -> tuple[str, int]:
    payload = json.loads(SELECTION.read_text(encoding="utf-8"))
    if payload.get("status") != "completed" or payload.get("final_sets_used_for_selection") is not False:
        raise RuntimeError("PPO DEV selection is not a valid selection-only result")
    profile = str(payload.get("selected_profile_id"))
    budget = int(payload.get("selected_budget_multiplier", -1))
    if (profile, budget) != (EXPECTED_PROFILE, EXPECTED_BUDGET):
        raise RuntimeError("DEV selection does not match the frozen selected endpoint")
    return profile, budget


def suite_cells() -> tuple[dict, ...]:
    cells = []
    for block_index, block in enumerate(FINAL_BLOCKS):
        for target_index, target in enumerate(TARGETS_N):
            scenario_id = FINAL_SCENARIO_ID_BASE + block_index * len(TARGETS_N) + target_index
            spec = v16p30_final_scenario(target, int(block["scenario_seed"]), scenario_id)
            cells.append({
                "suite": "matched_weak_curvature",
                "block_id": str(block["block_id"]),
                "target_force_n": float(target),
                "scenario_id": scenario_id,
                "scenario_seed": int(block["scenario_seed"]),
                "scenario_digest": final_scenario_digest(spec),
                "scenario_loader": v16p30_final_scenario,
            })
    for block in FACTOR_BLOCKS:
        for target in TARGETS_N:
            spec = factor_separated_scenario(str(block["block_id"]), target)
            cells.append({
                "suite": "factor_separated",
                "block_id": str(block["block_id"]),
                "target_force_n": float(target),
                "scenario_id": int(spec.scenario_id),
                "scenario_seed": int(spec.scenario_seed),
                "scenario_digest": factor_scenario_digest(spec),
                "scenario_loader": v19_factor_separated_scenario,
            })
    if len(cells) != 45:
        raise RuntimeError("confirmation roster must contain 45 cells per seed")
    return tuple(cells)


def run_episode(*, agent, seed: int, cell: dict) -> tuple[list[dict], dict, dict]:
    direct_env_module.nominal_direct_scenario = cell["scenario_loader"]
    target = float(cell["target_force_n"])
    env = V6DirectFirstPassEnv(
        target_force_n=target,
        scenario_seed=int(cell["scenario_seed"]),
        scenario_id=int(cell["scenario_id"]),
        config=DirectFirstPassConfig(),
    )
    observation, reset_info = env.reset(seed=int(cell["scenario_seed"]))
    rows = []
    episode_return = 0.0
    suite_index = 0 if cell["suite"] == "matched_weak_curvature" else 1
    cell_index = suite_cells().index(cell)
    planner_root = 929_600_000 + suite_index * 10_000_000 + TRAINING_SEEDS.index(seed) * 1_000_000 + cell_index * 10_000
    started = time.perf_counter()
    try:
        for step in range(env.config.maximum_steps):
            planner_seed = planner_root + step
            seed_step(planner_seed)
            action_started = time.perf_counter()
            action = agent.act(observation, deterministic=True)
            planning_ms = (time.perf_counter() - action_started) * 1000.0
            next_observation, reward, terminated, truncated, info = env.step(action)
            episode_return += float(reward)
            row = {
                "control_step": step,
                "suite": cell["suite"],
                "block_id": cell["block_id"],
                "target_force_n": target,
                "method_seed": seed,
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
        "profile_id": EXPECTED_PROFILE,
        "budget_multiplier": EXPECTED_BUDGET,
        "method_seed": seed,
        "suite": cell["suite"],
        "block_id": cell["block_id"],
        "target_force_n": target,
        "scenario_id": int(cell["scenario_id"]),
        "scenario_seed": int(cell["scenario_seed"]),
        "scenario_digest": cell["scenario_digest"],
        "episode_return": episode_return,
        "completed_dose_bins": int(last.get("completed_dose_bins", 0)),
        "progress": float(last.get("progress", 0.0)),
        "wall_time_s": time.perf_counter() - started,
        **metrics,
    }
    return rows, reset_info, summary


def aggregate(rows: list[dict], suite: str) -> dict:
    selected = [row for row in rows if row["suite"] == suite]
    compound = [
        bool(row["task_success"] and row["tracking_pass"] and row["safety_pass"] and row["authority_pass"])
        for row in selected
    ]
    return {
        "suite": suite,
        "evaluations": len(selected),
        "task_successes": sum(bool(row["task_success"]) for row in selected),
        "tracking_passes": sum(bool(row["tracking_pass"]) for row in selected),
        "safety_passes": sum(bool(row["safety_pass"]) for row in selected),
        "authority_passes": sum(bool(row["authority_pass"]) for row in selected),
        "compound_passes": sum(compound),
        "samples_gt_15n": sum(int(row["samples_gt_15n"]) for row in selected),
        "maximum_peak_force_n": max(float(row["peak_force_n"]) for row in selected if row["peak_force_n"] is not None),
    }


def main() -> int:
    profile, budget = selected_endpoint()
    cells = suite_cells()
    final = OUTPUT_ROOT / RUN_ID
    stage = OUTPUT_ROOT / f".{RUN_ID}.creating"
    if final.exists() or stage.exists():
        raise SystemExit(f"confirmation run already exists: {RUN_ID}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    stage.mkdir()
    atomic_json(stage / "RUN_DEFINITION.json", {
        "format": "forcewipe_v19_ppo_selected_confirmation_definition_v1",
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "selection_result": str(SELECTION.relative_to(ROOT)),
        "profile_id": profile,
        "budget_multiplier": budget,
        "method_seeds": list(TRAINING_SEEDS),
        "cells_per_seed": len(cells),
        "evaluations": len(cells) * len(TRAINING_SEEDS),
        "selection_data_excluded": True,
        "no_further_parameter_selection": True,
        "scenario_definitions": [
            {key: value for key, value in cell.items() if key != "scenario_loader"}
            for cell in cells
        ],
    })
    evaluations = []
    for seed in TRAINING_SEEDS:
        agent = load_agent(profile, seed, budget)
        for cell in cells:
            evaluation_id = f"seed{seed}_{cell['suite']}_{cell['block_id']}_{int(cell['target_force_n'])}N"
            directory = stage / evaluation_id
            directory.mkdir()
            rows, reset_info, summary = run_episode(agent=agent, seed=seed, cell=cell)
            trace = directory / "TRACE.jsonl"
            with trace.open("x", encoding="utf-8") as stream:
                for row in rows:
                    stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            atomic_json(directory / "EVALUATION_RECORD.json", {"reset_info": reset_info, **summary})
            evaluations.append(summary)
            print(json.dumps({
                "seed": seed,
                "suite": cell["suite"],
                "block": cell["block_id"],
                "target": cell["target_force_n"],
                "task": summary["task_success"],
                "compound": bool(summary["task_success"] and summary["tracking_pass"] and summary["safety_pass"] and summary["authority_pass"]),
            }, sort_keys=True), flush=True)
        del agent
        torch.cuda.empty_cache()
    if len(evaluations) != 225:
        raise RuntimeError("confirmation must contain exactly 225 evaluations")
    keys = {
        (row["suite"], row["block_id"], row["target_force_n"], row["method_seed"])
        for row in evaluations
    }
    if len(keys) != 225:
        raise RuntimeError("confirmation contains duplicate evaluation identities")
    summaries = [aggregate(evaluations, suite) for suite in ("matched_weak_curvature", "factor_separated")]
    atomic_json(stage / "RESULT.json", {
        "format": "forcewipe_v19_ppo_selected_confirmation_result_v1",
        "status": "completed",
        "profile_id": profile,
        "budget_multiplier": budget,
        "evaluations": evaluations,
        "suite_summaries": summaries,
    })
    atomic_json(stage / "RUN_STATE.json", {"status": "completed"})
    os.replace(stage, final)
    print(json.dumps({"suite_summaries": summaries}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
