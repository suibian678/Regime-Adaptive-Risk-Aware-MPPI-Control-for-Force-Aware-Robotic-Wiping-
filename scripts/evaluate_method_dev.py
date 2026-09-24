#!/usr/bin/env python3
"""Bounded 4-arm x 3-target development screen for the V19 method."""

from __future__ import annotations

import argparse
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
V16_ROOT = ROOT / "archive/direct_study"
for source in (ROOT / "src", ROOT / "scripts", ROOT / "vendor/tdmpc2", ROOT / "vendor"):
    sys.path.insert(0, str(source))

from forcewipe.direct.tdmpc2_direct_firstpass import DirectFirstPassConfig  # noqa: E402
import forcewipe.direct.tdmpc2_direct_sapien_env as direct_env_module  # noqa: E402
from forcewipe.direct.tdmpc2_direct_sapien_env import V6DirectFirstPassEnv  # noqa: E402
from forcewipe.learning.ood_scenarios import v16_ood_scenario  # noqa: E402
from forcewipe.risk_aware_mppi import (  # noqa: E402
    ForceConditionedRegimeAdaptiveRiskAwareTDMPC2,
    configure_regime_adaptive_risk_mppi,
)
from forcewipe.training import build_v19_training_config  # noqa: E402
from evaluation_common import atomic_json, sha256, summarize  # noqa: E402


SEED = 201
MAIN_SEEDS = (201, 202, 203, 204, 205)
ALL_TARGETS_N = (5.0, 8.0, 12.0)
ARMS = {
    "M1_fixed_objective_fixed_compute": (False, False),
    "M2_adaptive_objective_fixed_compute": (True, False),
    "M3_fixed_objective_adaptive_compute": (False, True),
    "M4_full_adaptive": (True, True),
}
RUN_ID_FULL = "v19_seed201_four_arm_ood_dev12_20260919_r1"
RUN_ID_TARGET12 = "v19_seed201_four_arm_target12_ood_dev4_20260919_r1"
OUTPUT_ROOT = ROOT / "results/dev"


def train_run(seed: int, bc_coefficient: float) -> Path:
    return ROOT / (
        f"results/train/v19_force_conditioned_seed{seed}_"
        f"bc{bc_coefficient:g}_u4000"
    )


def checkpoint(seed: int, bc_coefficient: float) -> Path:
    return train_run(seed, bc_coefficient) / "v19_force_conditioned_tdmpc2.pt"


def calibration(seed: int, bc_coefficient: float) -> Path:
    return train_run(seed, bc_coefficient) / "TRAIN_VALIDATION_CALIBRATION.json"


def load_radii(seed: int, bc_coefficient: float) -> dict[float, float]:
    payload = json.loads(
        calibration(seed, bc_coefficient).read_text(encoding="utf-8")
    )
    return {
        float(target): float(detail["calibrated_radius_n"])
        for target, detail in payload["targets"].items()
    }


def load_agent(
    seed: int,
    arm: str,
    work_dir: Path,
    bc_coefficient: float,
    risk_envelope_mode: str,
):
    adaptive_objective, adaptive_compute = ARMS[arm]
    initialization_seed = seed + 12
    random.seed(initialization_seed)
    np.random.seed(initialization_seed)
    torch.manual_seed(initialization_seed)
    torch.cuda.manual_seed_all(initialization_seed)
    cfg = configure_regime_adaptive_risk_mppi(
        build_v19_training_config(
            seed=seed,
            work_dir=work_dir,
            bc_coefficient=bc_coefficient,
        ),
        calibrated_radii_n=load_radii(seed, bc_coefficient),
        adaptive_objective=adaptive_objective,
        adaptive_compute=adaptive_compute,
        risk_envelope_mode=risk_envelope_mode,
    )
    agent = ForceConditionedRegimeAdaptiveRiskAwareTDMPC2(cfg)
    payload = torch.load(
        checkpoint(seed, bc_coefficient), map_location="cpu", weights_only=False
    )
    agent.model.load_state_dict(payload["model"])
    agent.capture_eval_ema()
    agent._eval_ema_model.load_state_dict(payload["eval_ema_model"])
    agent.model.eval()
    agent._eval_ema_model.eval()
    return agent


def scenario_identity(target: float) -> tuple[int, int]:
    index = ALL_TARGETS_N.index(float(target))
    return 88_100_000 + index, 8_168_000 + index


def run_episode(
    *, agent, training_seed: int, target: float, scenario_seed: int, scenario_id: int, config
):
    env = V6DirectFirstPassEnv(
        target_force_n=target,
        scenario_seed=scenario_seed,
        scenario_id=scenario_id,
        config=config,
    )
    observation, reset_info = env.reset(seed=scenario_seed)
    agent._prev_mean.zero_()
    rows, episode_return = [], 0.0
    try:
        while len(rows) < config.maximum_steps:
            # The same per-step random stream is used for every arm. Arms with
            # different sample budgets consume different suffix lengths only.
            step_seed = (
                9_190_000_000
                + training_seed * 10_000_000
                + scenario_id * 2_000
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
            rows.append(
                {
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
                }
            )
            observation = next_observation
            if terminated or truncated:
                break
    finally:
        env.close()
    return rows, reset_info, episode_return


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target-set", choices=("12", "lower", "full"), default="12"
    )
    parser.add_argument("--arm", choices=tuple(ARMS), default=None)
    parser.add_argument(
        "--seed-set", choices=("single", "main", "tail203"), default="single"
    )
    parser.add_argument(
        "--bc-coef", type=float, choices=(0.0, 0.5, 2.0), default=2.0
    )
    parser.add_argument(
        "--comparison", choices=("all", "fixed-vs-full"), default="all"
    )
    parser.add_argument(
        "--envelope-mode",
        choices=("force_only", "point_envelope", "calibrated"),
        default="calibrated",
    )
    args = parser.parse_args()
    bc_coefficient = float(args.bc_coef)
    bc_run_label = "" if bc_coefficient == 2.0 else f"bc{bc_coefficient:g}_"
    targets_n = {
        "12": (12.0,),
        "lower": (5.0, 8.0),
        "full": ALL_TARGETS_N,
    }[args.target_set]
    selected_seeds = {
        "single": (SEED,),
        "main": MAIN_SEEDS,
        "tail203": (203, 204, 205),
    }[args.seed_set]
    seed_label = {
        "single": "seed201",
        "main": "main5",
        "tail203": "tail203",
    }[args.seed_set]
    if args.arm:
        selected_arms = (args.arm,)
    elif args.comparison == "fixed-vs-full":
        selected_arms = (
            "M1_fixed_objective_fixed_compute",
            "M4_full_adaptive",
        )
    else:
        selected_arms = tuple(ARMS)
    if args.arm:
        evaluation_count = len(selected_seeds) * len(targets_n)
        arm_target_label = {
            "12": f"target12_ood_dev{evaluation_count}",
            "lower": f"target5_8_ood_dev{evaluation_count}",
            "full": f"ood_dev{evaluation_count}",
        }[args.target_set]
        envelope_label = (
            "" if args.envelope_mode == "calibrated" else f"{args.envelope_mode}_"
        )
        run_id = (
            f"v19_{seed_label}_"
            f"{bc_run_label}{envelope_label}{args.arm}_"
            f"{arm_target_label}_20260919_r1"
        )
    elif args.seed_set == "main" and args.comparison == "fixed-vs-full":
        target_label = {
            "12": "target12_ood_dev10",
            "lower": "target5_8_ood_dev20",
            "full": "ood_dev30",
        }[args.target_set]
        run_id = (
            "v19_main5_fixed_vs_full_"
            f"{target_label}_20260919_r1"
        )
    else:
        run_id = RUN_ID_TARGET12 if args.target_set == "12" else RUN_ID_FULL
    final = OUTPUT_ROOT / run_id
    staging = OUTPUT_ROOT / f".{run_id}.creating"
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if final.exists() or staging.exists():
        raise SystemExit(f"development run id already exists: {run_id}")
    staging.mkdir()
    direct_env_module.nominal_direct_scenario = v16_ood_scenario
    config = DirectFirstPassConfig()
    atomic_json(
        staging / "RUN_DEFINITION.json",
        {
            "format": "forcewipe_v19_four_arm_development_definition_v1",
            "run_id": run_id,
            "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "role": "DEVELOPMENT",
            "training_seeds": list(selected_seeds),
            "bc_coefficient": bc_coefficient,
            "risk_envelope_mode": args.envelope_mode,
            "checkpoints": {
                str(seed): {
                    "path": str(checkpoint(seed, bc_coefficient)),
                    "sha256": sha256(checkpoint(seed, bc_coefficient)),
                }
                for seed in selected_seeds
            },
            "calibrations": {
                str(seed): {
                    "path": str(calibration(seed, bc_coefficient)),
                    "sha256": sha256(calibration(seed, bc_coefficient)),
                }
                for seed in selected_seeds
            },
            "arms": ARMS,
            "targets_n": list(targets_n),
            "shared_scenarios": True,
            "classical_force_controller": False,
            "action_shield": False,
            "rewiping": False,
            "claim_boundary": "Repeated development scenarios; method selection evidence only.",
        },
    )
    evaluations = []
    for seed in selected_seeds:
        for arm in selected_arms:
            agent = load_agent(
                seed, arm, staging, bc_coefficient, args.envelope_mode
            )
            for target in targets_n:
                scenario_seed, scenario_id = scenario_identity(target)
                rows, reset_info, episode_return = run_episode(
                    agent=agent,
                    training_seed=seed,
                    target=target,
                    scenario_seed=scenario_seed,
                    scenario_id=scenario_id,
                    config=config,
                )
                trace = staging / (
                    f"seed_{seed}_{arm}_target_{int(target)}n_trace.jsonl"
                )
                with trace.open("w", encoding="utf-8", buffering=1) as stream:
                    for row in rows:
                        stream.write(
                            json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
                        )
                summary = summarize(rows, target=target, episode_return=episode_return)
                mean_force = summary["contact_mean_force_n"]
                rmse = summary["contact_rmse_n"]
                summary.update(
                    {
                        "training_seed": seed,
                        "arm": arm,
                        "scenario_seed": scenario_seed,
                        "scenario_id": scenario_id,
                        "reset_info": reset_info,
                        "trace": trace.name,
                        "trace_sha256": sha256(trace),
                        "mean_relative_error": (
                            abs(float(mean_force) - target) / target
                            if mean_force is not None
                            else None
                        ),
                        "target_normalized_rmse": (
                            float(rmse) / target if rmse is not None else None
                        ),
                        "mean_planning_samples": float(
                            np.mean(
                                [
                                    row["risk_diagnostics"]["budget"]["num_samples"]
                                    for row in rows
                                ]
                            )
                        ),
                        "mean_planning_iterations": float(
                            np.mean(
                                [
                                    row["risk_diagnostics"]["budget"]["iterations"]
                                    for row in rows
                                ]
                            )
                        ),
                    }
                )
                evaluations.append(summary)
                print(
                    json.dumps(
                        {
                            key: summary[key]
                            for key in (
                                "arm",
                                "training_seed",
                                "target_force_n",
                                "success",
                                "completed_dose_bins",
                                "contact_mean_force_n",
                                "contact_rmse_n",
                                "mean_relative_error",
                                "target_normalized_rmse",
                                "peak_force_n",
                                "force_limit_violation_samples",
                                "median_planning_ms",
                                "mean_planning_samples",
                            )
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            del agent
            torch.cuda.empty_cache()

    arms = []
    for arm in selected_arms:
        cell = [row for row in evaluations if row["arm"] == arm]
        arms.append(
            {
                "arm": arm,
                "training_seeds": sorted({row["training_seed"] for row in cell}),
                "evaluations": len(cell),
                "task_successes": int(sum(row["success"] for row in cell)),
                "tracking_gate_passes": int(
                    sum(
                        row["mean_relative_error"] is not None
                        and row["target_normalized_rmse"] is not None
                        and row["mean_relative_error"] <= 0.15
                        and row["target_normalized_rmse"] <= 0.20
                        for row in cell
                    )
                ),
                "force_limit_violation_samples": int(
                    sum(row["force_limit_violation_samples"] for row in cell)
                ),
                "maximum_peak_force_n": float(max(row["peak_force_n"] for row in cell)),
                "median_planning_ms_mean": float(
                    np.mean([row["median_planning_ms"] for row in cell])
                ),
                "mean_planning_samples": float(
                    np.mean([row["mean_planning_samples"] for row in cell])
                ),
            }
        )
    result = {
        "format": "forcewipe_v19_four_arm_development_result_v1",
        "status": "completed",
        "evaluations": evaluations,
        "arm_summaries": arms,
        "claim_boundary": "Repeated development scenarios; not frozen evaluation evidence.",
    }
    atomic_json(staging / "RESULT.json", result)
    atomic_json(staging / "RUN_STATE.json", {"status": "completed"})
    os.replace(staging, final)
    print(json.dumps({"arm_summaries": arms}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
