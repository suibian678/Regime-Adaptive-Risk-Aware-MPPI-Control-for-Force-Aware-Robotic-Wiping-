#!/usr/bin/env python3
"""Reconstruct the frozen direct-PPO training curves and fairness statement."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
V16 = ROOT.parent / "forcewipe_v16"
TRAIN = V16 / "results/train/v16p30_direct_ppo"
FINAL_ROOT = V16 / "results/final/v16p30_final_matched_simulation_20260830_r1"
OUT = ROOT / "results/analysis/v19_ppo_training_fairness"
SEEDS = (301, 302, 303, 304, 305)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    runs = {}
    endpoints = []
    for seed in SEEDS:
        directory = TRAIN / f"v16p30_direct_ppo_seed{seed}_transitions82774_20260830_r2"
        metrics = read_jsonl(directory / "TRAINING_METRICS.jsonl")
        episodes = read_jsonl(directory / "EPISODE_SUMMARIES.jsonl")
        result = json.loads((directory / "RESULT.json").read_text(encoding="utf-8"))
        runs[seed] = metrics
        endpoints.append({
            "seed": seed,
            "environment_transitions": int(result["environment_transitions"]),
            "optimizer_updates": int(result["optimizer_updates"]),
            "episodes_completed": int(result["episodes_completed"]),
            "training_task_successes": int(result["task_successes_during_training"]),
            "training_force_violation_episodes": int(result["force_violation_episodes_during_training"]),
            "training_peak_force_n": float(result["peak_force_n_during_training"]),
            "episode_log_rows": len(episodes),
        })

    with (OUT / "PPO_TRAINING_ENDPOINTS.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(endpoints[0]))
        writer.writeheader(); writer.writerows(endpoints)

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.65), constrained_layout=True)
    colours = plt.cm.Blues(np.linspace(0.38, 0.82, len(SEEDS)))
    common_x = None
    success_curves, violation_curves = [], []
    for seed_index, (colour, seed) in enumerate(zip(colours, SEEDS)):
        rows = runs[seed]
        x = np.asarray([row["transitions"] for row in rows], dtype=float)
        episodes = np.asarray([row["episodes_completed"] for row in rows], dtype=float)
        success = np.asarray([row["task_successes"] for row in rows], dtype=float) / episodes
        violation = np.asarray([row["force_violation_episodes"] for row in rows], dtype=float) / episodes
        if common_x is None:
            common_x = x
        if not np.array_equal(x, common_x):
            raise RuntimeError("PPO seeds do not share the same transition grid")
        success_curves.append(success); violation_curves.append(violation)
        axes[0].plot(
            x, success, color=colour, lw=0.9, alpha=0.8,
            label="Individual seeds" if seed_index == 0 else None,
        )
        axes[1].plot(x, violation, color=colour, lw=0.9, alpha=0.8)
    for axis, curves, ylabel in (
        (axes[0], success_curves, "Cumulative task-success rate"),
        (axes[1], violation_curves, "Cumulative force-violation rate"),
    ):
        mean = np.mean(np.stack(curves), axis=0)
        axis.plot(common_x, mean, color="#d95f02", lw=2.0, label="Five-seed mean")
        axis.set_xlabel("Environment transitions")
        axis.set_ylabel(ylabel)
        axis.set_xlim(0, 82_774)
        axis.grid(True, alpha=0.22, lw=0.6)
    axes[0].set_ylim(-0.001, 0.051)
    axes[1].set_ylim(0.60, 1.01)
    axes[0].legend(frameon=False, loc="upper left")
    fig.savefig(OUT / "ppo_training_curves.pdf", bbox_inches="tight")
    fig.savefig(OUT / "ppo_training_curves.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    with (FINAL_ROOT / "EVALUATION_METRICS.csv").open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        ppo_evals = [row for row in csv.DictReader(stream) if row["method"] == "M2"]
    summary = {
        "format": "forcewipe_v19_ppo_training_fairness_analysis_v1",
        "seeds": list(SEEDS),
        "same_environment_transition_budget_as_tdmpc2_source": True,
        "environment_transitions_per_seed": 82_774,
        "ppo_is_online_on_policy": True,
        "tdmpc2_uses_the_frozen_offline_source_dataset": True,
        "optimizer_update_counts_are_not_matched": True,
        "shared_deployment_observation_and_action_interface": True,
        "training_task_successes_total": sum(row["training_task_successes"] for row in endpoints),
        "training_episodes_total": sum(row["episodes_completed"] for row in endpoints),
        "training_force_violation_episodes_total": sum(row["training_force_violation_episodes"] for row in endpoints),
        "final_evaluations": len(ppo_evals),
        "final_task_successes": sum(row["task_success"] == "True" for row in ppo_evals),
        "interpretation": "This is a matched simulator-transition and deployment-interface baseline, not an algorithm-independent sample-efficiency comparison.",
    }
    (OUT / "PPO_TRAINING_FAIRNESS_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
