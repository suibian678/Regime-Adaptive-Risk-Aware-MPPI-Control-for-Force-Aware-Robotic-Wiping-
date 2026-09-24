#!/usr/bin/env python3
"""Paired descriptive analysis for the frozen deployment-gap stress panel."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
STRESS_RESULT = ROOT / "results/final/v19_deployment_gap_stress_20260920_r1/RESULT.json"
BASELINE_RESULT = ROOT / "results/final/v19_factor_separated_m1_m4_evaluation_20260920_r1/RESULT.json"
OUTPUT_ROOT = ROOT / "results/analysis/deployment_gap_stress_20260920"
SEEDS = (201, 202, 203, 204, 205)
TARGETS = (5.0, 8.0, 12.0)
BLOCKS = ("B0", "S1", "S2")
RESAMPLES = 10_000
BOOTSTRAP_SEED = 919_730_021


def _normalise(row: dict, *, condition_id: str) -> dict:
    target = float(row["target_force_n"])
    mean_force = row.get("contact_mean_force_n")
    rmse = row.get("contact_rmse_n")
    mre = (
        abs(float(mean_force) - target) / target
        if mean_force is not None else None
    )
    nrmse = float(rmse) / target if rmse is not None else None
    tracking = bool(
        mre is not None and nrmse is not None and mre <= 0.15 and nrmse <= 0.20
    )
    task = bool(row["success"])
    safe = int(row["force_limit_violation_samples"]) == 0
    return {
        "condition_id": condition_id,
        "block_id": str(row["block_id"]),
        "training_seed": int(row["training_seed"]),
        "target_force_n": target,
        "task_success": task,
        "tracking_pass": tracking,
        "safety_pass": safe,
        "compound_pass": bool(task and tracking and safe),
        "mre": mre,
        "nrmse": nrmse,
        "contact_fraction": float(row["contact_fraction"]),
        "peak_force_n": float(row["peak_force_n"]),
        "force_limit_violation_samples": int(row["force_limit_violation_samples"]),
        "violation_episode": not safe,
        "episode_length_samples": int(row["native_samples"]),
    }


def _mean(rows: list[dict], metric: str) -> float | None:
    values = [float(row[metric]) for row in rows if row[metric] is not None]
    return float(np.mean(values)) if values else None


def _paired_seed_bootstrap(
    stress: list[dict], nominal_lookup: dict[tuple, dict], metric: str,
) -> dict:
    seed_means = []
    for seed in SEEDS:
        differences = []
        for row in stress:
            if row["training_seed"] != seed:
                continue
            key = (row["block_id"], seed, row["target_force_n"])
            base = nominal_lookup[key]
            if row[metric] is None or base[metric] is None:
                continue
            differences.append(float(row[metric]) - float(base[metric]))
        if not differences:
            return {
                "point_estimate": None, "ci95_low": None, "ci95_high": None,
                "status": "not_estimable",
            }
        seed_means.append(float(np.mean(differences)))
    values = np.asarray(seed_means, dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = values[rng.integers(0, len(values), size=(RESAMPLES, len(values)))].mean(axis=1)
    return {
        "point_estimate": float(values.mean()),
        "ci95_low": float(np.percentile(draws, 2.5)),
        "ci95_high": float(np.percentile(draws, 97.5)),
        "status": "paired_seed_cluster_bootstrap_pointwise",
    }


def run() -> Path:
    stress_payload = json.loads(STRESS_RESULT.read_text(encoding="utf-8"))
    baseline_payload = json.loads(BASELINE_RESULT.read_text(encoding="utf-8"))
    if stress_payload.get("status") != "completed" or int(stress_payload.get("new_evaluations", -1)) != 375:
        raise RuntimeError("deployment-gap result is incomplete")
    stress = [_normalise(row, condition_id=str(row["condition_id"])) for row in stress_payload["evaluations"]]
    nominal = [
        _normalise(row, condition_id="nominal")
        for row in baseline_payload["evaluations"]
        if row.get("arm") == "M4_full_adaptive" and row.get("block_id") in BLOCKS
    ]
    if len(stress) != 375 or len(nominal) != 45:
        raise RuntimeError("unexpected stress or nominal denominator")
    stress_keys = {
        (row["condition_id"], row["block_id"], row["training_seed"], row["target_force_n"])
        for row in stress
    }
    nominal_keys = {
        (row["block_id"], row["training_seed"], row["target_force_n"])
        for row in nominal
    }
    if len(stress_keys) != 375 or len(nominal_keys) != 45:
        raise RuntimeError("duplicate deployment-gap identities")
    nominal_lookup = {
        (row["block_id"], row["training_seed"], row["target_force_n"]): row
        for row in nominal
    }
    metrics = (
        "compound_pass", "task_success", "tracking_pass", "safety_pass",
        "mre", "nrmse", "contact_fraction", "peak_force_n",
        "violation_episode", "episode_length_samples",
    )
    summaries = []
    interval_rows = []
    for block_id in BLOCKS:
        conditions = ["nominal"] + sorted({
            row["condition_id"] for row in stress if row["block_id"] == block_id
        })
        for condition in conditions:
            rows = nominal if condition == "nominal" else stress
            selected = [
                row for row in rows
                if row["block_id"] == block_id and row["condition_id"] == condition
            ]
            if len(selected) != 15:
                raise RuntimeError(f"{block_id}/{condition} must contain 15 evaluations")
            summary = {
                "block_id": block_id,
                "condition_id": condition,
                "evaluations": len(selected),
                "task_successes": sum(row["task_success"] for row in selected),
                "tracking_passes": sum(row["tracking_pass"] for row in selected),
                "safety_passes": sum(row["safety_pass"] for row in selected),
                "compound_passes": sum(row["compound_pass"] for row in selected),
                "mean_mre": _mean(selected, "mre"),
                "mean_nrmse": _mean(selected, "nrmse"),
                "mean_contact_fraction": _mean(selected, "contact_fraction"),
                "mean_peak_force_n": _mean(selected, "peak_force_n"),
                "maximum_peak_force_n": max(row["peak_force_n"] for row in selected),
                "force_limit_violation_samples": sum(row["force_limit_violation_samples"] for row in selected),
                "violation_episodes": sum(row["violation_episode"] for row in selected),
                "mean_episode_length_samples": _mean(selected, "episode_length_samples"),
            }
            summaries.append(summary)
            if condition != "nominal":
                for metric in metrics:
                    estimate = _paired_seed_bootstrap(selected, nominal_lookup, metric)
                    interval_rows.append({
                        "block_id": block_id,
                        "condition_id": condition,
                        "metric": metric,
                        "stress_minus_nominal": estimate["point_estimate"],
                        "ci95_low": estimate["ci95_low"],
                        "ci95_high": estimate["ci95_high"],
                        "status": estimate["status"],
                    })
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT_ROOT / "CONDITION_SUMMARY.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader(); writer.writerows(summaries)
    interval_path = OUTPUT_ROOT / "PAIRED_INTERVALS.csv"
    with interval_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(interval_rows[0]))
        writer.writeheader(); writer.writerows(interval_rows)
    output = OUTPUT_ROOT / "RESULT.json"
    output.write_text(json.dumps({
        "format": "forcewipe_v19_deployment_gap_analysis_v1",
        "status": "completed",
        "new_stress_evaluations": len(stress),
        "reused_nominal_evaluations": len(nominal),
        "reported_evaluations": len(stress) + len(nominal),
        "condition_summaries": summaries,
        "paired_intervals": interval_rows,
        "inference_scope": "fixed B0, S1, and S2 blocks with five checkpoint seeds; pointwise intervals",
        "claim_boundary": "simulation deployment-gap stress evidence, not hardware or Sim2Real validation",
    }, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return output


if __name__ == "__main__":
    print(run())
