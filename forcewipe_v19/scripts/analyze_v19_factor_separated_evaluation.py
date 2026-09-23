#!/usr/bin/env python3
"""Paired block--seed analysis for the frozen V19 factor-separated study."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/final/v19_factor_separated_m1_m4_evaluation_20260920_r1"
OUT = ROOT / "results/analysis/v19_factor_separated_m1_m4"
ARMS = ("M1_fixed_objective_fixed_compute", "M4_full_adaptive")


def tracking(row: dict) -> bool:
    return bool(
        row["mean_relative_error"] is not None
        and row["target_normalized_rmse"] is not None
        and row["mean_relative_error"] <= 0.15
        and row["target_normalized_rmse"] <= 0.20
    )


def value(row: dict, metric: str) -> float:
    if metric == "joint_pass":
        return float(bool(row["success"]) and tracking(row) and row["force_limit_violation_samples"] == 0)
    if metric == "task_success":
        return float(bool(row["success"]))
    if metric == "tracking_pass":
        return float(tracking(row))
    if metric == "mean_relative_error":
        return float(row["mean_relative_error"])
    if metric == "target_normalized_rmse":
        return float(row["target_normalized_rmse"])
    if metric == "absolute_signed_bias":
        return abs(float(row["signed_relative_bias"]))
    if metric == "peak_force_n":
        return float(row["peak_force_n"])
    if metric == "median_planning_ms":
        return float(row["median_planning_ms"])
    if metric == "mean_planning_samples":
        return float(row["mean_planning_samples"])
    raise KeyError(metric)


def main() -> int:
    result = json.loads((RUN / "RESULT.json").read_text(encoding="utf-8"))
    rows = result["evaluations"]
    lookup = {
        (int(row["training_seed"]), row["block_id"], float(row["target_force_n"]), row["arm"]): row
        for row in rows
    }
    seeds = sorted({key[0] for key in lookup})
    blocks = sorted({key[1] for key in lookup})
    targets = sorted({key[2] for key in lookup})
    if len(lookup) != 270 or len(seeds) != 5 or len(blocks) != 9 or len(targets) != 3:
        raise RuntimeError("result is not the frozen 5 x 9 x 3 x 2 cross")

    metrics = (
        "joint_pass", "task_success", "tracking_pass", "mean_relative_error",
        "target_normalized_rmse", "absolute_signed_bias", "peak_force_n",
        "median_planning_ms", "mean_planning_samples",
    )
    pairs = []
    for seed in seeds:
        for block in blocks:
            for target in targets:
                fixed = lookup[(seed, block, target, ARMS[0])]
                full = lookup[(seed, block, target, ARMS[1])]
                row = {"training_seed": seed, "block_id": block, "target_force_n": target}
                for metric in metrics:
                    row[f"fixed_{metric}"] = value(fixed, metric)
                    row[f"full_{metric}"] = value(full, metric)
                    row[f"difference_{metric}"] = value(full, metric) - value(fixed, metric)
                pairs.append(row)

    rng = np.random.default_rng(19_920_920)
    repetitions = 10_000
    interval_rows = []
    for metric in metrics:
        point = float(np.mean([row[f"difference_{metric}"] for row in pairs]))
        samples = np.empty(repetitions, dtype=np.float64)
        for index in range(repetitions):
            sampled_blocks = rng.choice(blocks, size=len(blocks), replace=True)
            sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
            differences = []
            for block in sampled_blocks:
                for seed in sampled_seeds:
                    for target in targets:
                        fixed = lookup[(int(seed), str(block), target, ARMS[0])]
                        full = lookup[(int(seed), str(block), target, ARMS[1])]
                        differences.append(value(full, metric) - value(fixed, metric))
            samples[index] = np.mean(differences)
        interval_rows.append({
            "metric": metric, "difference_full_minus_fixed": point,
            "interval_low": float(np.quantile(samples, 0.025)),
            "interval_high": float(np.quantile(samples, 0.975)),
            "bootstrap_repetitions": repetitions,
        })

    factor_rows = []
    for arm in ARMS:
        for factor, level in sorted({(row["factor"], row["level"]) for row in rows}):
            cell = [row for row in rows if row["arm"] == arm and row["factor"] == factor and row["level"] == level]
            factor_rows.append({
                "arm": arm, "factor": factor, "level": level,
                "evaluations": len(cell),
                "joint_passes": sum(
                    bool(row["success"]) and tracking(row)
                    and row["force_limit_violation_samples"] == 0 for row in cell
                ),
                "task_successes": sum(bool(row["success"]) for row in cell),
                "tracking_passes": sum(tracking(row) for row in cell),
                "mean_mre": float(np.mean([row["mean_relative_error"] for row in cell])),
                "mean_nrmse": float(np.mean([row["target_normalized_rmse"] for row in cell])),
                "maximum_peak_force_n": float(max(row["peak_force_n"] for row in cell)),
                "violation_samples": int(sum(row["force_limit_violation_samples"] for row in cell)),
            })

    OUT.mkdir(parents=True, exist_ok=True)
    for name, payload in (("PAIRED_DIFFERENCES.csv", pairs), ("PAIRED_INTERVALS.csv", interval_rows), ("FACTOR_LEVEL_SUMMARY.csv", factor_rows)):
        with (OUT / name).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(payload[0]))
            writer.writeheader(); writer.writerows(payload)
    machine = {
        "format": "forcewipe_v19_factor_separated_analysis_v1",
        "evaluations": len(rows), "paired_cells": len(pairs),
        "intervals": interval_rows, "factor_level_summaries": factor_rows,
    }
    (OUT / "FACTOR_SEPARATED_ANALYSIS.json").write_text(
        json.dumps(machine, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"paired_cells": len(pairs), "intervals": interval_rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
