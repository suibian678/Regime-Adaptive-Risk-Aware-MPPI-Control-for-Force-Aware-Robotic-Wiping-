#!/usr/bin/env python3
"""Analyse the matched historical actor--MPPI threshold sensitivity."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/sensitivity/v19_historical_gate_threshold_sensitivity_dev90_20260920_r2"
OUT = ROOT / "results/analysis/v19_historical_threshold_sensitivity"


def main() -> int:
    result = json.loads((RUN / "RESULT.json").read_text(encoding="utf-8"))
    rows = result["evaluations"]
    thresholds = sorted({float(row["activation_fraction"]) for row in rows})
    targets = sorted({float(row["target_force_n"]) for row in rows})
    summaries = []
    for threshold in thresholds:
        for target in targets:
            cell = [
                row for row in rows
                if float(row["activation_fraction"]) == threshold
                and float(row["target_force_n"]) == target
            ]
            summaries.append({
                "activation_fraction": threshold, "target_force_n": target,
                "evaluations": len(cell),
                "task_successes": sum(bool(row["success"]) for row in cell),
                "tracking_passes": sum(
                    row["mean_relative_error"] <= 0.15
                    and row["target_normalized_rmse"] <= 0.20
                    for row in cell
                ),
                "mean_mre": float(np.mean([row["mean_relative_error"] for row in cell])),
                "mean_nrmse": float(np.mean([row["target_normalized_rmse"] for row in cell])),
                "maximum_peak_force_n": float(max(row["peak_force_n"] for row in cell)),
                "violation_samples": int(sum(row["force_limit_violation_samples"] for row in cell)),
                "actor_step_fraction": float(
                    sum(row["actor_steps"] for row in cell)
                    / sum(row["native_samples"] for row in cell)
                ),
            })
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "THRESHOLD_TARGET_SUMMARY.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader(); writer.writerows(summaries)

    colours = {5.0: "#0072B2", 8.0: "#E69F00", 12.0: "#009E73"}
    markers = {5.0: "o", 8.0: "s", 12.0: "^"}
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.35), constrained_layout=True)
    for target in targets:
        cell = [row for row in summaries if row["target_force_n"] == target]
        x = [row["activation_fraction"] for row in cell]
        for axis, field in zip(
            axes, ("mean_mre", "mean_nrmse", "actor_step_fraction")
        ):
            axis.plot(
                x, [row[field] for row in cell], color=colours[target],
                marker=markers[target], lw=1.4, ms=4.0, label=f"{int(target)} N",
            )
    axes[0].set_ylabel("Mean relative error")
    axes[1].set_ylabel("Target-normalised RMSE")
    axes[2].set_ylabel("Actor step fraction")
    for axis in axes:
        axis.set_xlabel(r"Activation fraction $\alpha$")
        axis.set_xticks(thresholds)
        axis.grid(True, alpha=0.22, lw=0.6)
    axes[0].legend(frameon=False, ncol=1, loc="best")
    fig.savefig(OUT / "historical_threshold_sensitivity.pdf", bbox_inches="tight")
    fig.savefig(OUT / "historical_threshold_sensitivity.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    machine = {
        "format": "forcewipe_v19_historical_threshold_sensitivity_analysis_v1",
        "evaluations": len(rows), "thresholds": thresholds,
        "targets_n": targets, "threshold_target_summaries": summaries,
        "new_method_uses_hard_threshold": False,
        "interpretation": "Development sensitivity of the historical router; not a hyperparameter search for RA-RMPPI.",
    }
    (OUT / "THRESHOLD_SENSITIVITY_ANALYSIS.json").write_text(
        json.dumps(machine, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["threshold_summaries"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
