#!/usr/bin/env python3
"""Combine the force-only, point-envelope, and calibrated-envelope screens."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEV = ROOT / "results/dev"
OUT = ROOT / "results/analysis/v19_envelope_ablation"
SOURCES = {
    "force_only": [
        DEV / "v19_main5_force_only_M4_full_adaptive_ood_dev15_20260919_r1/RESULT.json"
    ],
    "point_envelope": [
        DEV / "v19_main5_point_envelope_M4_full_adaptive_ood_dev15_20260919_r1/RESULT.json"
    ],
    "calibrated": [
        DEV / "v19_main5_fixed_vs_full_target5_8_ood_dev20_20260919_r1/RESULT.json",
        DEV / "v19_main5_fixed_vs_full_target12_ood_dev10_20260919_r1/RESULT.json",
    ],
}


def main() -> int:
    evaluations = []
    for mode, paths in SOURCES.items():
        for path in paths:
            payload = json.loads(path.read_text(encoding="utf-8"))
            for row in payload["evaluations"]:
                if row["arm"] != "M4_full_adaptive":
                    continue
                evaluations.append({**row, "risk_envelope_mode": mode})
    keys = {
        (row["risk_envelope_mode"], row["training_seed"], row["target_force_n"])
        for row in evaluations
    }
    if len(evaluations) != 45 or len(keys) != 45:
        raise RuntimeError("envelope ablation is not the expected 3 x 5 x 3 cross")

    summaries = []
    for mode in SOURCES:
        for target in (5.0, 8.0, 12.0):
            cell = [
                row for row in evaluations
                if row["risk_envelope_mode"] == mode
                and float(row["target_force_n"]) == target
            ]
            summaries.append({
                "risk_envelope_mode": mode, "target_force_n": target,
                "evaluations": len(cell),
                "task_successes": sum(bool(row["success"]) for row in cell),
                "tracking_passes": sum(
                    row["mean_relative_error"] <= 0.15
                    and row["target_normalized_rmse"] <= 0.20
                    for row in cell
                ),
                "mean_force_n": float(np.mean([row["contact_mean_force_n"] for row in cell])),
                "mean_mre": float(np.mean([row["mean_relative_error"] for row in cell])),
                "mean_nrmse": float(np.mean([row["target_normalized_rmse"] for row in cell])),
                "maximum_peak_force_n": float(max(row["peak_force_n"] for row in cell)),
                "violation_samples": int(sum(row["force_limit_violation_samples"] for row in cell)),
            })
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "ENVELOPE_TARGET_SUMMARY.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader(); writer.writerows(summaries)

    modes = tuple(SOURCES)
    colours = {"force_only": "#999999", "point_envelope": "#E69F00", "calibrated": "#0072B2"}
    markers = {"force_only": "o", "point_envelope": "s", "calibrated": "^"}
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.35), constrained_layout=True)
    for mode in modes:
        cell = [row for row in summaries if row["risk_envelope_mode"] == mode]
        x = [row["target_force_n"] for row in cell]
        for axis, field in zip(axes, ("mean_mre", "mean_nrmse", "maximum_peak_force_n")):
            axis.plot(
                x, [row[field] for row in cell], color=colours[mode],
                marker=markers[mode], lw=1.4, ms=4.0,
                label=mode.replace("_", " "),
            )
    axes[0].set_ylabel("Mean relative error")
    axes[1].set_ylabel("Target-normalised RMSE")
    axes[2].set_ylabel("Maximum sampled force (N)")
    for axis in axes:
        axis.set_xlabel("Target force (N)")
        axis.set_xticks((5, 8, 12)); axis.grid(True, alpha=0.22, lw=0.6)
    axes[0].legend(frameon=False, fontsize=7)
    fig.savefig(OUT / "envelope_ablation.pdf", bbox_inches="tight")
    fig.savefig(OUT / "envelope_ablation.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    machine = {
        "format": "forcewipe_v19_envelope_ablation_analysis_v1",
        "evaluations": len(evaluations), "unique_keys": len(keys),
        "summaries": summaries,
    }
    (OUT / "ENVELOPE_ABLATION_ANALYSIS.json").write_text(
        json.dumps(machine, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(machine, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
