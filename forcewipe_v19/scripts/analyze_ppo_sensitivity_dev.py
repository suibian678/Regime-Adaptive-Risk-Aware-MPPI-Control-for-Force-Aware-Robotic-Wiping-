#!/usr/bin/env python3
"""Aggregate the complete 810-evaluation PPO DEV sensitivity matrix."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
TEACHER_ROOT = ROOT.parent
for source_root in (
    ROOT / "code",
    TEACHER_ROOT / "forcewipe_v16" / "code",
):
    sys.path.insert(0, str(source_root))

from forcewipe_v19.ppo_budget_sensitivity import (  # noqa: E402
    TRAINING_SEEDS,
    configuration_endpoints,
    selection_key,
)


DEV_ROOT = ROOT / "results" / "dev" / "ppo_budget_sensitivity"
OUTPUT_ROOT = ROOT / "results" / "analysis" / "ppo_budget_sensitivity"


def run() -> Path:
    rows = []
    for profile_id, multiplier in configuration_endpoints():
        for seed in TRAINING_SEEDS:
            result_path = DEV_ROOT / f"dev_{profile_id}_{multiplier}x_seed{seed}" / "RESULT.json"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if result.get("status") != "completed" or len(result.get("evaluations", [])) != 18:
                raise RuntimeError(f"incomplete PPO DEV result: {result_path}")
            rows.extend(result["evaluations"])
    if len(rows) != 810:
        raise RuntimeError("PPO DEV matrix must contain exactly 810 evaluations")
    keys = {
        (
            row["profile_id"], int(row["budget_multiplier"]), int(row["method_seed"]),
            row["block_id"], float(row["target_force_n"]),
        )
        for row in rows
    }
    if len(keys) != 810:
        raise RuntimeError("PPO DEV matrix contains duplicate identities")

    summaries = []
    for profile_id, multiplier in configuration_endpoints():
        selected = [
            row for row in rows
            if row["profile_id"] == profile_id
            and int(row["budget_multiplier"]) == multiplier
        ]
        compound = sum(
            row["task_success"] and row["tracking_pass"]
            and row["safety_pass"] and row["authority_pass"]
            for row in selected
        )
        summaries.append({
            "profile_id": profile_id,
            "budget_multiplier": multiplier,
            "evaluations": len(selected),
            "compound_passes": compound,
            "task_successes": sum(bool(row["task_success"]) for row in selected),
            "tracking_passes": sum(bool(row["tracking_pass"]) for row in selected),
            "safety_passes": sum(bool(row["safety_pass"]) for row in selected),
            "authority_passes": sum(bool(row["authority_pass"]) for row in selected),
            "completed_dose_bins": sum(int(row["completed_dose_bins"]) for row in selected),
            "mean_episode_return": sum(float(row["episode_return"]) for row in selected) / len(selected),
            "selection_key": list(selection_key(
                rows, profile_id=profile_id, budget_multiplier=multiplier
            )),
        })
    winner = max(summaries, key=lambda row: tuple(row["selection_key"]))
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_ROOT / "DEV_ENDPOINT_SUMMARY.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            name for name in summaries[0] if name != "selection_key"
        ])
        writer.writeheader()
        writer.writerows({k: v for k, v in row.items() if k != "selection_key"} for row in summaries)
    output = OUTPUT_ROOT / "DEV_SELECTION_RESULT.json"
    output.write_text(json.dumps({
        "format": "forcewipe_v19_ppo_sensitivity_dev_selection_v1",
        "status": "completed",
        "evaluations": len(rows),
        "endpoints": summaries,
        "selected_profile_id": winner["profile_id"],
        "selected_budget_multiplier": winner["budget_multiplier"],
        "selection_rule": "compound, task, bins, return, frozen endpoint order",
        "final_sets_used_for_selection": False,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


if __name__ == "__main__":
    print(run())
