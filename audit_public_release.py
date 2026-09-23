#!/usr/bin/env python3
"""Verify the compact public ForceWipe release without raw trace archives."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
V19 = ROOT / "forcewipe_v19"


def load(relative: str):
    return json.loads((V19 / relative).read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, label: str, checks: list[str]) -> None:
    if not condition:
        raise AssertionError(label)
    checks.append(label)


def main() -> int:
    checks: list[str] = []

    with (ROOT / "MANIFEST_SHA256.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    mismatches = []
    for row in rows:
        path = ROOT / row["path"]
        if (
            not path.is_file()
            or path.stat().st_size != int(row["bytes"])
            or sha256(path) != row["sha256"]
        ):
            mismatches.append(row["path"])
    require(not mismatches, f"repository manifest mismatch: {mismatches}", checks)

    with (ROOT / "MODEL_MANIFEST.csv").open(encoding="utf-8-sig", newline="") as stream:
        models = list(csv.DictReader(stream))
    require(
        len(models) == 5
        and {int(row["seed"]) for row in models} == {201, 202, 203, 204, 205}
        and all(len(row["sha256"]) == 64 for row in models),
        "five external checkpoint identities are complete",
        checks,
    )

    factor = load("results/factor_separated/RESULT.json")
    require(factor["status"] == "completed" and len(factor["evaluations"]) == 270,
            "factor-separated result contains 270 evaluations", checks)
    factor_keys = {
        (row["arm"], row["training_seed"], row["block_id"], row["target_force_n"])
        for row in factor["evaluations"]
    }
    require(len(factor_keys) == 270, "factor-separated keys are unique", checks)
    arms = {row["arm"]: row for row in factor["arm_summaries"]}
    require(
        arms["M1_fixed_objective_fixed_compute"]["joint_passes"] == 85
        and arms["M4_full_adaptive"]["joint_passes"] == 100,
        "compound-pass counts are 85/135 and 100/135",
        checks,
    )

    analysis = load("results/analysis/v19_factor_separated_m1_m4/FACTOR_SEPARATED_ANALYSIS.json")
    intervals = {row["metric"]: row for row in analysis["intervals"]}
    joint = intervals["joint_pass"]
    require(
        abs(joint["difference_full_minus_fixed"] - 0.1111111111111111) < 1e-12
        and abs(joint["interval_low"] - 0.037037037037037035) < 1e-12
        and abs(joint["interval_high"] - 0.18518518518518517) < 1e-12,
        "paired compound-pass interval is [3.70, 18.52] percentage points",
        checks,
    )
    require(
        abs(intervals["median_planning_ms"]["difference_full_minus_fixed"]
            - (-25.652220763210682)) < 1e-12,
        "median planning-time difference is -25.652 ms",
        checks,
    )

    ppo = load("results/ppo_confirmation/RESULT.json")
    suites = {row["suite"]: row for row in ppo["suite_summaries"]}
    require(
        ppo["status"] == "completed" and len(ppo["evaluations"]) == 225
        and suites["matched_weak_curvature"]["compound_passes"] == 0
        and suites["factor_separated"]["compound_passes"] == 6,
        "PPO confirmation closes 225 evaluations with 0/90 and 6/135 compound pass",
        checks,
    )

    gap = load("results/deployment_gap/RESULT.json")
    require(
        gap["status"] == "completed" and len(gap["evaluations"]) == 375
        and gap["compound_passes"]
        == sum(bool(row["compound_pass"]) for row in gap["evaluations"]),
        "deployment-gap aggregate closes all 375 new evaluations",
        checks,
    )

    crosssim = load("results/crosssim/RESULT.json")
    engines = {row["engine"]: row for row in crosssim["engine_summaries"]}
    require(
        crosssim["status"] == "completed" and len(crosssim["evaluations"]) == 45
        and engines["MuJoCo"]["joint_passes"] == 30
        and engines["PhysX"]["joint_passes"] == 24,
        "cross-engine result contains 45 pairs and the frozen engine summaries",
        checks,
    )
    crosssim_audit = load("results/analysis/v19_mujoco_zero_shot_transfer_independent_audit.json")
    require(
        crosssim_audit["status"] == "PASS"
        and crosssim_audit["checks_passed"] == crosssim_audit["checks_total"] == 12,
        "archived trace-level cross-engine audit passes 12/12",
        checks,
    )

    require((V19 / "paper/ras/manuscript_ras.pdf").is_file(),
            "RAS manuscript PDF is present", checks)
    require((V19 / "paper/ojcs/manuscript_ojcs.pdf").is_file(),
            "OJ-CS manuscript PDF is present", checks)
    require((V19 / "paper/ojcs/supplement_ojcs.pdf").is_file(),
            "OJ-CS supplement PDF is present", checks)

    print(f"PASS {len(checks)}/{len(checks)} compact public-release checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
