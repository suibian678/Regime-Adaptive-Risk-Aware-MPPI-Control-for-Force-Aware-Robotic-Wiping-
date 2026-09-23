#!/usr/bin/env python3
"""Independent numerical and artefact audit for the V19 MuJoCo transfer run."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/crosssim/v19_mujoco_zero_shot_transfer_20260921_r1"
PROTOCOL = ROOT / "config/V19_MUJOCO_ZERO_SHOT_TRANSFER_PROTOCOL_2026-09-21.json"
PHYSX = ROOT / "results/final/v19_factor_separated_m1_m4_evaluation_20260920_r1/RESULT.json"
OUTPUT = ROOT / "results/analysis/v19_mujoco_zero_shot_transfer_independent_audit.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tracking(row: dict) -> bool:
    return bool(
        row["mean_relative_error"] is not None
        and row["target_normalized_rmse"] is not None
        and row["mean_relative_error"] <= 0.15
        and row["target_normalized_rmse"] <= 0.20
    )


def joint(row: dict) -> bool:
    return bool(row["success"] and tracking(row) and row["force_limit_violation_samples"] == 0)


def main() -> int:
    checks: list[dict] = []

    def check(name: str, passed: bool, detail) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    result = json.loads((RUN / "RESULT.json").read_text(encoding="utf-8"))
    state = json.loads((RUN / "RUN_STATE.json").read_text(encoding="utf-8"))
    manifest = json.loads((RUN / "MANIFEST.json").read_text(encoding="utf-8"))
    physx_payload = json.loads(PHYSX.read_text(encoding="utf-8"))
    rows = result["evaluations"]
    keys = [
        (int(row["training_seed"]), str(row["block_id"]), float(row["target_force_n"]))
        for row in rows
    ]
    expected = {
        (seed, block, target)
        for seed in protocol["execution"]["training_seeds"]
        for block in protocol["execution"]["blocks"]
        for target in protocol["execution"]["targets_n"]
    }
    check("result_status", result.get("status") == "completed", result.get("status"))
    check("state_status", state.get("status") == "completed", state.get("status"))
    check("protocol_identity", result.get("protocol_sha256") == sha256(PROTOCOL), result.get("protocol_sha256"))
    check("evaluation_key_set", len(keys) == 45 and len(set(keys)) == 45 and set(keys) == expected, len(set(keys)))

    trace_failures = []
    numeric_failures = []
    planner_failures = []
    for row in rows:
        trace = RUN / row["trace"]
        if not trace.is_file() or sha256(trace) != row["trace_sha256"]:
            trace_failures.append(row["trace"])
            continue
        records = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
        if len(records) != int(row["native_samples"]):
            numeric_failures.append([row["trace"], "sample_count"])
            continue
        force = np.asarray([item["normal_force_n"] for item in records], dtype=float)
        contact = force >= 3.0
        target = float(row["target_force_n"])
        mean_force = float(force[contact].mean()) if np.any(contact) else None
        rmse = float(np.sqrt(np.mean(np.square(force[contact] - target)))) if np.any(contact) else None
        expected_mre = abs(mean_force - target) / target if mean_force is not None else None
        expected_nrmse = rmse / target if rmse is not None else None
        comparisons = (
            int(np.sum(force > 15.0)) == int(row["force_limit_violation_samples"]),
            math.isclose(float(force.max()), float(row["peak_force_n"]), abs_tol=1e-12),
            bool(records[-1]["success"]) == bool(row["success"]),
            (
                expected_mre is None and row["mean_relative_error"] is None
                or math.isclose(expected_mre, float(row["mean_relative_error"]), abs_tol=1e-12)
            ),
            (
                expected_nrmse is None and row["target_normalized_rmse"] is None
                or math.isclose(expected_nrmse, float(row["target_normalized_rmse"]), abs_tol=1e-12)
            ),
        )
        if not all(comparisons):
            numeric_failures.append([row["trace"], comparisons])
        for index, item in enumerate(records):
            expected_seed = (
                9_190_000_000
                + int(row["training_seed"]) * 10_000_000
                + int(row["scenario_id"]) * 2_000
                + index
            )
            if item["control_step"] != index or item["planner_seed"] != expected_seed:
                planner_failures.append([row["trace"], index])
                break
    check("trace_hashes_and_counts", not trace_failures, trace_failures)
    check("trace_metric_recomputation", not numeric_failures, numeric_failures)
    check("planner_seed_sequence", not planner_failures, planner_failures)

    manifest_failures = []
    for item in manifest["files"]:
        path = RUN / item["path"]
        if (
            not path.is_file()
            or path.stat().st_size != int(item["bytes"])
            or sha256(path) != item["sha256"]
        ):
            manifest_failures.append(item["path"])
    check("manifest_files", not manifest_failures, {
        "entries": len(manifest["files"]), "failures": manifest_failures
    })

    mujoco_summary = {
        "evaluations": len(rows),
        "task_successes": sum(bool(row["success"]) for row in rows),
        "tracking_passes": sum(tracking(row) for row in rows),
        "sampled_force_safety_passes": sum(row["force_limit_violation_samples"] == 0 for row in rows),
        "joint_passes": sum(joint(row) for row in rows),
        "force_limit_violation_samples": sum(int(row["force_limit_violation_samples"]) for row in rows),
        "maximum_peak_force_n": max(float(row["peak_force_n"]) for row in rows),
    }
    stored_mujoco = dict(next(
        item for item in result["engine_summaries"] if item["engine"] == "MuJoCo"
    ))
    stored_mujoco.pop("engine")
    check("mujoco_aggregate", mujoco_summary == stored_mujoco, mujoco_summary)

    physx_rows = [
        row for row in physx_payload["evaluations"]
        if row["arm"] == "M4_full_adaptive" and row["block_id"] in {"B0", "S1", "S2"}
    ]
    physx_keys = {
        (int(row["training_seed"]), str(row["block_id"]), float(row["target_force_n"]))
        for row in physx_rows
    }
    check("physx_pair_key_set", len(physx_rows) == 45 and physx_keys == expected, len(physx_rows))
    check("pair_count", result["paired_analysis"]["pairs"] == 45, result["paired_analysis"]["pairs"])
    check("no_staging", not (RUN.parent / f".{RUN.name}.creating").exists(), True)

    passed = all(item["passed"] for item in checks)
    audit = {
        "format": "forcewipe_v19_mujoco_zero_shot_transfer_independent_audit_v1",
        "status": "PASS" if passed else "FAIL",
        "checks_passed": sum(item["passed"] for item in checks),
        "checks_total": len(checks),
        "checks": checks,
        "result_sha256": sha256(RUN / "RESULT.json"),
        "manifest_sha256": sha256(RUN / "MANIFEST.json"),
        "mujoco_summary": mujoco_summary,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
