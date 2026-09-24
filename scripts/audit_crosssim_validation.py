#!/usr/bin/env python3
"""Independent numerical audit of the frozen R3 cross-simulator result."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/calibration/mujoco_crosssim_calibration_r3_v1"
OUTPUT = ROOT / "results/analysis/crosssim_r3_independent_audit.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    result_path = RUN / "RESULT.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    checks = []

    def check(name: str, condition: bool, evidence: object) -> None:
        checks.append({"check": name, "pass": bool(condition), "evidence": evidence})

    check("terminal status", result["status"] == "completed_validation_fail", result["status"])
    check("no checkpoint", result["policy_checkpoint_used"] is False, result["policy_checkpoint_used"])
    check("development passed", result["development"]["within_all_tolerances"] is True, result["development"])
    check("held-out failed", result["held_out_validation"]["within_all_tolerances"] is False, result["held_out_validation"])
    axis_rows = []
    for row in result["contact_metrics"]:
        motion_id = row["motion_id"]
        ref_path = RUN / f"PHYSX_{motion_id}.json"
        can_path = RUN / f"MUJOCO_{motion_id}.json"
        ref = json.loads(ref_path.read_text(encoding="utf-8"))
        can = json.loads(can_path.read_text(encoding="utf-8"))
        ref_pos = np.asarray(ref["tool_position_world_m"], dtype=float)
        can_pos = np.asarray(can["tool_position_world_m"], dtype=float)
        error = (can_pos - can_pos[0]) - (ref_pos - ref_pos[0])
        axis_rmse = np.sqrt(np.mean(np.square(error), axis=0))
        recomputed = float(np.sqrt(np.mean(np.sum(np.square(error), axis=1))))
        check(
            f"position metric {motion_id}",
            abs(recomputed - float(row["position_rmse_m"])) <= 1e-12,
            {"stored": row["position_rmse_m"], "recomputed": recomputed},
        )
        axis_rows.append({
            "motion_id": motion_id,
            "role": row["role"],
            "axis_position_rmse_m": [float(value) for value in axis_rmse],
            "combined_position_rmse_m": recomputed,
            "physx_sha256": sha256(ref_path),
            "mujoco_sha256": sha256(can_path),
        })
    failed = [row for row in result["contact_metrics"] if not row["within_all_tolerances"]]
    check(
        "single held-out failure",
        len(failed) == 1
        and failed[0]["motion_id"] == "validation_bidirectional_tangent_release",
        [row["motion_id"] for row in failed],
    )
    payload = {
        "format": "forcewipe_v19_crosssim_r3_independent_audit_v1",
        "status": "PASS" if all(row["pass"] for row in checks) else "FAIL",
        "checks_passed": sum(row["pass"] for row in checks),
        "checks_total": len(checks),
        "result_sha256": sha256(result_path),
        "axis_metrics": axis_rows,
        "checks": checks,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
