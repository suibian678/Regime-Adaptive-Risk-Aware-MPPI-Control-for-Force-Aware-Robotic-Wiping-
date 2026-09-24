#!/usr/bin/env python3
"""Train and calibrate the remaining frozen V19 main-method seeds."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / "scripts/train_force_conditioned_seed.py"
CALIBRATE_SCRIPT = ROOT / "scripts/calibrate_trained_seed.py"
MAIN_SEEDS = (201, 202, 203, 204, 205)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bc-coef", type=float, choices=(0.0, 0.5, 2.0), required=True)
    args = parser.parse_args()
    bc_coefficient = float(args.bc_coef)
    bc_tag = f"{bc_coefficient:g}"
    seeds = MAIN_SEEDS if bc_coefficient != 2.0 else (202, 203, 204, 205)
    output = ROOT / (
        "results/train/V19_MAIN_ROSTER_TRAINING_SUMMARY.json"
        if bc_coefficient == 2.0
        else f"results/train/V19_BC{bc_tag}_MAIN_ROSTER_TRAINING_SUMMARY.json"
    )
    if output.exists():
        raise SystemExit(f"roster summary already exists: {output}")
    rows = []
    for seed in seeds:
        run_id = f"v19_force_conditioned_seed{seed}_bc{bc_tag}_u4000"
        run = ROOT / "results/train" / run_id
        training = subprocess.run(
            [
                sys.executable,
                str(TRAIN_SCRIPT),
                "--seed",
                str(seed),
                "--bc-coef",
                bc_tag,
            ],
            check=False,
        )
        result_path = run / "RESULT.json"
        if not result_path.exists():
            raise RuntimeError(f"seed {seed} produced no training result")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        # The roster is unfiltered. Calibration is attempted for every trained
        # checkpoint; an engineering gate miss is reported, never used to
        # replace a seed.
        calibration = subprocess.run(
            [
                sys.executable,
                str(CALIBRATE_SCRIPT),
                "--seed",
                str(seed),
                "--bc-coef",
                bc_tag,
            ],
            check=False,
        )
        calibration_code = calibration.returncode
        row = {
            "seed": seed,
            "training_return_code": training.returncode,
            "offline_training_gate_passed": bool(result["offline_training_gate_passed"]),
            "one_step_force_rmse_n": result["audit"]["one_step_force_rmse_n"],
            "actor_teacher_action_rmse": result["audit"]["actor_teacher_action_rmse"],
            "calibration_return_code": calibration_code,
        }
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    payload = {
        "format": "forcewipe_v19_main_roster_training_summary_v1",
        "completed_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "bc_coefficient": bc_coefficient,
        "seeds": list(seeds),
        "seed_results": rows,
        "all_calibrations_completed": all(
            row["calibration_return_code"] == 0 for row in rows
        ),
        "all_offline_engineering_gates_passed": all(
            row["offline_training_gate_passed"] for row in rows
        ),
        "claim_boundary": "TRAIN and TRAIN-validation only; no simulator evaluation.",
    }
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["all_calibrations_completed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
