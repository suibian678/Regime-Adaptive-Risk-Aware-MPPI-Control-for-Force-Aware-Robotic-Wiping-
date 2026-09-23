#!/usr/bin/env python3
"""Combine the matched BC=0/0.5/2 V19 development evidence."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TEACHER_ROOT = ROOT.parent
for source in (
    ROOT / "code",
    TEACHER_ROOT / "forcewipe_v15" / "scripts",
):
    sys.path.insert(0, str(source))

from run_v15_trust_mppi_fresh_dev3 import summarize  # noqa: E402


DEV = ROOT / "results/dev"
BC2_RESULTS = (
    DEV / "v19_main5_fixed_vs_full_target12_ood_dev10_20260919_r1/RESULT.json",
    DEV / "v19_main5_fixed_vs_full_target5_8_ood_dev20_20260919_r1/RESULT.json",
)
BC0_RESULT = (
    DEV / "v19_main5_bc0_M4_full_adaptive_ood_dev15_20260919_r1/RESULT.json"
)
BC05_HEAD = DEV / (
    "v19_main5_bc0.5_M4_full_adaptive_ood_dev15_20260919_r1_"
    "ABORTED_CUDA_AFTER7"
)
BC05_TAIL_RESULT = (
    DEV / "v19_tail203_bc0.5_M4_full_adaptive_ood_dev9_20260919_r1/RESULT.json"
)
OUTPUT = DEV / "V19_BC_ABLATION_DEVELOPMENT_RESULT.json"
TRACE_PATTERN = re.compile(
    r"seed_(?P<seed>\d+)_M4_full_adaptive_target_(?P<target>\d+)n_trace\.jsonl"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def add_common(row: dict, *, bc: float) -> dict:
    item = dict(row)
    item["bc_coefficient"] = bc
    return item


def summarize_head_trace(path: Path) -> dict:
    match = TRACE_PATTERN.fullmatch(path.name)
    if not match:
        raise ValueError(f"unexpected trace name: {path.name}")
    seed = int(match.group("seed"))
    target = float(match.group("target"))
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    episode_return = float(sum(float(row["reward"]) for row in rows))
    result = summarize(rows, target=target, episode_return=episode_return)
    mean_force = result["contact_mean_force_n"]
    rmse = result["contact_rmse_n"]
    result.update(
        {
            "training_seed": seed,
            "arm": "M4_full_adaptive",
            "bc_coefficient": 0.5,
            "trace": str(path),
            "trace_sha256": sha256(path),
            "mean_relative_error": (
                abs(float(mean_force) - target) / target
                if mean_force is not None
                else None
            ),
            "target_normalized_rmse": (
                float(rmse) / target if rmse is not None else None
            ),
            "mean_planning_samples": float(
                np.mean(
                    [row["risk_diagnostics"]["budget"]["num_samples"] for row in rows]
                )
            ),
            "mean_planning_iterations": float(
                np.mean(
                    [row["risk_diagnostics"]["budget"]["iterations"] for row in rows]
                )
            ),
        }
    )
    return result


def group_summary(rows: list[dict], bc: float) -> dict:
    contact = [row for row in rows if row["contact_mean_force_n"] is not None]
    return {
        "bc_coefficient": bc,
        "evaluations": len(rows),
        "task_successes": sum(bool(row["success"]) for row in rows),
        "tracking_gate_passes": sum(
            row["mean_relative_error"] is not None
            and row["target_normalized_rmse"] is not None
            and row["mean_relative_error"] <= 0.15
            and row["target_normalized_rmse"] <= 0.20
            for row in rows
        ),
        "no_contact_evaluations": len(rows) - len(contact),
        "force_limit_violation_samples": sum(
            int(row["force_limit_violation_samples"]) for row in rows
        ),
        "maximum_peak_force_n": max(float(row["peak_force_n"]) for row in rows),
        "mean_mre_contact_evaluations": (
            float(np.mean([row["mean_relative_error"] for row in contact]))
            if contact
            else None
        ),
        "mean_nrmse_contact_evaluations": (
            float(np.mean([row["target_normalized_rmse"] for row in contact]))
            if contact
            else None
        ),
        "mean_planning_samples": float(
            np.mean([row["mean_planning_samples"] for row in rows])
        ),
        "mean_median_planning_ms": float(
            np.mean([row["median_planning_ms"] for row in rows])
        ),
    }


def main() -> int:
    if OUTPUT.exists():
        raise SystemExit(f"output already exists: {OUTPUT}")
    required = [*BC2_RESULTS, BC0_RESULT, BC05_TAIL_RESULT]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    evaluations: list[dict] = []
    for path in BC2_RESULTS:
        evaluations.extend(
            add_common(row, bc=2.0)
            for row in read_json(path)["evaluations"]
            if row["arm"] == "M4_full_adaptive"
        )
    evaluations.extend(
        add_common(row, bc=0.0) for row in read_json(BC0_RESULT)["evaluations"]
    )
    # Seeds 201--202 completed before the CUDA infrastructure abort. Seed 203
    # was rerun identically in the tail result and is taken only from the tail.
    for path in sorted(BC05_HEAD.glob("seed_*_M4_full_adaptive_target_*n_trace.jsonl")):
        match = TRACE_PATTERN.fullmatch(path.name)
        if match and int(match.group("seed")) in (201, 202):
            evaluations.append(summarize_head_trace(path))
    evaluations.extend(
        add_common(row, bc=0.5)
        for row in read_json(BC05_TAIL_RESULT)["evaluations"]
    )

    keys = [
        (float(row["bc_coefficient"]), int(row["training_seed"]), float(row["target_force_n"]))
        for row in evaluations
    ]
    if len(keys) != 45 or len(set(keys)) != 45:
        raise RuntimeError(f"expected 45 unique BC x seed x target keys, got {len(set(keys))}")

    grouped: dict[float, list[dict]] = defaultdict(list)
    for row in evaluations:
        grouped[float(row["bc_coefficient"])].append(row)
    training = {}
    for bc in (0.0, 0.5, 2.0):
        audits = []
        calibrations = []
        for seed in range(201, 206):
            run = (
                ROOT
                / "results/train"
                / f"v19_force_conditioned_seed{seed}_bc{bc:g}_u4000"
            )
            result = read_json(run / "RESULT.json")
            audits.append(result["audit"])
            calibrations.append(read_json(run / "TRAIN_VALIDATION_CALIBRATION.json"))
        training[f"{bc:g}"] = {
            "mean_one_step_force_rmse_n": float(
                np.mean([row["one_step_force_rmse_n"] for row in audits])
            ),
            "mean_actor_teacher_action_rmse": float(
                np.mean([row["actor_teacher_action_rmse"] for row in audits])
            ),
            "all_target_conditions_distinct": all(
                bool(row["target_conditions_distinct"]) for row in calibrations
            ),
        }

    payload = {
        "format": "forcewipe_v19_bc_ablation_development_result_v1",
        "completed_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "role": "DEVELOPMENT",
        "unique_keys": len(set(keys)),
        "training_audit": training,
        "group_summaries": [group_summary(grouped[bc], bc) for bc in (0.0, 0.5, 2.0)],
        "evaluations": sorted(
            evaluations,
            key=lambda row: (
                float(row["bc_coefficient"]),
                int(row["training_seed"]),
                float(row["target_force_n"]),
            ),
        ),
        "source_identities": {str(path): sha256(path) for path in required},
        "aborted_head_definition_sha256": sha256(BC05_HEAD / "RUN_DEFINITION.json"),
        "claim_boundary": "Matched repeated development scenarios; not final evaluation evidence.",
    }
    temporary = OUTPUT.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, OUTPUT)
    print(json.dumps({"group_summaries": payload["group_summaries"], "training_audit": training}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
