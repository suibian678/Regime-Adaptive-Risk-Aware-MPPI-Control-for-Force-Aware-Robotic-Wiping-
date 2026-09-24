#!/usr/bin/env python3
"""Execute the frozen R3 held-out cross-simulator calibration once."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TEACHER_ROOT = ROOT.parent
for source in (ROOT / "src", ROOT / "scripts", ROOT / "vendor/tdmpc2", ROOT / "vendor"):
    sys.path.insert(0, str(source))

import forcewipe.direct.tdmpc2_direct_sapien_env as sapien_module  # noqa: E402
from forcewipe.crosssim_calibration import (  # noqa: E402
    MOTIONS as EXPOSED_CONTACT_MOTIONS,
    aggregate_motion_metrics,
    comparison_metrics,
)
from forcewipe.crosssim_validation import (  # noqa: E402
    ACTUATOR_FIT_MOTIONS,
    HELD_OUT_MOTIONS,
)
from forcewipe.factor_separated_scenarios import (  # noqa: E402
    SCENARIO_ID_BASE,
    SCENARIO_SEED,
    factor_separated_scenario,
    v19_factor_separated_scenario,
)
from forcewipe.mujoco_crosssim_env import (  # noqa: E402
    MuJoCoContactConfig,
    MuJoCoDirectFirstPassEnv,
)
from run_crosssim_calibration import atomic_json, run_motion  # noqa: E402


PROTOCOL = ROOT / "config/CROSS_SIMULATOR_CALIBRATION_PROTOCOL_R3_2026-09-21.json"
OUTPUT = ROOT / "results/calibration/mujoco_crosssim_calibration_r3_v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def position_rmse(reference: dict, candidate: dict) -> float:
    ref = np.asarray(reference["tool_position_world_m"], dtype=float)
    can = np.asarray(candidate["tool_position_world_m"], dtype=float)
    error = (can - can[0]) - (ref - ref[0])
    return float(np.sqrt(np.mean(np.sum(np.square(error), axis=1))))


def require_identity(path: Path, expected: str, label: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise SystemExit(f"{label} identity mismatch: {actual} != {expected}")


def main() -> int:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol.get("status") != "FROZEN_BEFORE_NEW_HELD_OUT_EXECUTION":
        raise SystemExit("R3 protocol is not frozen")
    identities = protocol["source_identities"]
    require_identity(
        ROOT / "src/forcewipe/mujoco_crosssim_env.py",
        identities["mujoco_crosssim_env_sha256"],
        "MuJoCo port",
    )
    require_identity(
        ROOT / "src/forcewipe/crosssim_validation.py",
        identities["motion_definition_sha256"],
        "R3 motion definition",
    )
    if OUTPUT.exists():
        raise SystemExit(f"R3 calibration output already exists: {OUTPUT}")
    staging = OUTPUT.with_name(f".{OUTPUT.name}.creating")
    if staging.exists():
        raise SystemExit(f"R3 calibration staging already exists: {staging}")
    staging.mkdir(parents=True)

    config = MuJoCoContactConfig(**protocol["frozen_contact_config"])
    spec = factor_separated_scenario("B0", 8.0)
    all_motions = (
        *ACTUATOR_FIT_MOTIONS,
        *EXPOSED_CONTACT_MOTIONS,
        *HELD_OUT_MOTIONS,
    )
    sapien_module.nominal_direct_scenario = v19_factor_separated_scenario
    physx = sapien_module.V6DirectFirstPassEnv(
        target_force_n=8.0,
        scenario_seed=SCENARIO_SEED,
        scenario_id=SCENARIO_ID_BASE + 1,
    )
    references = {}
    try:
        for motion in all_motions:
            trace = run_motion(physx, motion.actions(), engine="PhysX")
            references[motion.motion_id] = trace
            atomic_json(staging / f"PHYSX_{motion.motion_id}.json", trace)
    finally:
        physx.close()

    actuator_fit = []
    contact_rows = []
    held_out_ids = {motion.motion_id for motion in HELD_OUT_MOTIONS}
    for motion in all_motions:
        env = MuJoCoDirectFirstPassEnv(
            scenario_spec=spec,
            contact_config=config,
        )
        try:
            candidate = run_motion(env, motion.actions(), engine="MuJoCo")
        finally:
            env.close()
        atomic_json(staging / f"MUJOCO_{motion.motion_id}.json", candidate)
        if motion.role == "actuator_fit":
            actuator_fit.append({
                "motion_id": motion.motion_id,
                "position_rmse_m": position_rmse(
                    references[motion.motion_id], candidate
                ),
            })
        else:
            contact_rows.append({
                "motion_id": motion.motion_id,
                "role": (
                    "validation"
                    if motion.motion_id in held_out_ids
                    else "development"
                ),
                **comparison_metrics(references[motion.motion_id], candidate),
            })

    development = aggregate_motion_metrics(
        row for row in contact_rows if row["role"] != "validation"
    )
    held_out = aggregate_motion_metrics(
        row for row in contact_rows if row["role"] == "validation"
    )
    actuator_max = max(row["position_rmse_m"] for row in actuator_fit)
    status = (
        "completed_validation_pass"
        if actuator_max <= 0.0005
        and development["within_all_tolerances"]
        and held_out["within_all_tolerances"]
        else "completed_validation_fail"
    )
    result = {
        "format": "forcewipe_v19_crosssim_calibration_r3_result_v1",
        "status": status,
        "completed_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "protocol_sha256": sha256(PROTOCOL),
        "policy_checkpoint_used": False,
        "contact_config": asdict(config),
        "actuator_fit": {
            "maximum_position_rmse_m": actuator_max,
            "within_tolerance": actuator_max <= 0.0005,
            "per_motion": actuator_fit,
        },
        "development": development,
        "held_out_validation": held_out,
        "contact_metrics": contact_rows,
    }
    atomic_json(staging / "RESULT.json", result)
    atomic_json(staging / "RUN_STATE.json", {"status": status})
    os.replace(staging, OUTPUT)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
