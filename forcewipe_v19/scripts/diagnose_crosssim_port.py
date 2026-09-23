#!/usr/bin/env python3
"""Read-only diagnostics for the existing MuJoCo cross-simulator port.

This script does not select parameters or use a policy checkpoint.  It
replays the already frozen open-loop calibration motions and reports signed
position errors so that a subsequent port revision can distinguish gain from
dynamic-lag mismatch.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TEACHER_ROOT = ROOT.parent
for source in (
    ROOT / "code",
    ROOT / "scripts",
    TEACHER_ROOT / "forcewipe_v6" / "code",
    TEACHER_ROOT / "forcewipe_v4" / "code",
    ROOT / "release" / "runtime_source",
):
    sys.path.insert(0, str(source))

from forcewipe_v19.crosssim_calibration import MOTIONS  # noqa: E402
from forcewipe_v19.factor_separated_scenarios import factor_separated_scenario  # noqa: E402
from forcewipe_v19.mujoco_crosssim_env import (  # noqa: E402
    MuJoCoContactConfig,
    MuJoCoDirectFirstPassEnv,
)
from run_crosssim_calibration import run_motion  # noqa: E402


REFERENCE_ROOT = ROOT / "results/calibration/mujoco_crosssim_calibration_r2_v1"
OUTPUT = ROOT / "results/analysis/crosssim_port_r3_diagnostic_20260921.json"
SAMPLES = (25, 50, 75, 100, 125, 150, 175, 200, 205, 208, 212, 215, 220, 228, 235, 245, 255, 270, 285, 310, 350)


def main() -> int:
    config = MuJoCoContactConfig(
        actuation_gain=1.0,
        actuation_delay_steps=2,
        actuation_time_constant_s=0.08,
        contact_time_constant_s=0.03,
        contact_damping_ratio=0.5,
        contact_margin_m=0.0035,
        solver_iterations=50,
    )
    records = []
    for motion in MOTIONS:
        reference = json.loads(
            (REFERENCE_ROOT / f"PHYSX_{motion.motion_id}.json").read_text(encoding="utf-8")
        )
        env = MuJoCoDirectFirstPassEnv(
            scenario_spec=factor_separated_scenario("B0", 8.0),
            contact_config=config,
        )
        try:
            candidate = run_motion(env, motion.actions(), engine="MuJoCo")
        finally:
            env.close()
        ref_pos = np.asarray(reference["tool_position_world_m"], dtype=float)
        can_pos = np.asarray(candidate["tool_position_world_m"], dtype=float)
        ref_force = np.asarray(reference["normal_force_n"], dtype=float)
        can_force = np.asarray(candidate["normal_force_n"], dtype=float)
        ref_delta = ref_pos - ref_pos[0]
        can_delta = can_pos - can_pos[0]
        vector_error = can_delta - ref_delta
        rows = []
        for sample in SAMPLES:
            if sample > len(ref_pos):
                continue
            index = sample - 1
            rows.append({
                "sample": sample,
                "reference_delta_z_m": float(ref_delta[index, 2]),
                "candidate_delta_z_m": float(can_delta[index, 2]),
                "candidate_minus_reference_z_m": float(can_delta[index, 2] - ref_delta[index, 2]),
                "reference_force_n": float(ref_force[index]),
                "candidate_force_n": float(can_force[index]),
            })
        records.append({
            "motion_id": motion.motion_id,
            "role": motion.role,
            "axis_position_rmse_m": [
                float(np.sqrt(np.mean(np.square(vector_error[:, axis]))))
                for axis in range(3)
            ],
            "maximum_absolute_axis_error_m": [
                float(np.max(np.abs(vector_error[:, axis])))
                for axis in range(3)
            ],
            "samples": rows,
        })
    payload = {
        "format": "forcewipe_v19_crosssim_port_diagnostic_v1",
        "policy_checkpoint_used": False,
        "selection_performed": False,
        "existing_r2_config": config.__dict__,
        "motions": records,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
