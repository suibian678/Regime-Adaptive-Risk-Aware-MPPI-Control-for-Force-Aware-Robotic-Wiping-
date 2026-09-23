#!/usr/bin/env python3
"""Fit causal actuator dynamics using policy-free, contact-free motions."""

from __future__ import annotations

import json
import os
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

import forcewipe_v6.tdmpc2_direct_sapien_env as sapien_module  # noqa: E402
from forcewipe_v19.crosssim_calibration_r3 import ACTUATOR_FIT_MOTIONS  # noqa: E402
from forcewipe_v19.factor_separated_scenarios import (  # noqa: E402
    SCENARIO_ID_BASE,
    SCENARIO_SEED,
    factor_separated_scenario,
    v19_factor_separated_scenario,
)
from forcewipe_v19.mujoco_crosssim_env import (  # noqa: E402
    MuJoCoContactConfig,
    MuJoCoDirectFirstPassEnv,
)
from run_crosssim_calibration import atomic_json, run_motion  # noqa: E402


OUTPUT = ROOT / "results/analysis/crosssim_r3_actuator_fit_development"
DELAYS = tuple(range(0, 9))
TIME_CONSTANTS_S = (0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08)


def position_rmse(reference: dict, candidate: dict) -> float:
    ref = np.asarray(reference["tool_position_world_m"], dtype=float)
    can = np.asarray(candidate["tool_position_world_m"], dtype=float)
    error = (can - can[0]) - (ref - ref[0])
    return float(np.sqrt(np.mean(np.sum(np.square(error), axis=1))))


def main() -> int:
    if OUTPUT.exists():
        raise SystemExit(f"development output already exists: {OUTPUT}")
    staging = OUTPUT.with_name(f".{OUTPUT.name}.creating")
    if staging.exists():
        raise SystemExit(f"staging output already exists: {staging}")
    staging.mkdir(parents=True)
    sapien_module.nominal_direct_scenario = v19_factor_separated_scenario
    physx = sapien_module.V6DirectFirstPassEnv(
        target_force_n=8.0,
        scenario_seed=SCENARIO_SEED,
        scenario_id=SCENARIO_ID_BASE + 1,
    )
    references = {}
    try:
        for motion in ACTUATOR_FIT_MOTIONS:
            trace = run_motion(physx, motion.actions(), engine="PhysX")
            references[motion.motion_id] = trace
            atomic_json(staging / f"PHYSX_{motion.motion_id}.json", trace)
    finally:
        physx.close()

    spec = factor_separated_scenario("B0", 8.0)
    candidates = []
    for delay in DELAYS:
        for time_constant in TIME_CONSTANTS_S:
            contact = MuJoCoContactConfig(
                actuation_delay_steps=delay,
                actuation_time_constant_s=time_constant,
            )
            per_motion = []
            for motion in ACTUATOR_FIT_MOTIONS:
                env = MuJoCoDirectFirstPassEnv(
                    scenario_spec=spec,
                    contact_config=contact,
                )
                try:
                    trace = run_motion(env, motion.actions(), engine="MuJoCo")
                finally:
                    env.close()
                per_motion.append({
                    "motion_id": motion.motion_id,
                    "position_rmse_m": position_rmse(
                        references[motion.motion_id], trace
                    ),
                })
            candidates.append({
                "actuation_delay_steps": delay,
                "actuation_time_constant_s": time_constant,
                "maximum_position_rmse_m": max(
                    row["position_rmse_m"] for row in per_motion
                ),
                "mean_position_rmse_m": float(np.mean([
                    row["position_rmse_m"] for row in per_motion
                ])),
                "per_motion": per_motion,
            })
    candidates.sort(key=lambda row: (
        row["maximum_position_rmse_m"],
        row["mean_position_rmse_m"],
        row["actuation_delay_steps"],
        row["actuation_time_constant_s"],
    ))
    result = {
        "format": "forcewipe_v19_crosssim_r3_actuator_fit_development_v1",
        "policy_checkpoint_used": False,
        "contact_occurred": False,
        "candidate_cells": len(candidates),
        "selected": candidates[0],
        "top_candidates": candidates[:20],
    }
    atomic_json(staging / "RESULT.json", result)
    os.replace(staging, OUTPUT)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
