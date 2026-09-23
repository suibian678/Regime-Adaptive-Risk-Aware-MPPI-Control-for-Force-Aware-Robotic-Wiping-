#!/usr/bin/env python3
"""Fit only the exposed tangential-contact channel for cross-sim R4."""

from __future__ import annotations

from dataclasses import asdict
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

from forcewipe_v19.crosssim_calibration import (  # noqa: E402
    POSITION_TOLERANCE_M,
    comparison_metrics,
    motion_by_id,
)
from forcewipe_v19.crosssim_calibration_r3 import (  # noqa: E402
    ACTUATOR_FIT_MOTIONS,
    HELD_OUT_MOTIONS,
)
from forcewipe_v19.factor_separated_scenarios import factor_separated_scenario  # noqa: E402
from forcewipe_v19.mujoco_crosssim_env import (  # noqa: E402
    MuJoCoContactConfig,
    MuJoCoDirectFirstPassEnv,
)
from run_crosssim_calibration import run_motion  # noqa: E402


REFERENCE_ROOT = ROOT / "results/calibration/mujoco_crosssim_calibration_r3_v1"
OUTPUT = ROOT / "results/analysis/crosssim_r4_tangent_fit_development/RESULT.json"
TANGENTIAL_SCALES = (0.2, 0.3, 0.4, 0.5, 0.6, 0.8)
FRICTION_SCALES = (1.0, 1.5, 2.0, 2.5, 3.0)


def position_rmse(reference: dict, candidate: dict) -> float:
    ref = np.asarray(reference["tool_position_world_m"], dtype=float)
    can = np.asarray(candidate["tool_position_world_m"], dtype=float)
    error = (can - can[0]) - (ref - ref[0])
    return float(np.sqrt(np.mean(np.sum(np.square(error), axis=1))))


def main() -> int:
    tangent_fit = next(
        motion for motion in ACTUATOR_FIT_MOTIONS
        if motion.motion_id == "fit_free_space_tangent_reversal"
    )
    cross_fit = next(
        motion for motion in ACTUATOR_FIT_MOTIONS
        if motion.motion_id == "fit_free_space_cross_reversal"
    )
    forward_contact = motion_by_id("validation_contact_tangent_release")
    bidirectional_contact = next(
        motion for motion in HELD_OUT_MOTIONS
        if motion.motion_id == "validation_bidirectional_tangent_release"
    )
    motions = (tangent_fit, cross_fit, forward_contact, bidirectional_contact)
    references = {
        motion.motion_id: json.loads(
            (REFERENCE_ROOT / f"PHYSX_{motion.motion_id}.json").read_text(encoding="utf-8")
        )
        for motion in motions
    }
    spec = factor_separated_scenario("B0", 8.0)
    candidates = []
    for tangential_scale in TANGENTIAL_SCALES:
        for friction_scale in FRICTION_SCALES:
            config = MuJoCoContactConfig(
                actuation_delay_steps=2,
                actuation_time_constant_s=0.08,
                contact_time_constant_s=0.03,
                contact_damping_ratio=0.5,
                contact_margin_m=0.0035,
                tool_drive_scale=0.8,
                tool_tangential_drive_scale=tangential_scale,
                friction_scale=friction_scale,
            )
            rows = []
            ratios = []
            for motion in motions:
                env = MuJoCoDirectFirstPassEnv(
                    scenario_spec=spec,
                    contact_config=config,
                )
                try:
                    trace = run_motion(env, motion.actions(), engine="MuJoCo")
                finally:
                    env.close()
                reference = references[motion.motion_id]
                if motion in (tangent_fit, cross_fit):
                    rmse = position_rmse(reference, trace)
                    ratio = rmse / POSITION_TOLERANCE_M
                    row = {
                        "motion_id": motion.motion_id,
                        "position_rmse_m": rmse,
                        "maximum_tolerance_ratio": ratio,
                        "within_all_tolerances": ratio <= 1.0,
                    }
                else:
                    row = {
                        "motion_id": motion.motion_id,
                        **comparison_metrics(reference, trace),
                    }
                    ratio = row["maximum_tolerance_ratio"]
                rows.append(row)
                ratios.append(ratio)
            candidates.append({
                "contact_config": asdict(config),
                "maximum_tolerance_ratio": float(max(ratios)),
                "mean_tolerance_ratio": float(np.mean(ratios)),
                "within_all_tolerances": all(
                    row["within_all_tolerances"] for row in rows
                ),
                "per_motion": rows,
            })
    candidates.sort(key=lambda row: (
        row["maximum_tolerance_ratio"],
        row["mean_tolerance_ratio"],
        row["contact_config"]["tool_tangential_drive_scale"],
        row["contact_config"]["friction_scale"],
    ))
    result = {
        "format": "forcewipe_v19_crosssim_r4_tangent_fit_development_v1",
        "policy_checkpoint_used": False,
        "r3_held_out_is_exposed_development_data": True,
        "candidate_cells": len(candidates),
        "selected": candidates[0],
        "top_candidates": candidates[:20],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
