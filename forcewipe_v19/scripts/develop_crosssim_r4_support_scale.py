#!/usr/bin/env python3
"""Probe the exposed support-drive implementation mismatch in the MuJoCo port."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys


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
    aggregate_motion_metrics,
    comparison_metrics,
    motion_by_id,
)
from forcewipe_v19.crosssim_calibration_r3 import HELD_OUT_MOTIONS  # noqa: E402
from forcewipe_v19.factor_separated_scenarios import factor_separated_scenario  # noqa: E402
from forcewipe_v19.mujoco_crosssim_env import (  # noqa: E402
    MuJoCoContactConfig,
    MuJoCoDirectFirstPassEnv,
)
from run_crosssim_calibration import invalid_motion_metrics, run_motion  # noqa: E402


REFERENCE_ROOT = ROOT / "results/calibration/mujoco_crosssim_calibration_r3_v1"
OUTPUT = ROOT / "results/analysis/crosssim_r4_support_scale_development/RESULT.json"
SUPPORT_SCALES = (1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)


def main() -> int:
    motions = (
        motion_by_id("fit_light_load_hold_release"),
        motion_by_id("fit_medium_load_hold_release"),
        motion_by_id("validation_contact_tangent_release"),
        next(
            item for item in HELD_OUT_MOTIONS
            if item.motion_id == "validation_bidirectional_tangent_release"
        ),
    )
    references = {
        motion.motion_id: json.loads(
            (REFERENCE_ROOT / f"PHYSX_{motion.motion_id}.json").read_text(encoding="utf-8")
        )
        for motion in motions
    }
    spec = factor_separated_scenario("B0", 8.0)
    candidates = []
    for support_scale in SUPPORT_SCALES:
        config = MuJoCoContactConfig(
            actuation_delay_steps=2,
            actuation_time_constant_s=0.08,
            contact_time_constant_s=0.03,
            contact_damping_ratio=0.5,
            contact_margin_m=0.0035,
            tool_drive_scale=0.8,
            support_drive_scale=support_scale,
        )
        per_motion = []
        for motion in motions:
            env = MuJoCoDirectFirstPassEnv(
                scenario_spec=spec,
                contact_config=config,
            )
            try:
                try:
                    trace = run_motion(env, motion.actions(), engine="MuJoCo")
                    metrics = comparison_metrics(references[motion.motion_id], trace)
                except RuntimeError as exc:
                    metrics = invalid_motion_metrics(
                        motion.motion_id, "exposed_development", str(exc)
                    )
            finally:
                env.close()
            per_motion.append({
                "motion_id": motion.motion_id,
                **metrics,
            })
        candidates.append({
            "contact_config": asdict(config),
            "aggregate": aggregate_motion_metrics(per_motion),
            "per_motion": per_motion,
        })
    candidates.sort(key=lambda row: (
        row["aggregate"]["maximum_tolerance_ratio"],
        row["aggregate"]["mean_tolerance_ratio"],
        row["contact_config"]["support_drive_scale"],
    ))
    result = {
        "format": "forcewipe_v19_crosssim_r4_support_scale_development_v1",
        "policy_checkpoint_used": False,
        "motions_are_exposed_development_data": True,
        "candidate_cells": len(candidates),
        "selected": candidates[0],
        "candidates": candidates,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "candidate_cells": len(candidates),
        "selected": candidates[0],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
