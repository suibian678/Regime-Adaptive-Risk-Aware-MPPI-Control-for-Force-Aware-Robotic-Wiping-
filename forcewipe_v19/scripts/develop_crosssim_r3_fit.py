#!/usr/bin/env python3
"""Fit-only development sweep for a causal-lag MuJoCo port revision.

The four motions used here have all been exposed by the frozen R1/R2 studies;
none can serve as final validation again.  This script never loads a policy
checkpoint and never evaluates the new R3 held-out motions.
"""

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
    MOTIONS,
    aggregate_motion_metrics,
    comparison_metrics,
)
from forcewipe_v19.factor_separated_scenarios import factor_separated_scenario  # noqa: E402
from forcewipe_v19.mujoco_crosssim_env import (  # noqa: E402
    MuJoCoContactConfig,
    MuJoCoDirectFirstPassEnv,
)
from run_crosssim_calibration import invalid_motion_metrics, run_motion  # noqa: E402


REFERENCE_ROOT = ROOT / "results/calibration/mujoco_crosssim_calibration_r2_v1"
OUTPUT = ROOT / "results/analysis/crosssim_r3i_drive_fit_development/RESULT.json"
ACTUATION_DELAYS = (2,)
CONTACT_TIME_CONSTANTS_S = (0.03, 0.05, 0.10)
CONTACT_DAMPING_RATIOS = (0.5, 1.0)
CONTACT_MARGINS_M = (0.0025, 0.003, 0.0035, 0.004)
CONTACT_IMPEDANCE_MINS = (0.9,)
CONTACT_IMPEDANCE_WIDTHS_M = (0.001,)
TOOL_DRIVE_SCALES = (0.4, 0.6, 0.8, 1.0)


def main() -> int:
    references = {
        motion.motion_id: json.loads(
            (REFERENCE_ROOT / f"PHYSX_{motion.motion_id}.json").read_text(encoding="utf-8")
        )
        for motion in MOTIONS
    }
    spec = factor_separated_scenario("B0", 8.0)
    candidates = []
    for delay in ACTUATION_DELAYS:
        for contact_time in CONTACT_TIME_CONSTANTS_S:
            for damping in CONTACT_DAMPING_RATIOS:
                for margin in CONTACT_MARGINS_M:
                    for impedance_min in CONTACT_IMPEDANCE_MINS:
                        for impedance_width in CONTACT_IMPEDANCE_WIDTHS_M:
                            for tool_scale in TOOL_DRIVE_SCALES:
                                contact = MuJoCoContactConfig(
                                    actuation_gain=1.0,
                                    actuation_time_constant_s=0.08,
                                    actuation_delay_steps=delay,
                                    contact_time_constant_s=contact_time,
                                    contact_damping_ratio=damping,
                                    contact_margin_m=margin,
                                    contact_impedance_min=impedance_min,
                                    contact_impedance_width_m=impedance_width,
                                    tool_drive_scale=tool_scale,
                                    solver_iterations=50,
                                )
                                per_motion = []
                                for motion in MOTIONS:
                                    env = MuJoCoDirectFirstPassEnv(
                                        scenario_spec=spec,
                                        contact_config=contact,
                                    )
                                    try:
                                        try:
                                            trace = run_motion(env, motion.actions(), engine="MuJoCo")
                                            metrics = comparison_metrics(
                                                references[motion.motion_id], trace
                                            )
                                        except RuntimeError as exc:
                                            metrics = invalid_motion_metrics(
                                                motion.motion_id, "development", str(exc)
                                            )
                                    finally:
                                        env.close()
                                    per_motion.append({
                                        "motion_id": motion.motion_id,
                                        "role": "development",
                                        **metrics,
                                    })
                                aggregate = aggregate_motion_metrics(per_motion)
                                candidates.append({
                                    "contact_config": asdict(contact),
                                    "fit": aggregate,
                                    "per_motion": per_motion,
                                })
    candidates.sort(key=lambda row: (
        row["fit"]["maximum_tolerance_ratio"],
        row["fit"]["mean_tolerance_ratio"],
        row["contact_config"]["actuation_delay_steps"],
        row["contact_config"]["contact_time_constant_s"],
        row["contact_config"]["contact_damping_ratio"],
        row["contact_config"]["contact_margin_m"],
        row["contact_config"]["contact_impedance_min"],
        row["contact_config"]["contact_impedance_width_m"],
        row["contact_config"]["tool_drive_scale"],
    ))
    payload = {
        "format": "forcewipe_v19_crosssim_r3i_drive_fit_development_v1",
        "policy_checkpoint_used": False,
        "motions_are_previously_exposed": True,
        "candidate_cells": len(candidates),
        "selected": candidates[0],
        "top_candidates": candidates[:20],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "candidate_cells": len(candidates),
        "selected": candidates[0],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
