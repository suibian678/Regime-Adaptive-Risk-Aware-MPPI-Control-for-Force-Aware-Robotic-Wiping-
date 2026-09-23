#!/usr/bin/env python3
"""Fit MuJoCo contact parameters on policy-independent motions and validate held out."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TEACHER_ROOT = ROOT.parent
for source in (
    ROOT / "code",
    TEACHER_ROOT / "forcewipe_v6" / "code",
    TEACHER_ROOT / "forcewipe_v4" / "code",
    ROOT / "release" / "runtime_source",
):
    sys.path.insert(0, str(source))

import forcewipe_v6.tdmpc2_direct_sapien_env as sapien_module  # noqa: E402
from forcewipe_v19.crosssim_calibration import (  # noqa: E402
    MOTIONS,
    aggregate_motion_metrics,
    comparison_metrics,
)
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


PROTOCOL = ROOT / "config/CROSS_SIMULATOR_CALIBRATION_PROTOCOL_R2_2026-09-20.json"
OUTPUT = ROOT / "results/calibration/mujoco_crosssim_calibration_r2_v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run_motion(env, actions: np.ndarray, *, engine: str) -> dict:
    observation, reset_info = env.reset(seed=SCENARIO_SEED)
    forces: list[float] = []
    positions: list[list[float]] = []
    for step, action in enumerate(actions):
        observation, reward, terminated, truncated, info = env.step(action)
        if not np.all(np.isfinite(observation)) or not np.isfinite(reward):
            raise RuntimeError(f"{engine} emitted a non-finite value at step {step}")
        forces.append(float(info["normal_force_n"]))
        if engine == "PhysX":
            position = np.asarray(env._raw.v4_tool.pose.p).reshape(-1)
        else:
            position = np.asarray(info["tool_position_world_m"])
        positions.append([float(value) for value in position])
        if terminated or truncated:
            raise RuntimeError(
                f"{engine} calibration motion terminated at step {step + 1}; "
                "the policy-independent schedule is not within its intended range"
            )
    return {
        "engine": engine,
        "reset_info": reset_info,
        "normal_force_n": forces,
        "tool_position_world_m": positions,
    }


def config_grid(protocol: dict):
    grid = protocol["parameter_grid"]
    for gain in grid["actuation_gain"]:
        for time_constant in grid["contact_time_constant_s"]:
            for damping in grid["contact_damping_ratio"]:
                for margin in grid["contact_margin_m"]:
                    for iterations in grid["solver_iterations"]:
                        yield MuJoCoContactConfig(
                            actuation_gain=float(gain),
                            contact_time_constant_s=float(time_constant),
                            contact_damping_ratio=float(damping),
                            contact_margin_m=float(margin),
                            solver_iterations=int(iterations),
                        )


def invalid_motion_metrics(motion_id: str, role: str, reason: str) -> dict:
    """Represent a scientifically unusable grid cell without aborting the sweep."""
    return {
        "motion_id": motion_id,
        "role": role,
        "invalid_candidate": True,
        "invalid_reason": reason,
        "force_nrmse": None,
        "position_rmse_m": None,
        "contact_onset_error_samples": None,
        "peak_relative_error": None,
        "reference_contact_onset_sample": None,
        "candidate_contact_onset_sample": None,
        "reference_peak_force_n": None,
        "candidate_peak_force_n": None,
        "maximum_tolerance_ratio": 1000000000.0,
        "mean_tolerance_ratio": 1000000000.0,
        "within_all_tolerances": False,
    }


def main() -> int:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol["status"] != "FROZEN_BEFORE_POLICY_DEPLOYMENT":
        raise SystemExit("calibration protocol is not frozen")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    spec = factor_separated_scenario("B0", 8.0)
    sapien_module.nominal_direct_scenario = v19_factor_separated_scenario
    physx = sapien_module.V6DirectFirstPassEnv(
        target_force_n=8.0,
        scenario_seed=SCENARIO_SEED,
        scenario_id=SCENARIO_ID_BASE + 1,
    )
    references: dict[str, dict] = {}
    try:
        for motion in MOTIONS:
            trace = run_motion(physx, motion.actions(), engine="PhysX")
            references[motion.motion_id] = trace
            atomic_json(OUTPUT / f"PHYSX_{motion.motion_id}.json", trace)
    finally:
        physx.close()

    candidates = []
    for contact in config_grid(protocol):
        per_motion = []
        for motion in MOTIONS:
            env = MuJoCoDirectFirstPassEnv(
                scenario_spec=spec,
                contact_config=contact,
            )
            try:
                try:
                    trace = run_motion(env, motion.actions(), engine="MuJoCo")
                except RuntimeError as exc:
                    per_motion.append(invalid_motion_metrics(
                        motion.motion_id, motion.role, str(exc)
                    ))
                    continue
            finally:
                env.close()
            per_motion.append({
                "motion_id": motion.motion_id,
                "role": motion.role,
                **comparison_metrics(references[motion.motion_id], trace),
            })
        fit = aggregate_motion_metrics(row for row in per_motion if row["role"] == "fit")
        validation = aggregate_motion_metrics(
            row for row in per_motion if row["role"] == "validation"
        )
        candidates.append({
            "contact_config": asdict(contact),
            "fit": fit,
            "validation": validation,
            "per_motion": per_motion,
        })
    candidates.sort(key=lambda row: (
        row["fit"]["maximum_tolerance_ratio"],
        row["fit"]["mean_tolerance_ratio"],
        row["contact_config"]["contact_time_constant_s"],
        row["contact_config"]["contact_damping_ratio"],
        row["contact_config"]["contact_margin_m"],
        row["contact_config"]["actuation_gain"],
    ))
    selected = candidates[0]
    result = {
        "format": "forcewipe_v19_crosssim_calibration_result_v1",
        "status": (
            "completed_validation_pass"
            if selected["validation"]["within_all_tolerances"]
            else "completed_validation_fail"
        ),
        "completed_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "protocol_sha256": sha256(PROTOCOL),
        "policy_checkpoint_used": False,
        "candidate_cells": len(candidates),
        "selected": selected,
        "all_candidates": candidates,
    }
    atomic_json(OUTPUT / "RESULT.json", result)
    print(json.dumps({
        "status": result["status"],
        "candidate_cells": len(candidates),
        "selected": selected,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
