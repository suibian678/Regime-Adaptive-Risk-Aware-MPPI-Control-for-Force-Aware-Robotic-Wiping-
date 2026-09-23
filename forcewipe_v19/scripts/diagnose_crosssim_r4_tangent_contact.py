#!/usr/bin/env python3
"""Diagnose exposed PhysX/MuJoCo tangential-contact state trajectories.

This script is development-only.  It executes no policy checkpoint and uses
only the already exposed bidirectional calibration motion.
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
    TEACHER_ROOT / "forcewipe_v6" / "code",
    TEACHER_ROOT / "forcewipe_v4" / "code",
    ROOT / "release" / "runtime_source",
):
    sys.path.insert(0, str(source))

import forcewipe_v6.tdmpc2_direct_sapien_env as sapien_module  # noqa: E402
from forcewipe_v19.crosssim_calibration_r3 import HELD_OUT_MOTIONS  # noqa: E402
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


OUTPUT = ROOT / "results/analysis/crosssim_r4_tangent_contact_diagnostic.json"
SAMPLES = (239, 249, 259, 269, 279, 289, 299, 309, 319, 329, 339)


def _vec(value) -> list[float]:
    return [float(item) for item in np.asarray(value).reshape(-1)[:3]]


def _run_physx(actions: np.ndarray) -> list[dict]:
    sapien_module.nominal_direct_scenario = v19_factor_separated_scenario
    env = sapien_module.V6DirectFirstPassEnv(
        target_force_n=8.0,
        scenario_seed=SCENARIO_SEED,
        scenario_id=SCENARIO_ID_BASE + 1,
    )
    rows = []
    try:
        env.reset(seed=SCENARIO_SEED)
        for step, action in enumerate(actions):
            _, _, terminated, truncated, info = env.step(action)
            raw = env._raw
            rows.append({
                "step": step,
                "normal_force_n": float(info["normal_force_n"]),
                "tool_world_m": _vec(raw.v4_tool.pose.p),
                "surface_world_m": _vec(raw.wipe_pad.pose.p),
                "tcp_world_m": _vec(raw.agent.tcp.pose.p),
            })
            if terminated or truncated:
                raise RuntimeError(f"PhysX stopped at {step}")
    finally:
        env.close()
    return rows


def _run_mujoco(actions: np.ndarray) -> list[dict]:
    config = MuJoCoContactConfig(
        actuation_delay_steps=2,
        actuation_time_constant_s=0.08,
        contact_time_constant_s=0.03,
        contact_damping_ratio=0.5,
        contact_margin_m=0.0035,
        tool_drive_scale=0.8,
        tool_tangential_drive_scale=0.3,
        friction_scale=1.0,
    )
    env = MuJoCoDirectFirstPassEnv(
        scenario_spec=factor_separated_scenario("B0", 8.0),
        contact_config=config,
    )
    rows = []
    try:
        env.reset(seed=SCENARIO_SEED)
        for step, action in enumerate(actions):
            _, _, terminated, truncated, info = env.step(action)
            rows.append({
                "step": step,
                "normal_force_n": float(info["normal_force_n"]),
                "tool_world_m": list(info["tool_position_world_m"]),
                "surface_world_m": list(info["surface_position_world_m"]),
                "tcp_world_m": list(info["tcp_target_world_m"]),
            })
            if terminated or truncated:
                raise RuntimeError(f"MuJoCo stopped at {step}")
    finally:
        env.close()
    return rows


def _summarise(rows: list[dict]) -> dict:
    tool = np.asarray([row["tool_world_m"] for row in rows], dtype=float)
    surface = np.asarray([row["surface_world_m"] for row in rows], dtype=float)
    tcp = np.asarray([row["tcp_world_m"] for row in rows], dtype=float)
    return {
        "selected_samples": [rows[index] for index in SAMPLES],
        "tool_x_range_m": [float(tool[:, 0].min()), float(tool[:, 0].max())],
        "surface_x_range_m": [float(surface[:, 0].min()), float(surface[:, 0].max())],
        "relative_x_range_m": [
            float((tool[:, 0] - surface[:, 0]).min()),
            float((tool[:, 0] - surface[:, 0]).max()),
        ],
        "tcp_x_range_m": [float(tcp[:, 0].min()), float(tcp[:, 0].max())],
    }


def main() -> int:
    motion = next(
        item for item in HELD_OUT_MOTIONS
        if item.motion_id == "validation_bidirectional_tangent_release"
    )
    physx_rows = _run_physx(motion.actions())
    mujoco_rows = _run_mujoco(motion.actions())
    result = {
        "format": "forcewipe_v19_crosssim_r4_tangent_contact_diagnostic_v1",
        "policy_checkpoint_used": False,
        "data_role": "exposed_development_only",
        "physx": _summarise(physx_rows),
        "mujoco": _summarise(mujoco_rows),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
