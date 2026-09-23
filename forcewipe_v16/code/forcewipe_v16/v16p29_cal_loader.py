"""Canonical loader for the frozen V16.29 independent CAL blocks."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from forcewipe_v16.v16p28_qualification_loader_rev4 import (
    CAL_PLANNER_ROOT,
    TARGETS_N,
    v16p29_planner_step_seed,
)


ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "v16_protocol" / "v16p29_independent_cal_rev1"
BLOCKS_PATH = BUNDLE / "SCENARIO_BLOCKS_V16P29_R1.json"
SCENARIO_ID_BASE = 16_290_000


class CalScenarioError(ValueError):
    pass


def _load_blocks() -> dict[str, dict]:
    payload = json.loads(BLOCKS_PATH.read_text(encoding="utf-8"))
    if payload.get("format") != "forcewipe_v16p29_independent_cal_blocks_r1_v1":
        raise CalScenarioError("unexpected V16.29 block format")
    blocks = payload.get("blocks")
    if not isinstance(blocks, list) or len(blocks) != 10:
        raise CalScenarioError("V16.29 requires exactly ten CAL blocks")
    result = {str(block["block_id"]): block for block in blocks}
    if set(result) != {f"C{index}" for index in range(10)}:
        raise CalScenarioError("V16.29 block roster is not C0--C9")
    return result


def v16p29_cal_scenario(target_force_n: float, scenario_seed: int, scenario_id: int):
    from forcewipe_v4.scenarios import ScenarioSpec

    target = float(target_force_n)
    if target not in TARGETS_N:
        raise CalScenarioError("target must be 5, 8, or 12 N")
    cell = int(scenario_id) - SCENARIO_ID_BASE
    if not 0 <= cell < 30 or cell % 3 != TARGETS_N.index(target):
        raise CalScenarioError("scenario id does not match target")
    block = _load_blocks()[f"C{cell // 3}"]
    if int(scenario_seed) != int(block["scenario_seed"]):
        raise CalScenarioError("scenario seed does not match frozen CAL block")
    spec = ScenarioSpec(
        scenario_id=int(scenario_id), role="CAL", seed_namespace=str(block["seed_namespace"]),
        scenario_seed=int(scenario_seed), path_kind=str(block["path_kind"]),
        path_length_m=float(block["path_length_m"]), path_lateral_scale_m=float(block["path_lateral_scale_m"]),
        surface_kind=str(block["surface_kind"]), cylinder_radius_m=float(block["cylinder_radius_m"]),
        tool_kind=str(block["tool_kind"]), tool_width_m=float(block["tool_width_m"]),
        tool_footprint_length_m=float(block["tool_footprint_length_m"]),
        tool_normal_stiffness_n_m=float(block["tool_normal_stiffness_n_m"]),
        friction_coefficient=float(block["friction_coefficient"]),
        support_stiffness_n_m=float(block["support_stiffness_n_m"]),
        effective_mass_kg=float(block["effective_mass_kg"]), damping_ratio=float(block["damping_ratio"]),
        restitution=float(block["restitution"]), residual_family=str(block["residual_family"]),
        residual_seed=int(block["residual_seed"]), residual_severity=float(block["residual_severity"]),
        disturbance_kind=str(block["disturbance_kind"]), disturbance_scale=float(block["disturbance_scale"]),
        sensor_latency_steps=int(block["sensor_latency_steps"]), constraint_kind=str(block["constraint_kind"]),
        obstacle_center_s=float(block["obstacle_center_s"]), obstacle_half_width_s=float(block["obstacle_half_width_s"]),
        target_force_n=target, residual_cleanability=float(block["residual_cleanability"]),
    )
    spec.validate()
    return spec


def canonical_scenario_digest(spec) -> str:
    payload = json.dumps(asdict(spec), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def planner_step_seed(planner_seed_base: int, target_force_n: float, control_step: int) -> int:
    return v16p29_planner_step_seed(planner_seed_base, target_force_n, control_step)


__all__ = [
    "CAL_PLANNER_ROOT", "TARGETS_N", "canonical_scenario_digest", "planner_step_seed",
    "v16p29_cal_scenario",
]

