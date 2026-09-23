"""Canonical Q0--Q4 loader and collision-free RNG encoding for V16.28 rev3."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "v16_protocol" / "v16p28_paired_qualification_rev3"
BLOCKS_PATH = BUNDLE / "SCENARIO_BLOCKS_V16P28_R3.json"
TARGETS_N = (5.0, 8.0, 12.0)
SCENARIO_ID_BASE = 16_280_000
MAXIMUM_STEPS = 1200
QUALIFICATION_PLANNER_ROOT = 716_280_000
CAL_PLANNER_ROOT = 716_290_100


class QualificationScenarioError(ValueError):
    pass


def _load_blocks() -> dict[str, dict]:
    payload = json.loads(BLOCKS_PATH.read_text(encoding="utf-8"))
    if payload.get("format") != "forcewipe_v16p28_common_scenario_blocks_r3_v1":
        raise QualificationScenarioError("unexpected V16.28 rev3 block format")
    blocks = payload.get("blocks")
    if not isinstance(blocks, list) or len(blocks) != 5:
        raise QualificationScenarioError("V16.28 rev3 requires exactly five blocks")
    result = {str(block["block_id"]): block for block in blocks}
    if set(result) != {"Q0", "Q1", "Q2", "Q3", "Q4"}:
        raise QualificationScenarioError("V16.28 rev3 block roster is not Q0--Q4")
    return result


def v16p28_qualification_scenario(target_force_n: float, scenario_seed: int, scenario_id: int):
    from forcewipe_v4.scenarios import ScenarioSpec

    target = float(target_force_n)
    if target not in TARGETS_N:
        raise QualificationScenarioError("target must be 5, 8, or 12 N")
    cell = int(scenario_id) - SCENARIO_ID_BASE
    if not 0 <= cell < 15 or cell % 3 != TARGETS_N.index(target):
        raise QualificationScenarioError("scenario id does not match target")
    block = _load_blocks()[f"Q{cell // 3}"]
    if int(scenario_seed) != int(block["scenario_seed"]):
        raise QualificationScenarioError("scenario seed does not match frozen block")
    spec = ScenarioSpec(
        scenario_id=int(scenario_id), role="DEV", seed_namespace=str(block["seed_namespace"]),
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
        obstacle_center_s=0.5, obstacle_half_width_s=0.0, target_force_n=target,
        residual_cleanability=float(block["residual_cleanability"]),
    )
    spec.validate()
    return spec


def canonical_scenario_digest(spec) -> str:
    payload = json.dumps(asdict(spec), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _mixed_radix_seed(*, root: int, block_index: int, target_force_n: float, control_step: int, blocks: int) -> int:
    target = float(target_force_n)
    if target not in TARGETS_N or not 0 <= int(block_index) < int(blocks):
        raise QualificationScenarioError("invalid block/target RNG identity")
    if not 0 <= int(control_step) < MAXIMUM_STEPS:
        raise QualificationScenarioError("control step exceeds frozen 1200-step horizon")
    tuple_ordinal = (int(block_index) * len(TARGETS_N) + TARGETS_N.index(target)) * MAXIMUM_STEPS + int(control_step)
    return int(root) + tuple_ordinal


def planner_step_seed(planner_seed_base: int, target_force_n: float, control_step: int) -> int:
    block_index = int(planner_seed_base) - QUALIFICATION_PLANNER_ROOT
    return _mixed_radix_seed(
        root=QUALIFICATION_PLANNER_ROOT,
        block_index=block_index,
        target_force_n=target_force_n,
        control_step=control_step,
        blocks=5,
    )


def v16p29_planner_step_seed(planner_seed_base: int, target_force_n: float, control_step: int) -> int:
    block_index = int(planner_seed_base) - CAL_PLANNER_ROOT
    return _mixed_radix_seed(
        root=CAL_PLANNER_ROOT,
        block_index=block_index,
        target_force_n=target_force_n,
        control_step=control_step,
        blocks=10,
    )
