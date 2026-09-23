"""Canonical Q0--Q4 scenario loader for V16.28 qualification rev2."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "v16_protocol" / "v16p28_paired_qualification_rev2"
BLOCKS_PATH = BUNDLE / "SCENARIO_BLOCKS_V16P28_R2.json"
TARGETS_N = (5.0, 8.0, 12.0)
SCENARIO_ID_BASE = 16_280_000


class QualificationScenarioError(ValueError):
    """The requested qualification scenario is not exactly in the frozen roster."""


def _load_blocks() -> dict[str, dict]:
    payload = json.loads(BLOCKS_PATH.read_text(encoding="utf-8"))
    if payload.get("format") != "forcewipe_v16p28_common_scenario_blocks_r2_v1":
        raise QualificationScenarioError("unexpected V16.28 block format")
    blocks = payload.get("blocks")
    if not isinstance(blocks, list) or len(blocks) != 5:
        raise QualificationScenarioError("V16.28 requires exactly five blocks")
    result = {str(block["block_id"]): block for block in blocks}
    if set(result) != {"Q0", "Q1", "Q2", "Q3", "Q4"}:
        raise QualificationScenarioError("V16.28 block roster is not Q0--Q4")
    return result


def scenario_identity(block_id: str, target_force_n: float) -> tuple[int, int]:
    blocks = _load_blocks()
    if block_id not in blocks or float(target_force_n) not in TARGETS_N:
        raise QualificationScenarioError("unknown block/target")
    block = blocks[block_id]
    target_index = TARGETS_N.index(float(target_force_n))
    scenario_id = SCENARIO_ID_BASE + int(block["block_index"]) * 3 + target_index
    return int(block["scenario_seed"]), int(scenario_id)


def v16p28_qualification_scenario(
    target_force_n: float,
    scenario_seed: int,
    scenario_id: int,
):
    from forcewipe_v4.scenarios import ScenarioSpec

    target = float(target_force_n)
    if target not in TARGETS_N:
        raise QualificationScenarioError("target must be 5, 8, or 12 N")
    cell = int(scenario_id) - SCENARIO_ID_BASE
    if not 0 <= cell < 15 or cell % 3 != TARGETS_N.index(target):
        raise QualificationScenarioError("scenario id does not match target")
    block_id = f"Q{cell // 3}"
    block = _load_blocks()[block_id]
    if int(scenario_seed) != int(block["scenario_seed"]):
        raise QualificationScenarioError("scenario seed does not match frozen block")
    spec = ScenarioSpec(
        scenario_id=int(scenario_id),
        role="DEV",
        seed_namespace=str(block["seed_namespace"]),
        scenario_seed=int(scenario_seed),
        path_kind=str(block["path_kind"]),
        path_length_m=float(block["path_length_m"]),
        path_lateral_scale_m=float(block["path_lateral_scale_m"]),
        surface_kind=str(block["surface_kind"]),
        cylinder_radius_m=float(block["cylinder_radius_m"]),
        tool_kind=str(block["tool_kind"]),
        tool_width_m=float(block["tool_width_m"]),
        tool_footprint_length_m=float(block["tool_footprint_length_m"]),
        tool_normal_stiffness_n_m=float(block["tool_normal_stiffness_n_m"]),
        friction_coefficient=float(block["friction_coefficient"]),
        support_stiffness_n_m=float(block["support_stiffness_n_m"]),
        effective_mass_kg=float(block["effective_mass_kg"]),
        damping_ratio=float(block["damping_ratio"]),
        restitution=float(block["restitution"]),
        residual_family=str(block["residual_family"]),
        residual_seed=int(block["residual_seed"]),
        residual_severity=float(block["residual_severity"]),
        disturbance_kind=str(block["disturbance_kind"]),
        disturbance_scale=float(block["disturbance_scale"]),
        sensor_latency_steps=int(block["sensor_latency_steps"]),
        constraint_kind=str(block["constraint_kind"]),
        obstacle_center_s=0.5,
        obstacle_half_width_s=0.0,
        target_force_n=target,
        residual_cleanability=float(block["residual_cleanability"]),
    )
    spec.validate()
    return spec


def canonical_scenario_digest(spec) -> str:
    payload = json.dumps(
        asdict(spec), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def planner_step_seed(planner_seed_base: int, target_force_n: float, control_step: int) -> int:
    """Dedicated per-step random stream, independent of earlier actor/MPPI switches."""

    target = float(target_force_n)
    if target not in TARGETS_N or int(control_step) < 0:
        raise QualificationScenarioError("invalid planner stream identity")
    return int(planner_seed_base) + TARGETS_N.index(target) * 100_000 + int(control_step)
