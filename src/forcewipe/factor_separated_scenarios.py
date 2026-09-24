"""Fresh one-factor-at-a-time evaluation blocks for the V19 method study."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json

from forcewipe.simulation.scenarios import ScenarioSpec


TARGETS_N = (5.0, 8.0, 12.0)
SCENARIO_ID_BASE = 19_900_000
SCENARIO_SEED = 919_900_000
RESIDUAL_SEED = 919_901_000
SEED_NAMESPACE = "v5_cal_v19_factor_separated_20260920"


REFERENCE = {
    "path_kind": "line",
    "path_length_m": 0.160,
    "path_lateral_scale_m": 0.0,
    "surface_kind": "flat",
    "cylinder_radius_m": 1.5,
    "tool_kind": "wide_soft",
    "tool_width_m": 0.050,
    "tool_footprint_length_m": 0.030,
    "tool_normal_stiffness_n_m": 650.0,
    "friction_coefficient": 0.50,
    "support_stiffness_n_m": 2500.0,
    "effective_mass_kg": 0.39,
    "damping_ratio": 0.82,
    "restitution": 0.05,
    "residual_family": "single_blob",
    "residual_seed": RESIDUAL_SEED,
    "residual_severity": 0.80,
    "disturbance_kind": "none",
    "disturbance_scale": 0.0,
    "sensor_latency_steps": 0,
}


BLOCKS = (
    {"block_id": "B0", "factor": "reference", "level": "reference", "updates": {}},
    {"block_id": "S1", "factor": "surface", "level": "incline_10deg", "updates": {"surface_kind": "incline_10"}},
    {"block_id": "S2", "factor": "surface", "level": "cylinder_r1.2m", "updates": {"surface_kind": "cylinder_low", "cylinder_radius_m": 1.2}},
    {"block_id": "P1", "factor": "path", "level": "arc", "updates": {"path_kind": "arc", "path_lateral_scale_m": 0.014}},
    {"block_id": "P2", "factor": "path", "level": "s_curve", "updates": {"path_kind": "s_curve", "path_lateral_scale_m": 0.014}},
    {"block_id": "F1", "factor": "friction", "level": "mu_0.30", "updates": {"friction_coefficient": 0.30}},
    {"block_id": "F2", "factor": "friction", "level": "mu_0.70", "updates": {"friction_coefficient": 0.70}},
    {"block_id": "K1", "factor": "support_stiffness", "level": "1600_N_per_m", "updates": {"support_stiffness_n_m": 1600.0}},
    {"block_id": "K2", "factor": "support_stiffness", "level": "3400_N_per_m", "updates": {"support_stiffness_n_m": 3400.0}},
)


def block_definition(block_id: str) -> dict:
    matches = [row for row in BLOCKS if row["block_id"] == str(block_id)]
    if len(matches) != 1:
        raise ValueError("unknown factor-separated block")
    return dict(matches[0])


def factor_separated_scenario(block_id: str, target_force_n: float) -> ScenarioSpec:
    target = float(target_force_n)
    if target not in TARGETS_N:
        raise ValueError("target must be 5, 8, or 12 N")
    block = block_definition(block_id)
    block_index = [row["block_id"] for row in BLOCKS].index(block_id)
    values = dict(REFERENCE)
    values.update(block["updates"])
    spec = ScenarioSpec(
        scenario_id=SCENARIO_ID_BASE + block_index * len(TARGETS_N) + TARGETS_N.index(target),
        role="CAL", seed_namespace=SEED_NAMESPACE, scenario_seed=SCENARIO_SEED,
        target_force_n=target, residual_cleanability=0.85,
        constraint_kind="none", obstacle_center_s=0.5, obstacle_half_width_s=0.0,
        **values,
    )
    spec.validate()
    return spec


def v19_factor_separated_scenario(
    target_force_n: float, scenario_seed: int, scenario_id: int
) -> ScenarioSpec:
    if int(scenario_seed) != SCENARIO_SEED:
        raise ValueError("scenario seed does not match frozen factor-separated roster")
    cell = int(scenario_id) - SCENARIO_ID_BASE
    if not 0 <= cell < len(BLOCKS) * len(TARGETS_N):
        raise ValueError("scenario id is outside the factor-separated roster")
    block = BLOCKS[cell // len(TARGETS_N)]
    target = TARGETS_N[cell % len(TARGETS_N)]
    if float(target_force_n) != target:
        raise ValueError("scenario id does not match target force")
    return factor_separated_scenario(block["block_id"], target)


def scenario_digest(spec: ScenarioSpec) -> str:
    payload = json.dumps(
        asdict(spec), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def changed_fields(block_id: str) -> tuple[str, ...]:
    return tuple(sorted(block_definition(block_id)["updates"]))
