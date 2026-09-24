"""Disjoint TRAIN/DEV scenario families for the V16.30 direct-PPO baseline."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math

import numpy as np

from forcewipe.simulation.scenarios import ScenarioSpec


TARGETS_N = (5.0, 8.0, 12.0)
TRAIN_BLOCKS = 30
DEV_BLOCKS = 6
TRAIN_ID_BASE = 16_310_000
DEV_ID_BASE = 16_320_000
TRAIN_MASTER_SEED = 616_310_001
DEV_MASTER_SEED = 616_320_001


class PPOScenarioError(ValueError):
    pass


def _blocks(*, role: str) -> tuple[dict, ...]:
    role = str(role).upper()
    if role == "TRAIN":
        count, master, scenario_root, residual_root = (
            TRAIN_BLOCKS, TRAIN_MASTER_SEED, 917_310_000, 918_310_000
        )
    elif role == "DEV":
        count, master, scenario_root, residual_root = (
            DEV_BLOCKS, DEV_MASTER_SEED, 917_320_000, 918_320_000
        )
    else:
        raise PPOScenarioError("PPO scenario role must be TRAIN or DEV")
    rng = np.random.default_rng(master)
    paths = ("line", "diagonal", "arc", "s_curve", "polyline")
    surfaces = ("flat", "incline_5", "incline_10")
    tools = (
        ("narrow_soft", 0.027, 0.019, 650.0),
        ("wide_soft", 0.050, 0.030, 650.0),
        ("wide_hard", 0.050, 0.030, 1450.0),
    )
    disturbances = ("none", "lateral_impulse", "normal_impulse", "reference_noise", "sensor_noise_latency")
    residuals = ("single_blob", "islands", "edge_dense", "stripe", "multimodal", "gaussian_random_field")
    output = []
    for index in range(count):
        tool = tools[index % len(tools)]
        disturbance = disturbances[index % len(disturbances)]
        output.append(dict(
            block_id=f"{role[0]}{index:02d}", block_index=index,
            scenario_seed=scenario_root + index, residual_seed=residual_root + index,
            path_kind=paths[index % len(paths)],
            path_length_m=float(rng.uniform(0.13, 0.19)),
            path_lateral_scale_m=(0.0 if paths[index % len(paths)] == "line" else float(rng.uniform(0.009, 0.018))),
            surface_kind=surfaces[index % len(surfaces)], cylinder_radius_m=0.5,
            tool_kind=tool[0], tool_width_m=tool[1], tool_footprint_length_m=tool[2],
            tool_normal_stiffness_n_m=tool[3], friction_coefficient=float(rng.uniform(0.30, 0.72)),
            support_stiffness_n_m=float(rng.uniform(1900.0, 3100.0)),
            effective_mass_kg=float(rng.uniform(0.26, 0.48)),
            damping_ratio=float(rng.uniform(0.65, 1.00)), restitution=float(rng.uniform(0.03, 0.09)),
            residual_family=residuals[index % len(residuals)], residual_severity=float(rng.uniform(0.65, 0.95)),
            disturbance_kind=disturbance,
            disturbance_scale=(0.0 if disturbance == "none" else float(rng.uniform(0.10, 0.22))),
            sensor_latency_steps=(2 if disturbance == "sensor_noise_latency" else 0),
        ))
    return tuple(output)


TRAIN_SCENARIO_BLOCKS = _blocks(role="TRAIN")
DEV_SCENARIO_BLOCKS = _blocks(role="DEV")


def _role_identity(scenario_id: int) -> tuple[str, int, int]:
    value = int(scenario_id)
    if TRAIN_ID_BASE <= value < TRAIN_ID_BASE + TRAIN_BLOCKS * len(TARGETS_N):
        cell = value - TRAIN_ID_BASE
        return "TRAIN", cell // len(TARGETS_N), cell % len(TARGETS_N)
    if DEV_ID_BASE <= value < DEV_ID_BASE + DEV_BLOCKS * len(TARGETS_N):
        cell = value - DEV_ID_BASE
        return "DEV", cell // len(TARGETS_N), cell % len(TARGETS_N)
    raise PPOScenarioError("scenario id is outside the PPO TRAIN/DEV families")


def scenario_identity(*, role: str, block_index: int, target_force_n: float) -> tuple[int, int]:
    role = str(role).upper()
    target = float(target_force_n)
    if target not in TARGETS_N:
        raise PPOScenarioError("target must be 5, 8, or 12 N")
    blocks = TRAIN_SCENARIO_BLOCKS if role == "TRAIN" else DEV_SCENARIO_BLOCKS if role == "DEV" else None
    if blocks is None or not 0 <= int(block_index) < len(blocks):
        raise PPOScenarioError("invalid role/block")
    root = TRAIN_ID_BASE if role == "TRAIN" else DEV_ID_BASE
    scenario_id = root + int(block_index) * len(TARGETS_N) + TARGETS_N.index(target)
    return scenario_id, int(blocks[int(block_index)]["scenario_seed"])


def v16p30_ppo_scenario(target_force_n: float, scenario_seed: int, scenario_id: int) -> ScenarioSpec:
    role, block_index, target_index = _role_identity(scenario_id)
    target = float(target_force_n)
    if TARGETS_N[target_index] != target:
        raise PPOScenarioError("scenario id does not match target")
    blocks = TRAIN_SCENARIO_BLOCKS if role == "TRAIN" else DEV_SCENARIO_BLOCKS
    item = blocks[block_index]
    if int(scenario_seed) != int(item["scenario_seed"]):
        raise PPOScenarioError("scenario seed does not match PPO block")
    spec = ScenarioSpec(
        scenario_id=int(scenario_id), role=role,
        seed_namespace=f"v5_{role.lower()}_v16p30_direct_ppo_20260830",
        scenario_seed=int(scenario_seed), path_kind=item["path_kind"],
        path_length_m=item["path_length_m"], path_lateral_scale_m=item["path_lateral_scale_m"],
        surface_kind=item["surface_kind"], cylinder_radius_m=item["cylinder_radius_m"],
        tool_kind=item["tool_kind"], tool_width_m=item["tool_width_m"],
        tool_footprint_length_m=item["tool_footprint_length_m"],
        tool_normal_stiffness_n_m=item["tool_normal_stiffness_n_m"],
        friction_coefficient=item["friction_coefficient"],
        support_stiffness_n_m=item["support_stiffness_n_m"],
        effective_mass_kg=item["effective_mass_kg"], damping_ratio=item["damping_ratio"],
        restitution=item["restitution"], residual_family=item["residual_family"],
        residual_seed=item["residual_seed"], residual_severity=item["residual_severity"],
        disturbance_kind=item["disturbance_kind"], disturbance_scale=item["disturbance_scale"],
        sensor_latency_steps=item["sensor_latency_steps"], constraint_kind="none",
        obstacle_center_s=0.5, obstacle_half_width_s=0.0, target_force_n=target,
        residual_cleanability=0.85,
    )
    spec.validate()
    return spec


def scenario_digest(spec: ScenarioSpec) -> str:
    payload = json.dumps(asdict(spec), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def training_cell_schedule() -> tuple[tuple[int, float], ...]:
    """One fixed 90-cell cycle, repeated without dependence on policy seed."""

    cells = [(block, target) for block in range(TRAIN_BLOCKS) for target in TARGETS_N]
    rng = np.random.default_rng(616_310_090)
    order = rng.permutation(len(cells))
    return tuple(cells[int(index)] for index in order)
