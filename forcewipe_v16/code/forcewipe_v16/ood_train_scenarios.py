"""Balanced TRAIN-only domain randomization for V16.8."""

from __future__ import annotations

import numpy as np


TARGETS_N = (5.0, 8.0, 12.0)
PATHS = ("line", "diagonal", "arc", "s_curve", "polyline")
SURFACES = ("flat", "incline_5", "incline_10")
TOOLS = ("wide_soft", "narrow_soft", "wide_hard")
DISTURBANCES = ("none", "lateral_impulse", "normal_impulse", "reference_noise", "sensor_noise_latency")
RESIDUALS = ("single_blob", "islands", "edge_dense", "stripe", "multimodal", "gaussian_random_field")


def train_identity(target_force_n: float, replicate: int) -> tuple[int, int, int]:
    if float(target_force_n) not in TARGETS_N or not 0 <= int(replicate) < 15:
        raise ValueError("V16.8 TRAIN identity is outside the frozen 3 x 15 roster")
    ordinal = TARGETS_N.index(float(target_force_n)) * 15 + int(replicate)
    return ordinal, 88_200_000 + ordinal, 8_169_000 + ordinal


def v16p8_ood_training_scenario(target_force_n: float, scenario_seed: int, scenario_id: int):
    from forcewipe_v4.scenarios import ScenarioSpec

    ordinal = int(scenario_id) - 8_169_000
    if not 0 <= ordinal < 45:
        raise ValueError("invalid V16.8 TRAIN scenario identity")
    target_index, replicate = divmod(ordinal, 15)
    if float(target_force_n) != TARGETS_N[target_index]:
        raise ValueError("V16.8 TRAIN target does not match scenario identity")
    rng = np.random.default_rng(int(scenario_seed))
    path = PATHS[(replicate + 2 * target_index) % len(PATHS)]
    surface = SURFACES[(replicate + target_index) % len(SURFACES)]
    tool = TOOLS[(replicate + 2 * target_index) % len(TOOLS)]
    disturbance = DISTURBANCES[(replicate + 3 * target_index) % len(DISTURBANCES)]
    residual = RESIDUALS[(replicate + target_index) % len(RESIDUALS)]
    if tool == "wide_soft":
        width, footprint, tool_stiffness = 0.050, 0.030, 650.0
    elif tool == "narrow_soft":
        width, footprint, tool_stiffness = 0.027, 0.019, 650.0
    else:
        width, footprint, tool_stiffness = 0.050, 0.030, 1450.0
    spec = ScenarioSpec(
        scenario_id=int(scenario_id),
        role="TRAIN",
        seed_namespace="v5_train_v16p8_ood_aware_20260828",
        scenario_seed=int(scenario_seed),
        path_kind=path,
        path_length_m=float(rng.uniform(0.14, 0.18)),
        path_lateral_scale_m=0.0 if path == "line" else float(rng.uniform(0.012, 0.022)),
        surface_kind=surface,
        cylinder_radius_m=0.5,
        tool_kind=tool,
        tool_width_m=float(width * rng.uniform(0.96, 1.04)),
        tool_footprint_length_m=float(footprint * rng.uniform(0.96, 1.04)),
        tool_normal_stiffness_n_m=float(tool_stiffness * rng.uniform(0.94, 1.06)),
        friction_coefficient=float(rng.uniform(0.30, 0.70)),
        support_stiffness_n_m=float(rng.uniform(1800.0, 3200.0)),
        effective_mass_kg=float(rng.uniform(0.25, 0.45)),
        damping_ratio=float(rng.uniform(0.55, 1.05)),
        restitution=float(rng.uniform(0.03, 0.12)),
        residual_family=residual,
        residual_seed=int(scenario_seed) + 3_000_000,
        residual_severity=float(rng.uniform(0.55, 0.95)),
        disturbance_kind=disturbance,
        disturbance_scale=0.0 if disturbance == "none" else float(rng.uniform(0.15, 0.40)),
        sensor_latency_steps=2 if disturbance == "sensor_noise_latency" else 0,
        constraint_kind="none",
        obstacle_center_s=0.5,
        obstacle_half_width_s=0.0,
        target_force_n=float(target_force_n),
        residual_cleanability=float(rng.uniform(0.65, 1.0)),
    )
    spec.validate()
    return spec


def frozen_roster() -> list[dict[str, object]]:
    roster = []
    for target in TARGETS_N:
        for replicate in range(15):
            ordinal, scenario_seed, scenario_id = train_identity(target, replicate)
            roster.append({
                "episode_id": ordinal,
                "target_force_n": target,
                "replicate": replicate,
                "scenario_seed": scenario_seed,
                "scenario_id": scenario_id,
                "action_noise_sigma": 0.0 if replicate < 8 else 0.005,
            })
    return roster
