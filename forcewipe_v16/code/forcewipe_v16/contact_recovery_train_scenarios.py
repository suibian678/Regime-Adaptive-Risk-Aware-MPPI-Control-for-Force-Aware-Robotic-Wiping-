"""TRAIN-only contact-loss/reacquisition scenarios for direct TD-MPC2."""

from __future__ import annotations

import numpy as np


TARGETS_N = (5.0, 8.0, 12.0)
PATHS = ("line", "diagonal", "arc", "s_curve", "polyline")
SURFACES = ("flat", "incline_5", "incline_10")
TOOLS = ("narrow_soft", "wide_soft", "narrow_soft")
RESIDUALS = ("single_blob", "islands", "edge_dense", "stripe", "multimodal", "gaussian_random_field")
REPLICATES = 12


def recovery_train_identity(target_force_n: float, replicate: int) -> tuple[int, int, int]:
    if float(target_force_n) not in TARGETS_N or not 0 <= int(replicate) < REPLICATES:
        raise ValueError("contact-recovery TRAIN identity is outside the frozen roster")
    ordinal = TARGETS_N.index(float(target_force_n)) * REPLICATES + int(replicate)
    return 45 + ordinal, 88_300_000 + ordinal, 8_170_000 + ordinal


def contact_recovery_training_scenario(target_force_n: float, scenario_seed: int, scenario_id: int):
    from forcewipe_v4.scenarios import ScenarioSpec

    ordinal = int(scenario_id) - 8_170_000
    if not 0 <= ordinal < len(TARGETS_N) * REPLICATES:
        raise ValueError("invalid contact-recovery TRAIN scenario identity")
    target_index, replicate = divmod(ordinal, REPLICATES)
    if float(target_force_n) != TARGETS_N[target_index]:
        raise ValueError("contact-recovery TRAIN target does not match identity")
    rng = np.random.default_rng(int(scenario_seed))
    path = PATHS[(replicate + target_index) % len(PATHS)]
    surface = SURFACES[(replicate + 2 * target_index) % len(SURFACES)]
    tool = TOOLS[(replicate + target_index) % len(TOOLS)]
    if tool == "narrow_soft":
        width, footprint = 0.027, 0.019
    else:
        width, footprint = 0.050, 0.030
    spec = ScenarioSpec(
        scenario_id=int(scenario_id),
        role="TRAIN",
        seed_namespace="v5_train_v16_contact_recovery_20260828",
        scenario_seed=int(scenario_seed),
        path_kind=path,
        path_length_m=float(rng.uniform(0.14, 0.18)),
        path_lateral_scale_m=0.0 if path == "line" else float(rng.uniform(0.012, 0.022)),
        surface_kind=surface,
        cylinder_radius_m=0.5,
        tool_kind=tool,
        tool_width_m=float(width * rng.uniform(0.96, 1.04)),
        tool_footprint_length_m=float(footprint * rng.uniform(0.96, 1.04)),
        tool_normal_stiffness_n_m=float(rng.uniform(560.0, 760.0)),
        friction_coefficient=float(rng.uniform(0.30, 0.62)),
        support_stiffness_n_m=float(rng.uniform(1800.0, 2600.0)),
        effective_mass_kg=float(rng.uniform(0.34, 0.48)),
        damping_ratio=float(rng.uniform(0.55, 0.88)),
        restitution=float(rng.uniform(0.03, 0.10)),
        residual_family=RESIDUALS[(replicate + target_index) % len(RESIDUALS)],
        residual_seed=int(scenario_seed) + 4_000_000,
        residual_severity=float(rng.uniform(0.60, 0.95)),
        disturbance_kind="normal_impulse",
        disturbance_scale=float(rng.uniform(0.30, 0.45)),
        sensor_latency_steps=0,
        constraint_kind="none",
        obstacle_center_s=0.5,
        obstacle_half_width_s=0.0,
        target_force_n=float(target_force_n),
        residual_cleanability=float(rng.uniform(0.68, 1.0)),
    )
    spec.validate()
    return spec


def frozen_recovery_roster() -> list[dict[str, object]]:
    rows = []
    for target in TARGETS_N:
        for replicate in range(REPLICATES):
            episode_id, scenario_seed, scenario_id = recovery_train_identity(target, replicate)
            rows.append({
                "episode_id": episode_id,
                "target_force_n": target,
                "replicate": replicate,
                "scenario_seed": scenario_seed,
                "scenario_id": scenario_id,
                "action_noise_sigma": 0.0,
            })
    return rows

