"""Non-cylindrical geometry/plant/tool/disturbance OOD screen for V16.7."""

from __future__ import annotations

import numpy as np

from forcewipe.learning.qualification_scenarios import TARGETS_N, TRAINING_SEEDS


PATHS = ("diagonal", "arc", "s_curve", "polyline")
SURFACES = ("flat", "incline_5", "incline_10")
TOOLS = ("wide_soft", "narrow_soft", "wide_hard")
DISTURBANCES = ("none", "lateral_impulse", "normal_impulse", "reference_noise", "sensor_noise_latency")
RESIDUALS = ("single_blob", "islands", "edge_dense", "stripe", "multimodal", "gaussian_random_field")


def _index(training_seed: int, target_force_n: float) -> int:
    if int(training_seed) not in TRAINING_SEEDS or float(target_force_n) not in TARGETS_N:
        raise ValueError("unknown V16 OOD training-seed/target pair")
    return (int(training_seed) - 141) * 3 + TARGETS_N.index(float(target_force_n))


def ood_identity(training_seed: int, target_force_n: float) -> tuple[int, int]:
    index = _index(training_seed, target_force_n)
    return 88_100_000 + index, 8_168_000 + index


def v16_ood_scenario(target_force_n: float, scenario_seed: int, scenario_id: int):
    from forcewipe.simulation.scenarios import ScenarioSpec

    # The identity is sufficient to recover the frozen factor roster.
    index = int(scenario_id) - 8_168_000
    if not 0 <= index < 15 or float(target_force_n) not in TARGETS_N:
        raise ValueError("invalid V16 OOD scenario identity")
    rng = np.random.default_rng(int(scenario_seed))
    path = PATHS[index % len(PATHS)]
    surface = SURFACES[(index // 2) % len(SURFACES)]
    tool = TOOLS[(index // 3) % len(TOOLS)]
    disturbance = DISTURBANCES[index % len(DISTURBANCES)]
    residual = RESIDUALS[index % len(RESIDUALS)]
    if tool == "wide_soft":
        width, footprint, tool_stiffness = 0.050, 0.030, 650.0
    elif tool == "narrow_soft":
        width, footprint, tool_stiffness = 0.027, 0.019, 650.0
    else:
        width, footprint, tool_stiffness = 0.050, 0.030, 1450.0
    spec = ScenarioSpec(
        scenario_id=int(scenario_id),
        role="DEV",
        seed_namespace="v5_dev_v16_direct_tdmpc2_ood_20260828",
        scenario_seed=int(scenario_seed),
        path_kind=path,
        path_length_m=float(rng.uniform(0.14, 0.18)),
        path_lateral_scale_m=float(rng.uniform(0.012, 0.022)),
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
        residual_seed=int(scenario_seed) + 2_000_000,
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
