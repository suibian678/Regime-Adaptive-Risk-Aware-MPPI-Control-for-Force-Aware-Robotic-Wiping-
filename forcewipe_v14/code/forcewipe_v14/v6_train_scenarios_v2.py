"""Fresh V14.1 TRAIN-only scenario namespace."""

from __future__ import annotations

import numpy as np


def v14_v2_training_scenario(target_force_n: float, scenario_seed: int, scenario_id: int):
    from forcewipe_v4.scenarios import ScenarioSpec

    if float(target_force_n) not in {5.0, 8.0, 12.0}:
        raise ValueError("V14.1 target must be 5, 8, or 12 N")
    rng = np.random.default_rng(int(scenario_seed))
    spec = ScenarioSpec(
        scenario_id=int(scenario_id),
        role="TRAIN",
        seed_namespace="v5_train_v14p1_v6_direct_closed_loop_20260828",
        scenario_seed=int(scenario_seed),
        path_kind="line",
        path_length_m=float(rng.uniform(0.155, 0.165)),
        path_lateral_scale_m=0.0,
        surface_kind="flat",
        cylinder_radius_m=0.5,
        tool_kind="wide_soft",
        tool_width_m=float(rng.uniform(0.045, 0.052)),
        tool_footprint_length_m=float(rng.uniform(0.026, 0.031)),
        tool_normal_stiffness_n_m=float(rng.uniform(610.0, 690.0)),
        friction_coefficient=float(rng.uniform(0.45, 0.55)),
        support_stiffness_n_m=float(rng.uniform(2350.0, 2650.0)),
        effective_mass_kg=float(rng.uniform(0.33, 0.37)),
        damping_ratio=float(rng.uniform(0.75, 0.85)),
        restitution=float(rng.uniform(0.035, 0.065)),
        residual_family="single_blob",
        residual_seed=int(scenario_seed) + 1_000_000,
        residual_severity=float(rng.uniform(0.55, 0.85)),
        disturbance_kind="none",
        disturbance_scale=0.0,
        sensor_latency_steps=0,
        constraint_kind="none",
        obstacle_center_s=0.5,
        obstacle_half_width_s=0.0,
        target_force_n=float(target_force_n),
        residual_cleanability=1.0,
    )
    spec.validate()
    return spec
