"""Fresh, non-overlapping nominal qualification scenarios for V16.7."""

from __future__ import annotations

import numpy as np


TRAINING_SEEDS = (141, 142, 143, 144, 145)
TARGETS_N = (5.0, 8.0, 12.0)


def qualification_identity(training_seed: int, target_force_n: float) -> tuple[int, int]:
    if int(training_seed) not in TRAINING_SEEDS or float(target_force_n) not in TARGETS_N:
        raise ValueError("unknown V16 qualification training-seed/target pair")
    target_code = {5.0: 5, 8.0: 8, 12.0: 12}[float(target_force_n)]
    offset = (int(training_seed) - 141) * 100 + target_code
    return 87_100_000 + offset, 8_167_000 + offset


def v16_qualification_scenario(target_force_n: float, scenario_seed: int, scenario_id: int):
    from forcewipe_v4.scenarios import ScenarioSpec

    if float(target_force_n) not in TARGETS_N:
        raise ValueError("V16 qualification target must be 5, 8, or 12 N")
    rng = np.random.default_rng(int(scenario_seed))
    spec = ScenarioSpec(
        scenario_id=int(scenario_id),
        role="DEV",
        seed_namespace="v5_dev_v16_direct_tdmpc2_qualification_20260828",
        scenario_seed=int(scenario_seed),
        path_kind="line",
        path_length_m=float(rng.uniform(0.150, 0.170)),
        path_lateral_scale_m=0.0,
        surface_kind="flat",
        cylinder_radius_m=0.5,
        tool_kind="wide_soft",
        tool_width_m=float(rng.uniform(0.044, 0.053)),
        tool_footprint_length_m=float(rng.uniform(0.025, 0.032)),
        tool_normal_stiffness_n_m=float(rng.uniform(600.0, 700.0)),
        friction_coefficient=float(rng.uniform(0.42, 0.58)),
        support_stiffness_n_m=float(rng.uniform(2300.0, 2700.0)),
        effective_mass_kg=float(rng.uniform(0.32, 0.38)),
        damping_ratio=float(rng.uniform(0.72, 0.88)),
        restitution=float(rng.uniform(0.030, 0.070)),
        residual_family="single_blob",
        residual_seed=int(scenario_seed) + 1_000_000,
        residual_severity=float(rng.uniform(0.50, 0.90)),
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
