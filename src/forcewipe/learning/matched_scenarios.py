"""Frozen six-block weak-curvature roster for the V16.30 final study.

These blocks are evaluation-only.  They must never be imported by a training
or tuning runner.  Each block is crossed with all three force targets and all
four methods by the eventual V16.30 execution manifest.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json

from forcewipe.simulation.scenarios import ScenarioSpec


TARGETS_N = (5.0, 8.0, 12.0)
SCENARIO_ID_BASE = 16_300_000
SEED_NAMESPACE = "v5_cal_v16p30_final_matched_20260830"


# Exactly three tool profiles are used, twice each.  Cylinder radii of
# 1.2--1.8 m yield only weak static curvature over the 0.13--0.205 m paths.
FINAL_BLOCKS = (
    dict(block_id="F0", path_kind="arc", path_length_m=0.130,
         path_lateral_scale_m=0.010, cylinder_radius_m=1.20,
         tool_kind="narrow_soft", tool_width_m=0.027,
         tool_footprint_length_m=0.019, tool_normal_stiffness_n_m=650.0,
         friction_coefficient=0.34, support_stiffness_n_m=2050.0,
         effective_mass_kg=0.29, damping_ratio=0.72, restitution=0.035,
         residual_family="single_blob", residual_seed=918_300_100,
         residual_severity=0.68, disturbance_kind="none",
         disturbance_scale=0.0, sensor_latency_steps=0,
         scenario_seed=917_300_100),
    dict(block_id="F1", path_kind="s_curve", path_length_m=0.145,
         path_lateral_scale_m=0.012, cylinder_radius_m=1.40,
         tool_kind="wide_soft", tool_width_m=0.050,
         tool_footprint_length_m=0.030, tool_normal_stiffness_n_m=650.0,
         friction_coefficient=0.43, support_stiffness_n_m=2380.0,
         effective_mass_kg=0.33, damping_ratio=0.80, restitution=0.045,
         residual_family="islands", residual_seed=918_300_101,
         residual_severity=0.74, disturbance_kind="lateral_impulse",
         disturbance_scale=0.18, sensor_latency_steps=0,
         scenario_seed=917_300_101),
    dict(block_id="F2", path_kind="polyline", path_length_m=0.160,
         path_lateral_scale_m=0.014, cylinder_radius_m=1.60,
         tool_kind="wide_hard", tool_width_m=0.050,
         tool_footprint_length_m=0.030, tool_normal_stiffness_n_m=1450.0,
         friction_coefficient=0.52, support_stiffness_n_m=2710.0,
         effective_mass_kg=0.37, damping_ratio=0.88, restitution=0.055,
         residual_family="edge_dense", residual_seed=918_300_102,
         residual_severity=0.80, disturbance_kind="normal_impulse",
         disturbance_scale=0.18, sensor_latency_steps=0,
         scenario_seed=917_300_102),
    dict(block_id="F3", path_kind="arc", path_length_m=0.175,
         path_lateral_scale_m=0.016, cylinder_radius_m=1.80,
         tool_kind="narrow_soft", tool_width_m=0.027,
         tool_footprint_length_m=0.019, tool_normal_stiffness_n_m=650.0,
         friction_coefficient=0.61, support_stiffness_n_m=3040.0,
         effective_mass_kg=0.41, damping_ratio=0.96, restitution=0.065,
         residual_family="stripe", residual_seed=918_300_103,
         residual_severity=0.86, disturbance_kind="reference_noise",
         disturbance_scale=0.16, sensor_latency_steps=0,
         scenario_seed=917_300_103),
    dict(block_id="F4", path_kind="s_curve", path_length_m=0.190,
         path_lateral_scale_m=0.018, cylinder_radius_m=1.30,
         tool_kind="wide_soft", tool_width_m=0.050,
         tool_footprint_length_m=0.030, tool_normal_stiffness_n_m=650.0,
         friction_coefficient=0.70, support_stiffness_n_m=2220.0,
         effective_mass_kg=0.45, damping_ratio=0.68, restitution=0.075,
         residual_family="multimodal", residual_seed=918_300_104,
         residual_severity=0.92, disturbance_kind="sensor_noise_latency",
         disturbance_scale=0.14, sensor_latency_steps=2,
         scenario_seed=917_300_104),
    dict(block_id="F5", path_kind="polyline", path_length_m=0.205,
         path_lateral_scale_m=0.015, cylinder_radius_m=1.50,
         tool_kind="wide_hard", tool_width_m=0.050,
         tool_footprint_length_m=0.030, tool_normal_stiffness_n_m=1450.0,
         friction_coefficient=0.47, support_stiffness_n_m=2890.0,
         effective_mass_kg=0.49, damping_ratio=0.84, restitution=0.085,
         residual_family="gaussian_random_field", residual_seed=918_300_105,
         residual_severity=0.76, disturbance_kind="lateral_impulse",
         disturbance_scale=0.22, sensor_latency_steps=0,
         scenario_seed=917_300_105),
)


class FinalScenarioError(ValueError):
    pass


def block(block_id: str) -> dict:
    matches = [item for item in FINAL_BLOCKS if item["block_id"] == str(block_id)]
    if len(matches) != 1:
        raise FinalScenarioError("unknown V16.30 final block")
    return dict(matches[0])


def v16p30_final_scenario(
    target_force_n: float,
    scenario_seed: int,
    scenario_id: int,
) -> ScenarioSpec:
    target = float(target_force_n)
    if target not in TARGETS_N:
        raise FinalScenarioError("target must be 5, 8, or 12 N")
    cell = int(scenario_id) - SCENARIO_ID_BASE
    if not 0 <= cell < len(FINAL_BLOCKS) * len(TARGETS_N):
        raise FinalScenarioError("scenario id is outside the final roster")
    if cell % len(TARGETS_N) != TARGETS_N.index(target):
        raise FinalScenarioError("scenario id does not match target")
    item = FINAL_BLOCKS[cell // len(TARGETS_N)]
    if int(scenario_seed) != int(item["scenario_seed"]):
        raise FinalScenarioError("scenario seed does not match final block")
    spec = ScenarioSpec(
        scenario_id=int(scenario_id), role="CAL", seed_namespace=SEED_NAMESPACE,
        scenario_seed=int(scenario_seed), path_kind=str(item["path_kind"]),
        path_length_m=float(item["path_length_m"]),
        path_lateral_scale_m=float(item["path_lateral_scale_m"]),
        surface_kind="cylinder_low",
        cylinder_radius_m=float(item["cylinder_radius_m"]),
        tool_kind=str(item["tool_kind"]), tool_width_m=float(item["tool_width_m"]),
        tool_footprint_length_m=float(item["tool_footprint_length_m"]),
        tool_normal_stiffness_n_m=float(item["tool_normal_stiffness_n_m"]),
        friction_coefficient=float(item["friction_coefficient"]),
        support_stiffness_n_m=float(item["support_stiffness_n_m"]),
        effective_mass_kg=float(item["effective_mass_kg"]),
        damping_ratio=float(item["damping_ratio"]), restitution=float(item["restitution"]),
        residual_family=str(item["residual_family"]), residual_seed=int(item["residual_seed"]),
        residual_severity=float(item["residual_severity"]),
        disturbance_kind=str(item["disturbance_kind"]),
        disturbance_scale=float(item["disturbance_scale"]),
        sensor_latency_steps=int(item["sensor_latency_steps"]),
        constraint_kind="none", obstacle_center_s=0.5, obstacle_half_width_s=0.0,
        target_force_n=target, residual_cleanability=0.85,
    )
    spec.validate()
    return spec


def scenario_digest(spec: ScenarioSpec) -> str:
    payload = json.dumps(
        asdict(spec), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def final_roster() -> tuple[ScenarioSpec, ...]:
    rows = []
    for block_index, item in enumerate(FINAL_BLOCKS):
        for target_index, target in enumerate(TARGETS_N):
            rows.append(v16p30_final_scenario(
                target, int(item["scenario_seed"]),
                SCENARIO_ID_BASE + block_index * len(TARGETS_N) + target_index,
            ))
    return tuple(rows)
