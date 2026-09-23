"""Frozen, result-independent generator rules for the disabled V16.29 CAL."""

from __future__ import annotations

import numpy as np


MASTER_SEED = 916_290_000
PATHS = ("line", "diagonal", "arc", "s_curve", "polyline")
SURFACES = ("flat", "incline_5", "incline_10", "flat", "incline_5")
TOOLS = ("wide_soft", "narrow_soft", "wide_hard", "wide_soft", "narrow_soft")
RESIDUALS = ("single_blob", "islands", "edge_dense", "multimodal", "gaussian_random_field")
DISTURBANCES = ("none", "lateral_impulse", "normal_impulse", "reference_noise", "sensor_noise_latency")


def _latin_hypercube(count: int, dimensions: int, rng: np.random.Generator) -> np.ndarray:
    result = np.empty((count, dimensions), dtype=np.float64)
    for dimension in range(dimensions):
        strata = (np.arange(count, dtype=np.float64) + rng.random(count)) / count
        result[:, dimension] = strata[rng.permutation(count)]
    return result


def frozen_cal_blocks() -> tuple[dict, ...]:
    """Return ten deterministic blocks without executing or inspecting physics."""

    rng = np.random.default_rng(MASTER_SEED)
    randomized = _latin_hypercube(5, 8, rng)
    blocks = []
    for index in range(10):
        roster = index % 5
        nominal = index < 5
        row = randomized[roster]
        tool = TOOLS[roster]
        width, footprint, stiffness = {
            "wide_soft": (0.050, 0.030, 650.0),
            "narrow_soft": (0.027, 0.019, 650.0),
            "wide_hard": (0.050, 0.030, 1450.0),
        }[tool]
        blocks.append({
            "block_id": f"C{index}",
            "block_index": index,
            "panel": "multipath_nominal_plant" if nominal else "multipath_randomized_plant",
            "scenario_seed": int(916_290_100 + index),
            "residual_seed": int(917_290_100 + index),
            "planner_seed": int(716_290_100 + index),
            "path_kind": PATHS[roster],
            "path_length_m": float(0.14 + 0.04 * (0.5 if nominal else row[0])),
            "path_lateral_scale_m": 0.0 if PATHS[roster] == "line" else float(0.012 + 0.010 * (0.5 if nominal else row[1])),
            "surface_kind": SURFACES[roster],
            "tool_kind": tool,
            "tool_width_m": width,
            "tool_footprint_length_m": footprint,
            "tool_normal_stiffness_n_m": stiffness,
            "friction_coefficient": 0.50 if nominal else float(0.30 + 0.40 * row[2]),
            "support_stiffness_n_m": 2500.0 if nominal else float(1800.0 + 1400.0 * row[3]),
            "effective_mass_kg": 0.35 if nominal else float(0.25 + 0.20 * row[4]),
            "damping_ratio": 0.80 if nominal else float(0.55 + 0.50 * row[5]),
            "restitution": 0.06 if nominal else float(0.03 + 0.09 * row[6]),
            "residual_family": RESIDUALS[roster],
            "residual_severity": float(0.65 + 0.25 * (0.5 if nominal else row[7])),
            "residual_cleanability": 0.85,
            "disturbance_kind": "none" if nominal else DISTURBANCES[roster],
            "disturbance_scale": 0.0 if nominal or DISTURBANCES[roster] == "none" else 0.25,
            "sensor_latency_steps": 2 if not nominal and DISTURBANCES[roster] == "sensor_noise_latency" else 0,
        })
    return tuple(blocks)
