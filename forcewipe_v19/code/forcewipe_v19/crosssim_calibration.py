"""Policy-independent calibration definitions for the MuJoCo task-dynamics port."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np


CONTACT_THRESHOLD_N = 0.1
POSITION_TOLERANCE_M = 0.0005  # one maximum normal command increment
ONSET_TOLERANCE_SAMPLES = 5  # 50 ms at the frozen 100-Hz control rate
FORCE_NRMSE_TOLERANCE = 0.20  # matches the frozen task-level NRMSE scale
PEAK_RELATIVE_TOLERANCE = 0.20


@dataclass(frozen=True)
class CalibrationMotion:
    motion_id: str
    role: str
    segments: tuple[tuple[int, tuple[float, float, float]], ...]

    def actions(self) -> np.ndarray:
        values: list[tuple[float, float, float]] = []
        for count, action in self.segments:
            if count <= 0:
                raise ValueError("calibration segment length must be positive")
            values.extend([action] * count)
        result = np.asarray(values, dtype=np.float32)
        if result.ndim != 2 or result.shape[1] != 3:
            raise ValueError("calibration motion must contain three-axis actions")
        if np.any(np.abs(result) > 1.0) or not np.all(np.isfinite(result)):
            raise ValueError("calibration action is outside the normalised action box")
        return result


MOTIONS = (
    CalibrationMotion(
        "fit_light_load_hold_release",
        "fit",
        ((215, (0.0, 0.0, 1.0)), (40, (0.0, 0.0, 0.0)), (20, (0.0, 0.0, -1.0))),
    ),
    CalibrationMotion(
        "fit_medium_load_hold_release",
        "fit",
        ((228, (0.0, 0.0, 1.0)), (40, (0.0, 0.0, 0.0)), (24, (0.0, 0.0, -1.0))),
    ),
    CalibrationMotion(
        "validation_contact_tangent_release",
        "validation",
        (
            (222, (0.0, 0.0, 1.0)),
            (20, (0.0, 0.0, 0.0)),
            (80, (0.2, 0.0, 0.0)),
            (20, (0.0, 0.0, -1.0)),
        ),
    ),
    CalibrationMotion(
        "validation_medium_unload",
        "validation",
        ((230, (0.0, 0.0, 1.0)), (30, (0.0, 0.0, -1.0)), (20, (0.0, 0.0, 0.0))),
    ),
)


def motion_by_id(motion_id: str) -> CalibrationMotion:
    rows = [row for row in MOTIONS if row.motion_id == motion_id]
    if len(rows) != 1:
        raise ValueError("unknown calibration motion")
    return rows[0]


def _onset(force: np.ndarray) -> int | None:
    indices = np.flatnonzero(np.asarray(force, dtype=float) > CONTACT_THRESHOLD_N)
    return int(indices[0]) if indices.size else None


def comparison_metrics(reference: dict, candidate: dict) -> dict:
    reference_force = np.asarray(reference["normal_force_n"], dtype=float)
    candidate_force = np.asarray(candidate["normal_force_n"], dtype=float)
    reference_position = np.asarray(reference["tool_position_world_m"], dtype=float)
    candidate_position = np.asarray(candidate["tool_position_world_m"], dtype=float)
    if reference_force.shape != candidate_force.shape or reference_position.shape != candidate_position.shape:
        raise ValueError("calibration traces have different shapes")
    if reference_position.shape != (reference_force.size, 3):
        raise ValueError("calibration position trace has an invalid shape")
    if not all(np.all(np.isfinite(row)) for row in (
        reference_force, candidate_force, reference_position, candidate_position
    )):
        raise ValueError("calibration trace contains NaN or Inf")
    union = (reference_force > CONTACT_THRESHOLD_N) | (candidate_force > CONTACT_THRESHOLD_N)
    force_scale = max(float(np.max(reference_force)), 1.0)
    force_rmse = float(np.sqrt(np.mean(np.square(
        reference_force[union] - candidate_force[union]
    )))) if np.any(union) else 0.0
    position_error = (
        (candidate_position - candidate_position[0])
        - (reference_position - reference_position[0])
    )
    position_rmse = float(np.sqrt(np.mean(np.sum(np.square(position_error), axis=1))))
    reference_peak = float(np.max(reference_force))
    candidate_peak = float(np.max(candidate_force))
    peak_relative_error = abs(candidate_peak - reference_peak) / max(reference_peak, 1.0)
    reference_onset = _onset(reference_force)
    candidate_onset = _onset(candidate_force)
    onset_error = (
        abs(candidate_onset - reference_onset)
        if reference_onset is not None and candidate_onset is not None
        else reference_force.size
    )
    normalised = {
        "force_nrmse": force_rmse / force_scale,
        "position_rmse_m": position_rmse,
        "contact_onset_error_samples": int(onset_error),
        "peak_relative_error": peak_relative_error,
    }
    ratios = (
        normalised["force_nrmse"] / FORCE_NRMSE_TOLERANCE,
        normalised["position_rmse_m"] / POSITION_TOLERANCE_M,
        normalised["contact_onset_error_samples"] / ONSET_TOLERANCE_SAMPLES,
        normalised["peak_relative_error"] / PEAK_RELATIVE_TOLERANCE,
    )
    normalised.update({
        "reference_contact_onset_sample": reference_onset,
        "candidate_contact_onset_sample": candidate_onset,
        "reference_peak_force_n": reference_peak,
        "candidate_peak_force_n": candidate_peak,
        "maximum_tolerance_ratio": float(max(ratios)),
        "mean_tolerance_ratio": float(np.mean(ratios)),
        "within_all_tolerances": bool(max(ratios) <= 1.0),
    })
    return normalised


def aggregate_motion_metrics(rows: Iterable[dict]) -> dict:
    values = list(rows)
    if not values:
        raise ValueError("at least one motion metric is required")
    return {
        "motions": len(values),
        "maximum_tolerance_ratio": float(max(row["maximum_tolerance_ratio"] for row in values)),
        "mean_tolerance_ratio": float(np.mean([row["mean_tolerance_ratio"] for row in values])),
        "within_all_tolerances": all(bool(row["within_all_tolerances"]) for row in values),
    }

