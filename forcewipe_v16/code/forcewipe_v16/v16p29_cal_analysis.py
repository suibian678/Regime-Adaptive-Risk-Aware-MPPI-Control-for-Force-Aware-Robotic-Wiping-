"""Frozen V16.29 CAL aggregation over the V16.28 per-evaluation metrics."""

from __future__ import annotations

from collections.abc import Iterable
import math

import numpy as np

from forcewipe_v16.v16p28_qualification_analysis import analyze_evaluation


TARGETS_N = (5.0, 8.0, 12.0)
BLOCKS = tuple(f"C{index}" for index in range(10))
PANELS = {
    "multipath_nominal_plant": tuple(f"C{index}" for index in range(5)),
    "multipath_randomized_plant": tuple(f"C{index}" for index in range(5, 10)),
}


def _finite_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _mean_sd(values: Iterable[object]) -> tuple[float | None, float | None]:
    finite = np.asarray([value for value in (_finite_float(item) for item in values) if value is not None], dtype=np.float64)
    if not finite.size:
        return None, None
    return float(finite.mean()), float(finite.std(ddof=1)) if finite.size > 1 else 0.0


def cluster_bootstrap(
    evaluations: list[dict], *, metric: str, target_force_n: float,
    resamples: int = 10_000, seed: int = 616_290_001,
) -> dict:
    cells = {
        block: [
            _finite_float(row.get(metric)) for row in evaluations
            if row["block_id"] == block and float(row["target_force_n"]) == float(target_force_n)
        ]
        for block in BLOCKS
    }
    if any(len(values) != 5 or any(value is None for value in values) for values in cells.values()):
        return {"metric": metric, "target_force_n": target_force_n, "available": False}
    rng = np.random.default_rng(seed + int(target_force_n) * 100)
    draws = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        sampled = rng.integers(0, len(BLOCKS), len(BLOCKS))
        values = [value for position in sampled for value in cells[BLOCKS[int(position)]]]
        draws[index] = float(np.mean(values))
    point = float(np.mean([value for values in cells.values() for value in values]))
    return {
        "metric": metric, "target_force_n": target_force_n, "available": True,
        "point_estimate": point,
        "pointwise_unadjusted_95pct_interval": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
        "cluster_unit": "pre-specified fixed CAL block", "clusters": 10,
        "resamples": resamples, "seed": seed + int(target_force_n) * 100,
    }


def _summary_cell(rows: list[dict], *, label: str) -> dict:
    mean_force, mean_force_sd = _mean_sd(row.get("contact_mean_force_n") for row in rows)
    rmse, rmse_sd = _mean_sd(row.get("contact_rmse_n") for row in rows)
    contact, contact_sd = _mean_sd(row.get("contact_fraction") for row in rows)
    peaks = [row["peak_force_n"] for row in rows if row.get("peak_force_n") is not None]
    return {
        "label": label, "evaluations": len(rows),
        "evaluation_gate_passes": sum(bool(row["evaluation_gate_passed"]) for row in rows),
        "task_gate_passes": sum(bool(row["task_gate_passed"]) for row in rows),
        "tracking_gate_passes": sum(bool(row["tracking_gate_passed"]) for row in rows),
        "safety_gate_passes": sum(bool(row["safety_gate_passed"]) for row in rows),
        "native_samples_gt_14p5n": sum(int(row["native_samples_gt_14p5n"]) for row in rows),
        "native_samples_gt_15n": sum(int(row["native_samples_gt_15n"]) for row in rows),
        "contact_mean_force_n_mean": mean_force, "contact_mean_force_n_sd": mean_force_sd,
        "contact_rmse_n_mean": rmse, "contact_rmse_n_sd": rmse_sd,
        "contact_fraction_mean": contact, "contact_fraction_sd": contact_sd,
        "maximum_peak_force_n": max(peaks) if peaks else None,
    }


def summarize_cal(evaluations: list[dict]) -> dict:
    if len(evaluations) != 150:
        raise ValueError("V16.29 finalizer requires exactly 150 evaluations")
    keys = {(int(row["checkpoint_seed"]), str(row["block_id"]), float(row["target_force_n"])) for row in evaluations}
    if len(keys) != 150 or {str(row["block_id"]) for row in evaluations} != set(BLOCKS):
        raise ValueError("V16.29 fully crossed key closure failed")
    target_cells = []
    intervals = []
    for target in TARGETS_N:
        rows = [row for row in evaluations if float(row["target_force_n"]) == target]
        if len(rows) != 50:
            raise ValueError("each V16.29 target requires 50 evaluations")
        cell = _summary_cell(rows, label=f"target_{int(target)}N")
        cell["target_force_n"] = target
        target_cells.append(cell)
        for metric in ("contact_mean_force_n", "signed_bias_fraction", "contact_rmse_n", "target_normalized_rmse", "contact_fraction"):
            intervals.append(cluster_bootstrap(evaluations, metric=metric, target_force_n=target))
    panel_cells = []
    for panel, blocks in PANELS.items():
        rows = [row for row in evaluations if row["block_id"] in blocks]
        panel_cells.append(_summary_cell(rows, label=panel))
    gate = all(bool(row["evaluation_gate_passed"]) for row in evaluations)
    peaks = [row["peak_force_n"] for row in evaluations if row.get("peak_force_n") is not None]
    return {
        "status": "completed_pass" if gate else "completed_scientific_fail",
        "cal_gate_passed": gate, "evaluations": evaluations,
        "target_cells": target_cells, "panel_cells": panel_cells,
        "cluster_bootstrap_intervals": intervals, "total_evaluations": 150,
        "fixed_cal_blocks": 10,
        "total_native_samples": sum(int(row["native_samples"]) for row in evaluations),
        "total_native_samples_gt_14p5n": sum(int(row["native_samples_gt_14p5n"]) for row in evaluations),
        "total_native_samples_gt_15n": sum(int(row["native_samples_gt_15n"]) for row in evaluations),
        "total_authority_violations": sum(int(row["authority_violations"]) for row in evaluations),
        "maximum_peak_force_n": max(peaks) if peaks else None,
        "independence_warning": "150 evaluations are repeated measurements over ten pre-specified fixed CAL blocks; they are not 150 independent environments or a random deployment sample.",
    }


__all__ = ["analyze_evaluation", "cluster_bootstrap", "summarize_cal"]

