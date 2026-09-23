"""Machine-unique metrics and gates for V16.28 qualification rev2."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


CONTACT_THRESHOLD_N = 3.0
HARD_FORCE_LIMIT_N = 15.0
HEADROOM_THRESHOLD_N = 14.5
MRE_LIMIT = 0.15
NRMSE_LIMIT = 0.20
ACTION_TOLERANCE = 1e-7


def _finite_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _percentile(values: np.ndarray, percentile: float) -> float | None:
    return float(np.percentile(values, percentile)) if values.size else None


def analyze_evaluation(
    rows: list[dict],
    *,
    target_force_n: float,
    expected_evaluation: dict,
) -> dict:
    """Analyze one complete trace; missing/invalid science is a gate failure, not an exception."""

    target = float(target_force_n)
    reasons: list[str] = []
    if target not in {5.0, 8.0, 12.0}:
        raise ValueError("target must be 5, 8, or 12 N")
    if not rows:
        return {
            **expected_evaluation,
            "target_force_n": target,
            "native_samples": 0,
            "metric_valid": False,
            "contact_samples": 0,
            "contact_fraction": 0.0,
            "contact_mean_force_n": None,
            "signed_bias_n": None,
            "signed_bias_fraction": None,
            "mean_relative_error": None,
            "contact_rmse_n": None,
            "target_normalized_rmse": None,
            "peak_force_n": None,
            "native_samples_gt_14p5n": 0,
            "native_samples_gt_15n": 0,
            "native_samples_equal_15n": 0,
            "authority_violations": 0,
            "maximum_action_record_mismatch": None,
            "task_gate_passed": False,
            "authority_gate_passed": False,
            "safety_gate_passed": False,
            "tracking_gate_passed": False,
            "evaluation_gate_passed": False,
            "science_failure_reasons": ["empty_trace", "no_contact_samples"],
        }

    force_values = [_finite_float(row.get("normal_force_n")) for row in rows]
    nonfinite_force = sum(value is None for value in force_values)
    finite_force = np.asarray([value for value in force_values if value is not None], dtype=np.float64)
    if nonfinite_force:
        reasons.append("nonfinite_force_sample")
    contact_values = finite_force[finite_force >= CONTACT_THRESHOLD_N]
    if contact_values.size == 0:
        reasons.append("no_contact_samples")

    mean_force = float(contact_values.mean()) if contact_values.size else None
    rmse = (
        float(np.sqrt(np.mean(np.square(contact_values - target))))
        if contact_values.size
        else None
    )
    signed_bias = mean_force - target if mean_force is not None else None
    signed_bias_fraction = signed_bias / target if signed_bias is not None else None
    mre = abs(signed_bias_fraction) if signed_bias_fraction is not None else None
    nrmse = rmse / target if rmse is not None else None

    authority_violations = 0
    action_mismatches: list[float] = []
    for row in rows:
        if row.get("direct_tdmpc2_action_authority") is not True:
            authority_violations += 1
        if row.get("force_dependent_action_projection") is not False:
            authority_violations += 1
        if row.get("rewiping_enabled") is not False:
            authority_violations += 1
        proposed = row.get("action")
        recorded = row.get("tdmpc2_action")
        try:
            left = np.asarray(proposed, dtype=np.float64).reshape(-1)
            right = np.asarray(recorded, dtype=np.float64).reshape(-1)
            if left.shape != (3,) or right.shape != (3,) or not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
                raise ValueError
            mismatch = float(np.max(np.abs(left - right)))
            action_mismatches.append(mismatch)
            authority_violations += int(mismatch > ACTION_TOLERANCE)
        except (TypeError, ValueError):
            authority_violations += 1
    if authority_violations:
        reasons.append("authority_contract_violation")

    last = rows[-1]
    progress = _finite_float(last.get("progress"))
    bins = last.get("completed_dose_bins")
    task_pass = bool(
        last.get("success") is True
        and progress is not None
        and progress >= 0.98
        and isinstance(bins, int)
        and bins == 20
    )
    if not task_pass:
        reasons.append("first_pass_task_gate_failed")
    safety_pass = bool(nonfinite_force == 0 and int(np.sum(finite_force > HARD_FORCE_LIMIT_N)) == 0)
    if not safety_pass:
        reasons.append("sampled_force_gate_failed")
    tracking_pass = bool(
        nonfinite_force == 0
        and mre is not None
        and nrmse is not None
        and mre <= MRE_LIMIT
        and nrmse <= NRMSE_LIMIT
    )
    if not tracking_pass:
        reasons.append("tracking_gate_failed")

    planning = np.asarray(
        [value for value in (_finite_float(row.get("planning_ms")) for row in rows) if value is not None],
        dtype=np.float64,
    )
    if planning.size != len(rows):
        reasons.append("nonfinite_planning_latency")
    steady = planning[1:] if planning.size > 1 else np.asarray([], dtype=np.float64)
    metric_valid = bool(nonfinite_force == 0 and contact_values.size > 0)
    evaluation_pass = bool(task_pass and authority_violations == 0 and safety_pass and tracking_pass)
    return {
        **expected_evaluation,
        "target_force_n": target,
        "native_samples": len(rows),
        "nonfinite_force_samples": nonfinite_force,
        "metric_valid": metric_valid,
        "contact_samples": int(contact_values.size),
        "contact_fraction": float(contact_values.size / len(rows)),
        "contact_mean_force_n": mean_force,
        "signed_bias_n": signed_bias,
        "signed_bias_fraction": signed_bias_fraction,
        "mean_relative_error": mre,
        "contact_rmse_n": rmse,
        "target_normalized_rmse": nrmse,
        "peak_force_n": float(finite_force.max()) if finite_force.size else None,
        "native_samples_gt_14p5n": int(np.sum(finite_force > HEADROOM_THRESHOLD_N)),
        "native_samples_gt_15n": int(np.sum(finite_force > HARD_FORCE_LIMIT_N)),
        "native_samples_equal_15n": int(np.sum(finite_force == HARD_FORCE_LIMIT_N)),
        "authority_violations": int(authority_violations),
        "maximum_action_record_mismatch": max(action_mismatches) if action_mismatches else None,
        "success": bool(last.get("success") is True),
        "progress": progress,
        "completed_dose_bins": bins,
        "task_gate_passed": task_pass,
        "authority_gate_passed": authority_violations == 0,
        "safety_gate_passed": safety_pass,
        "tracking_gate_passed": tracking_pass,
        "evaluation_gate_passed": evaluation_pass,
        "first_action_planning_ms": float(planning[0]) if planning.size else None,
        "steady_planning_median_ms": _percentile(steady, 50.0),
        "steady_planning_p95_ms": _percentile(steady, 95.0),
        "steady_planning_maximum_ms": float(steady.max()) if steady.size else None,
        "science_failure_reasons": sorted(set(reasons)),
    }


def _mean_sd(values: Iterable[object]) -> tuple[float | None, float | None]:
    finite = np.asarray(
        [value for value in (_finite_float(item) for item in values) if value is not None],
        dtype=np.float64,
    )
    if not finite.size:
        return None, None
    return float(finite.mean()), float(finite.std(ddof=1)) if finite.size > 1 else 0.0


def cluster_bootstrap(
    evaluations: list[dict],
    *,
    metric: str,
    target_force_n: float,
    resamples: int = 10_000,
    seed: int = 616_280_001,
) -> dict:
    blocks = sorted({str(row["block_id"]) for row in evaluations})
    if blocks != ["Q0", "Q1", "Q2", "Q3", "Q4"]:
        raise ValueError("cluster bootstrap requires frozen Q0--Q4 blocks")
    cells = {
        block: [
            _finite_float(row.get(metric))
            for row in evaluations
            if row["block_id"] == block and float(row["target_force_n"]) == float(target_force_n)
        ]
        for block in blocks
    }
    if any(len(values) != 5 or any(value is None for value in values) for values in cells.values()):
        return {"metric": metric, "target_force_n": target_force_n, "available": False}
    rng = np.random.default_rng(seed + int(target_force_n) * 100)
    draws = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        sampled = rng.integers(0, len(blocks), len(blocks))
        values = [value for position in sampled for value in cells[blocks[int(position)]]]
        draws[index] = float(np.mean(values))
    point = float(np.mean([value for values in cells.values() for value in values]))
    return {
        "metric": metric,
        "target_force_n": target_force_n,
        "available": True,
        "point_estimate": point,
        "pointwise_unadjusted_95pct_interval": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
        "cluster_unit": "pre-specified fixed scenario block",
        "clusters": 5,
        "resamples": resamples,
        "seed": seed + int(target_force_n) * 100,
    }


def summarize_qualification(evaluations: list[dict]) -> dict:
    if len(evaluations) != 75:
        raise ValueError("V16.28 finalizer requires exactly 75 evaluations")
    target_cells = []
    intervals = []
    for target in (5.0, 8.0, 12.0):
        cell = [row for row in evaluations if float(row["target_force_n"]) == target]
        if len(cell) != 25:
            raise ValueError("each target must contain 25 evaluations")
        mean_force, mean_force_sd = _mean_sd(row.get("contact_mean_force_n") for row in cell)
        rmse, rmse_sd = _mean_sd(row.get("contact_rmse_n") for row in cell)
        contact, contact_sd = _mean_sd(row.get("contact_fraction") for row in cell)
        peaks = [row["peak_force_n"] for row in cell if row["peak_force_n"] is not None]
        target_cells.append({
            "target_force_n": target,
            "evaluations": 25,
            "evaluation_gate_passes": sum(bool(row["evaluation_gate_passed"]) for row in cell),
            "task_gate_passes": sum(bool(row["task_gate_passed"]) for row in cell),
            "tracking_gate_passes": sum(bool(row["tracking_gate_passed"]) for row in cell),
            "native_samples_gt_14p5n": sum(int(row["native_samples_gt_14p5n"]) for row in cell),
            "native_samples_gt_15n": sum(int(row["native_samples_gt_15n"]) for row in cell),
            "contact_mean_force_n_mean": mean_force,
            "contact_mean_force_n_sd": mean_force_sd,
            "contact_rmse_n_mean": rmse,
            "contact_rmse_n_sd": rmse_sd,
            "contact_fraction_mean": contact,
            "contact_fraction_sd": contact_sd,
            "maximum_peak_force_n": max(peaks) if peaks else None,
        })
        for metric in ("contact_mean_force_n", "signed_bias_fraction", "contact_rmse_n", "target_normalized_rmse", "contact_fraction"):
            intervals.append(cluster_bootstrap(evaluations, metric=metric, target_force_n=target))
    gate = all(bool(row["evaluation_gate_passed"]) for row in evaluations)
    all_peaks = [row["peak_force_n"] for row in evaluations if row["peak_force_n"] is not None]
    return {
        "status": "completed_pass" if gate else "completed_scientific_fail",
        "qualification_gate_passed": gate,
        "evaluations": evaluations,
        "target_cells": target_cells,
        "cluster_bootstrap_intervals": intervals,
        "total_evaluations": 75,
        "fixed_scenario_blocks": 5,
        "total_native_samples": sum(int(row["native_samples"]) for row in evaluations),
        "total_native_samples_gt_14p5n": sum(int(row["native_samples_gt_14p5n"]) for row in evaluations),
        "total_native_samples_gt_15n": sum(int(row["native_samples_gt_15n"]) for row in evaluations),
        "total_authority_violations": sum(int(row["authority_violations"]) for row in evaluations),
        "maximum_peak_force_n": max(all_peaks) if all_peaks else None,
        "independence_warning": "75 evaluations are repeated measurements over five pre-specified different fixed scenario blocks; they are not 75 independent environments or a random sample from a deployment population.",
    }
