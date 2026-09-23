"""Pre-result acceptance rule for the V4 Study-C multi-force method screen."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class StudyCScreenThresholds:
    minimum_band_fraction: float = 0.75
    minimum_mean_force_ratio: float = 0.85
    maximum_mean_force_ratio: float = 1.15
    maximum_rmse_fraction: float = 0.15
    minimum_rmse_n: float = 0.50
    maximum_native_peak_n: float = 14.50
    maximum_solver_failure_fraction: float = 0.0
    maximum_deadline_miss_fraction: float = 0.01


@dataclass(frozen=True)
class StudyCScreenCell:
    method: str
    target_force_n: float
    lifecycle_success: bool
    stable_mean_force_n: float | None
    stable_rmse_n: float | None
    stable_band_fraction: float | None
    native_peak_force_n: float | None
    native_force_limit_exceedances: int
    solver_calls: int
    solver_failures: int
    solver_deadline_misses: int


@dataclass(frozen=True)
class StudyCScreenDecision:
    passed: bool
    reasons: tuple[str, ...]


def evaluate_study_c_screen_cell(
    cell: StudyCScreenCell,
    *,
    thresholds: StudyCScreenThresholds = StudyCScreenThresholds(),
) -> StudyCScreenDecision:
    """Evaluate one force-tier cell; this is a DEV admission rule, not a claim gate."""

    reasons: list[str] = []
    target = float(cell.target_force_n)
    if not math.isfinite(target) or target <= 0:
        raise ValueError("target force must be finite and positive")
    metrics = (
        cell.stable_mean_force_n,
        cell.stable_rmse_n,
        cell.stable_band_fraction,
        cell.native_peak_force_n,
    )
    if not cell.lifecycle_success:
        reasons.append("lifecycle_failure")
    if any(value is None or not math.isfinite(float(value)) for value in metrics):
        reasons.append("missing_or_nonfinite_tracking_metric")
    else:
        mean = float(cell.stable_mean_force_n)
        rmse = float(cell.stable_rmse_n)
        band = float(cell.stable_band_fraction)
        peak = float(cell.native_peak_force_n)
        ratio = mean / target
        if not thresholds.minimum_mean_force_ratio <= ratio <= thresholds.maximum_mean_force_ratio:
            reasons.append("mean_force_ratio")
        rmse_limit = max(
            thresholds.minimum_rmse_n,
            thresholds.maximum_rmse_fraction * target,
        )
        if rmse > rmse_limit:
            reasons.append("rmse")
        if band < thresholds.minimum_band_fraction:
            reasons.append("band_fraction")
        if peak > thresholds.maximum_native_peak_n:
            reasons.append("native_peak_headroom")
    if int(cell.native_force_limit_exceedances) != 0:
        reasons.append("native_force_limit_exceedance")
    if int(cell.solver_calls) < 0 or int(cell.solver_failures) < 0:
        raise ValueError("solver counts must be nonnegative")
    if cell.solver_calls:
        if cell.solver_failures / cell.solver_calls > thresholds.maximum_solver_failure_fraction:
            reasons.append("solver_failure_fraction")
        if (
            cell.solver_deadline_misses / cell.solver_calls
            > thresholds.maximum_deadline_miss_fraction
        ):
            reasons.append("deadline_miss_fraction")
    elif cell.solver_failures or cell.solver_deadline_misses:
        raise ValueError("solver failures/misses require positive solver calls")
    return StudyCScreenDecision(not reasons, tuple(reasons))


def aggregate_study_c_method_decision(
    cells: tuple[StudyCScreenCell, ...],
) -> StudyCScreenDecision:
    if not cells:
        raise ValueError("method screen requires force-tier cells")
    methods = {cell.method for cell in cells}
    targets = {float(cell.target_force_n) for cell in cells}
    if len(methods) != 1 or targets != {5.0, 8.0, 12.0}:
        raise ValueError("method screen requires exactly one 5/8/12 N cell per method")
    reasons: list[str] = []
    for cell in sorted(cells, key=lambda item: item.target_force_n):
        decision = evaluate_study_c_screen_cell(cell)
        reasons.extend(f"{cell.target_force_n:g}N:{reason}" for reason in decision.reasons)
    return StudyCScreenDecision(not reasons, tuple(reasons))

