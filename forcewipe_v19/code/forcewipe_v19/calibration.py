"""Training-partition calibration for force-prediction uncertainty radii."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping


def _finite_nonnegative(values: Iterable[float]) -> list[float]:
    cleaned = [float(value) for value in values]
    if not cleaned:
        raise ValueError("at least one residual is required")
    if any(not math.isfinite(value) or value < 0.0 for value in cleaned):
        raise ValueError("residuals must be finite and non-negative")
    return cleaned


def conformal_upper_radius(residuals: Iterable[float], *, coverage: float) -> float:
    """Return a finite-sample upper residual radius.

    The order statistic is ``ceil((n + 1) * coverage)`` with clipping at the
    largest observed residual.  Residuals must come only from the frozen
    training or validation partition; evaluation traces are not admissible.
    """

    if not 0.5 < float(coverage) < 1.0:
        raise ValueError("coverage must lie strictly between 0.5 and 1")
    values = sorted(_finite_nonnegative(residuals))
    rank = min(len(values), math.ceil((len(values) + 1) * float(coverage)))
    return values[rank - 1]


def fit_target_residual_radii(
    records: Iterable[Mapping[str, float]],
    *,
    coverage: float,
) -> dict[float, float]:
    """Fit one calibrated absolute-residual radius per target-force tier."""

    grouped: dict[float, list[float]] = defaultdict(list)
    for record in records:
        if "target_force_n" not in record or "absolute_residual_n" not in record:
            raise KeyError("records require target_force_n and absolute_residual_n")
        target = float(record["target_force_n"])
        residual = float(record["absolute_residual_n"])
        if not math.isfinite(target) or target <= 0.0:
            raise ValueError("target_force_n must be finite and positive")
        grouped[target].append(residual)
    if not grouped:
        raise ValueError("at least one calibration record is required")
    return {
        target: conformal_upper_radius(values, coverage=coverage)
        for target, values in sorted(grouped.items())
    }


def select_target_radius(
    target_force_n: float,
    radii_by_target: Mapping[float, float],
    *,
    tolerance_n: float = 1e-6,
) -> float:
    """Select the calibrated radius for an explicitly supported force tier."""

    target = float(target_force_n)
    matches = [
        float(radius)
        for key, radius in radii_by_target.items()
        if abs(float(key) - target) <= float(tolerance_n)
    ]
    if len(matches) != 1:
        raise ValueError(f"target {target_force_n!r} has no unique calibrated radius")
    radius = matches[0]
    if not math.isfinite(radius) or radius < 0.0:
        raise ValueError("calibrated radius must be finite and non-negative")
    return radius
