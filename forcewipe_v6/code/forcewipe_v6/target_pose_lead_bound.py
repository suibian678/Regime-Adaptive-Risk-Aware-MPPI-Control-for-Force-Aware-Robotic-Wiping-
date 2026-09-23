"""Bound positive target-delta increments by current target-pose lead."""

from __future__ import annotations

import math


class TargetPoseLeadBoundError(ValueError):
    pass


def apply_target_pose_lead_bound(
    normal_increment_m: float,
    detail: dict,
    *,
    target_lead_inward_m: float,
    maximum_target_lead_m: float = 0.01,
) -> tuple[float, dict]:
    """Project an inward increment so the one-step lead cannot exceed a bound.

    Outward and zero increments pass through unchanged.  This function changes
    neither the force law nor its gains; it limits only the integrated target
    setpoint relative to the measured TCP position.
    """

    increment = float(normal_increment_m)
    lead = float(target_lead_inward_m)
    limit = float(maximum_target_lead_m)
    if not all(math.isfinite(value) for value in (increment, lead, limit)):
        raise TargetPoseLeadBoundError("target-lead inputs must be finite")
    if lead < 0.0 or limit <= 0.0:
        raise TargetPoseLeadBoundError("target lead must be nonnegative and limit positive")
    projected = increment
    active = False
    if increment > 0.0:
        projected = min(increment, max(0.0, limit - lead))
        active = projected < increment - 1.0e-15
    enriched = dict(detail)
    enriched["target_lead_bound_m"] = limit
    enriched["target_lead_projection_active"] = active
    enriched["pre_lead_bound_increment_m"] = increment
    enriched["projected_increment_m"] = projected
    enriched["projection_active"] = bool(enriched.get("projection_active", False) or active)
    return projected, enriched
