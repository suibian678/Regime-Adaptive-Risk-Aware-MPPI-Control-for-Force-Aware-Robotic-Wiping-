"""Causal one-step inward hold for abrupt sampled force drops."""

from __future__ import annotations

from typing import Callable


def guarded_normal_increment(
    base_controller: Callable[..., tuple[float, dict]],
    *,
    target_force_n: float,
    measured_force_n: float,
    previous_force_n: float,
    target_lead_down_m: float,
    drop_rate_threshold_n_s: float = -150.0,
) -> tuple[float, dict]:
    increment, detail = base_controller(
        target_force_n=target_force_n,
        measured_force_n=measured_force_n,
        previous_force_n=previous_force_n,
        target_lead_down_m=target_lead_down_m,
    )
    rate = float(detail["force_rate_n_s"])
    if rate <= float(drop_rate_threshold_n_s) and float(increment) > 0.0:
        updated = dict(detail)
        updated.update(
            {
                "mode": "FORCE_DROP_HOLD",
                "unguarded_increment_m": float(increment),
                "projected_increment_m": 0.0,
                "projection_active": True,
                "force_drop_inward_hold_active": True,
            }
        )
        return 0.0, updated
    updated = dict(detail)
    updated["force_drop_inward_hold_active"] = False
    return float(increment), updated


__all__ = ["guarded_normal_increment"]
