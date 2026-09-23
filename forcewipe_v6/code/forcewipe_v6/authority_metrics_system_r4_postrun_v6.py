"""Corrected post-run authority reconstruction for stripped System R4 rows.

The frozen v5 metric reconstructed the geometry-realignment ledger but did
not mirror the controller's reset when geometric tracking became ready.  It
therefore joined separate legal realignment intervals and reported six false
budget violations in the frozen R4 development run.  This module enriches a
copy of each row with the exact sequential ledger before delegating all other
checks to the frozen metric.  Frozen run files are never modified.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .authority_metrics_system_r4 import authority_violations_system_r4
from .controller import SupervisorConfig, SupervisorState


AUTHORITY_METRIC_SYSTEM_R4_POSTRUN_V6 = (
    "state_aware_nested_recovery_authority_v6_geometry_ready_reset"
)


def authority_violations_system_r4_postrun_v6(
    rows: Sequence[Mapping[str, Any]],
    *,
    config: SupervisorConfig | None = None,
) -> int:
    """Audit R4 rows after reconstructing the two omitted ledger fields."""

    enriched_rows: list[dict[str, Any]] = []
    ledgers: dict[tuple[Any, Any], tuple[int, int]] = {}
    for original in rows:
        row = dict(original)
        key = (row.get("run_id"), row.get("evaluation_id"))
        local_before, geometry_before = ledgers.get(key, (0, 0))
        reason = str(row["transition_reason"])
        geometry_counted = bool(
            reason == "track_cross_track_correction_projection"
            and not bool(row["geometric_track_ready"])
            and bool(row["in_recovery_episode_after"])
        )
        local_counted = reason == "track_contact_degradation_to_bounded_local_acquire"
        escalated = reason in {
            "local_reacquire_budget_to_full_recovery",
            "geometry_realign_budget_to_full_recovery",
        }
        local_after = local_before + int(local_counted)
        geometry_after = geometry_before + int(geometry_counted)
        if escalated:
            local_after = 0
            geometry_after = 0
        if bool(row.get("verified_return_completed", False)) or not bool(
            row["in_recovery_episode_after"]
        ):
            local_after = 0
            geometry_after = 0
        elif str(row["state_after"]) in {
            SupervisorState.RECOVERY_LIFT.value,
            SupervisorState.REBOUND_GUARD.value,
            SupervisorState.HEADROOM.value,
            SupervisorState.SAFE_HOLD.value,
        }:
            local_after = 0
            geometry_after = 0
        elif bool(row["geometric_track_ready"]):
            # This reset exists in R4's _normalize_subattempt_state() and was
            # the sole omission in the frozen stripped-row reconstruction.
            geometry_after = 0
        row.update(
            {
                "local_reacquire_attempts_before": local_before,
                "local_reacquire_attempts_after": local_after,
                "geometry_realign_samples_before": geometry_before,
                "geometry_realign_samples_after": geometry_after,
                "local_reacquire_attempt_counted_this_step": local_counted,
                "geometry_realign_sample_counted_this_step": geometry_counted,
                "full_recovery_escalated_this_step": escalated,
            }
        )
        enriched_rows.append(row)
        ledgers[key] = (local_after, geometry_after)
    return authority_violations_system_r4(enriched_rows, config=config)


__all__ = [
    "AUTHORITY_METRIC_SYSTEM_R4_POSTRUN_V6",
    "authority_violations_system_r4_postrun_v6",
]
