"""State-aware authority and nested-budget audit for System R4."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .authority_metrics_s3 import authority_violations_s3
from .controller import SupervisorConfig, SupervisorState
from .controller_s5 import S5_CROSS_TRACK_CORRECTION_ENTER_M
from .controller_system_r4 import (
    R4_GEOMETRY_REALIGN_MAX_ERROR_M,
    R4_GEOMETRY_REALIGN_MAX_SAMPLES,
    R4_LOCAL_REACQUIRE_MAX_ATTEMPTS_PER_CYCLE,
    V6SystemContactSupervisorR4,
)


AUTHORITY_METRIC_SYSTEM_R4 = "state_aware_nested_recovery_authority_v5"


def authority_violations_system_r4(
    rows: Sequence[Mapping[str, Any]],
    *,
    config: SupervisorConfig | None = None,
) -> int:
    """Count task-authority and R4 nested-budget contract violations."""

    if config is None:
        config = V6SystemContactSupervisorR4().config
    count = authority_violations_s3(rows, config=config)
    tolerance = 1e-12
    derived_local: dict[tuple[Any, Any], int] = {}
    derived_geometry: dict[tuple[Any, Any], int] = {}
    for row in rows:
        reason = str(row["transition_reason"])
        cross_track = bool(row.get("cross_track_correction_permitted", False))
        normal = float(row["executed_normal_step_m"])
        tangent = float(row["executed_tangential_step_m"])
        progress_delta = float(row["committed_progress_after"]) - float(
            row["committed_progress_before"]
        )
        integral_delta = float(row["integral_after_n_s"]) - float(
            row["integral_before_n_s"]
        )
        r4_geometry_realign = bool(
            reason == "track_cross_track_correction_projection"
            and not bool(row["geometric_track_ready"])
            and bool(row["in_recovery_episode_after"])
        )
        key = (row.get("run_id"), row.get("evaluation_id"))
        local_before = derived_local.get(key, 0)
        geometry_before = derived_geometry.get(key, 0)
        enriched = "local_reacquire_attempts_after" in row
        if enriched:
            local_before = int(row["local_reacquire_attempts_before"])
            geometry_before = int(row["geometry_realign_samples_before"])
            local_after = int(row["local_reacquire_attempts_after"])
            geometry_after = int(row["geometry_realign_samples_after"])
        else:
            local_after = local_before
            geometry_after = geometry_before
            if reason == "track_contact_degradation_to_bounded_local_acquire":
                local_after += 1
            if r4_geometry_realign:
                geometry_after += 1
            if reason in {
                "local_reacquire_budget_to_full_recovery",
                "geometry_realign_budget_to_full_recovery",
            }:
                local_after = 0
                geometry_after = 0
            if bool(row.get("verified_return_completed", False)) or str(
                row["state_after"]
            ) in {
                SupervisorState.RECOVERY_LIFT.value,
                SupervisorState.REBOUND_GUARD.value,
                SupervisorState.HEADROOM.value,
                SupervisorState.SAFE_HOLD.value,
            }:
                local_after = 0
                geometry_after = 0

        if reason == "track_cross_track_correction_projection":
            minimum_error = (
                config.track_geometric_tolerance_m
                if r4_geometry_realign
                else S5_CROSS_TRACK_CORRECTION_ENTER_M
            )
            valid = bool(
                str(row["state_before"]) == SupervisorState.TRACK.value
                and str(row["state_after"]) == SupervisorState.TRACK.value
                and not bool(row["transitioned"])
                and bool(row["contact_observed"])
                and minimum_error - tolerance
                <= float(row["geometric_track_error_m"])
                <= R4_GEOMETRY_REALIGN_MAX_ERROR_M + tolerance
                and cross_track
                and not bool(row["tangential_motion_permitted"])
                and abs(normal) <= tolerance
                and abs(tangent) <= tolerance
                and abs(progress_delta) <= tolerance
                and abs(integral_delta) <= tolerance
            )
            count += int(not valid)
        elif cross_track:
            count += 1

        local_counted = bool(
            row.get(
                "local_reacquire_attempt_counted_this_step",
                reason == "track_contact_degradation_to_bounded_local_acquire",
            )
        )
        geometry_counted = bool(
            row.get(
                "geometry_realign_sample_counted_this_step",
                r4_geometry_realign,
            )
        )
        escalated = bool(
            row.get(
                "full_recovery_escalated_this_step",
                reason
                in {
                    "local_reacquire_budget_to_full_recovery",
                    "geometry_realign_budget_to_full_recovery",
                },
            )
        )
        count += int(local_counted and geometry_counted)
        count += int(not 0 <= local_after <= R4_LOCAL_REACQUIRE_MAX_ATTEMPTS_PER_CYCLE)
        count += int(not 0 <= geometry_after <= R4_GEOMETRY_REALIGN_MAX_SAMPLES)
        if local_counted and not escalated:
            count += int(local_after != local_before + 1)
        if geometry_counted and not escalated:
            count += int(geometry_after != geometry_before + 1)
        if escalated:
            count += int(local_after != 0 or geometry_after != 0)
        if str(row["state_after"]) == SupervisorState.SAFE_HOLD.value:
            count += int(local_after != 0 or geometry_after != 0)

        tail_flag = bool(
            row.get(
                "recovery_tail_settling_interlock",
                reason
                in {
                    "track_recovery_tail_settling_interlock",
                    "track_high_envelope_settling_interlock",
                },
            )
        )
        tail_reason = reason == "track_recovery_tail_settling_interlock"
        count += int(tail_flag != (tail_reason or reason == "track_high_envelope_settling_interlock"))
        if tail_reason:
            count += int(
                not bool(row["in_recovery_episode_before"])
                or not bool(row["in_recovery_episode_after"])
                or not bool(row["recovery_sample_counted_this_step"])
                or bool(row["tangential_motion_permitted"])
                or cross_track
                or abs(normal) > tolerance
                or abs(tangent) > tolerance
                or abs(progress_delta) > tolerance
                or abs(integral_delta) > tolerance
            )
        derived_local[key] = local_after
        derived_geometry[key] = geometry_after
    return count


__all__ = [
    "AUTHORITY_METRIC_SYSTEM_R4",
    "authority_violations_system_r4",
]
