"""Versioned authority audit for the V6-S3 split-authority controller.

S2's ``state_aware_integrator_v2`` treated every non-stable TRACK sample as
an integral hold.  S3 deliberately separates normal-force regulation from
task authority: a below-band sample may update the normal integral only when
contact, geometry, state, final-command, headroom, and rebound gates all
permit it.  Tangential motion and progress remain owned exclusively by the
original ``stable_track`` predicate.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

from .controller import SupervisorConfig, SupervisorState


AUTHORITY_METRIC_SPLIT_V3 = "state_aware_split_authority_v3"


def normal_regulation_permitted(
    row: Mapping[str, Any],
    config: SupervisorConfig = SupervisorConfig(),
) -> bool:
    """Reconstruct the exact S3 normal-regulation permission from one row."""

    config.validate()
    target = float(row["target_force_n"])
    force = float(row["measured_force_n"])
    contact_loss_threshold = max(
        config.contact_loss_floor_n,
        config.contact_loss_target_fraction * target,
    )
    return bool(
        str(row["state_before"]) == SupervisorState.TRACK.value
        and str(row["state_after"]) == SupervisorState.TRACK.value
        and not bool(row["transitioned"])
        and bool(row["contact_observed"])
        and force > contact_loss_threshold
        and bool(row["geometric_track_ready"])
        and not bool(row["projection_active"])
        and not bool(row["rebound_guard_triggered"])
        and not bool(row["headroom_triggered"])
        and math.isclose(
            float(row["executed_normal_step_m"]),
            float(row["nominal_normal_step_m"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    )


def authority_violations_s3(
    diagnostic_rows: Sequence[Mapping[str, Any]],
    *,
    config: SupervisorConfig = SupervisorConfig(),
) -> int:
    """Count S3 authority/integrator contract violations.

    A permitted normal-regulation step must commit the exactly reconstructible
    clipped integral candidate.  A nonpermitted TRACK hold must preserve the
    prior integral, and every state transition or non-TRACK state must clear
    it.  These normal rules never weaken task-tangential or progress rules.
    """

    config.validate()
    count = 0
    tolerance = 1e-12
    for row in diagnostic_rows:
        stable = bool(row["stable_track"])
        task_permission = bool(row["tangential_motion_permitted"])
        transitioned = bool(row["transitioned"])
        state_before = str(row["state_before"])
        state_after = str(row["state_after"])
        committed_delta = float(row["committed_progress_after"]) - float(
            row["committed_progress_before"]
        )
        integral_before = float(row["integral_before_n_s"])
        integral_after = float(row["integral_after_n_s"])
        violation = task_permission != stable
        violation = violation or (transitioned != (state_before != state_after))
        violation = violation or (
            stable
            and (
                state_after != SupervisorState.TRACK.value
                or not bool(row["force_track_ready"])
                or not bool(row["geometric_track_ready"])
            )
        )
        violation = violation or (
            not task_permission
            and abs(float(row["executed_tangential_step_m"])) > tolerance
        )
        violation = violation or (not stable and abs(committed_delta) > tolerance)
        violation = violation or (
            task_permission and bool(row["recovery_reposition_permitted"])
        )
        violation = violation or (
            transitioned
            and (
                task_permission
                or abs(float(row["executed_tangential_step_m"])) > tolerance
                or abs(committed_delta) > tolerance
            )
        )

        if normal_regulation_permitted(row, config):
            candidate = max(
                -config.integral_clip_n_s,
                min(
                    config.integral_clip_n_s,
                    integral_before
                    + (float(row["target_force_n"]) - float(row["measured_force_n"]))
                    * config.dt_s,
                ),
            )
            violation = violation or abs(integral_after - candidate) > tolerance
        elif (
            state_before == SupervisorState.TRACK.value
            and state_after == SupervisorState.TRACK.value
            and not transitioned
        ):
            violation = violation or abs(integral_after - integral_before) > tolerance
        else:
            violation = violation or abs(integral_after) > tolerance

        violation = violation or (
            state_after == SupervisorState.SAFE_HOLD.value
            and (
                abs(float(row["executed_normal_step_m"])) > tolerance
                or abs(float(row["executed_tangential_step_m"])) > tolerance
                or bool(row["recovery_reposition_permitted"])
                or abs(integral_after) > tolerance
                or abs(committed_delta) > tolerance
            )
        )
        count += int(violation)
    return count


__all__ = [
    "AUTHORITY_METRIC_SPLIT_V3",
    "authority_violations_s3",
    "normal_regulation_permitted",
]
