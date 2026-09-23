"""Authority audit for the V6-S5 path-frame projection candidate."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .authority_metrics_s3 import authority_violations_s3
from .controller import SupervisorConfig, SupervisorState
from .controller_s5 import (
    S5_CROSS_TRACK_CORRECTION_ENTER_M,
    S5_MINIMUM_ACTIONABLE_INWARD_STEP_M,
)


AUTHORITY_METRIC_PATH_FRAME_V4 = "state_aware_path_frame_authority_v4"


def authority_violations_s5(
    diagnostic_rows: Sequence[Mapping[str, Any]],
    *,
    config: SupervisorConfig = SupervisorConfig(),
) -> int:
    """Extend the zero-target S3 audit with S5-specific permissions."""

    count = authority_violations_s3(diagnostic_rows, config=config)
    tolerance = 1e-12
    for row in diagnostic_rows:
        reason = str(row["transition_reason"])
        recovery = bool(row["recovery_reposition_permitted"])
        cross_track = bool(row.get("cross_track_correction_permitted", False))
        normal = float(row["executed_normal_step_m"])
        tangent = float(row["executed_tangential_step_m"])
        progress_delta = float(row["committed_progress_after"]) - float(
            row["committed_progress_before"]
        )
        integral_delta = float(row["integral_after_n_s"]) - float(
            row["integral_before_n_s"]
        )

        if reason == "track_cross_track_correction_projection":
            valid = bool(
                str(row["state_before"]) == SupervisorState.TRACK.value
                and str(row["state_after"]) == SupervisorState.TRACK.value
                and not bool(row["transitioned"])
                and bool(row["force_track_ready"])
                and bool(row["geometric_track_ready"])
                and float(row["geometric_track_error_m"])
                >= S5_CROSS_TRACK_CORRECTION_ENTER_M
                and not recovery
                and cross_track
                and bool(row["projection_active"])
                and not bool(row["tangential_motion_permitted"])
                and abs(normal) <= tolerance
                and abs(tangent) <= tolerance
                and abs(progress_delta) <= tolerance
                and abs(integral_delta) <= tolerance
            )
            count += int(not valid)
        elif reason == "track_subresolution_inward_hold":
            projected = float(row["projected_normal_step_m"])
            valid = bool(
                str(row["state_before"]) == SupervisorState.TRACK.value
                and str(row["state_after"]) == SupervisorState.TRACK.value
                and not bool(row["transitioned"])
                and not recovery
                and not cross_track
                and bool(row["projection_active"])
                and not bool(row["tangential_motion_permitted"])
                and 0.0 < projected <= S5_MINIMUM_ACTIONABLE_INWARD_STEP_M
                and abs(normal) <= tolerance
                and abs(tangent) <= tolerance
                and abs(progress_delta) <= tolerance
                and abs(integral_delta) <= tolerance
            )
            count += int(not valid)
        elif (recovery or cross_track) and str(row["state_after"]) == SupervisorState.TRACK.value:
            count += 1
    return count


__all__ = [
    "AUTHORITY_METRIC_PATH_FRAME_V4",
    "authority_violations_s5",
]
