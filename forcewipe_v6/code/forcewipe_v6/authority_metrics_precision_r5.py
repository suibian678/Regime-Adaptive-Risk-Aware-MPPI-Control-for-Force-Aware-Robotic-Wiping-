"""Authority audit for precision-r5 orthogonal low-level composition."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .authority_metrics_s5 import authority_violations_s5
from .controller import SupervisorConfig, SupervisorState
from .controller_precision_r5 import PRECISION_R5_COMBINED_REASON
from .controller_s5 import S5_CROSS_TRACK_CORRECTION_ENTER_M


AUTHORITY_METRIC_PRECISION_R5 = "state_aware_precision_r5_orthogonal_v1"


def authority_violations_precision_r5(
    diagnostic_rows: Sequence[Mapping[str, Any]],
    *,
    config: SupervisorConfig = SupervisorConfig(),
) -> int:
    """Audit r5 while reusing the complete S5 contract for all other rows."""

    transformed: list[dict[str, Any]] = []
    combined: list[Mapping[str, Any]] = []
    for source in diagnostic_rows:
        row = dict(source)
        if str(row["transition_reason"]) == PRECISION_R5_COMBINED_REASON:
            combined.append(source)
            row["transition_reason"] = "track_cross_track_correction_projection"
            row["executed_normal_step_m"] = 0.0
            row["integral_after_n_s"] = row["integral_before_n_s"]
        transformed.append(row)

    count = authority_violations_s5(transformed, config=config)
    tolerance = 1e-12
    for row in combined:
        normal = float(row["executed_normal_step_m"])
        progress_delta = float(row["committed_progress_after"]) - float(
            row["committed_progress_before"]
        )
        valid = bool(
            str(row["state_before"]) == SupervisorState.TRACK.value
            and str(row["state_after"]) == SupervisorState.TRACK.value
            and not bool(row["transitioned"])
            and bool(row["force_track_ready"])
            and bool(row["geometric_track_ready"])
            and float(row["geometric_track_error_m"])
            >= S5_CROSS_TRACK_CORRECTION_ENTER_M
            and not bool(row["recovery_reposition_permitted"])
            and bool(row.get("cross_track_correction_permitted", False))
            and bool(row["projection_active"])
            and not bool(row["tangential_motion_permitted"])
            and abs(float(row["executed_tangential_step_m"])) <= tolerance
            and abs(progress_delta) <= tolerance
            and -config.max_lift_step_m - tolerance
            <= normal
            <= config.track_max_press_step_m + tolerance
            and float(row["envelope_command_upper_n"])
            <= config.safety_projection_bound_n + tolerance
        )
        count += int(not valid)
    return count


__all__ = [
    "AUTHORITY_METRIC_PRECISION_R5",
    "authority_violations_precision_r5",
]
