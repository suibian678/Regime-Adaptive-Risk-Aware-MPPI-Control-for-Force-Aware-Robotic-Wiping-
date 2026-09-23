"""Precision TRACK r7 with continuously speed-scheduled recovery capture.

R6 demonstrated that a fixed 0.5-mm recovery cap was too conservative to
re-establish contact.  R7 changes only that cap: the inherited recovery ramp
is retained at zero/outward velocity, and its admissible inward increment is
reduced continuously as measured inward speed approaches 0.08 m/s.  At or
above that inward speed the command is a zero-increment hold.

The R6 contact-verification support, rapid-drop phase hold, force envelope,
task-authority allocation, and all downstream permission boundaries remain
unchanged.
"""

from __future__ import annotations

from .controller import SupervisorCommand, SupervisorInput, SupervisorSnapshot
from .controller_precision_r6 import (
    V6PrecisionContactSupervisorR6,
)


PRECISION_R7_MAX_CAPTURE_INWARD_SPEED_M_S = 0.080
PRECISION_R7_RECOVERY_REASON = "recovery_acquire_speed_scheduled_ramp"


class V6PrecisionContactSupervisorR7(V6PrecisionContactSupervisorR6):
    """Single R7 candidate with continuous recovery authority scheduling."""

    controller_revision = "V6-precision-track-r7-speed-scheduled-capture"

    @staticmethod
    def _capture_authority_scale(normal_velocity_outward_m_s: float) -> float:
        inward_speed = max(-float(normal_velocity_outward_m_s), 0.0)
        return max(
            0.0,
            min(
                1.0,
                1.0
                - inward_speed / PRECISION_R7_MAX_CAPTURE_INWARD_SPEED_M_S,
            ),
        )

    def _velocity_limited_recovery_acquire(
        self,
        *,
        before: SupervisorSnapshot,
        observation: SupervisorInput,
        candidate: SupervisorCommand,
    ) -> SupervisorCommand:
        nominal = max(0.0, float(candidate.nominal_normal_step_m))
        authority_scale = self._capture_authority_scale(
            observation.normal_velocity_outward_m_s
        )
        dynamic_limit = self.config.recovery_acquire_max_step_m * authority_scale
        actuator_limited = min(nominal, dynamic_limit)
        executed = min(
            actuator_limited,
            max(0.0, float(candidate.safe_press_limit_m)),
        )
        return self._replace_normal_candidate(
            before=before,
            candidate=candidate,
            executed=executed,
            nominal=nominal,
            actuator_limited=actuator_limited,
            reason=PRECISION_R7_RECOVERY_REASON,
            freeze_integral=True,
        )


__all__ = [
    "PRECISION_R7_MAX_CAPTURE_INWARD_SPEED_M_S",
    "PRECISION_R7_RECOVERY_REASON",
    "V6PrecisionContactSupervisorR7",
]
