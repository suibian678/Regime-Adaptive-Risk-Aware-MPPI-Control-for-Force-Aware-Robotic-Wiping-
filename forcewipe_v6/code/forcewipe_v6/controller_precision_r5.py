"""Precision TRACK r5 with simultaneous orthogonal low-level regulation.

R3 established a force-ready recovery handoff but froze normal regulation on
every cross-track correction sample.  R4 time-division improved force tracking
at the cost of much faster geometric-recovery exhaustion.  R5 instead composes
the two *low-level* Cartesian components in one TRACK hold: exact normal-force
regulation along the instantaneous surface normal and the already bounded
cross-track projection in its orthogonal tangent plane.  Task-path progress is
still frozen and no high-level Cartesian authority is introduced.
"""

from __future__ import annotations

from dataclasses import replace

from .controller import SupervisorCommand, SupervisorInput, SupervisorSnapshot
from .controller_precision_r3 import V6PrecisionContactSupervisorR3
from .precision_tracking import (
    ExactTargetForceServo,
    PrecisionTrackingInput,
    PrecisionTrackingState,
)


PRECISION_R5_COMBINED_REASON = "track_cross_track_with_exact_normal"


class V6PrecisionContactSupervisorR5(V6PrecisionContactSupervisorR3):
    """Compose orthogonal normal and cross-track low-level commands."""

    controller_revision = "V6-precision-track-r5-orthogonal-composition"

    def _precision_preview(
        self,
        *,
        before: SupervisorSnapshot,
        observation: SupervisorInput,
        candidate: SupervisorCommand,
    ):
        servo = ExactTargetForceServo(self._precision_config())
        servo.restore(
            PrecisionTrackingState(
                integral_error_n_s=before.integral_error_n_s,
                previous_executed_normal_step_m=before.previous_normal_command_m,
            )
        )
        return servo.step(
            PrecisionTrackingInput(
                target_force_n=float(observation.target_force_n),
                measured_force_n=float(observation.measured_force_n),
                force_rate_n_s=float(candidate.force_rate_n_s),
                envelope_base_upper_n=float(candidate.envelope_base_upper_n),
                contact_observed=bool(observation.contact_observed),
                geometric_track_ready=True,
                requested_tangential_step_m=0.0,
            )
        )

    def command(self, observation: SupervisorInput) -> SupervisorCommand:
        before = self.snapshot()
        candidate = super().command(observation)
        if not (
            candidate.transition_reason
            == "track_cross_track_correction_projection"
            and candidate.cross_track_correction_permitted
        ):
            return candidate

        precise = self._precision_preview(
            before=before,
            observation=observation,
            candidate=candidate,
        )
        self._integral_error_n_s = precise.integral_after_n_s
        self._previous_previous_normal_command_m = before.previous_normal_command_m
        self._previous_normal_command_m = precise.executed_normal_step_m
        self._committed_progress = before.committed_progress
        self._in_recovery_episode = candidate.in_recovery_episode_after
        self._pre_recovery_highwater_progress = (
            candidate.pre_recovery_highwater_progress_after
        )
        self._verified_return_count = candidate.verified_return_count

        envelope_command = candidate.envelope_base_upper_n + (
            self.config.envelope_candidate_press_gain_n_per_m
            * max(
                precise.executed_normal_step_m - precise.projection_reference_m,
                0.0,
            )
        )
        return replace(
            candidate,
            transition_reason=PRECISION_R5_COMBINED_REASON,
            stable_track=False,
            p_step_m=precise.p_step_m,
            i_step_m=precise.i_step_m,
            d_step_m=precise.one_sided_d_step_m,
            nominal_normal_step_m=precise.raw_normal_step_m,
            actuator_limited_normal_step_m=precise.actuator_limited_normal_step_m,
            projected_normal_step_m=precise.executed_normal_step_m,
            executed_normal_step_m=precise.executed_normal_step_m,
            envelope_command_upper_n=envelope_command,
            safe_press_limit_m=precise.safe_press_limit_m,
            projection_active=True,
            actuator_saturation_active=precise.actuator_saturation_active,
            safety_projection_active=precise.safety_projection_active,
            integral_after_n_s=precise.integral_after_n_s,
            tangential_motion_permitted=False,
            recovery_reposition_permitted=False,
            cross_track_correction_permitted=True,
            executed_tangential_step_m=0.0,
            committed_progress_after=before.committed_progress,
        )


__all__ = [
    "PRECISION_R5_COMBINED_REASON",
    "V6PrecisionContactSupervisorR5",
]
