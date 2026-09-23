"""Precision TRACK r4 with non-overlapping force/geometry scheduling.

R3 completed the 5-N development lifecycle, but its 510 cross-track holds
froze normal force regulation and averaged only 3.927 N.  R4 retains the
existing mutually exclusive Cartesian authorities.  When geometry requests a
cross-track correction, the exact-target normal servo gets priority whenever
its final command is physically actionable.  Cross-track correction executes
only after the normal command has settled below actuator resolution.  Task
tangential progress remains frozen in both cases.
"""

from __future__ import annotations

from dataclasses import replace

from .controller import SupervisorCommand, SupervisorInput, SupervisorSnapshot
from .controller_precision_r3 import V6PrecisionContactSupervisorR3
from .controller_s5 import S5_MINIMUM_ACTIONABLE_INWARD_STEP_M
from .precision_tracking import (
    ExactTargetForceServo,
    PrecisionTrackingInput,
    PrecisionTrackingState,
)


PRECISION_R4_SCHEDULER_PERIOD = 4
PRECISION_R4_NORMAL_PRIORITY_SLOTS = 3


class V6PrecisionContactSupervisorR4(V6PrecisionContactSupervisorR3):
    """Force-priority scheduler that preserves strict authority separation."""

    controller_revision = "V6-precision-track-r4-force-priority-scheduling"

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

    def _apply_force_priority(
        self,
        *,
        before: SupervisorSnapshot,
        observation: SupervisorInput,
        candidate: SupervisorCommand,
    ) -> SupervisorCommand:
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
            transition_reason="track_exact_normal_priority_over_cross_track",
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
            projection_active=(
                precise.actuator_saturation_active
                or precise.safety_projection_active
            ),
            actuator_saturation_active=precise.actuator_saturation_active,
            safety_projection_active=precise.safety_projection_active,
            integral_after_n_s=precise.integral_after_n_s,
            tangential_motion_permitted=False,
            recovery_reposition_permitted=False,
            cross_track_correction_permitted=False,
            executed_tangential_step_m=0.0,
            committed_progress_after=before.committed_progress,
        )

    def command(self, observation: SupervisorInput) -> SupervisorCommand:
        before = self.snapshot()
        candidate = super().command(observation)
        if (
            candidate.transition_reason
            == "track_cross_track_correction_projection"
            and candidate.cross_track_correction_permitted
        ):
            precise = self._precision_preview(
                before=before,
                observation=observation,
                candidate=candidate,
            )
            phase = before.state_dwell_samples % PRECISION_R4_SCHEDULER_PERIOD
            normal_slot = phase < PRECISION_R4_NORMAL_PRIORITY_SLOTS
            if (
                normal_slot
                and abs(precise.executed_normal_step_m)
                > S5_MINIMUM_ACTIONABLE_INWARD_STEP_M + 1.0e-15
            ):
                return self._apply_force_priority(
                    before=before,
                    observation=observation,
                    candidate=candidate,
                )
        return candidate


__all__ = [
    "PRECISION_R4_NORMAL_PRIORITY_SLOTS",
    "PRECISION_R4_SCHEDULER_PERIOD",
    "V6PrecisionContactSupervisorR4",
]
