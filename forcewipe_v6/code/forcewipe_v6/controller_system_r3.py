"""Bounded System R3 candidate for code-only development.

R3 is a versioned child of the frozen System R2 controller.  It changes only
two mechanisms identified by the completed R2 development SANDBOX:

* a low-static-margin settling interlock suppresses an inward TRACK command
  when the causal envelope is already close to the projection bound and the
  tool still has material outward velocity; and
* contact degradation that retains measured contact and valid path geometry
  enters bounded local ``RECOVERY_ACQUIRE`` directly instead of first lifting
  and hover-repositioning.  True contact loss, geometry loss, headroom, and
  rebound paths retain their inherited priority.

This module is independent of SAPIEN.  It is not physical evidence and does
not authorize qualification, CAL, TRAIN, or TEST execution.
"""

from __future__ import annotations

from dataclasses import replace

from .controller import (
    SupervisorCommand,
    SupervisorInput,
    SupervisorSnapshot,
    SupervisorState,
)
from .controller_system_r2 import V6SystemContactSupervisorR2


R3_LOW_STATIC_MARGIN_N = 3.0
R3_CONTACT_LOSS_TARGET_FRACTION = 0.50
R3_SETTLING_ENVELOPE_ENTER_N = 14.0
R3_SETTLING_OUTWARD_VELOCITY_M_S = 0.025
R3_SETTLING_ESTABLISHED_FORCE_FRACTION = 0.85
R3_SETTLING_RECENT_PRESS_M = 0.001
R3_LOCAL_REACQUIRE_MIN_TARGET_FRACTION = 0.15


class V6SystemContactSupervisorR3(V6SystemContactSupervisorR2):
    """Single non-swept System R3 candidate."""

    controller_revision = "V6-system-contact-candidate-v3-settling-local-acquire"

    def __init__(self, config=None) -> None:
        super().__init__(config)
        # R3 makes the local-acquire boundary and the frozen authority audit
        # use the same threshold.  This is part of the local-acquire
        # mechanism, not a caller-selectable parameter sweep.
        self.config = replace(
            self.config,
            contact_loss_target_fraction=R3_CONTACT_LOSS_TARGET_FRACTION,
        )
        self.config.validate()
        self._r3_settling_interlock = False

    def _phase_hold_eligible(
        self,
        *,
        before: SupervisorSnapshot,
        observation: SupervisorInput,
        candidate: SupervisorCommand,
    ) -> bool:
        """Extend the inherited hold with one high-envelope settling gate."""

        self._r3_settling_interlock = False
        if super()._phase_hold_eligible(
            before=before,
            observation=observation,
            candidate=candidate,
        ):
            return True

        low_static_margin = (
            self.config.force_limit_n - observation.target_force_n
            <= R3_LOW_STATIC_MARGIN_N + 1e-12
        )
        established = (
            observation.measured_force_n
            >= R3_SETTLING_ESTABLISHED_FORCE_FRACTION
            * observation.target_force_n
        )
        recent_press = max(
            before.previous_normal_command_m,
            before.previous_previous_normal_command_m,
        ) >= R3_SETTLING_RECENT_PRESS_M
        eligible = bool(
            before.state == SupervisorState.TRACK.value
            and not before.in_recovery_episode
            and low_static_margin
            and established
            and recent_press
            and observation.normal_velocity_outward_m_s
            >= R3_SETTLING_OUTWARD_VELOCITY_M_S
            and candidate.envelope_command_upper_n
            >= R3_SETTLING_ENVELOPE_ENTER_N
            and candidate.state_before == SupervisorState.TRACK.value
            and candidate.state_after == SupervisorState.TRACK.value
            and not candidate.transitioned
            and candidate.force_track_ready
            and candidate.geometric_track_ready
            and not candidate.headroom_triggered
            and not candidate.rebound_guard_triggered
            and candidate.executed_normal_step_m > 0.0
        )
        self._r3_settling_interlock = eligible
        return eligible

    def _local_reacquire_eligible(
        self,
        *,
        before: SupervisorSnapshot,
        observation: SupervisorInput,
        candidate: SupervisorCommand,
    ) -> bool:
        """Return whether degraded contact may reacquire without a lift."""

        minimum_force = max(
            self.config.contact_threshold_n,
            R3_LOCAL_REACQUIRE_MIN_TARGET_FRACTION
            * observation.target_force_n,
        )
        return bool(
            before.state == SupervisorState.TRACK.value
            and candidate.state_before == SupervisorState.TRACK.value
            and candidate.state_after == SupervisorState.RECOVERY_LIFT.value
            and candidate.transitioned
            and candidate.transition_reason == "track_contact_loss_to_recovery_lift"
            and observation.contact_observed
            and observation.measured_force_n > minimum_force
            and observation.geometric_track_error_m
            <= self.config.track_geometric_tolerance_m
            and not candidate.headroom_triggered
            and not candidate.rebound_guard_triggered
        )

    def _apply_local_reacquire(
        self,
        *,
        before: SupervisorSnapshot,
        observation: SupervisorInput,
        candidate: SupervisorCommand,
    ) -> SupervisorCommand:
        """Replace one full-recovery transition with bounded local acquire."""

        self.restore(before)
        self._integral_error_n_s = 0.0
        if self._in_recovery_episode:
            recovery_opened = True
            self._verified_return_count = 0
        else:
            recovery_opened = self._start_recovery_cycle()

        if recovery_opened and self._state is not SupervisorState.SAFE_HOLD:
            self._transition(SupervisorState.RECOVERY_ACQUIRE)
            reason = "track_contact_degradation_to_local_acquire"
        else:
            reason = "episode_recovery_cycle_budget_exhausted"

        counted, exhausted = self._account_open_recovery_sample(
            recovery_active_this_step=bool(
                self._in_recovery_episode
                and before.state != SupervisorState.SAFE_HOLD.value
            )
        )
        if exhausted:
            reason = "episode_recovery_sample_budget_exhausted"

        self._state_dwell_samples = 1
        self._update_history(
            force=float(observation.measured_force_n),
            executed_normal_step_m=0.0,
        )
        return replace(
            candidate,
            state_after=self._state.value,
            transition_reason=reason,
            transitioned=self._state.value != before.state,
            state_dwell_samples=1,
            stable_track=False,
            p_step_m=0.0,
            i_step_m=0.0,
            d_step_m=0.0,
            nominal_normal_step_m=0.0,
            actuator_limited_normal_step_m=0.0,
            projected_normal_step_m=0.0,
            executed_normal_step_m=0.0,
            envelope_command_upper_n=candidate.envelope_base_upper_n,
            safe_press_limit_m=0.0,
            projection_active=True,
            actuator_saturation_active=False,
            safety_projection_active=False,
            integral_after_n_s=0.0,
            tangential_motion_permitted=False,
            recovery_reposition_permitted=False,
            cross_track_correction_permitted=False,
            executed_tangential_step_m=0.0,
            committed_progress_after=before.committed_progress,
            in_recovery_episode_after=self._in_recovery_episode,
            recovery_sample_counted_this_step=counted,
            recovery_sample_budget_exhausted=exhausted,
            pre_recovery_highwater_progress_after=(
                self._pre_recovery_highwater_progress
            ),
            verified_return_count=0,
            verified_return_required_progress=None,
            verified_return_progress_gate_passed=False,
            verified_return_completed=False,
            episode_total_recovery_cycles=self._episode_total_recovery_cycles,
            episode_total_recovery_samples=self._episode_total_recovery_samples,
        )

    def command(self, observation: SupervisorInput) -> SupervisorCommand:
        before = self.snapshot()
        self._r3_settling_interlock = False
        candidate = super().command(observation)
        if (
            self._r3_settling_interlock
            and candidate.transition_reason
            == "track_low_static_margin_phase_hold"
        ):
            candidate = replace(
                candidate,
                transition_reason="track_high_envelope_settling_interlock",
            )
        if self._local_reacquire_eligible(
            before=before,
            observation=observation,
            candidate=candidate,
        ):
            return self._apply_local_reacquire(
                before=before,
                observation=observation,
                candidate=candidate,
            )
        return candidate


__all__ = [
    "R3_CONTACT_LOSS_TARGET_FRACTION",
    "R3_LOCAL_REACQUIRE_MIN_TARGET_FRACTION",
    "R3_LOW_STATIC_MARGIN_N",
    "R3_SETTLING_ENVELOPE_ENTER_N",
    "R3_SETTLING_ESTABLISHED_FORCE_FRACTION",
    "R3_SETTLING_OUTWARD_VELOCITY_M_S",
    "R3_SETTLING_RECENT_PRESS_M",
    "V6SystemContactSupervisorR3",
]
