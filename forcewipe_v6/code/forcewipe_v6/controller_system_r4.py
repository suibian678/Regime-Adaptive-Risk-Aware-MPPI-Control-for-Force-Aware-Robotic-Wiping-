"""Bounded System R4 controller for code-only development.

R4 is a versioned child of the completed System R3-rev2 physical SANDBOX.
It addresses only three mechanisms observed in that frozen run:

* the low-static-margin settling interlock is also active while a verified
  return is still open;
* contact-preserving marginal geometry drift is handled by a short, bounded
  TRACK realignment before escalating to lift/hover recovery; and
* repeated local force reacquisition inside one open recovery uses a bounded
  subattempt counter instead of consuming one complete episode recovery cycle
  per transient dropout.

The global ten-cycle and 4000-sample recovery ceilings remain unchanged.  This
module is independent of SAPIEN and provides no physical or safety claim.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

from .controller import (
    SupervisorCommand,
    SupervisorInput,
    SupervisorSnapshot,
    SupervisorState,
    V6ControlError,
)
from .controller_system_r2 import V6SystemContactSupervisorR2
from .controller_system_r3 import (
    R3_LOCAL_REACQUIRE_MIN_TARGET_FRACTION,
    R3_LOW_STATIC_MARGIN_N,
    R3_SETTLING_ENVELOPE_ENTER_N,
    R3_SETTLING_ESTABLISHED_FORCE_FRACTION,
    R3_SETTLING_OUTWARD_VELOCITY_M_S,
    R3_SETTLING_RECENT_PRESS_M,
)
from .controller_system_r3_rev2 import V6SystemContactSupervisorR3Rev2


R4_GEOMETRY_REALIGN_MAX_ERROR_M = 0.006
R4_GEOMETRY_REALIGN_MAX_SAMPLES = 20
R4_LOCAL_REACQUIRE_MAX_ATTEMPTS_PER_CYCLE = 4


@dataclass(frozen=True, eq=False)
class R4SupervisorSnapshot(SupervisorSnapshot):
    """Serializable R4 state required for exact replay/resume."""

    local_reacquire_attempts_in_cycle: int
    geometry_realign_samples: int

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SupervisorSnapshot):
            return NotImplemented
        base_names = SupervisorSnapshot.__dataclass_fields__
        if any(getattr(self, name) != getattr(other, name) for name in base_names):
            return False
        if isinstance(other, R4SupervisorSnapshot):
            return bool(
                self.local_reacquire_attempts_in_cycle
                == other.local_reacquire_attempts_in_cycle
                and self.geometry_realign_samples
                == other.geometry_realign_samples
            )
        # The generic causal adapter compares a new controller with the base
        # fresh-reset snapshot.  Only an actually empty R4 ledger is equivalent
        # to that legacy snapshot; nonzero R4 state remains distinguishable.
        return bool(
            self.local_reacquire_attempts_in_cycle == 0
            and self.geometry_realign_samples == 0
        )


@dataclass(frozen=True)
class R4SupervisorCommand(SupervisorCommand):
    """Base command plus the two bounded R4 subattempt ledgers."""

    local_reacquire_attempts_before: int = 0
    local_reacquire_attempts_after: int = 0
    geometry_realign_samples_before: int = 0
    geometry_realign_samples_after: int = 0
    local_reacquire_attempt_counted_this_step: bool = False
    geometry_realign_sample_counted_this_step: bool = False
    full_recovery_escalated_this_step: bool = False
    recovery_tail_settling_interlock: bool = False


class V6SystemContactSupervisorR4(V6SystemContactSupervisorR3Rev2):
    """One non-swept R4 candidate with nested, monotone recovery budgets."""

    controller_revision = (
        "V6-system-contact-candidate-v4-bounded-nested-recovery"
    )

    def reset(self) -> None:
        super().reset()
        self._local_reacquire_attempts_in_cycle = 0
        self._geometry_realign_samples = 0

    def snapshot(self) -> R4SupervisorSnapshot:
        return R4SupervisorSnapshot(
            **asdict(super().snapshot()),
            local_reacquire_attempts_in_cycle=(
                self._local_reacquire_attempts_in_cycle
            ),
            geometry_realign_samples=self._geometry_realign_samples,
        )

    def restore(self, snapshot: SupervisorSnapshot) -> None:
        if not isinstance(snapshot, R4SupervisorSnapshot):
            raise V6ControlError("R4 restore requires an R4 snapshot")
        extra_counts = (
            snapshot.local_reacquire_attempts_in_cycle,
            snapshot.geometry_realign_samples,
        )
        if any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
            for value in extra_counts
        ):
            raise V6ControlError("R4 subattempt counts must be nonnegative integers")
        if (
            snapshot.local_reacquire_attempts_in_cycle
            > R4_LOCAL_REACQUIRE_MAX_ATTEMPTS_PER_CYCLE
        ):
            raise V6ControlError("R4 local-reacquire subattempt budget is exceeded")
        if snapshot.geometry_realign_samples > R4_GEOMETRY_REALIGN_MAX_SAMPLES:
            raise V6ControlError("R4 geometry-realignment budget is exceeded")
        if not snapshot.in_recovery_episode and any(extra_counts):
            raise V6ControlError("R4 subattempt state requires an open recovery")
        if (
            snapshot.geometry_realign_samples
            and snapshot.state != SupervisorState.TRACK.value
        ):
            raise V6ControlError("R4 geometry-realignment state is outside TRACK")
        super().restore(snapshot)
        self._local_reacquire_attempts_in_cycle = (
            snapshot.local_reacquire_attempts_in_cycle
        )
        self._geometry_realign_samples = snapshot.geometry_realign_samples

    def _settling_interlock_eligible(
        self,
        *,
        before: R4SupervisorSnapshot,
        observation: SupervisorInput,
        candidate: SupervisorCommand,
    ) -> bool:
        """Retain the R3 interlock and extend it through verified return."""

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
        return bool(
            before.state == SupervisorState.TRACK.value
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

    def _geometry_realign_eligible(
        self,
        *,
        before: R4SupervisorSnapshot,
        observation: SupervisorInput,
        candidate: SupervisorCommand,
    ) -> bool:
        return bool(
            before.state == SupervisorState.TRACK.value
            and candidate.state_after == SupervisorState.RECOVERY_LIFT.value
            and candidate.transition_reason == "track_geometry_loss_to_recovery_lift"
            and observation.contact_observed
            and observation.measured_force_n
            >= self._contact_verify_threshold(observation.target_force_n)
            and self.config.track_geometric_tolerance_m
            < observation.geometric_track_error_m
            <= R4_GEOMETRY_REALIGN_MAX_ERROR_M
            and not candidate.headroom_triggered
            and not candidate.rebound_guard_triggered
        )

    def _local_reacquire_eligible(
        self,
        *,
        before: R4SupervisorSnapshot,
        observation: SupervisorInput,
        candidate: SupervisorCommand,
    ) -> bool:
        minimum_force = max(
            self.config.contact_threshold_n,
            R3_LOCAL_REACQUIRE_MIN_TARGET_FRACTION
            * observation.target_force_n,
        )
        return bool(
            before.state == SupervisorState.TRACK.value
            and candidate.state_after == SupervisorState.RECOVERY_LIFT.value
            and candidate.transition_reason == "track_contact_loss_to_recovery_lift"
            and observation.contact_observed
            and observation.measured_force_n > minimum_force
            and observation.geometric_track_error_m
            <= self.config.track_geometric_tolerance_m
            and not candidate.headroom_triggered
            and not candidate.rebound_guard_triggered
        )

    def _annotate(
        self,
        command: SupervisorCommand,
        *,
        before: R4SupervisorSnapshot,
        local_counted: bool = False,
        geometry_counted: bool = False,
        escalated: bool = False,
        tail_interlock: bool = False,
    ) -> R4SupervisorCommand:
        return R4SupervisorCommand(
            **asdict(command),
            local_reacquire_attempts_before=(
                before.local_reacquire_attempts_in_cycle
            ),
            local_reacquire_attempts_after=(
                self._local_reacquire_attempts_in_cycle
            ),
            geometry_realign_samples_before=before.geometry_realign_samples,
            geometry_realign_samples_after=self._geometry_realign_samples,
            local_reacquire_attempt_counted_this_step=local_counted,
            geometry_realign_sample_counted_this_step=geometry_counted,
            full_recovery_escalated_this_step=escalated,
            recovery_tail_settling_interlock=tail_interlock,
        )

    def _zero_authority_override(
        self,
        *,
        before: R4SupervisorSnapshot,
        observation: SupervisorInput,
        candidate: SupervisorCommand,
        reason: str,
        state: SupervisorState,
        normal_step_m: float,
        cross_track: bool,
        counted: bool,
        exhausted: bool,
    ) -> SupervisorCommand:
        integral_after = (
            before.integral_error_n_s
            if state is SupervisorState.TRACK else 0.0
        )
        self._integral_error_n_s = integral_after
        self._state = state
        self._state_dwell_samples = (
            before.state_dwell_samples + 1
            if state.value == before.state else 1
        )
        self._verified_return_count = 0
        self._committed_progress = before.committed_progress
        self._update_history(
            force=float(observation.measured_force_n),
            executed_normal_step_m=normal_step_m,
        )
        return replace(
            candidate,
            state_after=self._state.value,
            transition_reason=reason,
            transitioned=self._state.value != before.state,
            state_dwell_samples=self._state_dwell_samples,
            stable_track=False,
            p_step_m=0.0,
            i_step_m=0.0,
            d_step_m=0.0,
            nominal_normal_step_m=normal_step_m,
            actuator_limited_normal_step_m=normal_step_m,
            projected_normal_step_m=normal_step_m,
            executed_normal_step_m=normal_step_m,
            envelope_command_upper_n=candidate.envelope_base_upper_n,
            safe_press_limit_m=0.0,
            projection_active=True,
            actuator_saturation_active=False,
            safety_projection_active=normal_step_m < 0.0,
            integral_after_n_s=integral_after,
            tangential_motion_permitted=False,
            recovery_reposition_permitted=False,
            cross_track_correction_permitted=cross_track,
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

    def _apply_geometry_realign(
        self,
        *,
        before: R4SupervisorSnapshot,
        observation: SupervisorInput,
        candidate: SupervisorCommand,
    ) -> tuple[SupervisorCommand, bool]:
        self.restore(before)
        if not self._in_recovery_episode and not self._start_recovery_cycle():
            counted, exhausted = self._account_open_recovery_sample(
                recovery_active_this_step=False
            )
            return self._zero_authority_override(
                before=before,
                observation=observation,
                candidate=candidate,
                reason="episode_recovery_cycle_budget_exhausted",
                state=SupervisorState.SAFE_HOLD,
                normal_step_m=0.0,
                cross_track=False,
                counted=counted,
                exhausted=exhausted,
            ), True

        next_count = self._geometry_realign_samples + 1
        escalated = next_count > R4_GEOMETRY_REALIGN_MAX_SAMPLES
        if escalated:
            self._geometry_realign_samples = 0
            self._local_reacquire_attempts_in_cycle = 0
            if before.in_recovery_episode and not self._start_recovery_cycle():
                state = SupervisorState.SAFE_HOLD
                reason = "episode_recovery_cycle_budget_exhausted"
                normal = 0.0
            else:
                state = SupervisorState.RECOVERY_LIFT
                reason = "geometry_realign_budget_to_full_recovery"
                normal = -self.config.recovery_lift_step_m
            cross_track = False
        else:
            self._geometry_realign_samples = next_count
            state = SupervisorState.TRACK
            # Reuse the already registered S5 executor request type.  R4
            # realignment remains unambiguous in evidence because geometry is
            # not TRACK-ready and the recovery episode is open.
            reason = "track_cross_track_correction_projection"
            normal = 0.0
            cross_track = True

        counted, exhausted = self._account_open_recovery_sample(
            recovery_active_this_step=(
                self._in_recovery_episode
                and before.state != SupervisorState.SAFE_HOLD.value
            )
        )
        if exhausted:
            state = SupervisorState.SAFE_HOLD
            reason = "episode_recovery_sample_budget_exhausted"
            normal = 0.0
            cross_track = False
        return self._zero_authority_override(
            before=before,
            observation=observation,
            candidate=candidate,
            reason=reason,
            state=state,
            normal_step_m=normal,
            cross_track=cross_track,
            counted=counted,
            exhausted=exhausted,
        ), escalated

    def _apply_local_reacquire(
        self,
        *,
        before: R4SupervisorSnapshot,
        observation: SupervisorInput,
        candidate: SupervisorCommand,
    ) -> tuple[SupervisorCommand, bool]:
        self.restore(before)
        if not self._in_recovery_episode:
            if not self._start_recovery_cycle():
                counted, exhausted = self._account_open_recovery_sample(
                    recovery_active_this_step=False
                )
                return self._zero_authority_override(
                    before=before,
                    observation=observation,
                    candidate=candidate,
                    reason="episode_recovery_cycle_budget_exhausted",
                    state=SupervisorState.SAFE_HOLD,
                    normal_step_m=0.0,
                    cross_track=False,
                    counted=counted,
                    exhausted=exhausted,
                ), True
            self._local_reacquire_attempts_in_cycle = 0

        next_attempt = self._local_reacquire_attempts_in_cycle + 1
        escalated = (
            next_attempt > R4_LOCAL_REACQUIRE_MAX_ATTEMPTS_PER_CYCLE
        )
        self._geometry_realign_samples = 0
        if escalated:
            self._local_reacquire_attempts_in_cycle = 0
            if before.in_recovery_episode and not self._start_recovery_cycle():
                state = SupervisorState.SAFE_HOLD
                reason = "episode_recovery_cycle_budget_exhausted"
                normal = 0.0
            else:
                state = SupervisorState.RECOVERY_LIFT
                reason = "local_reacquire_budget_to_full_recovery"
                normal = -self.config.recovery_lift_step_m
        else:
            self._local_reacquire_attempts_in_cycle = next_attempt
            state = SupervisorState.RECOVERY_ACQUIRE
            reason = "track_contact_degradation_to_bounded_local_acquire"
            normal = 0.0

        counted, exhausted = self._account_open_recovery_sample(
            recovery_active_this_step=(
                self._in_recovery_episode
                and before.state != SupervisorState.SAFE_HOLD.value
            )
        )
        if exhausted:
            state = SupervisorState.SAFE_HOLD
            reason = "episode_recovery_sample_budget_exhausted"
            normal = 0.0
        return self._zero_authority_override(
            before=before,
            observation=observation,
            candidate=candidate,
            reason=reason,
            state=state,
            normal_step_m=normal,
            cross_track=False,
            counted=counted,
            exhausted=exhausted,
        ), escalated

    def _normalize_subattempt_state(self, command: SupervisorCommand) -> None:
        if command.verified_return_completed or not command.in_recovery_episode_after:
            self._local_reacquire_attempts_in_cycle = 0
            self._geometry_realign_samples = 0
            return
        if command.state_after in {
            SupervisorState.RECOVERY_LIFT.value,
            SupervisorState.REBOUND_GUARD.value,
            SupervisorState.HEADROOM.value,
            SupervisorState.SAFE_HOLD.value,
        }:
            self._local_reacquire_attempts_in_cycle = 0
            self._geometry_realign_samples = 0
        elif command.geometric_track_ready:
            self._geometry_realign_samples = 0

    def command(self, observation: SupervisorInput) -> R4SupervisorCommand:
        before = self.snapshot()

        # Skip the R3/R3-rev2 postprocessors and obtain the corrected System-R2
        # command.  Dynamic dispatch still retains the frozen S5/S4/core safety
        # pipeline and the R2 recovery-accounting repair.
        candidate = V6SystemContactSupervisorR2.command(self, observation)

        # Closed-recovery TRACK retains the already-frozen R3 naming and
        # behavior.  The inherited S4 hold has already committed the zero
        # command in this branch.
        if (
            not before.in_recovery_episode
            and candidate.transition_reason
            in {
                "track_low_static_margin_phase_hold",
                "track_outward_velocity_phase_hold",
            }
        ):
            return self._annotate(
                replace(
                    candidate,
                    transition_reason="track_high_envelope_settling_interlock",
                ),
                before=before,
                tail_interlock=True,
            )

        if self._settling_interlock_eligible(
            before=before,
            observation=observation,
            candidate=candidate,
        ):
            reason = (
                "track_recovery_tail_settling_interlock"
                if before.in_recovery_episode
                else "track_high_envelope_settling_interlock"
            )
            held = self._restore_track_hold(
                before=before,
                observation=observation,
                candidate=candidate,
                reason=reason,
                cross_track_reposition=False,
            )
            return self._annotate(
                held,
                before=before,
                tail_interlock=True,
            )

        if self._geometry_realign_eligible(
            before=before,
            observation=observation,
            candidate=candidate,
        ):
            command, escalated = self._apply_geometry_realign(
                before=before,
                observation=observation,
                candidate=candidate,
            )
            return self._annotate(
                command,
                before=before,
                geometry_counted=True,
                escalated=escalated,
            )

        if self._local_reacquire_eligible(
            before=before,
            observation=observation,
            candidate=candidate,
        ):
            command, escalated = self._apply_local_reacquire(
                before=before,
                observation=observation,
                candidate=candidate,
            )
            return self._annotate(
                command,
                before=before,
                local_counted=True,
                escalated=escalated,
            )

        self._normalize_subattempt_state(candidate)
        return self._annotate(candidate, before=before)


__all__ = [
    "R4_GEOMETRY_REALIGN_MAX_ERROR_M",
    "R4_GEOMETRY_REALIGN_MAX_SAMPLES",
    "R4_LOCAL_REACQUIRE_MAX_ATTEMPTS_PER_CYCLE",
    "R4SupervisorCommand",
    "R4SupervisorSnapshot",
    "V6SystemContactSupervisorR4",
]
