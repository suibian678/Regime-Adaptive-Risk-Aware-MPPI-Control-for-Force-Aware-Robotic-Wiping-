"""Versioned recovery-accounting correction for the V6 system candidate.

Historical S5 and system-v1 sources remain byte-identical to the files bound
by their earlier artifacts.  This module alone owns the corrected semantics:
post-processed non-stable TRACK steps retain inherited open-recovery sample
accounting, reset consecutive return confirmation, and undo only stable-TRACK
effects invalidated by the final command override.
"""

from __future__ import annotations

from dataclasses import replace

from .controller import SupervisorCommand, SupervisorInput, SupervisorSnapshot, SupervisorState
from .controller_system import V6SystemContactSupervisor


class V6SystemContactSupervisorR2(V6SystemContactSupervisor):
    """System candidate v2 with monotone recovery accounting."""

    controller_revision = "V6-system-contact-candidate-v2-accounting"

    def _commit_nonstable_track_override_state(
        self,
        *,
        before: SupervisorSnapshot,
        observation: SupervisorInput,
        executed_normal_step_m: float,
    ) -> None:
        """Keep base accounting and undo invalid stable-TRACK side effects."""

        self._integral_error_n_s = before.integral_error_n_s
        self._committed_progress = before.committed_progress
        self._in_recovery_episode = before.in_recovery_episode
        self._pre_recovery_highwater_progress = before.pre_recovery_highwater_progress
        self._verified_return_count = 0
        self._previous_force_n = before.previous_force_n
        self._previous_previous_force_n = before.previous_previous_force_n
        self._previous_normal_command_m = before.previous_normal_command_m
        self._previous_previous_normal_command_m = before.previous_previous_normal_command_m
        self._update_history(
            force=float(observation.measured_force_n),
            executed_normal_step_m=executed_normal_step_m,
        )

    def _restore_track_hold(
        self,
        *,
        before: SupervisorSnapshot,
        observation: SupervisorInput,
        candidate: SupervisorCommand,
        reason: str,
        cross_track_reposition: bool,
        preserve_candidate_projection: bool = False,
    ) -> SupervisorCommand:
        self._commit_nonstable_track_override_state(
            before=before,
            observation=observation,
            executed_normal_step_m=0.0,
        )
        return replace(
            candidate,
            transition_reason=reason,
            state_dwell_samples=self._state_dwell_samples,
            stable_track=False,
            projected_normal_step_m=(
                candidate.projected_normal_step_m
                if preserve_candidate_projection else 0.0
            ),
            executed_normal_step_m=0.0,
            envelope_command_upper_n=(
                candidate.envelope_command_upper_n
                if preserve_candidate_projection else candidate.envelope_base_upper_n
            ),
            safe_press_limit_m=(
                candidate.safe_press_limit_m if preserve_candidate_projection else 0.0
            ),
            projection_active=True,
            integral_after_n_s=before.integral_error_n_s,
            tangential_motion_permitted=False,
            recovery_reposition_permitted=False,
            cross_track_correction_permitted=cross_track_reposition,
            executed_tangential_step_m=0.0,
            committed_progress_after=before.committed_progress,
            in_recovery_episode_after=before.in_recovery_episode,
            recovery_sample_counted_this_step=candidate.recovery_sample_counted_this_step,
            recovery_sample_budget_exhausted=candidate.recovery_sample_budget_exhausted,
            pre_recovery_highwater_progress_after=before.pre_recovery_highwater_progress,
            verified_return_count=0,
            verified_return_required_progress=None,
            verified_return_progress_gate_passed=False,
            verified_return_completed=False,
            episode_total_recovery_cycles=candidate.episode_total_recovery_cycles,
            episode_total_recovery_samples=candidate.episode_total_recovery_samples,
        )

    def _apply_preemptive_brake(
        self,
        *,
        before: SupervisorSnapshot,
        observation: SupervisorInput,
        candidate: SupervisorCommand,
        drop_n: float,
    ) -> SupervisorCommand:
        brake = -self.config.max_lift_step_m
        self._commit_nonstable_track_override_state(
            before=before,
            observation=observation,
            executed_normal_step_m=brake,
        )
        return replace(
            candidate,
            transition_reason="rapid_drop_preemptive_brake",
            transitioned=False,
            state_after=SupervisorState.TRACK.value,
            state_dwell_samples=self._state_dwell_samples,
            stable_track=False,
            nominal_normal_step_m=brake,
            actuator_limited_normal_step_m=brake,
            projected_normal_step_m=brake,
            executed_normal_step_m=brake,
            envelope_command_upper_n=candidate.envelope_base_upper_n,
            safe_press_limit_m=0.0,
            projection_active=True,
            actuator_saturation_active=False,
            safety_projection_active=True,
            integral_after_n_s=before.integral_error_n_s,
            rebound_drop_n=drop_n,
            rebound_guard_triggered=False,
            tangential_motion_permitted=False,
            recovery_reposition_permitted=False,
            cross_track_correction_permitted=False,
            executed_tangential_step_m=0.0,
            committed_progress_after=before.committed_progress,
            in_recovery_episode_after=before.in_recovery_episode,
            recovery_sample_counted_this_step=candidate.recovery_sample_counted_this_step,
            recovery_sample_budget_exhausted=candidate.recovery_sample_budget_exhausted,
            pre_recovery_highwater_progress_after=before.pre_recovery_highwater_progress,
            verified_return_count=0,
            verified_return_required_progress=None,
            verified_return_progress_gate_passed=False,
            verified_return_completed=False,
            episode_total_recovery_cycles=candidate.episode_total_recovery_cycles,
            episode_total_recovery_samples=candidate.episode_total_recovery_samples,
        )


__all__ = ["V6SystemContactSupervisorR2"]
