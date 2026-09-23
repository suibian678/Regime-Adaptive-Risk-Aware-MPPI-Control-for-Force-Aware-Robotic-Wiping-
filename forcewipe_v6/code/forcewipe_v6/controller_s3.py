"""V6-S3 code-only liveness revision of the unified supervisor.

V6-S2 is retained unchanged for provenance.  S3 changes no force, geometry,
actuator, recovery-budget, or safety-envelope threshold.  It closes two
deadlocks observed in the frozen S2 physical traces:

* contact verification cannot grant TRACK while path geometry is invalid;
* normal-force regulation may converge inside TRACK before the stricter
  force band grants task-tangential and progress authority.

The controller remains independent of SAPIEN.  A passing code-only audit is
not evidence of closed-loop safety, tracking, recovery, or liveness.
"""

from __future__ import annotations

from .controller import (
    INWARD_CAPABLE_STATES,
    SupervisorCommand,
    SupervisorInput,
    SupervisorState,
    UnifiedCausalForceRecoverySupervisor,
    V6ControlError,
)


class V6S3LivenessSupervisor(UnifiedCausalForceRecoverySupervisor):
    """Single S3 mechanism candidate with unchanged S2 configuration."""

    controller_revision = "V6-S3-code-only-liveness-v1"

    def command(self, observation: SupervisorInput) -> SupervisorCommand:
        self._validate_input(observation)
        cfg = self.config
        before = self._state
        dwell_before = self._state_dwell_samples
        integral_before = self._integral_error_n_s
        committed_before = self._committed_progress
        previous_force = self._previous_force_n
        previous_previous_force = self._previous_previous_force_n
        previous_command = self._previous_normal_command_m
        previous_previous_command = self._previous_previous_normal_command_m
        in_recovery_before = self._in_recovery_episode
        highwater_before = self._pre_recovery_highwater_progress
        verified_return_count_before = self._verified_return_count
        force = float(observation.measured_force_n)
        target = float(observation.target_force_n)
        force_rate = self._force_rate(force)
        rebound, force_drop = self._rebound_guard_eligible(observation)
        sampled_violation = force > cfg.force_limit_n
        force_track_ready = bool(
            observation.contact_observed
            and abs(force - target) <= cfg.track_force_band_fraction * target
        )
        geometric_track_ready = bool(
            observation.geometric_track_error_m <= cfg.track_geometric_tolerance_m
        )
        force_regulation_ready = bool(
            observation.contact_observed
            and force > self._contact_loss_threshold(target)
            and geometric_track_ready
        )

        reason = "state_hold"
        p_step = i_step = d_step = nominal = 0.0
        actuator_limited = projected = executed = 0.0
        envelope_base = self._envelope_base(
            force=force,
            force_rate=force_rate,
            outward_velocity=observation.normal_velocity_outward_m_s,
        )
        envelope_command = envelope_base
        safe_press_limit = cfg.track_max_press_step_m
        projection_active = False
        actuator_saturation_active = False
        safety_projection_active = False
        rebound_triggered = False
        headroom_triggered = False
        recovery_reposition = False
        verified_completed = False
        verified_return_required_progress: float | None = None
        verified_return_progress_gate_passed = False
        recovery_sample_counted = False
        recovery_sample_budget_exhausted = False

        measured_headroom = force >= cfg.headroom_enter_n
        predicted_headroom = envelope_base >= cfg.safety_projection_bound_n
        if self._state is not SupervisorState.SAFE_HOLD and (
            measured_headroom or predicted_headroom
        ):
            headroom_triggered = True
            self._integral_error_n_s = 0.0
            recovery_opened = self._in_recovery_episode or self._start_recovery_cycle()
            if recovery_opened and self._state is not SupervisorState.SAFE_HOLD:
                self._transition(SupervisorState.HEADROOM)
                self._headroom_exit_count = 0
                reason = (
                    "measured_headroom_preemption"
                    if measured_headroom else "predicted_headroom_preemption"
                )
            else:
                reason = "episode_recovery_cycle_budget_exhausted"

        if (
            not headroom_triggered
            and self._state in INWARD_CAPABLE_STATES
            and rebound
        ):
            rebound_triggered = True
            self._integral_error_n_s = 0.0
            if self._start_recovery_cycle():
                self._transition(SupervisorState.REBOUND_GUARD)
                reason = "causal_collapse_rebound_guard"
            else:
                reason = "episode_recovery_cycle_budget_exhausted"

        if self._state is SupervisorState.SAFE_HOLD:
            if before is SupervisorState.SAFE_HOLD:
                reason = "safe_hold_terminal"

        elif self._state is SupervisorState.HEADROOM:
            nominal = -cfg.max_lift_step_m
            if headroom_triggered:
                self._headroom_exit_count = 0
                if before is SupervisorState.HEADROOM:
                    reason = (
                        "measured_headroom_outward_hold"
                        if measured_headroom else "predicted_headroom_outward_hold"
                    )
            elif force <= cfg.headroom_exit_n:
                self._headroom_exit_count += 1
                reason = "headroom_exit_confirmation"
                if self._headroom_exit_count >= cfg.headroom_exit_confirm_samples:
                    self._transition(SupervisorState.CONTACT_VERIFY)
                    self._in_recovery_episode = True
                    if self._pre_recovery_highwater_progress is None:
                        self._pre_recovery_highwater_progress = self._committed_progress
                    nominal = 0.0
                    reason = "headroom_exit_to_contact_verify"
            else:
                self._headroom_exit_count = 0
                reason = "headroom_outward_hold"

        elif self._state is SupervisorState.REBOUND_GUARD:
            self._integral_error_n_s = 0.0
            nominal = -cfg.max_lift_step_m
            if before is SupervisorState.REBOUND_GUARD:
                reason = "rebound_guard_outward_hold"
            rebound_dwell_before = (
                dwell_before if before is SupervisorState.REBOUND_GUARD else 0
            )
            if rebound_dwell_before + 1 >= cfg.rebound_guard_samples:
                self._transition(SupervisorState.RECOVERY_LIFT)
                reason = "rebound_guard_complete_to_recovery_lift"

        elif self._state is SupervisorState.RECOVERY_LIFT:
            self._integral_error_n_s = 0.0
            released = force <= cfg.recovery_release_force_n
            high_enough = observation.recovery_clearance_m >= (
                cfg.recovery_hover_clearance_m - cfg.recovery_hover_tolerance_m
            )
            bounded_exit = dwell_before + 1 >= cfg.recovery_lift_max_samples
            if (released and high_enough) or bounded_exit:
                self._transition(SupervisorState.RECOVERY_HOVER)
                reason = (
                    "recovery_lift_complete_to_hover"
                    if released and high_enough
                    else "recovery_lift_bounded_exit_to_hover"
                )
            else:
                nominal = -cfg.recovery_lift_step_m
                reason = "recovery_lift_outward"

        elif self._state is SupervisorState.RECOVERY_HOVER:
            self._integral_error_n_s = 0.0
            recovery_reposition = True
            at_hover = (
                observation.recovery_hover_pose_error_m
                <= cfg.recovery_hover_tolerance_m
            )
            bounded_exit = dwell_before + 1 >= cfg.recovery_hover_max_samples
            if at_hover or bounded_exit:
                self._transition(SupervisorState.RECOVERY_ACQUIRE)
                recovery_reposition = False
                reason = (
                    "recovery_hover_complete_to_acquire"
                    if at_hover else "recovery_hover_bounded_exit_to_acquire"
                )
            else:
                reason = "recovery_hover_reposition"

        elif self._state is SupervisorState.RECOVERY_ACQUIRE:
            self._integral_error_n_s = 0.0
            if (
                observation.contact_observed
                and force >= self._contact_verify_threshold(target)
            ):
                self._transition(SupervisorState.CONTACT_VERIFY)
                reason = "recovery_acquire_to_contact_verify"
            elif dwell_before + 1 >= cfg.recovery_acquire_max_samples:
                if self._start_recovery_cycle():
                    self._transition(SupervisorState.RECOVERY_LIFT)
                    reason = "recovery_acquire_timeout_to_new_cycle"
                else:
                    reason = "episode_recovery_cycle_budget_exhausted"
            else:
                nominal = min(
                    cfg.recovery_acquire_max_step_m,
                    cfg.recovery_acquire_initial_step_m
                    + cfg.recovery_acquire_increment_m * dwell_before,
                )
                reason = "recovery_acquire_bounded_ramp"

        elif self._state is SupervisorState.APPROACH:
            self._integral_error_n_s = 0.0
            if (
                observation.contact_observed
                and force >= self._contact_verify_threshold(target)
            ):
                self._transition(SupervisorState.CONTACT_VERIFY)
                reason = "approach_to_contact_verify"
            else:
                nominal = min(
                    cfg.approach_max_step_m,
                    cfg.approach_initial_step_m
                    + cfg.approach_increment_m * dwell_before,
                )
                reason = "approach_bounded_ramp"

        elif self._state is SupervisorState.CONTACT_VERIFY:
            self._integral_error_n_s = 0.0
            verified = bool(
                observation.contact_observed
                and force >= self._contact_verify_threshold(target)
            )
            if verified:
                self._contact_verify_count += 1
                reason = "contact_verify_confirmation"
                if self._contact_verify_count >= cfg.contact_verify_samples:
                    if geometric_track_ready:
                        self._transition(SupervisorState.TRACK)
                        reason = "contact_and_geometry_verify_complete_to_track"
                    elif self._start_recovery_cycle():
                        self._transition(SupervisorState.RECOVERY_LIFT)
                        nominal = -cfg.recovery_lift_step_m
                        reason = "contact_verified_geometry_invalid_to_recovery_lift"
                    else:
                        reason = "episode_recovery_cycle_budget_exhausted"
            else:
                self._contact_verify_count = 0
                destination = (
                    SupervisorState.RECOVERY_ACQUIRE
                    if self._in_recovery_episode else SupervisorState.APPROACH
                )
                self._transition(destination)
                reason = f"contact_verify_lost_to_{destination.value.lower()}"

        elif self._state is SupervisorState.TRACK:
            if (
                not observation.contact_observed
                or force <= self._contact_loss_threshold(target)
            ):
                self._integral_error_n_s = 0.0
                if self._start_recovery_cycle():
                    self._transition(SupervisorState.RECOVERY_LIFT)
                    nominal = -cfg.recovery_lift_step_m
                    reason = "track_contact_loss_to_recovery_lift"
                else:
                    reason = "episode_recovery_cycle_budget_exhausted"
            elif not geometric_track_ready:
                self._integral_error_n_s = 0.0
                if self._start_recovery_cycle():
                    self._transition(SupervisorState.RECOVERY_LIFT)
                    nominal = -cfg.recovery_lift_step_m
                    reason = "track_geometry_loss_to_recovery_lift"
                else:
                    reason = "episode_recovery_cycle_budget_exhausted"
            else:
                error = target - force
                candidate_integral = max(
                    -cfg.integral_clip_n_s,
                    min(
                        cfg.integral_clip_n_s,
                        self._integral_error_n_s + error * cfg.dt_s,
                    ),
                )
                p_step = cfg.kp_m_per_n * error
                i_step = cfg.ki_m_per_n_s * candidate_integral
                d_step = -cfg.kd_m_s_per_n * max(force_rate, 0.0)
                nominal = p_step + i_step + d_step
                reason = "track_raw_pi_candidate"

        state_before_recovery_accounting = self._state
        recovery_active_this_step = bool(
            self._in_recovery_episode and before is not SupervisorState.SAFE_HOLD
        )
        recovery_sample_counted, recovery_sample_budget_exhausted = (
            self._account_open_recovery_sample(
                recovery_active_this_step=recovery_active_this_step,
            )
        )
        if recovery_sample_budget_exhausted:
            self._integral_error_n_s = 0.0
            p_step = i_step = d_step = nominal = 0.0
            recovery_reposition = False
            self._verified_return_count = 0
            if state_before_recovery_accounting is not SupervisorState.SAFE_HOLD:
                reason = "episode_recovery_sample_budget_exhausted"

        actuator_limited = max(
            -cfg.max_lift_step_m,
            min(cfg.track_max_press_step_m, nominal),
        )
        actuator_saturation_active = abs(actuator_limited - nominal) > 1e-15
        projected = actuator_limited
        if self._state in INWARD_CAPABLE_STATES and actuator_limited > 0.0:
            gain = cfg.envelope_candidate_press_gain_n_per_m
            safe_press_limit = min(
                cfg.track_max_press_step_m,
                max(0.0, (cfg.safety_projection_bound_n - envelope_base) / gain),
            )
            projected = min(actuator_limited, safe_press_limit)
            safety_projection_active = projected < actuator_limited - 1e-15
            if safety_projection_active:
                reason = f"{self._state.value.lower()}_candidate_safety_projected"
        executed = projected
        envelope_command = envelope_base + (
            cfg.envelope_candidate_press_gain_n_per_m * max(executed, 0.0)
        )
        projection_active = bool(
            actuator_saturation_active or safety_projection_active
        )

        transitioned = before is not self._state
        stable_track = bool(
            before is SupervisorState.TRACK
            and self._state is SupervisorState.TRACK
            and not transitioned
            and not projection_active
            and not rebound_triggered
            and not headroom_triggered
            and force_track_ready
            and geometric_track_ready
        )
        force_regulation_track = bool(
            before is SupervisorState.TRACK
            and self._state is SupervisorState.TRACK
            and not transitioned
            and not projection_active
            and not rebound_triggered
            and not headroom_triggered
            and force_regulation_ready
        )

        if before is SupervisorState.TRACK and self._state is SupervisorState.TRACK:
            if "candidate_integral" in locals():
                if force_regulation_track and abs(executed - nominal) <= 1e-15:
                    self._integral_error_n_s = candidate_integral
                    reason = (
                        "track_stable_pi"
                        if stable_track else "track_force_regulation_pi_task_frozen"
                    )
                else:
                    self._integral_error_n_s = integral_before
                    reason = (
                        "track_final_command_limited"
                        if projection_active else "track_regulation_ineligible_integrator_frozen"
                    )
        tangential_permitted = stable_track
        executed_tangent = (
            float(observation.requested_tangential_step_m)
            if tangential_permitted else 0.0
        )
        if tangential_permitted:
            self._committed_progress = max(
                self._committed_progress,
                float(observation.instantaneous_progress),
            )

        if stable_track and self._in_recovery_episode:
            self._verified_return_count += 1
            if self._pre_recovery_highwater_progress is None:
                raise V6ControlError("open recovery episode lacks a high-water mark")
            highwater = self._pre_recovery_highwater_progress
            verified_return_required_progress = (
                highwater + cfg.verified_return_progress_fraction
            )
            verified_return_progress_gate_passed = bool(
                self._committed_progress + 1e-12
                >= verified_return_required_progress
            )
            if (
                self._verified_return_count >= cfg.verified_return_samples
                and verified_return_progress_gate_passed
            ):
                self._in_recovery_episode = False
                self._verified_return_count = 0
                self._pre_recovery_highwater_progress = None
                verified_completed = True
        elif self._in_recovery_episode:
            self._verified_return_count = 0

        if self._state is before:
            self._state_dwell_samples += 1
        else:
            self._state_dwell_samples = 1

        command = SupervisorCommand(
            state_before=before.value,
            state_after=self._state.value,
            transition_reason=reason,
            transitioned=transitioned,
            state_dwell_samples=self._state_dwell_samples,
            measured_force_n=force,
            target_force_n=target,
            contact_observed=observation.contact_observed,
            previous_force_n=previous_force,
            previous_previous_force_n=previous_previous_force,
            force_rate_n_s=force_rate,
            normal_velocity_outward_m_s=float(observation.normal_velocity_outward_m_s),
            previous_normal_command_m=previous_command,
            previous_previous_normal_command_m=previous_previous_command,
            geometric_track_error_m=float(observation.geometric_track_error_m),
            recovery_hover_pose_error_m=float(observation.recovery_hover_pose_error_m),
            recovery_clearance_m=float(observation.recovery_clearance_m),
            force_track_ready=force_track_ready,
            geometric_track_ready=geometric_track_ready,
            stable_track=stable_track,
            p_step_m=float(p_step),
            i_step_m=float(i_step),
            d_step_m=float(d_step),
            nominal_normal_step_m=float(nominal),
            actuator_limited_normal_step_m=float(actuator_limited),
            projected_normal_step_m=float(projected),
            executed_normal_step_m=float(executed),
            envelope_base_upper_n=float(envelope_base),
            envelope_command_upper_n=float(envelope_command),
            safe_press_limit_m=float(safe_press_limit),
            projection_active=projection_active,
            actuator_saturation_active=actuator_saturation_active,
            safety_projection_active=safety_projection_active,
            integral_before_n_s=float(integral_before),
            integral_after_n_s=float(self._integral_error_n_s),
            rebound_drop_n=float(force_drop),
            rebound_guard_triggered=rebound_triggered,
            headroom_triggered=headroom_triggered,
            tangential_motion_permitted=tangential_permitted,
            recovery_reposition_permitted=recovery_reposition,
            requested_tangential_step_m=float(observation.requested_tangential_step_m),
            executed_tangential_step_m=float(executed_tangent),
            instantaneous_progress=float(observation.instantaneous_progress),
            committed_progress_before=float(committed_before),
            committed_progress_after=float(self._committed_progress),
            in_recovery_episode_before=in_recovery_before,
            in_recovery_episode_after=self._in_recovery_episode,
            recovery_sample_counted_this_step=recovery_sample_counted,
            recovery_sample_budget_exhausted=recovery_sample_budget_exhausted,
            pre_recovery_highwater_progress_before=highwater_before,
            pre_recovery_highwater_progress_after=self._pre_recovery_highwater_progress,
            verified_return_count_before=verified_return_count_before,
            verified_return_count=int(self._verified_return_count),
            verified_return_required_progress=verified_return_required_progress,
            verified_return_progress_gate_passed=verified_return_progress_gate_passed,
            verified_return_completed=verified_completed,
            episode_total_recovery_cycles=int(self._episode_total_recovery_cycles),
            episode_total_recovery_samples=int(self._episode_total_recovery_samples),
            sampled_force_violation_observed=sampled_violation,
        )
        self._update_history(force=force, executed_normal_step_m=executed)
        return command


__all__ = ["V6S3LivenessSupervisor"]
