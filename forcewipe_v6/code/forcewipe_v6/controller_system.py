"""Single system-level V6 contact candidate for mutable SANDBOX development.

This candidate addresses two mechanisms observed in the frozen S5-r2 rev2
development run: marginal path-frame errors repeatedly escalated to full
recovery, and a low-margin force collapse was recognized one sample too late.
No physical, qualification, CAL, TRAIN, or TEST claim is implied.
"""

from __future__ import annotations

from dataclasses import replace

from .controller import SupervisorCommand, SupervisorConfig, SupervisorInput, SupervisorSnapshot, SupervisorState
from .controller_s5 import V6S5PathFrameSupervisor


SYSTEM_GEOMETRIC_TRACK_TOLERANCE_M = 0.0045
SYSTEM_CROSS_TRACK_CORRECTION_ENTER_M = 0.0035
SYSTEM_CONTACT_VERIFY_TARGET_FRACTION = 0.55
SYSTEM_CONTACT_LOSS_TARGET_FRACTION = 0.45
SYSTEM_MAXIMUM_EPISODE_RECOVERY_SAMPLES = 4000
SYSTEM_RAPID_DROP_RATE_N_S = -60.0
SYSTEM_RAPID_DROP_MIN_N = 0.50
SYSTEM_RAPID_DROP_PRIOR_FORCE_FRACTION = 0.85
SYSTEM_OUTWARD_PHASE_HOLD_VELOCITY_M_S = 0.040


def v6_system_supervisor_config(base: SupervisorConfig | None = None) -> SupervisorConfig:
    source = SupervisorConfig() if base is None else base
    return replace(
        source,
        track_geometric_tolerance_m=SYSTEM_GEOMETRIC_TRACK_TOLERANCE_M,
        contact_verify_target_fraction=SYSTEM_CONTACT_VERIFY_TARGET_FRACTION,
        contact_loss_target_fraction=SYSTEM_CONTACT_LOSS_TARGET_FRACTION,
        maximum_episode_recovery_samples=SYSTEM_MAXIMUM_EPISODE_RECOVERY_SAMPLES,
    )


class V6SystemContactSupervisor(V6S5PathFrameSupervisor):
    """One non-swept system candidate with causal preemption and bounded recovery."""

    controller_revision = "V6-system-contact-candidate-v1"

    def __init__(self, config: SupervisorConfig | None = None) -> None:
        super().__init__(v6_system_supervisor_config(config))
        self._system_velocity_phase_hold = False

    def _rapid_drop_eligible(self, observation: SupervisorInput) -> tuple[bool, float]:
        if self._previous_force_n is None:
            return False, 0.0
        target = float(observation.target_force_n)
        force = float(observation.measured_force_n)
        drop = max(float(self._previous_force_n) - force, 0.0)
        rate = max(
            -self.config.force_rate_clip_n_s,
            min(self.config.force_rate_clip_n_s, (force - self._previous_force_n) / self.config.dt_s),
        )
        low_static_margin = self.config.force_limit_n - target <= 3.0 + 1e-12
        established = self._previous_force_n >= SYSTEM_RAPID_DROP_PRIOR_FORCE_FRACTION * target
        before_contact_loss = force > self._contact_loss_threshold(target)
        return bool(
            low_static_margin
            and established
            and before_contact_loss
            and drop >= SYSTEM_RAPID_DROP_MIN_N
            and rate <= SYSTEM_RAPID_DROP_RATE_N_S
        ), float(drop)

    def _apply_preemptive_brake(
        self,
        *,
        before: SupervisorSnapshot,
        observation: SupervisorInput,
        candidate: SupervisorCommand,
        drop_n: float,
    ) -> SupervisorCommand:
        """Commit one causal outward TRACK brake without opening a recovery cycle."""

        brake = -self.config.max_lift_step_m
        self.restore(before)
        self._state_dwell_samples = before.state_dwell_samples + 1
        self._update_history(force=float(observation.measured_force_n), executed_normal_step_m=brake)
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
            recovery_sample_counted_this_step=False,
            recovery_sample_budget_exhausted=False,
            pre_recovery_highwater_progress_after=before.pre_recovery_highwater_progress,
            verified_return_count=before.verified_return_count,
            verified_return_required_progress=None,
            verified_return_progress_gate_passed=False,
            verified_return_completed=False,
            episode_total_recovery_cycles=before.episode_total_recovery_cycles,
            episode_total_recovery_samples=before.episode_total_recovery_samples,
        )

    def _phase_hold_eligible(
        self,
        *,
        before: SupervisorSnapshot,
        observation: SupervisorInput,
        candidate: SupervisorCommand,
    ) -> bool:
        if super()._phase_hold_eligible(before=before, observation=observation, candidate=candidate):
            self._system_velocity_phase_hold = False
            return True
        low_static_margin = self.config.force_limit_n - observation.target_force_n <= 3.0 + 1e-12
        established = observation.measured_force_n >= 0.85 * observation.target_force_n
        recent_press = max(before.previous_normal_command_m, before.previous_previous_normal_command_m) >= 0.001
        eligible = bool(
            before.state == SupervisorState.TRACK.value
            and not before.in_recovery_episode
            and low_static_margin
            and established
            and recent_press
            and observation.normal_velocity_outward_m_s >= SYSTEM_OUTWARD_PHASE_HOLD_VELOCITY_M_S
            and candidate.state_before == candidate.state_after == SupervisorState.TRACK.value
            and not candidate.transitioned
            and not candidate.headroom_triggered
            and not candidate.rebound_guard_triggered
            and candidate.executed_normal_step_m > 0.0
        )
        self._system_velocity_phase_hold = eligible
        return eligible

    def _cross_track_correction_eligible(
        self,
        *,
        before: SupervisorSnapshot,
        observation: SupervisorInput,
        candidate: SupervisorCommand,
    ) -> bool:
        return bool(
            observation.geometric_track_error_m >= SYSTEM_CROSS_TRACK_CORRECTION_ENTER_M
            and super()._cross_track_correction_eligible(
                before=before, observation=observation, candidate=candidate
            )
        )

    def command(self, observation: SupervisorInput) -> SupervisorCommand:
        before = self.snapshot()
        rapid_drop, rapid_drop_n = self._rapid_drop_eligible(observation)
        self._system_velocity_phase_hold = False
        candidate = super().command(observation)
        if (
            rapid_drop
            and before.state == SupervisorState.TRACK.value
            and candidate.state_before == candidate.state_after == SupervisorState.TRACK.value
            and not candidate.transitioned
            and not candidate.headroom_triggered
            and not candidate.rebound_guard_triggered
        ):
            return self._apply_preemptive_brake(
                before=before,
                observation=observation,
                candidate=candidate,
                drop_n=rapid_drop_n,
            )
        if self._system_velocity_phase_hold and candidate.transition_reason == "track_low_static_margin_phase_hold":
            return replace(candidate, transition_reason="track_outward_velocity_phase_hold")
        return candidate


__all__ = [
    "SYSTEM_CONTACT_VERIFY_TARGET_FRACTION",
    "SYSTEM_CONTACT_LOSS_TARGET_FRACTION",
    "SYSTEM_CROSS_TRACK_CORRECTION_ENTER_M",
    "SYSTEM_GEOMETRIC_TRACK_TOLERANCE_M",
    "SYSTEM_MAXIMUM_EPISODE_RECOVERY_SAMPLES",
    "SYSTEM_OUTWARD_PHASE_HOLD_VELOCITY_M_S",
    "SYSTEM_RAPID_DROP_MIN_N",
    "SYSTEM_RAPID_DROP_PRIOR_FORCE_FRACTION",
    "SYSTEM_RAPID_DROP_RATE_N_S",
    "V6SystemContactSupervisor",
    "v6_system_supervisor_config",
]
