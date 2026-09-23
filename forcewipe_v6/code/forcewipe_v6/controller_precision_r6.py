"""Precision TRACK r6 with bounded contact capture and phase-aware drop hold.

The frozen r5 development run exposed two distinct mechanisms: repeated
high-speed recovery acquisition at 8/12 N, and a 12-N force-limit event one
sample after a maximum outward rapid-drop command was issued while the tool
was already moving outward.  R6 makes one bounded change for each mechanism:

* recovery acquisition is velocity limited and contact verification retains a
  small causal normal-force command instead of dropping immediately to zero;
* a rapid force drop while the tool is already moving outward produces a
  zero-increment pose hold, not another maximum outward displacement.

The changes affect only low-level normal authority.  Tangential task motion,
progress ownership, the 14.5-N internal projection bound, and the 15-N native
audit remain unchanged.  This module is code-only until a repeatable physical
development SANDBOX is explicitly authorized and executed.
"""

from __future__ import annotations

from dataclasses import replace

from .controller import SupervisorCommand, SupervisorInput, SupervisorSnapshot, SupervisorState
from .controller_precision_r5 import V6PrecisionContactSupervisorR5


PRECISION_R6_CAPTURE_MAX_STEP_M = 0.0005
PRECISION_R6_CAPTURE_INWARD_SPEED_HOLD_M_S = -0.040
PRECISION_R6_DROP_OUTWARD_HOLD_SPEED_M_S = 0.020
PRECISION_R6_DROP_MAX_OUTWARD_STEP_M = 0.0010
PRECISION_R6_RECOVERY_REASON = "recovery_acquire_velocity_limited_ramp"
PRECISION_R6_DROP_HOLD_REASON = "rapid_drop_outward_motion_zero_increment_hold"
PRECISION_R6_DROP_BOUNDED_REASON = "rapid_drop_bounded_outward_command"


def _clip(value: float, lower: float, upper: float) -> float:
    return max(float(lower), min(float(upper), float(value)))


class V6PrecisionContactSupervisorR6(V6PrecisionContactSupervisorR5):
    """Single non-swept r6 development candidate."""

    controller_revision = "V6-precision-track-r6-bounded-capture"

    def _commit_normal_override_history(
        self,
        *,
        before: SupervisorSnapshot,
        executed_normal_step_m: float,
    ) -> None:
        self._previous_previous_normal_command_m = before.previous_normal_command_m
        self._previous_normal_command_m = float(executed_normal_step_m)

    def _replace_normal_candidate(
        self,
        *,
        before: SupervisorSnapshot,
        candidate: SupervisorCommand,
        executed: float,
        nominal: float,
        actuator_limited: float,
        reason: str,
        freeze_integral: bool,
    ) -> SupervisorCommand:
        cfg = self.config
        reference = max(
            0.0,
            min(cfg.track_max_press_step_m, before.previous_normal_command_m),
        )
        envelope_command = candidate.envelope_base_upper_n + (
            cfg.envelope_candidate_press_gain_n_per_m
            * max(float(executed) - reference, 0.0)
        )
        if envelope_command > cfg.safety_projection_bound_n + 1e-12:
            raise RuntimeError("r6 normal override exceeds the causal force envelope")
        self._commit_normal_override_history(
            before=before,
            executed_normal_step_m=executed,
        )
        if freeze_integral:
            self._integral_error_n_s = 0.0
        return replace(
            candidate,
            transition_reason=reason,
            stable_track=False,
            p_step_m=nominal,
            i_step_m=0.0,
            d_step_m=0.0,
            nominal_normal_step_m=nominal,
            actuator_limited_normal_step_m=actuator_limited,
            projected_normal_step_m=executed,
            executed_normal_step_m=executed,
            envelope_command_upper_n=envelope_command,
            safe_press_limit_m=max(executed, 0.0),
            projection_active=True,
            actuator_saturation_active=(
                abs(actuator_limited - nominal) > 1e-15
            ),
            safety_projection_active=(
                abs(executed - actuator_limited) > 1e-15
            ),
            integral_after_n_s=(
                0.0 if freeze_integral else candidate.integral_after_n_s
            ),
            tangential_motion_permitted=False,
            recovery_reposition_permitted=False,
            cross_track_correction_permitted=False,
            executed_tangential_step_m=0.0,
            committed_progress_after=before.committed_progress,
        )

    def _apply_preemptive_brake(
        self,
        *,
        before: SupervisorSnapshot,
        observation: SupervisorInput,
        candidate: SupervisorCommand,
        drop_n: float,
    ) -> SupervisorCommand:
        already_outward = bool(
            observation.normal_velocity_outward_m_s
            >= PRECISION_R6_DROP_OUTWARD_HOLD_SPEED_M_S
        )
        executed = (
            0.0
            if already_outward
            else -min(
                self.config.max_lift_step_m,
                PRECISION_R6_DROP_MAX_OUTWARD_STEP_M,
            )
        )
        self._commit_nonstable_track_override_state(
            before=before,
            observation=observation,
            executed_normal_step_m=executed,
        )
        return replace(
            candidate,
            transition_reason=(
                PRECISION_R6_DROP_HOLD_REASON
                if already_outward
                else PRECISION_R6_DROP_BOUNDED_REASON
            ),
            transitioned=False,
            state_after=SupervisorState.TRACK.value,
            state_dwell_samples=self._state_dwell_samples,
            stable_track=False,
            nominal_normal_step_m=executed,
            actuator_limited_normal_step_m=executed,
            projected_normal_step_m=executed,
            executed_normal_step_m=executed,
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

    def _velocity_limited_recovery_acquire(
        self,
        *,
        before: SupervisorSnapshot,
        observation: SupervisorInput,
        candidate: SupervisorCommand,
    ) -> SupervisorCommand:
        nominal = max(0.0, float(candidate.nominal_normal_step_m))
        actuator_limited = min(nominal, PRECISION_R6_CAPTURE_MAX_STEP_M)
        executed = actuator_limited
        if (
            observation.normal_velocity_outward_m_s
            <= PRECISION_R6_CAPTURE_INWARD_SPEED_HOLD_M_S
        ):
            executed = 0.0
        executed = min(executed, max(0.0, float(candidate.safe_press_limit_m)))
        return self._replace_normal_candidate(
            before=before,
            candidate=candidate,
            executed=executed,
            nominal=nominal,
            actuator_limited=actuator_limited,
            reason=PRECISION_R6_RECOVERY_REASON,
            freeze_integral=True,
        )

    def _contact_capture(
        self,
        *,
        before: SupervisorSnapshot,
        observation: SupervisorInput,
        candidate: SupervisorCommand,
    ) -> SupervisorCommand:
        cfg = self.config
        error = float(observation.target_force_n - observation.measured_force_n)
        p_step = cfg.kp_m_per_n * error
        d_step = -cfg.kd_m_s_per_n * max(float(candidate.force_rate_n_s), 0.0)
        nominal = p_step + d_step
        actuator_limited = _clip(
            nominal,
            -PRECISION_R6_CAPTURE_MAX_STEP_M,
            PRECISION_R6_CAPTURE_MAX_STEP_M,
        )
        if (
            observation.normal_velocity_outward_m_s
            <= PRECISION_R6_CAPTURE_INWARD_SPEED_HOLD_M_S
            and actuator_limited > 0.0
        ):
            actuator_limited = 0.0
        reference = max(
            0.0,
            min(cfg.track_max_press_step_m, before.previous_normal_command_m),
        )
        safe_increment = max(
            0.0,
            (
                cfg.safety_projection_bound_n
                - float(candidate.envelope_base_upper_n)
            )
            / cfg.envelope_candidate_press_gain_n_per_m,
        )
        safe_press_limit = min(
            PRECISION_R6_CAPTURE_MAX_STEP_M,
            reference + safe_increment,
        )
        executed = min(actuator_limited, safe_press_limit)
        command = self._replace_normal_candidate(
            before=before,
            candidate=candidate,
            executed=executed,
            nominal=nominal,
            actuator_limited=actuator_limited,
            reason=candidate.transition_reason,
            freeze_integral=True,
        )
        return replace(
            command,
            p_step_m=p_step,
            d_step_m=d_step,
            safe_press_limit_m=safe_press_limit,
        )

    @staticmethod
    def _contact_capture_eligible(
        observation: SupervisorInput,
        candidate: SupervisorCommand,
    ) -> bool:
        in_verify = bool(
            candidate.state_after == SupervisorState.CONTACT_VERIFY.value
            or (
                candidate.state_before == SupervisorState.CONTACT_VERIFY.value
                and candidate.state_after == SupervisorState.TRACK.value
            )
        )
        return bool(
            in_verify
            and observation.contact_observed
            and not candidate.headroom_triggered
            and not candidate.rebound_guard_triggered
        )

    def command(self, observation: SupervisorInput) -> SupervisorCommand:
        before = self.snapshot()
        candidate = super().command(observation)
        if candidate.transition_reason == "recovery_acquire_bounded_ramp":
            return self._velocity_limited_recovery_acquire(
                before=before,
                observation=observation,
                candidate=candidate,
            )
        if self._contact_capture_eligible(observation, candidate):
            return self._contact_capture(
                before=before,
                observation=observation,
                candidate=candidate,
            )
        return candidate


__all__ = [
    "PRECISION_R6_CAPTURE_INWARD_SPEED_HOLD_M_S",
    "PRECISION_R6_CAPTURE_MAX_STEP_M",
    "PRECISION_R6_DROP_BOUNDED_REASON",
    "PRECISION_R6_DROP_HOLD_REASON",
    "PRECISION_R6_DROP_MAX_OUTWARD_STEP_M",
    "PRECISION_R6_DROP_OUTWARD_HOLD_SPEED_M_S",
    "PRECISION_R6_RECOVERY_REASON",
    "V6PrecisionContactSupervisorR6",
]
