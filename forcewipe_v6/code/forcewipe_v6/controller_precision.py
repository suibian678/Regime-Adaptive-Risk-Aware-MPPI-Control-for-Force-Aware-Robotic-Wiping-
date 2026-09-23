"""Exact-target TRACK integration for bounded V6 development.

This versioned child leaves the recovery and global safety states of System
R3-rev2 unchanged.  It replaces only an uninterrupted TRACK candidate with
the tolerance-free exact-target servo in :mod:`precision_tracking`.

The class is SAPIEN-independent.  Passing its tests is code evidence only.
"""

from __future__ import annotations

from dataclasses import replace

from .controller import (
    SupervisorCommand,
    SupervisorInput,
    SupervisorSnapshot,
    SupervisorState,
)
from .controller_system_r3_rev2 import V6SystemContactSupervisorR3Rev2
from .precision_tracking import (
    ExactTargetForceServo,
    PrecisionTrackingConfig,
    PrecisionTrackingInput,
    PrecisionTrackingState,
)


_OVERRIDABLE_TRACK_REASONS = frozenset(
    {
        "track_projected_pi",
        "track_stable_pi",
        "track_final_command_limited",
        "track_force_regulation_pi_task_frozen",
        "track_regulation_ineligible_integrator_frozen",
        "track_return_ineligible_integrator_frozen",
    }
)


class V6PrecisionContactSupervisor(V6SystemContactSupervisorR3Rev2):
    """System R3-rev2 with continuous exact-target TRACK regulation."""

    controller_revision = "V6-precision-track-r1-code-only"

    def _precision_config(self) -> PrecisionTrackingConfig:
        cfg = self.config
        return PrecisionTrackingConfig(
            dt_s=cfg.dt_s,
            force_limit_n=cfg.force_limit_n,
            safety_projection_bound_n=cfg.safety_projection_bound_n,
            contact_floor_n=cfg.contact_loss_floor_n,
            max_inward_step_m=cfg.track_max_press_step_m,
            max_outward_step_m=cfg.max_lift_step_m,
            headroom_outward_step_m=min(0.001, cfg.max_lift_step_m),
            kp_m_per_n=cfg.kp_m_per_n,
            ki_m_per_n_s=cfg.ki_m_per_n_s,
            kd_m_s_per_n=cfg.kd_m_s_per_n,
            integral_clip_n_s=cfg.integral_clip_n_s,
            envelope_incremental_press_gain_n_per_m=(
                cfg.envelope_candidate_press_gain_n_per_m
            ),
        )

    @staticmethod
    def _eligible(
        *,
        before: SupervisorSnapshot,
        observation: SupervisorInput,
        candidate: SupervisorCommand,
    ) -> bool:
        return bool(
            before.state == SupervisorState.TRACK.value
            and candidate.state_before == SupervisorState.TRACK.value
            and candidate.state_after == SupervisorState.TRACK.value
            and not candidate.transitioned
            and candidate.transition_reason in _OVERRIDABLE_TRACK_REASONS
            and not candidate.headroom_triggered
            and not candidate.rebound_guard_triggered
            and observation.contact_observed
            and candidate.geometric_track_ready
        )

    def _apply_precision_track(
        self,
        *,
        before: SupervisorSnapshot,
        observation: SupervisorInput,
        candidate: SupervisorCommand,
    ) -> SupervisorCommand:
        servo = ExactTargetForceServo(self._precision_config())
        servo.restore(
            PrecisionTrackingState(
                integral_error_n_s=before.integral_error_n_s,
                previous_executed_normal_step_m=(
                    before.previous_normal_command_m
                ),
            )
        )
        precise = servo.step(
            PrecisionTrackingInput(
                target_force_n=float(observation.target_force_n),
                measured_force_n=float(observation.measured_force_n),
                force_rate_n_s=float(candidate.force_rate_n_s),
                envelope_base_upper_n=float(candidate.envelope_base_upper_n),
                contact_observed=bool(observation.contact_observed),
                geometric_track_ready=bool(candidate.geometric_track_ready),
                requested_tangential_step_m=float(
                    observation.requested_tangential_step_m
                ),
            )
        )

        stable_track = bool(
            precise.servo_active
            and not precise.actuator_saturation_active
            and not precise.safety_projection_active
            and precise.executed_tangential_step_m != 0.0
        )
        executed_tangent = (
            precise.executed_tangential_step_m if stable_track else 0.0
        )
        committed_progress = before.committed_progress
        if stable_track:
            committed_progress = max(
                committed_progress,
                float(observation.instantaneous_progress),
            )

        verified_count = 0
        verified_required = None
        verified_gate = False
        verified_completed = False
        recovery_after = before.in_recovery_episode
        highwater_after = before.pre_recovery_highwater_progress
        if stable_track and before.in_recovery_episode:
            if highwater_after is None:
                raise RuntimeError("open recovery episode lacks high-water progress")
            verified_count = before.verified_return_count + 1
            verified_required = (
                highwater_after + self.config.verified_return_progress_fraction
            )
            verified_gate = committed_progress + 1.0e-12 >= verified_required
            if (
                verified_count >= self.config.verified_return_samples
                and verified_gate
            ):
                recovery_after = False
                highwater_after = None
                verified_count = 0
                verified_completed = True

        self._integral_error_n_s = precise.integral_after_n_s
        self._previous_previous_normal_command_m = (
            before.previous_normal_command_m
        )
        self._previous_normal_command_m = precise.executed_normal_step_m
        self._committed_progress = committed_progress
        self._verified_return_count = verified_count
        self._in_recovery_episode = recovery_after
        self._pre_recovery_highwater_progress = highwater_after

        envelope_command = candidate.envelope_base_upper_n + (
            self.config.envelope_candidate_press_gain_n_per_m
            * max(
                precise.executed_normal_step_m
                - precise.projection_reference_m,
                0.0,
            )
        )
        return replace(
            candidate,
            transition_reason=precise.reason,
            force_track_ready=precise.servo_active,
            stable_track=stable_track,
            p_step_m=precise.p_step_m,
            i_step_m=precise.i_step_m,
            d_step_m=precise.one_sided_d_step_m,
            nominal_normal_step_m=precise.raw_normal_step_m,
            actuator_limited_normal_step_m=(
                precise.actuator_limited_normal_step_m
            ),
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
            tangential_motion_permitted=stable_track,
            executed_tangential_step_m=executed_tangent,
            committed_progress_after=committed_progress,
            in_recovery_episode_after=recovery_after,
            pre_recovery_highwater_progress_after=highwater_after,
            verified_return_count=verified_count,
            verified_return_required_progress=verified_required,
            verified_return_progress_gate_passed=verified_gate,
            verified_return_completed=verified_completed,
        )

    def command(self, observation: SupervisorInput) -> SupervisorCommand:
        before = self.snapshot()
        candidate = super().command(observation)
        if self._eligible(
            before=before,
            observation=observation,
            candidate=candidate,
        ):
            return self._apply_precision_track(
                before=before,
                observation=observation,
                candidate=candidate,
            )
        return candidate


__all__ = ["V6PrecisionContactSupervisor"]
