"""Minimal recovery-cycle accounting correction for System R3.

The original R3 code-only closure is retained unchanged.  This revision only
ensures that every new TRACK-to-local-acquire dropout consumes one monotone
episode recovery cycle, including dropouts that occur before the prior recovery
episode has completed its verified-return window.
"""

from __future__ import annotations

from dataclasses import replace

from .controller import (
    SupervisorCommand,
    SupervisorInput,
    SupervisorSnapshot,
    SupervisorState,
)
from .controller_system_r3 import V6SystemContactSupervisorR3


class V6SystemContactSupervisorR3Rev2(V6SystemContactSupervisorR3):
    """System R3 with non-bypassable local-reacquire cycle accounting."""

    controller_revision = (
        "V6-system-contact-candidate-v3-rev2-local-cycle-accounting"
    )

    def _apply_local_reacquire(
        self,
        *,
        before: SupervisorSnapshot,
        observation: SupervisorInput,
        candidate: SupervisorCommand,
    ) -> SupervisorCommand:
        self.restore(before)
        self._integral_error_n_s = 0.0

        # A TRACK dropout is a new recovery attempt even while the episode is
        # still awaiting its verified-return closure.  The inherited helper
        # preserves the original high-water mark when the episode is open.
        recovery_opened = self._start_recovery_cycle()
        if recovery_opened and self._state is not SupervisorState.SAFE_HOLD:
            self._transition(SupervisorState.RECOVERY_ACQUIRE)
            reason = "track_contact_degradation_to_local_acquire"
        else:
            reason = "episode_recovery_cycle_budget_exhausted"

        counted, exhausted = self._account_open_recovery_sample(
            recovery_active_this_step=bool(
                recovery_opened
                and self._in_recovery_episode
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


__all__ = ["V6SystemContactSupervisorR3Rev2"]
