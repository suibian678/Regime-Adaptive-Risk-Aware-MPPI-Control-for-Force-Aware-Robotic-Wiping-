"""V6-S5 path-frame consistency and bounded cross-track correction.

S5 is a code-only development candidate built after the frozen S4 SANDBOX
failure.  It retains every S4 force, headroom, rebound, recovery-budget, and
force-limit parameter.  It adds two final-command restrictions:

* before the existing 4-mm geometry-loss transition, TRACK may grant one
  bounded low-level Cartesian correction while revoking normal integral,
  task-tangential, and progress authority; and
* an isolated inward TRACK command no larger than the frozen actuator
  resolution is represented as zero before it reaches the executor.

The Cartesian correction vector is computed by the S5 physical binding from
the same causal pre-step geometry.  No learned or future signal is used.
"""

from __future__ import annotations

from dataclasses import replace

from .controller import (
    SupervisorCommand,
    SupervisorConfig,
    SupervisorInput,
    SupervisorSnapshot,
    SupervisorState,
)
from .controller_s4 import V6S4BoundedPhaseSupervisor


S5_CROSS_TRACK_CORRECTION_ENTER_M = 0.003
S5_CROSS_TRACK_CORRECTION_MAX_STEP_M = 0.0005
S5_MINIMUM_ACTIONABLE_INWARD_STEP_M = 0.00005


class V6S5PathFrameSupervisor(V6S4BoundedPhaseSupervisor):
    """One bounded S5 controller candidate; no physical claim is implied."""

    controller_revision = "V6-S5-path-frame-cross-track-v2-dual-frame"

    def __init__(self, config: SupervisorConfig | None = None) -> None:
        super().__init__(config)

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
        self.restore(before)
        self._state_dwell_samples = before.state_dwell_samples + 1
        self._update_history(
            force=float(observation.measured_force_n),
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
                if preserve_candidate_projection
                else candidate.envelope_base_upper_n
            ),
            safe_press_limit_m=(
                candidate.safe_press_limit_m
                if preserve_candidate_projection else 0.0
            ),
            projection_active=True,
            integral_after_n_s=before.integral_error_n_s,
            tangential_motion_permitted=False,
            recovery_reposition_permitted=False,
            cross_track_correction_permitted=cross_track_reposition,
            executed_tangential_step_m=0.0,
            committed_progress_after=before.committed_progress,
            in_recovery_episode_after=before.in_recovery_episode,
            recovery_sample_counted_this_step=False,
            recovery_sample_budget_exhausted=False,
            pre_recovery_highwater_progress_after=(
                before.pre_recovery_highwater_progress
            ),
            verified_return_count=before.verified_return_count,
            verified_return_required_progress=None,
            verified_return_progress_gate_passed=False,
            verified_return_completed=False,
            episode_total_recovery_cycles=before.episode_total_recovery_cycles,
            episode_total_recovery_samples=before.episode_total_recovery_samples,
        )

    def _cross_track_correction_eligible(
        self,
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
            and candidate.force_track_ready
            and candidate.geometric_track_ready
            and observation.geometric_track_error_m
            >= S5_CROSS_TRACK_CORRECTION_ENTER_M
            and observation.contact_observed
            and not candidate.headroom_triggered
            and not candidate.rebound_guard_triggered
            and not candidate.projection_active
            and not candidate.recovery_reposition_permitted
            and not candidate.cross_track_correction_permitted
        )

    def _subresolution_hold_eligible(
        self,
        *,
        candidate: SupervisorCommand,
    ) -> bool:
        return bool(
            candidate.state_before == SupervisorState.TRACK.value
            and candidate.state_after == SupervisorState.TRACK.value
            and not candidate.transitioned
            and not candidate.headroom_triggered
            and not candidate.rebound_guard_triggered
            and not candidate.recovery_reposition_permitted
            and not candidate.cross_track_correction_permitted
            and abs(candidate.executed_tangential_step_m) <= 1e-15
            and 0.0 < candidate.executed_normal_step_m
            <= S5_MINIMUM_ACTIONABLE_INWARD_STEP_M
        )

    def command(self, observation: SupervisorInput) -> SupervisorCommand:
        before = self.snapshot()
        candidate = super().command(observation)
        if self._cross_track_correction_eligible(
            before=before,
            observation=observation,
            candidate=candidate,
        ):
            return self._restore_track_hold(
                before=before,
                observation=observation,
                candidate=candidate,
                reason="track_cross_track_correction_projection",
                cross_track_reposition=True,
            )
        if self._subresolution_hold_eligible(candidate=candidate):
            return self._restore_track_hold(
                before=before,
                observation=observation,
                candidate=candidate,
                reason="track_subresolution_inward_hold",
                cross_track_reposition=False,
                preserve_candidate_projection=True,
            )
        return candidate


__all__ = [
    "S5_CROSS_TRACK_CORRECTION_ENTER_M",
    "S5_CROSS_TRACK_CORRECTION_MAX_STEP_M",
    "S5_MINIMUM_ACTIONABLE_INWARD_STEP_M",
    "V6S5PathFrameSupervisor",
]
