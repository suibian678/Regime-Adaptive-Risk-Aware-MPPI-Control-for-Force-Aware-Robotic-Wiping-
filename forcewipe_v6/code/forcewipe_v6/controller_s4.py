"""V6-S4 bounded recovery and low-margin phase-hold candidate.

This module is a development-only revision of :mod:`controller_s3`.  It is
kept in a separate file so that the frozen S2/S3 controllers and their
artifacts remain byte-identical.

S4 makes exactly two mechanism changes:

* the monotone episode recovery-cycle ceiling is raised from three to ten,
  while the unchanged 1,500-sample episode recovery budget remains a hard
  terminal bound; and
* in low-static-margin TRACK operation, a causal phase-mismatch signature
  projects an otherwise inward final command to zero for one sample and
  revokes task-tangential/progress authority for that sample.

The phase hold uses only the current observation and stored command/force
history.  It is not a proof that the next force sample will remain safe.
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
from .controller_s3 import V6S3LivenessSupervisor


S4_MAXIMUM_EPISODE_RECOVERY_CYCLES = 10
S4_LOW_STATIC_MARGIN_N = 3.0
S4_PHASE_HOLD_FORCE_FRACTION = 0.85
S4_PHASE_HOLD_FORCE_RATE_N_S = -15.0
S4_PHASE_HOLD_OUTWARD_VELOCITY_M_S = 0.03
S4_PHASE_HOLD_RECENT_PRESS_M = 0.001


def v6_s4_supervisor_config(
    base: SupervisorConfig | None = None,
) -> SupervisorConfig:
    """Return the single frozen S4 development configuration."""

    source = SupervisorConfig() if base is None else base
    return replace(
        source,
        maximum_episode_recovery_cycles=S4_MAXIMUM_EPISODE_RECOVERY_CYCLES,
    )


class V6S4BoundedPhaseSupervisor(V6S3LivenessSupervisor):
    """One S4 candidate; no parameter sweep or downstream claim is implied."""

    controller_revision = "V6-S4-bounded-recovery-phase-hold-v1"

    def __init__(self, config: SupervisorConfig | None = None) -> None:
        # The S4 cycle ceiling is part of this controller revision, not a
        # caller-selectable sweep parameter.  All other supplied parameters
        # are preserved exactly.
        super().__init__(v6_s4_supervisor_config(config))

    def _phase_hold_eligible(
        self,
        *,
        before: SupervisorSnapshot,
        observation: SupervisorInput,
        candidate: SupervisorCommand,
    ) -> bool:
        """Return whether the causal low-margin final-command hold applies."""

        if before.state != SupervisorState.TRACK.value:
            return False
        if before.in_recovery_episode:
            return False
        if self.config.force_limit_n - observation.target_force_n > (
            S4_LOW_STATIC_MARGIN_N + 1e-12
        ):
            return False
        if observation.measured_force_n < (
            S4_PHASE_HOLD_FORCE_FRACTION * observation.target_force_n
        ):
            return False
        if candidate.force_rate_n_s >= S4_PHASE_HOLD_FORCE_RATE_N_S:
            return False
        if observation.normal_velocity_outward_m_s <= (
            S4_PHASE_HOLD_OUTWARD_VELOCITY_M_S
        ):
            return False
        if max(
            before.previous_normal_command_m,
            before.previous_previous_normal_command_m,
        ) <= S4_PHASE_HOLD_RECENT_PRESS_M:
            return False
        return bool(
            candidate.state_before == SupervisorState.TRACK.value
            and candidate.state_after == SupervisorState.TRACK.value
            and not candidate.transitioned
            and not candidate.headroom_triggered
            and not candidate.rebound_guard_triggered
            and candidate.executed_normal_step_m > 0.0
        )

    def _apply_phase_hold(
        self,
        *,
        before: SupervisorSnapshot,
        observation: SupervisorInput,
        candidate: SupervisorCommand,
    ) -> SupervisorCommand:
        """Rollback the S3 candidate and commit the S4 zero-command hold."""

        # S4 restricts this projection to a closed recovery episode.  Restoring
        # the pre-command snapshot is therefore sufficient to undo the S3
        # integral, progress, dwell, and force/command history updates without
        # dropping an open recovery-sample accounting event.
        self.restore(before)
        self._state_dwell_samples = before.state_dwell_samples + 1
        self._update_history(
            force=float(observation.measured_force_n),
            executed_normal_step_m=0.0,
        )
        return replace(
            candidate,
            transition_reason="track_low_static_margin_phase_hold",
            state_dwell_samples=self._state_dwell_samples,
            stable_track=False,
            projected_normal_step_m=0.0,
            executed_normal_step_m=0.0,
            envelope_command_upper_n=candidate.envelope_base_upper_n,
            safe_press_limit_m=0.0,
            projection_active=True,
            safety_projection_active=True,
            integral_after_n_s=before.integral_error_n_s,
            tangential_motion_permitted=False,
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

    def command(self, observation: SupervisorInput) -> SupervisorCommand:
        before = self.snapshot()
        candidate = super().command(observation)
        if not self._phase_hold_eligible(
            before=before,
            observation=observation,
            candidate=candidate,
        ):
            return candidate
        return self._apply_phase_hold(
            before=before,
            observation=observation,
            candidate=candidate,
        )


__all__ = [
    "S4_LOW_STATIC_MARGIN_N",
    "S4_MAXIMUM_EPISODE_RECOVERY_CYCLES",
    "S4_PHASE_HOLD_FORCE_FRACTION",
    "S4_PHASE_HOLD_FORCE_RATE_N_S",
    "S4_PHASE_HOLD_OUTWARD_VELOCITY_M_S",
    "S4_PHASE_HOLD_RECENT_PRESS_M",
    "V6S4BoundedPhaseSupervisor",
    "v6_s4_supervisor_config",
]
