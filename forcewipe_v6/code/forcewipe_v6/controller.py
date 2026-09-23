"""Unified causal force-and-recovery supervisor for V6 code-only development.

The module is deliberately independent of SAPIEN.  It defines the causal
state, authority, progress, anti-windup, and recovery-accounting contracts
that must pass before any repeatable physics SANDBOX is opened.

Positive normal commands point inward.  Positive normal velocity points
outward, so ``max(-velocity, 0)`` is the causal inward-speed magnitude.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math
from numbers import Real


class V6ControlError(ValueError):
    """Raised when the code-only causal control contract is violated."""


def _is_finite_real(value: object) -> bool:
    """Accept numeric scalars, but never bool/string coercions."""

    return bool(
        isinstance(value, Real)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


class SupervisorState(str, Enum):
    APPROACH = "APPROACH"
    CONTACT_VERIFY = "CONTACT_VERIFY"
    TRACK = "TRACK"
    REBOUND_GUARD = "REBOUND_GUARD"
    HEADROOM = "HEADROOM"
    RECOVERY_LIFT = "RECOVERY_LIFT"
    RECOVERY_HOVER = "RECOVERY_HOVER"
    RECOVERY_ACQUIRE = "RECOVERY_ACQUIRE"
    SAFE_HOLD = "SAFE_HOLD"


RECOVERY_STATES = frozenset(
    {
        SupervisorState.REBOUND_GUARD,
        SupervisorState.RECOVERY_LIFT,
        SupervisorState.RECOVERY_HOVER,
        SupervisorState.RECOVERY_ACQUIRE,
    }
)

INWARD_CAPABLE_STATES = frozenset(
    {
        SupervisorState.APPROACH,
        SupervisorState.TRACK,
        SupervisorState.RECOVERY_ACQUIRE,
    }
)


@dataclass(frozen=True)
class SupervisorConfig:
    """Mutable-SANDBOX defaults; these are not qualification parameters."""

    dt_s: float = 0.01
    force_limit_n: float = 15.0
    safety_projection_bound_n: float = 14.5
    headroom_enter_n: float = 13.5
    headroom_exit_n: float = 12.0
    headroom_exit_confirm_samples: int = 3
    contact_threshold_n: float = 0.2
    contact_verify_floor_n: float = 3.0
    contact_verify_target_fraction: float = 0.65
    contact_verify_samples: int = 3
    contact_loss_floor_n: float = 3.0
    contact_loss_target_fraction: float = 0.50
    track_force_band_fraction: float = 0.25
    track_geometric_tolerance_m: float = 0.004
    approach_initial_step_m: float = 0.0025
    approach_increment_m: float = 0.0005
    approach_max_step_m: float = 0.0040
    track_max_press_step_m: float = 0.0100
    max_lift_step_m: float = 0.0035
    kp_m_per_n: float = 0.0008
    ki_m_per_n_s: float = 0.0012
    kd_m_s_per_n: float = 0.000006
    integral_clip_n_s: float = 15.0
    force_rate_clip_n_s: float = 300.0
    envelope_margin_n: float = 0.30
    envelope_positive_rate_horizon_s: float = 0.02
    envelope_inward_velocity_gain_n_per_m_s: float = 4.0
    envelope_previous_press_gain_n_per_m: float = 120.0
    envelope_candidate_press_gain_n_per_m: float = 500.0
    rebound_drop_floor_n: float = 4.0
    rebound_drop_target_fraction: float = 0.30
    rebound_prior_force_target_fraction: float = 0.65
    rebound_recent_press_min_m: float = 0.0010
    rebound_inward_speed_min_m_s: float = 0.02
    rebound_guard_samples: int = 3
    recovery_lift_step_m: float = 0.0035
    recovery_lift_max_samples: int = 100
    recovery_release_force_n: float = 1.25
    recovery_hover_clearance_m: float = 0.035
    recovery_hover_tolerance_m: float = 0.004
    recovery_hover_max_samples: int = 100
    recovery_acquire_initial_step_m: float = 0.0010
    recovery_acquire_increment_m: float = 0.00025
    recovery_acquire_max_step_m: float = 0.0040
    recovery_acquire_max_samples: int = 250
    maximum_episode_recovery_cycles: int = 3
    maximum_episode_recovery_samples: int = 1500
    verified_return_samples: int = 50
    verified_return_progress_fraction: float = 0.02

    def validate(self) -> None:
        positive_reals = (
            "dt_s", "force_limit_n", "safety_projection_bound_n",
            "headroom_enter_n", "headroom_exit_n", "contact_threshold_n",
            "contact_verify_floor_n", "contact_verify_target_fraction",
            "contact_loss_floor_n", "contact_loss_target_fraction",
            "track_force_band_fraction", "track_geometric_tolerance_m",
            "approach_initial_step_m", "approach_increment_m",
            "approach_max_step_m", "track_max_press_step_m",
            "max_lift_step_m", "kp_m_per_n", "ki_m_per_n_s",
            "kd_m_s_per_n", "integral_clip_n_s", "force_rate_clip_n_s",
            "envelope_margin_n", "envelope_positive_rate_horizon_s",
            "envelope_inward_velocity_gain_n_per_m_s",
            "envelope_previous_press_gain_n_per_m",
            "envelope_candidate_press_gain_n_per_m",
            "rebound_drop_floor_n", "rebound_drop_target_fraction",
            "rebound_prior_force_target_fraction", "rebound_recent_press_min_m",
            "rebound_inward_speed_min_m_s", "recovery_lift_step_m",
            "recovery_release_force_n", "recovery_hover_clearance_m",
            "recovery_hover_tolerance_m", "recovery_acquire_initial_step_m",
            "recovery_acquire_increment_m", "recovery_acquire_max_step_m",
            "verified_return_progress_fraction",
        )
        for name in positive_reals:
            raw_value = getattr(self, name)
            if not _is_finite_real(raw_value) or float(raw_value) <= 0.0:
                raise V6ControlError(f"{name} must be finite and positive")
        positive_ints = (
            "headroom_exit_confirm_samples", "contact_verify_samples",
            "rebound_guard_samples", "recovery_lift_max_samples",
            "recovery_hover_max_samples", "recovery_acquire_max_samples",
            "maximum_episode_recovery_cycles",
            "maximum_episode_recovery_samples", "verified_return_samples",
        )
        for name in positive_ints:
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise V6ControlError(f"{name} must be a positive integer")
        if not (
            self.contact_threshold_n
            < self.headroom_exit_n
            < self.headroom_enter_n
            < self.safety_projection_bound_n
            < self.force_limit_n
        ):
            raise V6ControlError("force thresholds must be strictly ordered")
        if self.contact_loss_target_fraction >= self.contact_verify_target_fraction:
            raise V6ControlError("contact-loss fraction must be below verify fraction")
        if self.track_force_band_fraction >= 1.0:
            raise V6ControlError("track force band fraction must be below one")
        if self.recovery_hover_tolerance_m >= self.recovery_hover_clearance_m:
            raise V6ControlError("hover tolerance must be below hover clearance")

    def as_dict(self) -> dict:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class SupervisorInput:
    measured_force_n: float
    target_force_n: float
    normal_velocity_outward_m_s: float
    contact_observed: bool
    instantaneous_progress: float
    geometric_track_error_m: float
    requested_tangential_step_m: float = 0.0
    recovery_hover_pose_error_m: float = 0.0
    recovery_clearance_m: float = 0.0


@dataclass(frozen=True)
class SupervisorSnapshot:
    state: str
    state_dwell_samples: int
    contact_verify_count: int
    headroom_exit_count: int
    previous_force_n: float | None
    previous_previous_force_n: float | None
    previous_normal_command_m: float
    previous_previous_normal_command_m: float
    integral_error_n_s: float
    committed_progress: float
    episode_total_recovery_cycles: int
    episode_total_recovery_samples: int
    pre_recovery_highwater_progress: float | None
    verified_return_count: int
    in_recovery_episode: bool


@dataclass(frozen=True)
class SupervisorCommand:
    state_before: str
    state_after: str
    transition_reason: str
    transitioned: bool
    state_dwell_samples: int
    measured_force_n: float
    target_force_n: float
    contact_observed: bool
    previous_force_n: float | None
    previous_previous_force_n: float | None
    force_rate_n_s: float
    normal_velocity_outward_m_s: float
    previous_normal_command_m: float
    previous_previous_normal_command_m: float
    geometric_track_error_m: float
    recovery_hover_pose_error_m: float
    recovery_clearance_m: float
    force_track_ready: bool
    geometric_track_ready: bool
    stable_track: bool
    p_step_m: float
    i_step_m: float
    d_step_m: float
    nominal_normal_step_m: float
    actuator_limited_normal_step_m: float
    projected_normal_step_m: float
    executed_normal_step_m: float
    envelope_base_upper_n: float
    envelope_command_upper_n: float
    safe_press_limit_m: float
    projection_active: bool
    actuator_saturation_active: bool
    safety_projection_active: bool
    integral_before_n_s: float
    integral_after_n_s: float
    rebound_drop_n: float
    rebound_guard_triggered: bool
    headroom_triggered: bool
    tangential_motion_permitted: bool
    recovery_reposition_permitted: bool
    requested_tangential_step_m: float
    executed_tangential_step_m: float
    instantaneous_progress: float
    committed_progress_before: float
    committed_progress_after: float
    in_recovery_episode_before: bool
    in_recovery_episode_after: bool
    recovery_sample_counted_this_step: bool
    recovery_sample_budget_exhausted: bool
    pre_recovery_highwater_progress_before: float | None
    pre_recovery_highwater_progress_after: float | None
    verified_return_count_before: int
    verified_return_count: int
    verified_return_required_progress: float | None
    verified_return_progress_gate_passed: bool
    verified_return_completed: bool
    episode_total_recovery_cycles: int
    episode_total_recovery_samples: int
    sampled_force_violation_observed: bool
    # TRACK-only Cartesian authority is distinct from recovery repositioning.
    # The vector itself is recomputed and bounded by the physical adapter.
    cross_track_correction_permitted: bool = False

    def to_log_row(self, *, evaluation_id: int, step: int) -> dict:
        if not isinstance(evaluation_id, int) or isinstance(evaluation_id, bool):
            raise V6ControlError("evaluation_id must be an integer")
        if not isinstance(step, int) or isinstance(step, bool) or step < 0:
            raise V6ControlError("step must be a nonnegative integer")
        return {"evaluation_id": evaluation_id, "step": step, **asdict(self)}


class UnifiedCausalForceRecoverySupervisor:
    """One mutually exclusive controller state owns all low-level authority."""

    def __init__(self, config: SupervisorConfig = SupervisorConfig()) -> None:
        config.validate()
        self.config = config
        self.reset()

    @property
    def state(self) -> SupervisorState:
        return self._state

    def reset(self) -> None:
        self._state = SupervisorState.APPROACH
        self._state_dwell_samples = 0
        self._previous_force_n: float | None = None
        self._previous_previous_force_n: float | None = None
        self._previous_normal_command_m = 0.0
        self._previous_previous_normal_command_m = 0.0
        self._integral_error_n_s = 0.0
        self._committed_progress = 0.0
        self._contact_verify_count = 0
        self._headroom_exit_count = 0
        self._episode_total_recovery_cycles = 0
        self._episode_total_recovery_samples = 0
        self._pre_recovery_highwater_progress: float | None = None
        self._verified_return_count = 0
        self._in_recovery_episode = False

    def snapshot(self) -> SupervisorSnapshot:
        return SupervisorSnapshot(
            state=self._state.value,
            state_dwell_samples=self._state_dwell_samples,
            contact_verify_count=self._contact_verify_count,
            headroom_exit_count=self._headroom_exit_count,
            previous_force_n=self._previous_force_n,
            previous_previous_force_n=self._previous_previous_force_n,
            previous_normal_command_m=self._previous_normal_command_m,
            previous_previous_normal_command_m=self._previous_previous_normal_command_m,
            integral_error_n_s=self._integral_error_n_s,
            committed_progress=self._committed_progress,
            episode_total_recovery_cycles=self._episode_total_recovery_cycles,
            episode_total_recovery_samples=self._episode_total_recovery_samples,
            pre_recovery_highwater_progress=self._pre_recovery_highwater_progress,
            verified_return_count=self._verified_return_count,
            in_recovery_episode=self._in_recovery_episode,
        )

    def restore(self, snapshot: SupervisorSnapshot) -> None:
        try:
            state = SupervisorState(snapshot.state)
        except ValueError as exc:
            raise V6ControlError("snapshot contains an unknown state") from exc
        finite_optional = (
            snapshot.previous_force_n,
            snapshot.previous_previous_force_n,
            snapshot.pre_recovery_highwater_progress,
        )
        if any(
            value is not None and not _is_finite_real(value)
            for value in finite_optional
        ):
            raise V6ControlError(
                "snapshot optional scalars must be finite real numbers"
            )
        finite = (
            snapshot.previous_normal_command_m,
            snapshot.previous_previous_normal_command_m,
            snapshot.integral_error_n_s,
            snapshot.committed_progress,
        )
        if not all(_is_finite_real(value) for value in finite):
            raise V6ControlError("snapshot scalars must be finite real numbers")
        counts = (
            snapshot.state_dwell_samples,
            snapshot.contact_verify_count,
            snapshot.headroom_exit_count,
            snapshot.episode_total_recovery_cycles,
            snapshot.episode_total_recovery_samples,
            snapshot.verified_return_count,
        )
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in counts):
            raise V6ControlError("snapshot counts must be nonnegative integers")
        if not isinstance(snapshot.in_recovery_episode, bool):
            raise V6ControlError("snapshot recovery flag must be boolean")
        committed_progress = float(snapshot.committed_progress)
        integral_error = float(snapshot.integral_error_n_s)
        if not 0.0 <= committed_progress <= 1.0:
            raise V6ControlError("snapshot committed progress is outside [0,1]")
        if abs(integral_error) > self.config.integral_clip_n_s:
            raise V6ControlError("snapshot integral exceeds the configured clip")
        if any(
            value is not None and float(value) < 0.0
            for value in (
                snapshot.previous_force_n,
                snapshot.previous_previous_force_n,
            )
        ):
            raise V6ControlError("snapshot force history must be nonnegative")
        if any(
            not -self.config.max_lift_step_m <= float(value)
            <= self.config.track_max_press_step_m
            for value in (
                snapshot.previous_normal_command_m,
                snapshot.previous_previous_normal_command_m,
            )
        ):
            raise V6ControlError("snapshot command history exceeds actuator bounds")
        if snapshot.pre_recovery_highwater_progress is not None:
            highwater = float(snapshot.pre_recovery_highwater_progress)
            if not 0.0 <= highwater <= committed_progress:
                raise V6ControlError(
                    "snapshot recovery high-water mark is outside committed progress"
                )
        if (
            snapshot.previous_force_n is None
            and snapshot.previous_previous_force_n is not None
        ):
            raise V6ControlError("snapshot force history order is inconsistent")
        if snapshot.in_recovery_episode != (
            snapshot.pre_recovery_highwater_progress is not None
        ):
            raise V6ControlError("snapshot recovery flag and high-water mark disagree")
        if snapshot.in_recovery_episode and snapshot.episode_total_recovery_cycles < 1:
            raise V6ControlError("open recovery requires at least one recovery cycle")
        if not snapshot.in_recovery_episode and snapshot.verified_return_count != 0:
            raise V6ControlError("closed recovery cannot retain a return count")
        if state in RECOVERY_STATES | {SupervisorState.HEADROOM} and not (
            snapshot.in_recovery_episode
        ):
            raise V6ControlError("recovery and HEADROOM states require an open recovery episode")
        if state is SupervisorState.APPROACH and snapshot.in_recovery_episode:
            raise V6ControlError("APPROACH cannot carry an open recovery episode")
        if (
            state is SupervisorState.SAFE_HOLD
            and not snapshot.in_recovery_episode
            and snapshot.episode_total_recovery_cycles
            != self.config.maximum_episode_recovery_cycles
        ):
            raise V6ControlError(
                "closed-recovery SAFE_HOLD requires an exhausted cycle budget"
            )
        if snapshot.episode_total_recovery_cycles > (
            self.config.maximum_episode_recovery_cycles
        ):
            raise V6ControlError("snapshot exceeds the episode recovery-cycle budget")
        maximum_snapshot_samples = self.config.maximum_episode_recovery_samples + (
            1
            if state is SupervisorState.SAFE_HOLD
            and snapshot.in_recovery_episode
            else 0
        )
        if snapshot.episode_total_recovery_samples > maximum_snapshot_samples:
            raise V6ControlError("snapshot exceeds the episode recovery-sample budget")
        if (
            snapshot.episode_total_recovery_samples
            < snapshot.episode_total_recovery_cycles
        ):
            raise V6ControlError("snapshot has fewer recovery samples than cycles")
        if (
            snapshot.episode_total_recovery_cycles == 0
            and snapshot.episode_total_recovery_samples != 0
        ):
            raise V6ControlError("snapshot has recovery samples without a cycle")
        if (
            snapshot.verified_return_count
            > snapshot.episode_total_recovery_samples
        ):
            raise V6ControlError("snapshot return count exceeds recovery samples")
        if (
            state is not SupervisorState.TRACK
            and snapshot.verified_return_count != 0
        ):
            raise V6ControlError("return confirmation count is outside TRACK")
        if state is not SupervisorState.TRACK and integral_error != 0.0:
            raise V6ControlError("integral state is nonzero outside TRACK")
        if state is not SupervisorState.APPROACH and snapshot.state_dwell_samples < 1:
            raise V6ControlError("non-APPROACH state has no completed dwell sample")
        maximum_reachable_dwell = {
            SupervisorState.REBOUND_GUARD: self.config.rebound_guard_samples - 1,
            SupervisorState.RECOVERY_LIFT: max(
                1, self.config.recovery_lift_max_samples - 1
            ),
            SupervisorState.RECOVERY_HOVER: max(
                1, self.config.recovery_hover_max_samples - 1
            ),
            SupervisorState.RECOVERY_ACQUIRE: max(
                1, self.config.recovery_acquire_max_samples - 1
            ),
        }
        if (
            state in maximum_reachable_dwell
            and snapshot.state_dwell_samples > maximum_reachable_dwell[state]
        ):
            raise V6ControlError("snapshot recovery dwell already requires transition")
        if snapshot.contact_verify_count >= self.config.contact_verify_samples:
            raise V6ControlError("snapshot contact confirmation should already transition")
        if (
            state is not SupervisorState.CONTACT_VERIFY
            and snapshot.contact_verify_count != 0
        ):
            raise V6ControlError("contact confirmation count is outside CONTACT_VERIFY")
        maximum_local_confirmation_count = max(
            0, snapshot.state_dwell_samples - 1
        )
        if (
            snapshot.contact_verify_count
            > maximum_local_confirmation_count
        ):
            raise V6ControlError(
                "contact confirmation count exceeds state dwell history"
            )
        if snapshot.headroom_exit_count >= self.config.headroom_exit_confirm_samples:
            raise V6ControlError("snapshot HEADROOM confirmation should already transition")
        if state is not SupervisorState.HEADROOM and snapshot.headroom_exit_count != 0:
            raise V6ControlError("HEADROOM exit count is outside HEADROOM")
        if snapshot.headroom_exit_count > maximum_local_confirmation_count:
            raise V6ControlError(
                "HEADROOM confirmation count exceeds state dwell history"
            )
        if snapshot.verified_return_count > maximum_local_confirmation_count:
            raise V6ControlError(
                "return confirmation count exceeds state dwell history"
            )
        self._state = state
        self._state_dwell_samples = snapshot.state_dwell_samples
        self._contact_verify_count = snapshot.contact_verify_count
        self._headroom_exit_count = snapshot.headroom_exit_count
        self._previous_force_n = (
            None if snapshot.previous_force_n is None
            else float(snapshot.previous_force_n)
        )
        self._previous_previous_force_n = (
            None if snapshot.previous_previous_force_n is None
            else float(snapshot.previous_previous_force_n)
        )
        self._previous_normal_command_m = float(snapshot.previous_normal_command_m)
        self._previous_previous_normal_command_m = float(
            snapshot.previous_previous_normal_command_m
        )
        self._integral_error_n_s = integral_error
        self._committed_progress = committed_progress
        self._episode_total_recovery_cycles = snapshot.episode_total_recovery_cycles
        self._episode_total_recovery_samples = snapshot.episode_total_recovery_samples
        self._pre_recovery_highwater_progress = (
            None if snapshot.pre_recovery_highwater_progress is None
            else float(snapshot.pre_recovery_highwater_progress)
        )
        self._verified_return_count = snapshot.verified_return_count
        self._in_recovery_episode = snapshot.in_recovery_episode

    def _validate_input(self, observation: SupervisorInput) -> None:
        scalars = (
            observation.measured_force_n, observation.target_force_n,
            observation.normal_velocity_outward_m_s,
            observation.instantaneous_progress,
            observation.geometric_track_error_m,
            observation.requested_tangential_step_m,
            observation.recovery_hover_pose_error_m,
            observation.recovery_clearance_m,
        )
        if not all(_is_finite_real(value) for value in scalars):
            raise V6ControlError(
                "all supervisor inputs must be finite real numbers"
            )
        if observation.measured_force_n < 0.0 or observation.target_force_n <= 0.0:
            raise V6ControlError("force inputs must be physically valid")
        if not 0.0 <= observation.instantaneous_progress <= 1.0:
            raise V6ControlError("instantaneous progress is outside [0,1]")
        if observation.requested_tangential_step_m < 0.0:
            raise V6ControlError("requested tangential step must be nonnegative")
        if observation.recovery_hover_pose_error_m < 0.0:
            raise V6ControlError("hover pose error must be nonnegative")
        if observation.geometric_track_error_m < 0.0:
            raise V6ControlError("geometric track error must be nonnegative")
        if not isinstance(observation.contact_observed, bool):
            raise V6ControlError("contact_observed must be boolean")

    def _transition(self, state: SupervisorState) -> bool:
        if state is self._state:
            return False
        self._state = state
        self._state_dwell_samples = 0
        self._contact_verify_count = 0
        self._headroom_exit_count = 0
        return True

    def _force_rate(self, force: float) -> float:
        if self._previous_force_n is None:
            return 0.0
        return max(
            -self.config.force_rate_clip_n_s,
            min(
                self.config.force_rate_clip_n_s,
                (force - self._previous_force_n) / self.config.dt_s,
            ),
        )

    def _contact_verify_threshold(self, target: float) -> float:
        return max(
            self.config.contact_verify_floor_n,
            self.config.contact_verify_target_fraction * target,
        )

    def _contact_loss_threshold(self, target: float) -> float:
        return max(
            self.config.contact_loss_floor_n,
            self.config.contact_loss_target_fraction * target,
        )

    def _envelope_base(
        self, *, force: float, force_rate: float,
        outward_velocity: float,
    ) -> float:
        cfg = self.config
        return (
            force
            + cfg.envelope_margin_n
            + cfg.envelope_positive_rate_horizon_s * max(force_rate, 0.0)
            + cfg.envelope_inward_velocity_gain_n_per_m_s
            * max(-outward_velocity, 0.0)
            + cfg.envelope_previous_press_gain_n_per_m
            * max(self._previous_normal_command_m, 0.0)
        )

    def _rebound_guard_eligible(self, observation: SupervisorInput) -> tuple[bool, float]:
        if self._previous_force_n is None:
            return False, 0.0
        target = observation.target_force_n
        drop = max(self._previous_force_n - observation.measured_force_n, 0.0)
        established = self._previous_force_n >= (
            self.config.rebound_prior_force_target_fraction * target
        )
        energetic = bool(
            max(
                self._previous_normal_command_m,
                self._previous_previous_normal_command_m,
            )
            >= self.config.rebound_recent_press_min_m
            or -observation.normal_velocity_outward_m_s
            >= self.config.rebound_inward_speed_min_m_s
        )
        threshold = max(
            self.config.rebound_drop_floor_n,
            self.config.rebound_drop_target_fraction * target,
        )
        return bool(established and energetic and drop >= threshold), float(drop)

    def _start_recovery_cycle(self) -> bool:
        next_count = self._episode_total_recovery_cycles + 1
        if next_count > self.config.maximum_episode_recovery_cycles:
            self._transition(SupervisorState.SAFE_HOLD)
            return False
        self._episode_total_recovery_cycles = next_count
        if not self._in_recovery_episode:
            self._pre_recovery_highwater_progress = self._committed_progress
        self._in_recovery_episode = True
        self._verified_return_count = 0
        return True

    def _account_open_recovery_sample(
        self, *, recovery_active_this_step: bool,
    ) -> tuple[bool, bool]:
        if not recovery_active_this_step:
            return False, False
        self._episode_total_recovery_samples += 1
        if self._episode_total_recovery_samples > self.config.maximum_episode_recovery_samples:
            if self._state is not SupervisorState.SAFE_HOLD:
                self._transition(SupervisorState.SAFE_HOLD)
            return True, True
        return True, False

    def _update_history(self, *, force: float, executed_normal_step_m: float) -> None:
        self._previous_previous_force_n = self._previous_force_n
        self._previous_force_n = force
        self._previous_previous_normal_command_m = self._previous_normal_command_m
        self._previous_normal_command_m = executed_normal_step_m

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

        # Measured or causally predicted headroom has global priority in every
        # nonterminal state.  This decision is independent of the sign of the
        # state-specific candidate, which has not yet been constructed.
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

        # Collapse/rebound preemption applies to every state that can issue an
        # inward candidate.  It is evaluated before state-specific candidates
        # are constructed and uses only causal history.
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
            at_hover = observation.recovery_hover_pose_error_m <= cfg.recovery_hover_tolerance_m
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
                    cfg.approach_initial_step_m + cfg.approach_increment_m * dwell_before,
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
                    self._transition(SupervisorState.TRACK)
                    reason = "contact_verify_complete_to_track"
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

        # One centralized episode-level accounting point covers every sample
        # from recovery opening through strict closure, including HEADROOM,
        # CONTACT_VERIFY, and TRACK return confirmation.  It prevents state-
        # local gaps and double counting.
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

        # Global final-command pipeline.  Every state-specific candidate first
        # passes the actuator bound.  Every remaining inward candidate from an
        # inward-capable state then passes the same causal safety envelope.
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

        # TRACK integral state is committed only during the same stable TRACK
        # condition that owns task authority, and only when the raw candidate
        # equals the final executed command.
        if before is SupervisorState.TRACK and self._state is SupervisorState.TRACK:
            if "candidate_integral" in locals():
                if stable_track and abs(executed - nominal) <= 1e-15:
                    self._integral_error_n_s = candidate_integral
                    reason = "track_projected_pi"
                else:
                    self._integral_error_n_s = integral_before
                    reason = (
                        "track_final_command_limited"
                        if projection_active else "track_return_ineligible_integrator_frozen"
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
