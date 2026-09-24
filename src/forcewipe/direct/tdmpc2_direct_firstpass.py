"""Direct first-pass interface for a native TD-MPC2 ForceWipe baseline.

The learner owns a three-dimensional continuous command in the instantaneous
task frame: along-path, cross-track, and inward-normal displacement.  This
module contains no V6 supervisor, force-dependent command projection, action
shield, or re-wiping transition.  The environment-side transformation into a
Cartesian delta is a fixed actuator interface.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


class DirectFirstPassContractError(ValueError):
    """The direct-control observation/action contract was violated."""


@dataclass(frozen=True)
class DirectFirstPassConfig:
    control_rate_hz: int = 100
    force_limit_n: float = 15.0
    safe_contact_min_n: float = 3.0
    path_bins: int = 20
    dose_samples_per_bin: int = 2
    success_progress: float = 0.98
    maximum_steps: int = 1200
    tangent_action_scale_m: float = 0.0005
    cross_track_action_scale_m: float = 0.00025
    inward_normal_action_scale_m: float = 0.0005
    cross_track_normalization_m: float = 0.02
    normal_offset_normalization_m: float = 0.02
    velocity_normalization_m_s: float = 0.25
    force_rate_normalization_n_s: float = 300.0
    position_action_scale_m: float = 0.1

    def validate(self) -> None:
        if self.control_rate_hz != 100:
            raise DirectFirstPassContractError("the direct baseline is frozen at 100 Hz")
        positive = (
            self.force_limit_n,
            self.safe_contact_min_n,
            self.path_bins,
            self.dose_samples_per_bin,
            self.success_progress,
            self.maximum_steps,
            self.tangent_action_scale_m,
            self.cross_track_action_scale_m,
            self.inward_normal_action_scale_m,
            self.cross_track_normalization_m,
            self.normal_offset_normalization_m,
            self.velocity_normalization_m_s,
            self.force_rate_normalization_n_s,
            self.position_action_scale_m,
        )
        if not all(math.isfinite(float(value)) and float(value) > 0.0 for value in positive):
            raise DirectFirstPassContractError("configuration values must be positive")
        if self.safe_contact_min_n >= self.force_limit_n:
            raise DirectFirstPassContractError("contact threshold must be below force limit")
        if not 0.0 < self.success_progress <= 1.0:
            raise DirectFirstPassContractError("success progress must be in (0, 1]")
        maximum_delta = math.sqrt(
            self.tangent_action_scale_m**2
            + self.cross_track_action_scale_m**2
            + self.inward_normal_action_scale_m**2
        )
        if maximum_delta >= self.position_action_scale_m:
            raise DirectFirstPassContractError("direct command exceeds actuator scale")


@dataclass(frozen=True)
class DirectFirstPassState:
    measured_force_n: float
    target_force_n: float
    force_rate_n_s: float
    progress: float
    signed_cross_track_error_m: float
    signed_normal_offset_m: float
    tangent_velocity_m_s: float
    cross_track_velocity_m_s: float
    outward_normal_velocity_m_s: float
    previous_action: tuple[float, float, float]
    elapsed_steps: int
    completed_dose_bins: int
    minimum_bin_dose: int


@dataclass(frozen=True)
class DirectRewardOutcome:
    reward: float
    success: bool
    force_limit_violation: bool
    new_dose_bins: int
    progress_delta: float


class SafeContactDoseTracker:
    """Online first-pass dose state; no re-wiping state or tolerance band."""

    def __init__(self, config: DirectFirstPassConfig) -> None:
        config.validate()
        self.config = config
        self.counts = np.zeros(config.path_bins, dtype=np.int64)

    def update(self, progress: float, measured_force_n: float) -> int:
        if not math.isfinite(progress) or not math.isfinite(measured_force_n):
            raise DirectFirstPassContractError("dose inputs must be finite")
        before = int(np.sum(self.counts >= self.config.dose_samples_per_bin))
        if (
            0.0 <= progress <= 1.0
            and self.config.safe_contact_min_n
            <= measured_force_n
            <= self.config.force_limit_n
        ):
            index = min(int(progress * self.config.path_bins), self.config.path_bins - 1)
            self.counts[index] += 1
        after = int(np.sum(self.counts >= self.config.dose_samples_per_bin))
        return after - before

    @property
    def completed_bins(self) -> int:
        return int(np.sum(self.counts >= self.config.dose_samples_per_bin))

    @property
    def minimum_bin_dose(self) -> int:
        return int(self.counts.min())

    @property
    def complete(self) -> bool:
        return self.completed_bins == self.config.path_bins


def _finite_vector(value: object, *, name: str, length: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.shape != (length,) or not np.all(np.isfinite(array)):
        raise DirectFirstPassContractError(f"{name} must be one finite length-{length} vector")
    return array


def _unit(value: object, *, name: str) -> np.ndarray:
    array = _finite_vector(value, name=name, length=3)
    norm = float(np.linalg.norm(array))
    if norm <= np.finfo(float).eps:
        raise DirectFirstPassContractError(f"{name} must be nonzero")
    return array / norm


def task_frame_cartesian_delta(
    action: object,
    *,
    tangent_world: object,
    outward_normal_world: object,
    config: DirectFirstPassConfig,
) -> np.ndarray:
    """Map the learner's bounded task-frame command to a world-frame delta.

    This mapping is state-independent with respect to contact force.  It clips
    only the learner's normalized action to the declared actuator action box.
    """

    config.validate()
    normalized = np.clip(_finite_vector(action, name="action", length=3), -1.0, 1.0)
    outward = _unit(outward_normal_world, name="outward normal")
    tangent = _unit(tangent_world, name="path tangent")
    tangent = tangent - float(np.dot(tangent, outward)) * outward
    tangent = _unit(tangent, name="orthogonal path tangent")
    cross = np.cross(outward, tangent)
    cross = _unit(cross, name="cross-track axis")
    return (
        tangent * (normalized[0] * config.tangent_action_scale_m)
        + cross * (normalized[1] * config.cross_track_action_scale_m)
        - outward * (normalized[2] * config.inward_normal_action_scale_m)
    )


def action_from_executed_world_delta(
    executed_world_delta: object,
    *,
    tangent_world: object,
    outward_normal_world: object,
    config: DirectFirstPassConfig,
) -> np.ndarray:
    """Invert the fixed actuator map for an actually issued Cartesian delta."""

    config.validate()
    delta = _finite_vector(
        executed_world_delta,
        name="executed world delta",
        length=3,
    )
    outward = _unit(outward_normal_world, name="outward normal")
    tangent = _unit(tangent_world, name="path tangent")
    tangent = _unit(
        tangent - float(np.dot(tangent, outward)) * outward,
        name="orthogonal path tangent",
    )
    cross = _unit(np.cross(outward, tangent), name="cross-track axis")
    action = np.asarray(
        [
            float(np.dot(delta, tangent)) / config.tangent_action_scale_m,
            float(np.dot(delta, cross)) / config.cross_track_action_scale_m,
            -float(np.dot(delta, outward)) / config.inward_normal_action_scale_m,
        ],
        dtype=np.float32,
    )
    return np.clip(action, -1.0, 1.0)


def encode_observation(
    state: DirectFirstPassState,
    config: DirectFirstPassConfig,
) -> np.ndarray:
    """Encode the causal 16-D state used by the TD-MPC2 world model."""

    config.validate()
    previous = np.clip(
        _finite_vector(state.previous_action, name="previous action", length=3),
        -1.0,
        1.0,
    )
    scalars = (
        state.measured_force_n,
        state.target_force_n,
        state.force_rate_n_s,
        state.progress,
        state.signed_cross_track_error_m,
        state.signed_normal_offset_m,
        state.tangent_velocity_m_s,
        state.cross_track_velocity_m_s,
        state.outward_normal_velocity_m_s,
    )
    if not all(math.isfinite(float(value)) for value in scalars):
        raise DirectFirstPassContractError("observation scalars must be finite")
    if not 0 <= state.elapsed_steps <= config.maximum_steps:
        raise DirectFirstPassContractError("elapsed steps are outside the episode")
    if not 0 <= state.completed_dose_bins <= config.path_bins:
        raise DirectFirstPassContractError("completed dose-bin count is invalid")
    observation = np.asarray(
        [
            np.clip(state.measured_force_n / config.force_limit_n, 0.0, 2.0),
            state.target_force_n / config.force_limit_n,
            np.clip(
                state.force_rate_n_s / config.force_rate_normalization_n_s,
                -1.0,
                1.0,
            ),
            np.clip(state.progress, 0.0, 1.0),
            np.clip(
                state.signed_cross_track_error_m / config.cross_track_normalization_m,
                -2.0,
                2.0,
            ),
            np.clip(
                state.signed_normal_offset_m / config.normal_offset_normalization_m,
                -2.0,
                2.0,
            ),
            np.clip(
                state.tangent_velocity_m_s / config.velocity_normalization_m_s,
                -2.0,
                2.0,
            ),
            np.clip(
                state.cross_track_velocity_m_s / config.velocity_normalization_m_s,
                -2.0,
                2.0,
            ),
            np.clip(
                state.outward_normal_velocity_m_s / config.velocity_normalization_m_s,
                -2.0,
                2.0,
            ),
            *previous,
            state.elapsed_steps / config.maximum_steps,
            float(state.measured_force_n >= config.safe_contact_min_n),
            state.completed_dose_bins / config.path_bins,
            min(state.minimum_bin_dose, config.dose_samples_per_bin)
            / config.dose_samples_per_bin,
        ],
        dtype=np.float32,
    )
    if observation.shape != (16,) or not np.all(np.isfinite(observation)):
        raise DirectFirstPassContractError("encoded observation is invalid")
    return observation


def direct_first_pass_reward(
    *,
    previous_progress: float,
    current_progress: float,
    measured_force_n: float,
    target_force_n: float,
    previous_action: object,
    action: object,
    new_dose_bins: int,
    dose_complete: bool,
    config: DirectFirstPassConfig,
) -> DirectRewardOutcome:
    """Shared training reward and terminal task signal.

    Force error is a smooth training term, not an evaluation tolerance band.
    The success event depends only on safe-contact dose, progress, and the hard
    sampled-force limit.
    """

    previous = _finite_vector(previous_action, name="previous action", length=3)
    current = _finite_vector(action, name="action", length=3)
    numeric = (previous_progress, current_progress, measured_force_n, target_force_n)
    if not all(math.isfinite(float(value)) for value in numeric):
        raise DirectFirstPassContractError("reward inputs must be finite")
    if new_dose_bins < 0 or new_dose_bins > config.path_bins:
        raise DirectFirstPassContractError("new dose-bin count is invalid")
    force_violation = measured_force_n > config.force_limit_n
    progress_delta = max(0.0, current_progress - previous_progress)
    safe_contact = config.safe_contact_min_n <= measured_force_n <= config.force_limit_n
    force_error = abs(measured_force_n - target_force_n) / max(target_force_n, 1e-6)
    reward = -0.002
    reward += 2.0 * float(new_dose_bins)
    reward += 8.0 * progress_delta if safe_contact else 0.0
    reward -= 0.02 * force_error
    reward -= 0.03 if current_progress > 0.02 and not safe_contact else 0.0
    reward -= 0.002 * float(np.square(current - previous).sum())
    if force_violation:
        reward -= 25.0
    success = bool(
        dose_complete
        and current_progress >= config.success_progress
        and not force_violation
    )
    if success:
        reward += 20.0
    return DirectRewardOutcome(
        reward=float(reward),
        success=success,
        force_limit_violation=force_violation,
        new_dose_bins=int(new_dose_bins),
        progress_delta=float(progress_delta),
    )
