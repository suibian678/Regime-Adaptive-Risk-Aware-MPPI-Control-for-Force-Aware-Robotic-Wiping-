"""Shared non-privileged decision observation, action, and reward contracts."""

from __future__ import annotations

from dataclasses import dataclass, fields
import math

import numpy as np

from .authority import (
    AuthorityBounds,
    AuthorityInterface,
    AuthorityProposal,
)
from .information_flow import ObservationMode, PolicyObservation


class DecisionProtocolError(ValueError):
    pass


@dataclass(frozen=True)
class DecisionContext:
    path_progress: float
    remaining_budget: int
    maximum_budget: int
    elapsed_s: float
    episode_horizon_s: float
    last_primitive_id: int | None = None
    primitive_count: int = 1

    def validate(self) -> None:
        numeric = (self.path_progress, self.elapsed_s, self.episode_horizon_s)
        if not all(math.isfinite(float(value)) for value in numeric):
            raise DecisionProtocolError("decision context must be finite")
        if not 0.0 <= float(self.path_progress) <= 1.0:
            raise DecisionProtocolError("path progress must lie in [0, 1]")
        if not 0 <= int(self.remaining_budget) <= int(self.maximum_budget):
            raise DecisionProtocolError("remaining budget is invalid")
        if int(self.maximum_budget) <= 0 or int(self.primitive_count) <= 0:
            raise DecisionProtocolError("budgets and primitive count must be positive")
        if not 0 <= float(self.elapsed_s) <= float(self.episode_horizon_s):
            raise DecisionProtocolError("elapsed time is outside the episode horizon")
        if self.last_primitive_id is not None and not (
            0 <= int(self.last_primitive_id) < int(self.primitive_count)
        ):
            raise DecisionProtocolError("last primitive is outside the shared library")


@dataclass(frozen=True)
class DecisionEncodingConfig:
    residual_cell_count: int
    force_scale_n: float = 15.0
    speed_scale_m_s: float = 0.08
    maximum_force_age_samples: int = 100
    maximum_effective_traversal: float = 100.0

    def validate(self) -> None:
        if int(self.residual_cell_count) <= 0:
            raise DecisionProtocolError("residual cell count must be positive")
        numeric = (
            self.force_scale_n,
            self.speed_scale_m_s,
            self.maximum_effective_traversal,
        )
        if not all(math.isfinite(float(value)) and float(value) > 0 for value in numeric):
            raise DecisionProtocolError("encoding scales must be finite and positive")
        if int(self.maximum_force_age_samples) <= 0:
            raise DecisionProtocolError("force-age scale must be positive")


def _bounded_unit(value: float, name: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise DecisionProtocolError(f"{name} must be finite")
    return float(np.clip(numeric, 0.0, 1.0))


def encode_policy_decision_observation(
    observation: PolicyObservation,
    context: DecisionContext,
    *,
    config: DecisionEncodingConfig,
    allow_oracle: bool = False,
) -> np.ndarray:
    """Encode a fixed-size policy vector without environment truth or hidden plant data."""

    config.validate()
    context.validate()
    mode = ObservationMode(observation.mode)
    if observation.privileged_oracle and not allow_oracle:
        raise DecisionProtocolError("oracle observation is forbidden for a non-oracle policy")

    residual_count = int(config.residual_cell_count)
    residual = np.zeros(residual_count, dtype=np.float64)
    uncertainty = np.zeros(residual_count, dtype=np.float64)
    residual_available = observation.residual_estimate is not None
    if residual_available:
        values = np.asarray(observation.residual_estimate, dtype=np.float64)
        variances = np.asarray(observation.residual_uncertainty, dtype=np.float64)
        if values.shape != (residual_count,) or variances.shape != (residual_count,):
            raise DecisionProtocolError("residual estimate has the wrong fixed dimension")
        if not np.all(np.isfinite(values)) or not np.all(np.isfinite(variances)):
            raise DecisionProtocolError("residual estimate must be finite")
        if np.any(values < 0) or np.any(values > 1) or np.any(variances < 0):
            raise DecisionProtocolError("residual estimate/uncertainty is outside its domain")
        residual = values
        uncertainty = np.clip(variances, 0.0, 1.0)

    force_available = observation.tracking_force_n is not None
    force = 0.0
    force_age = 0.0
    force_held = 0.0
    if force_available:
        force = float(observation.tracking_force_n) / float(config.force_scale_n)
        if not math.isfinite(force):
            raise DecisionProtocolError("tracking force must be finite")
        age = observation.tracking_force_age_samples
        if age is None or int(age) < 0:
            raise DecisionProtocolError("available tracking force requires nonnegative age")
        force_age = min(
            float(age) / float(config.maximum_force_age_samples),
            1.0,
        )
        force_held = float(bool(observation.tracking_force_is_held))

    traversal = observation.cumulative_effective_traversal
    traversal_value = (
        0.0
        if traversal is None
        else float(
            np.clip(
                float(traversal) / float(config.maximum_effective_traversal),
                0.0,
                1.0,
            )
        )
    )
    modes = list(ObservationMode)
    mode_one_hot = np.zeros(len(modes), dtype=np.float64)
    mode_one_hot[modes.index(mode)] = 1.0
    last_available = context.last_primitive_id is not None
    last_id = (
        0.0
        if context.last_primitive_id is None or context.primitive_count == 1
        else float(context.last_primitive_id) / float(context.primitive_count - 1)
    )
    core = np.asarray(
        [
            force,
            float(force_available),
            force_age,
            force_held,
            float(observation.nominal_force_n) / float(config.force_scale_n),
            float(observation.tangential_speed_m_s) / float(config.speed_scale_m_s),
            traversal_value,
            float(context.path_progress),
            float(context.remaining_budget) / float(context.maximum_budget),
            float(context.elapsed_s) / float(context.episode_horizon_s),
            last_id,
            float(last_available),
            float(residual_available),
        ],
        dtype=np.float64,
    )
    vector = np.concatenate((mode_one_hot, core, residual, uncertainty))
    if not np.all(np.isfinite(vector)):
        raise DecisionProtocolError("encoded observation contains NaN/Inf")
    vector.setflags(write=False)
    return vector


@dataclass(frozen=True)
class PolicyAction:
    discrete_index: int | None = None
    normalized_vector: tuple[float, ...] | None = None


def _normalized_vector(action: PolicyAction, expected_size: int) -> np.ndarray:
    if action.normalized_vector is None:
        raise DecisionProtocolError("this authority interface requires a continuous vector")
    vector = np.asarray(action.normalized_vector, dtype=np.float64)
    if vector.shape != (expected_size,) or not np.all(np.isfinite(vector)):
        raise DecisionProtocolError("normalized action has the wrong shape or is nonfinite")
    if np.any(vector < -1.0) or np.any(vector > 1.0):
        raise DecisionProtocolError("normalized action must lie in [-1, 1]")
    return vector


def _affine(value: float, interval: tuple[float, float]) -> float:
    return float(interval[0] + 0.5 * (float(value) + 1.0) * (interval[1] - interval[0]))


def decode_authority_policy_action(
    action: PolicyAction,
    *,
    interface: AuthorityInterface,
    primitive_count: int,
    bounds: AuthorityBounds = AuthorityBounds(),
) -> AuthorityProposal:
    """Decode the same normalized action convention into one A1--A4 proposal."""

    interface = AuthorityInterface(interface)
    bounds.validate()
    if int(primitive_count) <= 0:
        raise DecisionProtocolError("primitive count must be positive")
    primitive_id = action.discrete_index
    if interface != AuthorityInterface.A4_CONTINUOUS:
        if primitive_id is None or not 0 <= int(primitive_id) < int(primitive_count):
            raise DecisionProtocolError("discrete primitive index is outside the library")
        primitive_id = int(primitive_id)
    elif primitive_id is not None:
        raise DecisionProtocolError("A4 continuous authority must not provide a primitive ID")

    if interface == AuthorityInterface.A1_PRIMITIVE:
        if action.normalized_vector is not None:
            raise DecisionProtocolError("A1 must not provide continuous parameters")
        return AuthorityProposal(interface, primitive_id=primitive_id)
    if interface == AuthorityInterface.A2_FORCE_PARAMETER:
        vector = _normalized_vector(action, 1)
        return AuthorityProposal(
            interface,
            primitive_id=primitive_id,
            target_force_n=_affine(vector[0], bounds.target_force_n),
        )
    if interface == AuthorityInterface.A3_IMPEDANCE_PARAMETER:
        vector = _normalized_vector(action, 3)
        return AuthorityProposal(
            interface,
            primitive_id=primitive_id,
            target_force_n=_affine(vector[0], bounds.target_force_n),
            stiffness_n_m=_affine(vector[1], bounds.stiffness_n_m),
            damping_n_s_m=_affine(vector[2], bounds.damping_n_s_m),
        )
    if interface == AuthorityInterface.A4_CONTINUOUS:
        vector = _normalized_vector(action, 6)
        cartesian = tuple(
            _affine(value, bounds.cartesian_delta_m) for value in vector[:3]
        )
        return AuthorityProposal(
            interface,
            target_force_n=_affine(vector[3], bounds.target_force_n),
            stiffness_n_m=_affine(vector[4], bounds.stiffness_n_m),
            damping_n_s_m=_affine(vector[5], bounds.damping_n_s_m),
            cartesian_delta_xyz_m=cartesian,
        )
    raise DecisionProtocolError("A0 classical authority does not accept a policy action")


def decode_discrete_action_lattice(
    action_index: int,
    lattice: tuple[AuthorityProposal, ...],
) -> AuthorityProposal:
    """Resolve a DQN-compatible discrete lattice without continuous decoding."""

    if not lattice:
        raise DecisionProtocolError("action lattice must be nonempty")
    if not 0 <= int(action_index) < len(lattice):
        raise DecisionProtocolError("action lattice index is out of range")
    interfaces = {proposal.interface for proposal in lattice}
    if len(interfaces) != 1:
        raise DecisionProtocolError("one action lattice may not mix authority interfaces")
    return lattice[int(action_index)]


@dataclass(frozen=True)
class ObservableDecisionOutcome:
    """Process outcome exposed to the shared simulation-training reward.

    ``training_residual_excess_potential_gain`` is the simulator-truth decrease
    in ``max(residual_ratio - target_ratio, 0)``.  It is deliberately restricted
    to training reward construction and offline evaluation; it is never included
    in a deployed policy observation.  All observation modalities receive this
    same scalar reward definition.
    """

    training_residual_excess_potential_gain: float
    duration_s: float
    shield_intervened: bool
    force_limit_violation: bool
    spatial_constraint_violation: bool
    physical_lifecycle_complete: bool
    training_terminal_residual_mass_ratio: float | None = None
    episode_terminated: bool = True


def residual_excess_potential_gain(
    residual_ratio_t: float,
    residual_ratio_next: float,
    *,
    target_ratio: float,
) -> float:
    """Return the decrease of residual mass above the accepted threshold."""

    values = (residual_ratio_t, residual_ratio_next, target_ratio)
    if not all(math.isfinite(float(value)) and 0.0 <= float(value) <= 1.0 for value in values):
        raise DecisionProtocolError("residual potential ratios must be finite and in [0, 1]")
    gain = max(float(residual_ratio_t) - float(target_ratio), 0.0) - max(
        float(residual_ratio_next) - float(target_ratio), 0.0
    )
    if gain < -1e-12:
        raise DecisionProtocolError("residual-excess potential may not increase")
    return float(max(gain, 0.0))


@dataclass(frozen=True)
class SharedRewardConfig:
    residual_excess_potential_gain_weight: float = 1.0
    duration_cost_per_s: float = 0.01
    primitive_cost: float = 0.01
    shield_intervention_cost: float = 0.02
    force_violation_cost: float = 2.0
    spatial_violation_cost: float = 1.0
    incomplete_lifecycle_cost: float = 0.5
    terminal_residual_target_ratio: float = 0.15
    terminal_residual_excess_cost: float = 2.0

    def validate(self) -> None:
        for field in fields(self):
            value = float(getattr(self, field.name))
            if not math.isfinite(value) or value < 0:
                raise DecisionProtocolError("reward weights must be finite and nonnegative")
        if self.terminal_residual_target_ratio > 1.0:
            raise DecisionProtocolError("terminal residual target ratio must be in [0, 1]")


def assert_training_reward_contract() -> None:
    """Guard the one explicitly permitted simulator-truth reward channel."""

    names = {field.name for field in fields(ObservableDecisionOutcome)}
    expected = {
        "training_residual_excess_potential_gain",
        "duration_s",
        "shield_intervened",
        "force_limit_violation",
        "spatial_constraint_violation",
        "physical_lifecycle_complete",
        "training_terminal_residual_mass_ratio",
        "episode_terminated",
    }
    if names != expected:
        raise DecisionProtocolError("shared simulation-training reward schema changed")


def compute_shared_decision_reward(
    outcome: ObservableDecisionOutcome,
    *,
    config: SharedRewardConfig = SharedRewardConfig(),
) -> float:
    """Return the common, explicitly truth-assisted simulation-training reward."""

    assert_training_reward_contract()
    config.validate()
    potential_gain = float(outcome.training_residual_excess_potential_gain)
    duration = float(outcome.duration_s)
    if not math.isfinite(potential_gain) or not 0.0 <= potential_gain <= 1.0:
        raise DecisionProtocolError(
            "training residual-excess potential gain must be finite and in [0, 1]"
        )
    if not math.isfinite(duration) or duration < 0:
        raise DecisionProtocolError("decision duration must be finite and nonnegative")
    terminal_ratio = outcome.training_terminal_residual_mass_ratio
    if outcome.episode_terminated:
        if terminal_ratio is None or not math.isfinite(float(terminal_ratio)):
            raise DecisionProtocolError(
                "every terminal training outcome requires a residual-mass ratio"
            )
        if not 0.0 <= float(terminal_ratio) <= 1.0:
            raise DecisionProtocolError("terminal residual-mass ratio must be in [0, 1]")
    elif terminal_ratio is not None:
        raise DecisionProtocolError(
            "nonterminal training outcomes may not expose residual-mass truth"
        )
    reward = config.residual_excess_potential_gain_weight * potential_gain
    reward -= config.duration_cost_per_s * duration
    reward -= config.primitive_cost
    reward -= config.shield_intervention_cost * float(outcome.shield_intervened)
    reward -= config.force_violation_cost * float(outcome.force_limit_violation)
    reward -= config.spatial_violation_cost * float(outcome.spatial_constraint_violation)
    reward -= config.incomplete_lifecycle_cost * float(
        outcome.episode_terminated and not outcome.physical_lifecycle_complete
    )
    if terminal_ratio is not None:
        reward -= config.terminal_residual_excess_cost * max(
            0.0, float(terminal_ratio) - config.terminal_residual_target_ratio
        )
    return float(reward)
