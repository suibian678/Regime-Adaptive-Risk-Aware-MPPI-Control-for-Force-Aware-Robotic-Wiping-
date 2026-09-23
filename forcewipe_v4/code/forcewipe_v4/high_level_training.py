"""Algorithm-neutral high-level lifecycle environment for V4 training."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol

import gymnasium as gym
import numpy as np

from .decision_protocol import (
    ObservableDecisionOutcome,
    SharedRewardConfig,
    compute_shared_decision_reward,
)
from .information_flow import ObservationMode
from .modality_adapter import MatchedModalityPacket
from .primitives import PrimitiveProposal


class HighLevelTrainingError(RuntimeError):
    pass


def canonical_spatial_primitive_library_18(
    *,
    nominal_force_placeholder_n: float = 8.0,
    stiffness_n_m: float = 2000.0,
    damping_n_s_m: float = 80.0,
) -> tuple[PrimitiveProposal, ...]:
    """Return 18 spatial templates; the backend binds force to the scenario.

    The action identity is nine normalized path centers times two window widths.
    The historical reverse-motion branch is not admitted here because its
    physical method smoke failed to advance under contact; that failure is
    retained separately rather than hidden inside policy training.
    ``target_force_n`` is deliberately a validated placeholder and must be
    replaced by the scenario nominal force by an A1 backend.  This keeps one
    categorical policy valid across the 5/8/12 N task tiers without silently
    granting force-parameter authority.
    """

    library = tuple(
        PrimitiveProposal(
            center_s=float(center),
            window_width_s=float(window_width),
            travel_length_m=0.03,
            direction=1,
            target_force_n=float(nominal_force_placeholder_n),
            stiffness_n_m=float(stiffness_n_m),
            damping_n_s_m=float(damping_n_s_m),
            tangential_speed_m_s=0.02,
        )
        for center in np.linspace(0.20, 0.80, 9)
        for window_width in (0.30, 0.40)
    )
    for primitive in library:
        primitive.validate_finite()
    if len(library) != 18 or len(set(library)) != 18:
        raise HighLevelTrainingError("canonical spatial library is not 18 unique actions")
    return library


@dataclass(frozen=True)
class BackendReset:
    packet: MatchedModalityPacket
    first_pass_native_samples: int
    first_pass_force_violation: bool
    first_pass_peak_force_n: float = 0.0


@dataclass(frozen=True)
class BackendTransition:
    packet: MatchedModalityPacket
    training_residual_excess_potential_gain: float
    duration_s: float
    shield_intervened: bool
    force_limit_violation: bool
    spatial_constraint_violation: bool
    physical_lifecycle_complete: bool
    backend_terminal: bool
    executed_primitive_index: int | None
    native_samples: int
    native_peak_force_n: float = 0.0
    synthetic_coverage_success: bool | None = None
    synthetic_residual_mass_ratio: float | None = None

    def validate(self) -> None:
        numeric = (
            self.training_residual_excess_potential_gain,
            self.duration_s,
            self.native_peak_force_n,
        )
        if not all(math.isfinite(float(value)) and float(value) >= 0 for value in numeric):
            raise HighLevelTrainingError("backend transition metrics must be nonnegative and finite")
        if float(self.training_residual_excess_potential_gain) > 1.0:
            raise HighLevelTrainingError("residual-excess potential gain exceeds one")
        if int(self.native_samples) < 0:
            raise HighLevelTrainingError("backend native sample count must be nonnegative")
        if self.synthetic_residual_mass_ratio is not None:
            value = float(self.synthetic_residual_mass_ratio)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise HighLevelTrainingError("synthetic residual mass ratio is invalid")
            if self.synthetic_coverage_success is None:
                raise HighLevelTrainingError("terminal synthetic score lacks a success flag")


class PrimitiveLifecycleBackend(Protocol):
    mode: ObservationMode

    def reset(self, *, seed: int) -> BackendReset: ...

    def execute_primitive(
        self, primitive: PrimitiveProposal, *, primitive_index: int
    ) -> BackendTransition: ...

    def request_stop(self) -> BackendTransition: ...

    def close(self) -> None: ...


def packet_to_training_observation(packet: MatchedModalityPacket) -> dict[str, np.ndarray]:
    if packet.privileged_oracle:
        raise HighLevelTrainingError("oracle packet is forbidden in high-level training")
    decision = np.asarray(packet.decision_vector, dtype=np.float32)
    expected_decision = 18 + 2 * 64
    if decision.shape != (expected_decision,) or not np.all(np.isfinite(decision)):
        raise HighLevelTrainingError("decision packet differs from the frozen 64-cell schema")
    history = np.asarray(packet.force_history_flat, dtype=np.float32)
    mask = np.asarray(packet.force_history_mask, dtype=np.float32)
    if history.shape != (32 * 9,) or mask.shape != (32,):
        raise HighLevelTrainingError("force-history packet differs from the frozen schema")
    if not np.all(np.isfinite(history)) or not np.all((mask == 0) | (mask == 1)):
        raise HighLevelTrainingError("force-history packet is nonfinite or has a nonbinary mask")
    return {
        "common_core": decision[:18].copy(),
        "visual_features": decision[18:].copy(),
        "force_history": history.reshape(32, 9).copy(),
        "force_history_mask": mask.copy(),
    }


class HighLevelLifecycleEnv(gym.Env):
    """One high-level step executes one primitive; the final action requests stop."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        backend: PrimitiveLifecycleBackend,
        primitive_library: tuple[PrimitiveProposal, ...],
        *,
        maximum_primitives: int = 4,
        reward_config: SharedRewardConfig = SharedRewardConfig(),
        allow_stop_action: bool = True,
    ) -> None:
        super().__init__()
        if not primitive_library or int(maximum_primitives) <= 0:
            raise HighLevelTrainingError("primitive library and budget must be nonempty")
        for primitive in primitive_library:
            primitive.validate_finite()
        reward_config.validate()
        self.backend = backend
        self.primitive_library = tuple(primitive_library)
        self.maximum_primitives = int(maximum_primitives)
        self.reward_config = reward_config
        self.allow_stop_action = bool(allow_stop_action)
        self.stop_action = len(self.primitive_library) if self.allow_stop_action else None
        self.action_space = gym.spaces.Discrete(
            len(self.primitive_library) + int(self.allow_stop_action)
        )
        self.observation_space = gym.spaces.Dict(
            {
                "common_core": gym.spaces.Box(-np.inf, np.inf, (18,), np.float32),
                "visual_features": gym.spaces.Box(0.0, 1.0, (128,), np.float32),
                "force_history": gym.spaces.Box(-np.inf, np.inf, (32, 9), np.float32),
                "force_history_mask": gym.spaces.Box(0.0, 1.0, (32,), np.float32),
            }
        )
        self._primitive_count = 0
        self._closed = False
        self._episode_active = False

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        if self._closed:
            raise HighLevelTrainingError("cannot reset a closed lifecycle environment")
        if seed is None:
            raise HighLevelTrainingError("every training episode requires an explicit seed")
        result = self.backend.reset(seed=int(seed))
        if result.first_pass_force_violation:
            raise HighLevelTrainingError("first pass violated the force limit before policy control")
        self._primitive_count = 0
        self._episode_active = True
        observation = packet_to_training_observation(result.packet)
        if not self.observation_space.contains(observation):
            raise HighLevelTrainingError("backend reset packet is outside the observation space")
        return observation, {
            "first_pass_native_samples": int(result.first_pass_native_samples),
            "mode": ObservationMode(result.packet.mode).value,
            "first_pass_peak_force_n": float(result.first_pass_peak_force_n),
        }

    def step(self, action):
        if not self._episode_active or self._closed:
            raise HighLevelTrainingError("step requires one active nonclosed episode")
        if not self.action_space.contains(action):
            raise HighLevelTrainingError("high-level action is outside the frozen action space")
        action_index = int(action)
        stop_requested = bool(
            self.allow_stop_action and action_index == self.stop_action
        )
        if stop_requested:
            transition = self.backend.request_stop()
        else:
            transition = self.backend.execute_primitive(
                self.primitive_library[action_index],
                primitive_index=action_index,
            )
            self._primitive_count += 1
        transition.validate()
        budget_exhausted = self._primitive_count >= self.maximum_primitives
        terminated = bool(
            transition.backend_terminal
            or transition.physical_lifecycle_complete
            or stop_requested
            or budget_exhausted
            or transition.force_limit_violation
            or transition.spatial_constraint_violation
        )
        outcome = ObservableDecisionOutcome(
            training_residual_excess_potential_gain=float(
                transition.training_residual_excess_potential_gain
            ),
            duration_s=float(transition.duration_s),
            shield_intervened=bool(transition.shield_intervened),
            force_limit_violation=bool(transition.force_limit_violation),
            spatial_constraint_violation=bool(transition.spatial_constraint_violation),
            physical_lifecycle_complete=bool(
                transition.physical_lifecycle_complete
            ),
            training_terminal_residual_mass_ratio=(
                transition.synthetic_residual_mass_ratio if terminated else None
            ),
            episode_terminated=terminated,
        )
        reward = compute_shared_decision_reward(outcome, config=self.reward_config)
        observation = packet_to_training_observation(transition.packet)
        if not self.observation_space.contains(observation):
            raise HighLevelTrainingError("backend transition packet is outside the observation space")
        self._episode_active = not terminated
        info = {
            "primitive_count": self._primitive_count,
            "budget_exhausted": budget_exhausted,
            "stop_requested": stop_requested,
            "shield_intervened": bool(transition.shield_intervened),
            "force_limit_violation": bool(transition.force_limit_violation),
            "spatial_constraint_violation": bool(
                transition.spatial_constraint_violation
            ),
            "physical_lifecycle_complete": bool(
                transition.physical_lifecycle_complete
            ),
            "synthetic_coverage_success": transition.synthetic_coverage_success,
            "synthetic_residual_mass_ratio": transition.synthetic_residual_mass_ratio,
            "training_residual_excess_potential_gain": float(
                transition.training_residual_excess_potential_gain
            ),
            "executed_primitive_index": transition.executed_primitive_index,
            "native_samples": int(transition.native_samples),
            "native_peak_force_n": float(transition.native_peak_force_n),
        }
        return observation, float(reward), terminated, False, info

    def close(self) -> None:
        if not self._closed:
            self.backend.close()
        self._closed = True
        self._episode_active = False


class ContractSmokeBackend:
    """Deterministic nonphysical backend used only for API and trainer smoke tests."""

    def __init__(self, mode: ObservationMode, *, maximum_primitives: int = 4) -> None:
        self.mode = ObservationMode(mode)
        if self.mode not in {
            ObservationMode.VISION_ONLY,
            ObservationMode.FORCE_ONLY,
            ObservationMode.FUSION,
        }:
            raise HighLevelTrainingError("smoke backend supports matched nonoracle modes")
        self.maximum_primitives = int(maximum_primitives)
        self._step = 0
        self._closed = False

    def _packet(self) -> MatchedModalityPacket:
        visual = self.mode in {ObservationMode.VISION_ONLY, ObservationMode.FUSION}
        force = self.mode in {ObservationMode.FORCE_ONLY, ObservationMode.FUSION}
        core = np.zeros(18, dtype=np.float64)
        core[list(ObservationMode).index(self.mode)] = 1.0
        residual = np.linspace(0.8, 0.2, 64) * max(0.0, 1.0 - 0.15 * self._step)
        visual_features = np.concatenate((residual, np.full(64, 0.01))) if visual else np.zeros(128)
        decision = np.concatenate((core, visual_features))
        history = np.zeros((32, 9), dtype=np.float64)
        mask = np.zeros(32, dtype=np.float64)
        if force:
            history[-min(self._step + 1, 32) :, 0] = 8.0 / 15.0
            mask[-min(self._step + 1, 32) :] = 1.0
        return MatchedModalityPacket(
            mode=self.mode,
            sample_index=5 * (self._step + 1),
            time_ns=50_000_000 * (self._step + 1),
            decision_vector=tuple(decision),
            force_history_flat=tuple(history.reshape(-1)),
            force_history_mask=tuple(mask),
            force_history_length=32,
            force_history_feature_count=9,
            visual_frame=None,
            visual_available=visual,
            visual_source_sample_index=5 * (self._step + 1) if visual else None,
            visual_age_samples=0 if visual else None,
            privileged_oracle=False,
        )

    def reset(self, *, seed: int) -> BackendReset:
        if self._closed:
            raise HighLevelTrainingError("smoke backend is closed")
        self._step = 0
        return BackendReset(self._packet(), 100, False, 8.0)

    def execute_primitive(
        self, primitive: PrimitiveProposal, *, primitive_index: int
    ) -> BackendTransition:
        primitive.validate_finite()
        self._step += 1
        terminal = self._step >= self.maximum_primitives
        ratio_before = max(0.0, 0.5 - 0.1 * (self._step - 1))
        ratio_after = max(0.0, 0.5 - 0.1 * self._step)
        potential_gain = max(ratio_before - 0.15, 0.0) - max(
            ratio_after - 0.15, 0.0
        )
        residual_ratio = ratio_after if terminal else None
        return BackendTransition(
            packet=self._packet(),
            training_residual_excess_potential_gain=potential_gain,
            duration_s=1.0,
            shield_intervened=False,
            force_limit_violation=False,
            spatial_constraint_violation=False,
            physical_lifecycle_complete=terminal,
            backend_terminal=terminal,
            executed_primitive_index=int(primitive_index),
            native_samples=100,
            native_peak_force_n=8.0,
            synthetic_coverage_success=(
                residual_ratio <= 0.15 if residual_ratio is not None else None
            ),
            synthetic_residual_mass_ratio=residual_ratio,
        )

    def request_stop(self) -> BackendTransition:
        return BackendTransition(
            packet=self._packet(),
            training_residual_excess_potential_gain=0.0,
            duration_s=0.0,
            shield_intervened=False,
            force_limit_violation=False,
            spatial_constraint_violation=False,
            physical_lifecycle_complete=True,
            backend_terminal=True,
            executed_primitive_index=None,
            native_samples=0,
            native_peak_force_n=0.0,
            synthetic_coverage_success=True,
            synthetic_residual_mass_ratio=0.0,
        )

    def close(self) -> None:
        self._closed = True
