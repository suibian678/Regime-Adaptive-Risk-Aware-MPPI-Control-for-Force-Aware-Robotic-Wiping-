"""Auditable PPO components for the V4 matched observation study."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .decision_protocol import DecisionProtocolError
from .information_flow import ObservationMode
from .matched_policy_encoder import (
    MatchedPolicyEncoder,
    MatchedPolicyEncoderConfig,
    trainable_parameter_count,
)
from .training_evidence import ScientificEpisodeFailure


class ScientificRolloutExhaustion(DecisionProtocolError):
    """Scientific episode failures exhausted the frozen collection schedule."""


@dataclass(frozen=True)
class PPOConfig:
    hidden_dimension: int = 256
    action_count: int = 19
    clip_ratio: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    discount: float = 0.99
    gae_lambda: float = 0.95
    learning_rate: float = 3e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    adam_weight_decay: float = 0.0
    maximum_gradient_norm: float = 0.5
    update_epochs: int = 10
    minibatches_per_epoch: int = 2

    def validate(self) -> None:
        positive = (
            self.hidden_dimension,
            self.action_count,
            self.learning_rate,
            self.adam_epsilon,
            self.maximum_gradient_norm,
            self.update_epochs,
            self.minibatches_per_epoch,
        )
        if not all(float(value) > 0 for value in positive):
            raise DecisionProtocolError("PPO dimensions and optimizer settings must be positive")
        if not 0 < self.clip_ratio < 1 or not 0 < self.discount <= 1 or not 0 < self.gae_lambda <= 1:
            raise DecisionProtocolError("PPO clip/discount/GAE values are invalid")
        if self.value_coefficient < 0 or self.entropy_coefficient < 0:
            raise DecisionProtocolError("PPO loss coefficients must be nonnegative")
        if not 0 <= self.adam_beta1 < 1 or not 0 <= self.adam_beta2 < 1:
            raise DecisionProtocolError("Adam beta values must be in [0, 1)")
        if self.adam_weight_decay < 0:
            raise DecisionProtocolError("Adam weight decay must be nonnegative")


class PPOActorCritic(nn.Module):
    """Matched modality adapter followed by a shared actor-critic trunk."""

    def __init__(
        self,
        mode: ObservationMode,
        *,
        encoder_config: MatchedPolicyEncoderConfig = MatchedPolicyEncoderConfig(),
        ppo_config: PPOConfig = PPOConfig(),
    ) -> None:
        super().__init__()
        ppo_config.validate()
        self.mode = ObservationMode(mode)
        self.encoder_config = encoder_config
        self.ppo_config = ppo_config
        self.encoder = MatchedPolicyEncoder(self.mode, encoder_config)
        representation = encoder_config.common_core_dimension + encoder_config.embedding_dimension
        hidden = int(ppo_config.hidden_dimension)
        self.trunk = nn.Sequential(
            nn.Linear(representation, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden, int(ppo_config.action_count))
        self.critic = nn.Linear(hidden, 1)

    def forward(self, observation: Mapping[str, torch.Tensor]):
        encoded = self.encoder(
            observation["common_core"],
            visual_features=(
                observation["visual_features"]
                if self.mode in {ObservationMode.VISION_ONLY, ObservationMode.FUSION}
                else None
            ),
            force_history=(
                observation["force_history"]
                if self.mode in {ObservationMode.FORCE_ONLY, ObservationMode.FUSION}
                else None
            ),
            force_history_mask=(
                observation["force_history_mask"]
                if self.mode in {ObservationMode.FORCE_ONLY, ObservationMode.FUSION}
                else None
            ),
        )
        feature = self.trunk(encoded)
        return self.actor(feature), self.critic(feature).squeeze(-1)

    @torch.no_grad()
    def act(
        self,
        observation: Mapping[str, torch.Tensor],
        *,
        deterministic: bool = False,
        generator: torch.Generator | None = None,
    ):
        logits, value = self.forward(observation)
        distribution = Categorical(logits=logits)
        action = (
            torch.argmax(logits, dim=-1)
            if deterministic
            else torch.multinomial(
                torch.softmax(logits, dim=-1),
                num_samples=1,
                generator=generator,
            ).squeeze(-1)
        )
        return action, distribution.log_prob(action), value


@dataclass(frozen=True)
class PPORolloutBatch:
    observations: Mapping[str, torch.Tensor]
    actions: torch.Tensor
    old_log_probabilities: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor

    def validate(self) -> int:
        count = int(self.actions.shape[0])
        if count <= 0:
            raise DecisionProtocolError("PPO rollout batch must be nonempty")
        tensors = (
            self.actions,
            self.old_log_probabilities,
            self.returns,
            self.advantages,
        )
        if any(tensor.shape != (count,) for tensor in tensors):
            raise DecisionProtocolError("PPO rollout vectors have inconsistent shape")
        if any(value.shape[0] != count for value in self.observations.values()):
            raise DecisionProtocolError("PPO observation batch has inconsistent length")
        return count


def generalized_advantage_estimate(
    rewards: np.ndarray,
    values: np.ndarray,
    terminated: np.ndarray,
    *,
    bootstrap_value: float,
    discount: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    rewards = np.asarray(rewards, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    terminated = np.asarray(terminated, dtype=bool)
    if rewards.ndim != 1 or values.shape != rewards.shape or terminated.shape != rewards.shape:
        raise DecisionProtocolError("GAE arrays must be one-dimensional and aligned")
    advantages = np.zeros_like(rewards)
    next_advantage = 0.0
    next_value = float(bootstrap_value)
    for index in range(len(rewards) - 1, -1, -1):
        continuation = 0.0 if terminated[index] else 1.0
        delta = rewards[index] + discount * continuation * next_value - values[index]
        next_advantage = delta + discount * gae_lambda * continuation * next_advantage
        advantages[index] = next_advantage
        next_value = values[index]
    returns = advantages + values
    return advantages.astype(np.float32), returns.astype(np.float32)


class PPOTrainer:
    def __init__(self, model: PPOActorCritic, *, device: torch.device | str = "cpu"):
        self.model = model.to(device)
        self.device = torch.device(device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(model.ppo_config.learning_rate),
            betas=(
                float(model.ppo_config.adam_beta1),
                float(model.ppo_config.adam_beta2),
            ),
            eps=float(model.ppo_config.adam_epsilon),
            weight_decay=float(model.ppo_config.adam_weight_decay),
        )

    def update(self, batch: PPORolloutBatch, *, generator: torch.Generator) -> dict[str, float]:
        count = batch.validate()
        cfg = self.model.ppo_config
        advantages = batch.advantages.to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        actions = batch.actions.to(self.device)
        old_log_probabilities = batch.old_log_probabilities.to(self.device)
        returns = batch.returns.to(self.device)
        observations = {name: value.to(self.device) for name, value in batch.observations.items()}
        totals = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "updates": 0}
        minibatches = int(cfg.minibatches_per_epoch)
        if count < minibatches:
            raise DecisionProtocolError(
                "PPO rollout must contain at least one sample per frozen minibatch"
            )
        for _epoch in range(int(cfg.update_epochs)):
            order = torch.randperm(count, generator=generator)
            # Exactly two nearly equal minibatches are used for every formal
            # 128--131 transition rollout.  Episode-boundary extension cannot
            # create an additional optimizer step.
            for split in torch.tensor_split(order, minibatches):
                indices = split.to(self.device)
                selected = {name: value[indices] for name, value in observations.items()}
                logits, value = self.model(selected)
                distribution = Categorical(logits=logits)
                new_log_probability = distribution.log_prob(actions[indices])
                ratio = torch.exp(new_log_probability - old_log_probabilities[indices])
                unclipped = ratio * advantages[indices]
                clipped = torch.clamp(
                    ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio
                ) * advantages[indices]
                policy_loss = -torch.mean(torch.minimum(unclipped, clipped))
                value_loss = torch.mean((value - returns[indices]) ** 2)
                entropy = torch.mean(distribution.entropy())
                loss = (
                    policy_loss
                    + cfg.value_coefficient * value_loss
                    - cfg.entropy_coefficient * entropy
                )
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), cfg.maximum_gradient_norm
                )
                self.optimizer.step()
                totals["policy_loss"] += float(policy_loss.detach().cpu())
                totals["value_loss"] += float(value_loss.detach().cpu())
                totals["entropy"] += float(entropy.detach().cpu())
                totals["updates"] += 1
        updates = int(totals.pop("updates"))
        return {name: value / updates for name, value in totals.items()} | {
            "optimizer_updates": float(updates)
        }


@dataclass(frozen=True)
class PPORolloutAudit:
    transitions: int
    episodes_started: int
    episodes_completed: int
    physical_lifecycle_completions: int
    synthetic_coverage_successes: int
    force_limit_violations: int
    spatial_constraint_violations: int
    native_samples: int
    peak_force_n: float
    scientific_episode_failures: int
    scientific_failure_counts: Mapping[str, int]


def _tensor_observation(
    observation: Mapping[str, np.ndarray], *, device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        name: torch.as_tensor(value, dtype=torch.float32, device=device).unsqueeze(0)
        for name, value in observation.items()
    }


def collect_ppo_rollout(
    env: Any,
    model: PPOActorCritic,
    *,
    rollout_steps: int,
    episode_seeds: Iterable[int],
    device: torch.device | str = "cpu",
) -> tuple[PPORolloutBatch, PPORolloutAudit]:
    """Collect an exact on-policy batch with explicit per-episode seeds."""

    collector = PPORolloutCollector(
        env, model, episode_seeds=episode_seeds, device=device
    )
    return collector.collect(rollout_steps)


class PPORolloutCollector:
    """Stateful collector that preserves an unfinished physical episode."""

    def __init__(
        self,
        env: Any,
        model: PPOActorCritic,
        *,
        episode_seeds: Iterable[int],
        device: torch.device | str = "cpu",
        action_generator: torch.Generator | None = None,
    ) -> None:
        self.env = env
        self.model = model
        self.seed_iterator = iter(episode_seeds)
        self.device = torch.device(device)
        self.observation: Mapping[str, np.ndarray] | None = None
        self.episodes_started_total = 0
        self.episodes_completed_total = 0
        self.scientific_failures_total = 0
        self.scientific_failure_counts_total: dict[str, int] = {}
        if action_generator is None:
            generator_device = "cuda" if self.device.type == "cuda" else "cpu"
            action_generator = torch.Generator(device=generator_device)
            action_generator.manual_seed(torch.initial_seed())
        self.action_generator = action_generator

    def _reset(self) -> None:
        failures = 0
        while True:
            try:
                seed = int(next(self.seed_iterator))
            except StopIteration as exc:
                raise ScientificRolloutExhaustion(
                    "episode-seed sequence was exhausted"
                ) from exc
            self.episodes_started_total += 1
            try:
                self.observation, _reset_info = self.env.reset(seed=seed)
                return
            except ScientificEpisodeFailure as error:
                failures += 1
                self.episodes_completed_total += 1
                self.scientific_failures_total += 1
                key = error.code.name
                self.scientific_failure_counts_total[key] = (
                    self.scientific_failure_counts_total.get(key, 0) + 1
                )
                if failures >= 64:
                    raise ScientificRolloutExhaustion(
                        "64 consecutive recorded scientific failures produced no rollout transition"
                    ) from error

    def export_boundary_state(self) -> dict[str, Any]:
        if self.observation is not None:
            raise DecisionProtocolError("collector state is not at an episode boundary")
        return {
            "episodes_started_total": int(self.episodes_started_total),
            "episodes_completed_total": int(self.episodes_completed_total),
            "scientific_failures_total": int(self.scientific_failures_total),
            "scientific_failure_counts_total": dict(
                self.scientific_failure_counts_total
            ),
            "action_generator_state": self.action_generator.get_state().cpu(),
        }

    def load_boundary_state(self, state: Mapping[str, Any]) -> None:
        if self.observation is not None:
            raise DecisionProtocolError("cannot restore over an active collector episode")
        self.episodes_started_total = int(state["episodes_started_total"])
        self.episodes_completed_total = int(state["episodes_completed_total"])
        self.scientific_failures_total = int(state["scientific_failures_total"])
        self.scientific_failure_counts_total = {
            str(key): int(value)
            for key, value in dict(state["scientific_failure_counts_total"]).items()
        }
        generator_state = state["action_generator_state"]
        if not isinstance(generator_state, torch.Tensor):
            raise DecisionProtocolError("collector checkpoint lacks action RNG state")
        self.action_generator.set_state(generator_state.cpu())

    def collect(
        self, rollout_steps: int, *, finish_episode: bool = False
    ) -> tuple[PPORolloutBatch, PPORolloutAudit]:
        if int(rollout_steps) <= 0:
            raise DecisionProtocolError("PPO rollout length must be positive")
        episodes_started_before = self.episodes_started_total
        episodes_completed_before = self.episodes_completed_total
        scientific_failures_before = self.scientific_failures_total
        failure_counts_before = dict(self.scientific_failure_counts_total)
        if self.observation is None:
            self._reset()
        physical_lifecycle_completions = 0
        synthetic_coverage_successes = 0
        force_limit_violations = 0
        spatial_constraint_violations = 0
        native_samples = 0
        peak_force_n = 0.0
        observations: dict[str, list[torch.Tensor]] = {
            name: [] for name in self.observation
        }
        actions: list[torch.Tensor] = []
        log_probabilities: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        rewards: list[float] = []
        terminals: list[bool] = []
        last_terminal = False
        self.model.to(self.device)
        self.model.eval()
        _index = 0
        while _index < int(rollout_steps) or (finish_episode and not last_terminal):
            tensor_observation = _tensor_observation(
                self.observation, device=self.device
            )
            with torch.no_grad():
                action, log_probability, value = self.model.act(
                    tensor_observation, generator=self.action_generator
                )
            next_observation, reward, terminated, truncated, info = self.env.step(
                int(action.item())
            )
            if truncated:
                raise DecisionProtocolError(
                    "training backend must encode finite-horizon endings as terminals"
                )
            for name, tensor in tensor_observation.items():
                observations[name].append(tensor.squeeze(0).detach().cpu())
            actions.append(action.squeeze(0).detach().cpu())
            log_probabilities.append(log_probability.squeeze(0).detach().cpu())
            values.append(value.squeeze(0).detach().cpu())
            rewards.append(float(reward))
            last_terminal = bool(terminated)
            terminals.append(last_terminal)
            physical_lifecycle_completions += int(
                bool(info["physical_lifecycle_complete"])
            )
            synthetic_coverage_successes += int(
                info["synthetic_coverage_success"] is True
            )
            force_limit_violations += int(bool(info["force_limit_violation"]))
            spatial_constraint_violations += int(bool(info["spatial_constraint_violation"]))
            native_samples += int(info["native_samples"])
            peak_force_n = max(peak_force_n, float(info["native_peak_force_n"]))
            self.observation = next_observation
            if last_terminal:
                self.episodes_completed_total += 1
                if _index + 1 < int(rollout_steps):
                    self._reset()
            _index += 1
        if last_terminal:
            bootstrap_value = 0.0
            # Defer reset until the next collection so it is counted there.
            self.observation = None
        else:
            with torch.no_grad():
                _logits, final_value = self.model(
                    _tensor_observation(self.observation, device=self.device)
                )
            bootstrap_value = float(final_value.item())
        advantages, returns = generalized_advantage_estimate(
            np.asarray(rewards, dtype=np.float64),
            np.asarray([float(value.item()) for value in values], dtype=np.float64),
            np.asarray(terminals, dtype=bool),
            bootstrap_value=bootstrap_value,
            discount=float(self.model.ppo_config.discount),
            gae_lambda=float(self.model.ppo_config.gae_lambda),
        )
        batch = PPORolloutBatch(
            observations={name: torch.stack(rows) for name, rows in observations.items()},
            actions=torch.stack(actions).long(),
            old_log_probabilities=torch.stack(log_probabilities).float(),
            returns=torch.from_numpy(returns),
            advantages=torch.from_numpy(advantages),
        )
        batch.validate()
        return batch, PPORolloutAudit(
            transitions=len(actions),
            episodes_started=self.episodes_started_total - episodes_started_before,
            episodes_completed=self.episodes_completed_total - episodes_completed_before,
            physical_lifecycle_completions=physical_lifecycle_completions,
            synthetic_coverage_successes=synthetic_coverage_successes,
            force_limit_violations=force_limit_violations,
            spatial_constraint_violations=spatial_constraint_violations,
            native_samples=native_samples,
            peak_force_n=peak_force_n,
            scientific_episode_failures=(
                self.scientific_failures_total - scientific_failures_before
            ),
            scientific_failure_counts={
                key: value - failure_counts_before.get(key, 0)
                for key, value in self.scientific_failure_counts_total.items()
                if value - failure_counts_before.get(key, 0) > 0
            },
        )


def ppo_parameter_audit(
    *,
    encoder_config: MatchedPolicyEncoderConfig = MatchedPolicyEncoderConfig(),
    ppo_config: PPOConfig = PPOConfig(),
) -> dict[str, Any]:
    counts = {
        mode.value: trainable_parameter_count(
            PPOActorCritic(mode, encoder_config=encoder_config, ppo_config=ppo_config)
        )
        for mode in (
            ObservationMode.VISION_ONLY,
            ObservationMode.FORCE_ONLY,
            ObservationMode.FUSION,
        )
    }
    mean = float(np.mean(list(counts.values())))
    spread = float((max(counts.values()) - min(counts.values())) / mean)
    return {
        "trainable_parameters": counts,
        "mean_trainable_parameters": mean,
        "spread_fraction": spread,
        "pass_at_one_percent": bool(spread <= 0.01),
    }


def save_ppo_checkpoint(
    path: Path,
    model: PPOActorCritic,
    trainer: PPOTrainer,
    *,
    metadata: Mapping[str, Any],
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise DecisionProtocolError("immutable PPO checkpoint already exists")
    temporary = destination.with_name(destination.name + ".incomplete")
    if temporary.exists():
        raise DecisionProtocolError("stale incomplete PPO checkpoint exists")
    torch.save(
        {
            "format": "forcewipe_v4_ppo_observation_v1",
            "mode": model.mode.value,
            "encoder_config": asdict(model.encoder_config),
            "ppo_config": asdict(model.ppo_config),
            "model_state_dict": {
                name: value.detach().cpu() for name, value in model.state_dict().items()
            },
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "torch_cpu_rng_state": torch.get_rng_state(),
            "metadata": dict(metadata),
        },
        temporary,
    )
    os.replace(temporary, destination)


def ppo_checkpoint_matches(
    path: Path,
    model: PPOActorCritic,
    trainer: PPOTrainer,
    *,
    metadata: Mapping[str, Any],
) -> bool:
    """Return whether an orphan immutable checkpoint equals a replayed update."""

    source = Path(path)
    if not source.is_file():
        return False
    payload = torch.load(source, map_location="cpu", weights_only=True)
    if payload.get("format") != "forcewipe_v4_ppo_observation_v1":
        return False
    def equal(left, right) -> bool:
        if isinstance(left, torch.Tensor):
            return isinstance(right, torch.Tensor) and torch.equal(
                left.cpu(), right.cpu()
            )
        if isinstance(left, Mapping):
            return isinstance(right, Mapping) and set(left) == set(right) and all(
                equal(left[key], right[key]) for key in left
            )
        if isinstance(left, (list, tuple)):
            return isinstance(right, (list, tuple)) and len(left) == len(right) and all(
                equal(a, b) for a, b in zip(left, right)
            )
        return left == right

    expected = {
        "mode": model.mode.value,
        "encoder_config": asdict(model.encoder_config),
        "ppo_config": asdict(model.ppo_config),
        "metadata": dict(metadata),
    }
    if any(not equal(payload.get(key), value) for key, value in expected.items()):
        return False

    expected_model = {
        name: value.detach().cpu() for name, value in model.state_dict().items()
    }
    return bool(
        equal(payload.get("model_state_dict"), expected_model)
        and equal(payload.get("optimizer_state_dict"), trainer.optimizer.state_dict())
    )


def load_ppo_checkpoint(
    path: Path,
    *,
    expected_mode: ObservationMode | None = None,
    device: torch.device | str = "cpu",
) -> tuple[PPOActorCritic, PPOTrainer, dict[str, Any]]:
    """Load the tensor-only V4 checkpoint and validate its architecture identity."""

    source = Path(path)
    if not source.is_file():
        raise DecisionProtocolError("PPO checkpoint does not exist")
    payload = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != "forcewipe_v4_ppo_observation_v1":
        raise DecisionProtocolError("unsupported PPO checkpoint format")
    mode = ObservationMode(payload["mode"])
    if expected_mode is not None and mode is not ObservationMode(expected_mode):
        raise DecisionProtocolError("PPO checkpoint modality differs from the requested run")
    encoder_config = MatchedPolicyEncoderConfig(**dict(payload["encoder_config"]))
    ppo_config = PPOConfig(**dict(payload["ppo_config"]))
    model = PPOActorCritic(
        mode, encoder_config=encoder_config, ppo_config=ppo_config
    )
    trainer = PPOTrainer(model, device=device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    trainer.optimizer.load_state_dict(payload["optimizer_state_dict"])
    rng_state = payload.get("torch_cpu_rng_state")
    if not isinstance(rng_state, torch.Tensor):
        raise DecisionProtocolError("PPO checkpoint lacks the CPU RNG state")
    torch.set_rng_state(rng_state.cpu())
    metadata = dict(payload.get("metadata", {}))
    return model, trainer, metadata
