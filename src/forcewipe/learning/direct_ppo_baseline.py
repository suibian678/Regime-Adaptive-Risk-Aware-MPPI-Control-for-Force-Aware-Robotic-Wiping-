"""Continuous direct-PPO baseline matched to the V16 direct-control interface."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal


TDMPC2_TRAINING_SOURCE_TRANSITIONS = 82_774


class DirectPPOError(ValueError):
    pass


@dataclass(frozen=True)
class DirectPPOConfig:
    observation_dimension: int = 16
    action_dimension: int = 3
    hidden_dimension: int = 256
    total_environment_transitions: int = TDMPC2_TRAINING_SOURCE_TRANSITIONS
    rollout_transitions: int = 2048
    update_epochs: int = 10
    minibatches_per_epoch: int = 8
    discount: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.005
    learning_rate: float = 3e-4
    adam_epsilon: float = 1e-8
    maximum_gradient_norm: float = 0.5
    initial_log_standard_deviation: float = -0.5
    initial_inward_mean_bias: float = 1.0

    def validate(self) -> None:
        positive = (
            self.observation_dimension, self.action_dimension, self.hidden_dimension,
            self.total_environment_transitions, self.rollout_transitions,
            self.update_epochs, self.minibatches_per_epoch, self.learning_rate,
            self.adam_epsilon, self.maximum_gradient_norm,
        )
        if not all(float(value) > 0 for value in positive):
            raise DirectPPOError("PPO dimensions/budgets must be positive")
        if self.observation_dimension != 16 or self.action_dimension != 3:
            raise DirectPPOError("direct PPO must use the common 16-D/3-D interface")
        if self.total_environment_transitions != TDMPC2_TRAINING_SOURCE_TRANSITIONS:
            raise DirectPPOError("PPO transition budget must match TD-MPC2 source data")
        if not 0 < self.discount <= 1 or not 0 < self.gae_lambda <= 1:
            raise DirectPPOError("invalid discount/GAE")
        if not 0 < self.clip_ratio < 1:
            raise DirectPPOError("invalid PPO clip ratio")


def _squashed_log_probability(distribution: Normal, pre_tanh: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    correction = torch.log(torch.clamp(1.0 - action.square(), min=1e-6))
    return (distribution.log_prob(pre_tanh) - correction).sum(dim=-1)


class DirectPPOActorCritic(nn.Module):
    def __init__(self, config: DirectPPOConfig | None = None) -> None:
        super().__init__()
        self.config = config or DirectPPOConfig()
        self.config.validate()
        hidden = self.config.hidden_dimension
        self.trunk = nn.Sequential(
            nn.Linear(self.config.observation_dimension, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.actor_mean = nn.Linear(hidden, self.config.action_dimension)
        nn.init.zeros_(self.actor_mean.weight)
        nn.init.zeros_(self.actor_mean.bias)
        with torch.no_grad():
            self.actor_mean.bias[2] = self.config.initial_inward_mean_bias
        self.critic = nn.Linear(hidden, 1)
        self.log_standard_deviation = nn.Parameter(torch.full(
            (self.config.action_dimension,), self.config.initial_log_standard_deviation
        ))
        self._last_control_mode = "direct_ppo"

    def forward(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feature = self.trunk(observation)
        mean = self.actor_mean(feature)
        std = torch.exp(torch.clamp(self.log_standard_deviation, -5.0, 1.0)).expand_as(mean)
        value = self.critic(feature).squeeze(-1)
        return mean, std, value

    def sample(self, observation: torch.Tensor, *, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, std, value = self.forward(observation)
        noise = torch.randn(mean.shape, dtype=mean.dtype, device=mean.device, generator=generator)
        pre_tanh = mean + std * noise
        action = torch.tanh(pre_tanh)
        log_probability = _squashed_log_probability(Normal(mean, std), pre_tanh, action)
        return action, log_probability, value

    @torch.no_grad()
    def act(self, observation, *, deterministic: bool = True) -> np.ndarray:
        device = next(self.parameters()).device
        obs = torch.as_tensor(observation, dtype=torch.float32, device=device).reshape(1, -1)
        mean, std, _value = self.forward(obs)
        if deterministic:
            action = torch.tanh(mean)
        else:
            action = torch.tanh(mean + std * torch.randn_like(mean))
        self._last_control_mode = "direct_ppo"
        return action[0].cpu().numpy().astype(np.float32)

    def evaluate_actions(self, observation: torch.Tensor, action: torch.Tensor):
        mean, std, value = self.forward(observation)
        bounded = torch.clamp(action, -1.0 + 1e-6, 1.0 - 1e-6)
        pre_tanh = torch.atanh(bounded)
        distribution = Normal(mean, std)
        log_probability = _squashed_log_probability(distribution, pre_tanh, bounded)
        entropy = distribution.entropy().sum(dim=-1)
        return log_probability, entropy, value


def generalized_advantage_estimate(
    rewards: np.ndarray,
    values: np.ndarray,
    next_values: np.ndarray,
    terminated: np.ndarray,
    episode_ended: np.ndarray,
    *,
    discount: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    rewards = np.asarray(rewards, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    next_values = np.asarray(next_values, dtype=np.float64)
    terminated = np.asarray(terminated, dtype=bool)
    episode_ended = np.asarray(episode_ended, dtype=bool)
    if not (
        rewards.ndim == 1
        and values.shape == rewards.shape
        and next_values.shape == rewards.shape
        and terminated.shape == rewards.shape
        and episode_ended.shape == rewards.shape
    ):
        raise DirectPPOError("GAE arrays are not aligned")
    advantages = np.zeros_like(rewards)
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        bootstrap = 0.0 if terminated[index] else next_values[index]
        delta = rewards[index] + discount * bootstrap - values[index]
        continuation = 0.0 if episode_ended[index] else 1.0
        running = delta + discount * gae_lambda * continuation * running
        advantages[index] = running
    return advantages.astype(np.float32), (advantages + values).astype(np.float32)


class DirectPPOOptimizer:
    def __init__(self, model: DirectPPOActorCritic, *, device: str | torch.device = "cpu") -> None:
        self.model = model.to(device)
        self.device = torch.device(device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.model.config.learning_rate,
            eps=self.model.config.adam_epsilon,
        )

    def update(self, batch: dict[str, np.ndarray], *, generator: torch.Generator) -> dict[str, float]:
        cfg = self.model.config
        tensors = {
            name: torch.as_tensor(value, dtype=torch.float32, device=self.device)
            for name, value in batch.items()
        }
        count = int(tensors["action"].shape[0])
        if count < cfg.minibatches_per_epoch:
            raise DirectPPOError("rollout is smaller than frozen minibatch count")
        advantages = tensors["advantage"]
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        totals = dict(policy_loss=0.0, value_loss=0.0, entropy=0.0, updates=0)
        for _ in range(cfg.update_epochs):
            order = torch.randperm(count, generator=generator)
            for split in torch.tensor_split(order, cfg.minibatches_per_epoch):
                index = split.to(self.device)
                logp, entropy, value = self.model.evaluate_actions(
                    tensors["observation"][index], tensors["action"][index]
                )
                ratio = torch.exp(logp - tensors["old_log_probability"][index])
                raw = ratio * advantages[index]
                clipped = torch.clamp(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio) * advantages[index]
                policy_loss = -torch.minimum(raw, clipped).mean()
                value_loss = (value - tensors["return"][index]).square().mean()
                entropy_mean = entropy.mean()
                loss = policy_loss + cfg.value_coefficient * value_loss - cfg.entropy_coefficient * entropy_mean
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.maximum_gradient_norm)
                self.optimizer.step()
                totals["policy_loss"] += float(policy_loss.detach().cpu())
                totals["value_loss"] += float(value_loss.detach().cpu())
                totals["entropy"] += float(entropy_mean.detach().cpu())
                totals["updates"] += 1
        updates = int(totals.pop("updates"))
        return {name: value / updates for name, value in totals.items()} | {"optimizer_updates": updates}


def checkpoint_payload(model: DirectPPOActorCritic, optimizer: DirectPPOOptimizer, *, seed: int, transitions: int) -> dict:
    return {
        "format": "forcewipe_v16p30_direct_ppo_checkpoint_v1",
        "seed": int(seed), "environment_transitions": int(transitions),
        "config": asdict(model.config), "model": model.state_dict(),
        "optimizer": optimizer.optimizer.state_dict(),
    }
