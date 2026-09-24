"""Shared direct-PPO on-policy collector."""
from __future__ import annotations
import numpy as np
import torch
from forcewipe.direct.tdmpc2_direct_firstpass import DirectFirstPassConfig
from forcewipe.direct.tdmpc2_direct_sapien_env import V6DirectFirstPassEnv
from forcewipe.learning.direct_ppo_baseline import DirectPPOActorCritic, generalized_advantage_estimate
from forcewipe.learning.ppo_scenarios import scenario_identity, training_cell_schedule


class OnPolicyCollector:
    def __init__(self, model: DirectPPOActorCritic, *, device: torch.device, action_generator: torch.Generator):
        self.model = model
        self.device = device
        self.action_generator = action_generator
        self.schedule = training_cell_schedule()
        self.episode_ordinal = 0
        self.env = None
        self.observation = None
        self.current = None
        self.episode_return = 0.0
        self.episode_steps = 0
        self.episodes: list[dict] = []
        self.task_successes = 0
        self.force_violations = 0
        self.peak_force_n = 0.0

    def _start_episode(self) -> None:
        block_index, target = self.schedule[self.episode_ordinal % len(self.schedule)]
        scenario_id, scenario_seed = scenario_identity(
            role="TRAIN", block_index=block_index, target_force_n=target
        )
        self.env = V6DirectFirstPassEnv(
            target_force_n=target, scenario_seed=scenario_seed,
            scenario_id=scenario_id, config=DirectFirstPassConfig(),
        )
        self.observation, reset_info = self.env.reset(seed=scenario_seed)
        if reset_info.get("direct_tdmpc2_action_authority") is not True:
            raise RuntimeError("direct PPO environment did not expose direct action authority")
        self.current = dict(
            episode_ordinal=self.episode_ordinal, block_index=block_index,
            target_force_n=target, scenario_id=scenario_id, scenario_seed=scenario_seed,
        )
        self.episode_return = 0.0; self.episode_steps = 0
        self.episode_ordinal += 1

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
        self.env = None; self.observation = None

    def collect(self, transitions: int) -> dict[str, np.ndarray]:
        storage = {name: [] for name in (
            "observation", "action", "old_log_probability", "reward", "value",
            "next_value", "terminated", "episode_ended",
        )}
        for _ in range(int(transitions)):
            if self.env is None:
                self._start_episode()
            obs_tensor = torch.as_tensor(
                self.observation, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            with torch.no_grad():
                action_tensor, logp_tensor, value_tensor = self.model.sample(
                    obs_tensor, generator=self.action_generator
                )
            action = action_tensor[0].detach().cpu().numpy().astype(np.float32)
            next_observation, reward, terminated, truncated, info = self.env.step(action)
            ended = bool(terminated or truncated)
            next_tensor = torch.as_tensor(
                next_observation, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            with torch.no_grad():
                _mean, _std, next_value_tensor = self.model.forward(next_tensor)
            storage["observation"].append(np.asarray(self.observation, dtype=np.float32))
            storage["action"].append(action)
            storage["old_log_probability"].append(float(logp_tensor.item()))
            storage["reward"].append(float(reward))
            storage["value"].append(float(value_tensor.item()))
            storage["next_value"].append(float(next_value_tensor.item()))
            storage["terminated"].append(bool(terminated))
            storage["episode_ended"].append(ended)
            self.episode_return += float(reward); self.episode_steps += 1
            self.peak_force_n = max(self.peak_force_n, float(info["peak_force_n"]))
            self.force_violations += int(bool(info["force_limit_violation"]))
            self.observation = next_observation
            if ended:
                success = bool(info["success"])
                self.task_successes += int(success)
                self.episodes.append({
                    **self.current, "steps": self.episode_steps,
                    "return": self.episode_return, "success": success,
                    "force_limit_violation": bool(info["force_limit_violation"]),
                    "peak_force_n": float(info["peak_force_n"]),
                    "completed_dose_bins": int(info["completed_dose_bins"]),
                    "progress": float(info["progress"]),
                })
                self.close()
        arrays = {
            name: np.asarray(values, dtype=(bool if name in {"terminated", "episode_ended"} else np.float32))
            for name, values in storage.items()
        }
        advantage, returns = generalized_advantage_estimate(
            arrays["reward"], arrays["value"], arrays["next_value"],
            arrays["terminated"], arrays["episode_ended"],
            discount=self.model.config.discount, gae_lambda=self.model.config.gae_lambda,
        )
        return {
            "observation": arrays["observation"], "action": arrays["action"],
            "old_log_probability": arrays["old_log_probability"],
            "advantage": advantage, "return": returns,
        }
