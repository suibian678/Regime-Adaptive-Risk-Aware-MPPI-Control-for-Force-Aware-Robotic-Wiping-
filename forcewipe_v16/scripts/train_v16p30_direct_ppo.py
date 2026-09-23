#!/usr/bin/env python3
"""Train one direct-PPO seed with the matched 82,774-transition budget."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import traceback

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
TEACHER_ROOT = ROOT.parent
for source in (
    ROOT / "code", TEACHER_ROOT / "forcewipe_v6" / "code",
    TEACHER_ROOT / "forcewipe_v4" / "code",
    Path("/tmp/forcewipe_legacy_workspace/tdmpc2"),
    Path("/tmp/forcewipe_legacy_workspace"),
):
    sys.path.insert(0, str(source))

import forcewipe_v6.tdmpc2_direct_sapien_env as direct_env_module  # noqa: E402
from forcewipe_v6.tdmpc2_direct_firstpass import DirectFirstPassConfig  # noqa: E402
from forcewipe_v6.tdmpc2_direct_sapien_env import V6DirectFirstPassEnv  # noqa: E402
from forcewipe_v16.direct_ppo_baseline import (  # noqa: E402
    DirectPPOActorCritic, DirectPPOConfig, DirectPPOOptimizer,
    checkpoint_payload, generalized_advantage_estimate,
)
from forcewipe_v16.v16p30_ppo_scenarios import (  # noqa: E402
    scenario_identity, training_cell_schedule, v16p30_ppo_scenario,
)


ALLOWED_SEEDS = (301, 302, 303, 304, 305)
TRAIN_ROOT = ROOT / "results" / "train" / "v16p30_direct_ppo"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


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


def run(seed: int) -> Path:
    if int(seed) not in ALLOWED_SEEDS:
        raise ValueError(f"seed must be one of {ALLOWED_SEEDS}")
    run_id = f"v16p30_direct_ppo_seed{seed}_transitions82774_20260830_r2"
    final = TRAIN_ROOT / run_id
    stage = TRAIN_ROOT / f".{run_id}.creating"
    if final.exists() or stage.exists():
        raise RuntimeError("training run id already exists")
    TRAIN_ROOT.mkdir(parents=True, exist_ok=True)
    stage.mkdir()
    direct_env_module.nominal_direct_scenario = v16p30_ppo_scenario
    if direct_env_module.nominal_direct_scenario is not v16p30_ppo_scenario:
        raise RuntimeError("failed to bind frozen PPO TRAIN scenario loader")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = DirectPPOConfig()
    model = DirectPPOActorCritic(config)
    optimizer = DirectPPOOptimizer(model, device=device)
    action_generator = torch.Generator(device=device.type).manual_seed(716_330_000 + seed)
    update_generator = torch.Generator().manual_seed(816_330_000 + seed)
    collector = OnPolicyCollector(model, device=device, action_generator=action_generator)
    started = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    atomic_json(stage / "RUN_DEFINITION.json", {
        "format": "forcewipe_v16p30_direct_ppo_training_definition_v1",
        "run_id": run_id, "seed": seed, "started_utc": started,
        "device": str(device), "config": config.__dict__,
        "matched_budget": {
            "ppo_environment_transitions": config.total_environment_transitions,
            "tdmpc2_training_source_transitions": 82_774,
            "interpretation": "matched simulator/source transition count; optimizer updates and data provenance differ",
        },
        "observation_dimension": 16, "action_dimension": 3,
        "reward": "shared direct_first_pass_reward",
        "exploration_initialization": "zero actor-mean weights; fixed +1.0 pre-tanh inward-action bias",
        "training_scenario_family": "V16.30 PPO TRAIN only; no weak-curvature final blocks",
    })
    metrics = []
    transitions = 0; optimizer_updates = 0
    try:
        while transitions < config.total_environment_transitions:
            count = min(config.rollout_transitions, config.total_environment_transitions - transitions)
            batch = collector.collect(count)
            update = optimizer.update(batch, generator=update_generator)
            transitions += count; optimizer_updates += int(update["optimizer_updates"])
            row = {
                "rollout": len(metrics), "transitions": transitions,
                "optimizer_updates": optimizer_updates,
                "episodes_completed": len(collector.episodes),
                "task_successes": collector.task_successes,
                "force_violation_episodes": collector.force_violations,
                "peak_force_n": collector.peak_force_n, **update,
            }
            metrics.append(row); print(json.dumps(row, sort_keys=True), flush=True)
        collector.close()
        checkpoint = stage / f"direct_ppo_seed{seed}.pt"
        torch.save(checkpoint_payload(
            model, optimizer, seed=seed, transitions=transitions
        ), checkpoint)
        with (stage / "TRAINING_METRICS.jsonl").open("x", encoding="utf-8") as stream:
            for row in metrics:
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        with (stage / "EPISODE_SUMMARIES.jsonl").open("x", encoding="utf-8") as stream:
            for row in collector.episodes:
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        result = {
            "format": "forcewipe_v16p30_direct_ppo_training_result_v1",
            "status": "completed", "run_id": run_id, "seed": seed,
            "environment_transitions": transitions, "optimizer_updates": optimizer_updates,
            "episodes_completed": len(collector.episodes),
            "task_successes_during_training": collector.task_successes,
            "force_violation_episodes_during_training": collector.force_violations,
            "peak_force_n_during_training": collector.peak_force_n,
            "checkpoint": checkpoint.name, "checkpoint_sha256": sha256(checkpoint),
            "final_evaluation_used": False,
        }
        atomic_json(stage / "RESULT.json", result)
        files = sorted(path for path in stage.iterdir() if path.is_file())
        atomic_json(stage / "RUN_FILE_MANIFEST.json", {"files": [
            {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in files
        ]})
        os.replace(stage, final)
        return final
    except BaseException as exc:
        collector.close()
        atomic_json(stage / "INFRASTRUCTURE_FAILURE.json", {
            "status": "aborted_infrastructure", "exception_type": type(exc).__name__,
            "message": str(exc), "traceback": traceback.format_exc(),
            "transitions_completed": transitions,
        })
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, choices=ALLOWED_SEEDS)
    args = parser.parse_args()
    print(run(args.seed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
