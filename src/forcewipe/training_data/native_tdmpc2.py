"""Native TD-MPC2 training utilities for V6 direct-control trajectories."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
from forcewipe.paths import REPOSITORY_ROOT

import numpy as np
import torch
from tensordict.tensordict import TensorDict
from omegaconf import OmegaConf

from common import MODEL_SIZE
from common.buffer import Buffer
from common.parser import cfg_to_dataclass
from common import math as td_math
from tdmpc2.tdmpc2 import TDMPC2


def build_v14_config(*, seed: int, work_dir: Path, batch_size: int = 64):
    source = REPOSITORY_ROOT / "vendor" / "tdmpc2" / "config.yaml"
    cfg = OmegaConf.load(source)
    cfg.task = "v6-direct-forcewipe"
    cfg.obs = "state"
    cfg.episodic = True
    cfg.seed = int(seed)
    cfg.exp_name = "v14-v6-closed-loop-joint"
    cfg.model_size = 1
    cfg.steps = 50_000
    cfg.buffer_size = 50_000
    cfg.batch_size = int(batch_size)
    cfg.horizon = 20
    cfg.seed_steps = 0
    cfg.checkpoint = None
    cfg.data_dir = str(work_dir)
    cfg.wandb_project = "disabled"
    cfg.wandb_entity = "disabled"
    cfg.enable_wandb = False
    cfg.save_video = False
    cfg.save_agent = True
    cfg.compile = False
    cfg.mpc = True
    cfg.iterations = 4
    cfg.num_samples = 128
    cfg.num_elites = 16
    cfg.num_pi_trajs = 24
    cfg.min_std = 0.04
    cfg.max_std = 0.70
    cfg.temperature = 0.5
    cfg.reward_coef = 1.0
    cfg.value_coef = 0.5
    cfg.consistency_coef = 10.0
    cfg.termination_coef = 1.0
    cfg.force_pred = True
    cfg.force_coef = 8.0
    cfg.force_obs_idx = 0
    cfg.force_contact_balance = True
    cfg.force_contact_threshold_n = 0.20
    cfg.force_contact_weight = 4.0
    cfg.force_high_threshold_n = 0.667
    cfg.force_high_weight = 10.0
    cfg.force_weight_clip = 14.0
    cfg.envelope_pred = True
    cfg.envelope_coef = 2.0
    cfg.force_regime_pred = True
    cfg.force_regime_coef = 0.5
    cfg.force_regime_target_obs_idx = 1
    cfg.force_regime_band_n = 0.05
    cfg.force_regime_band_fraction = 0.0
    cfg.force_regime_min_force_n = 0.20
    cfg.force_regime_max_force_n = 1.0
    cfg.force_plan = True
    cfg.force_plan_coef = 4.0
    cfg.force_plan_target_obs_idx = 1
    cfg.force_plan_deadband = 0.03
    cfg.mppi_force_hard_cap = 0.98
    cfg.mppi_unsafe_penalty = 1_000.0
    cfg.mppi_actor_mean_init = True
    cfg.demo_bc_online_coef = 2.0
    cfg.demo_bc_online_batch_size = int(batch_size)
    cfg.entropy_coef = 1e-5
    cfg.vmin = -30.0
    cfg.vmax = 30.0
    cfg.num_bins = 121
    cfg.lr = 3e-4
    cfg.enc_lr_scale = 1.0
    cfg.pi_lr_scale = 1.0
    cfg.eval_ema = True
    cfg.eval_ema_tau = 0.995
    cfg.phase_policy = False
    cfg.target_policy = False
    cfg.freeze_pi_updates = False
    cfg.freeze_encoder_updates = False
    for key, value in MODEL_SIZE[1].items():
        cfg[key] = value
    cfg.work_dir = str(work_dir)
    cfg.task_title = "V6 Direct ForceWipe"
    cfg.multitask = False
    cfg.task_dim = 0
    cfg.tasks = [cfg.task]
    cfg.obs_shape = {"state": [16]}
    cfg.action_dim = 3
    cfg.episode_length = 1200
    cfg.obs_shapes = None
    cfg.action_dims = None
    cfg.episode_lengths = None
    cfg.bin_size = (cfg.vmax - cfg.vmin) / (cfg.num_bins - 1)
    parsed = cfg_to_dataclass(cfg)
    parsed.work_dir = Path(work_dir)
    parsed.obs_shape = {"state": (16,)}
    return parsed


def config_dict(cfg) -> dict:
    if is_dataclass(cfg):
        raw = asdict(cfg)
    else:
        raw = dict(vars(cfg))
    return {key: str(value) if isinstance(value, Path) else value for key, value in raw.items()}


def load_episode(path: Path) -> tuple[TensorDict, dict[str, torch.Tensor]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not rows:
        raise ValueError(f"empty episode: {path}")
    observation = torch.as_tensor(
        [rows[0]["observation"]] + [row["next_observation"] for row in rows],
        dtype=torch.float32,
    )
    action = torch.as_tensor(
        [[float("nan")] * 3] + [row["action"] for row in rows],
        dtype=torch.float32,
    )
    reward = torch.as_tensor(
        [float("nan")] + [row["reward"] for row in rows],
        dtype=torch.float32,
    )
    terminated = torch.as_tensor(
        [float("nan")] + [float(row["terminated"]) for row in rows],
        dtype=torch.float32,
    )
    if observation.shape != (len(rows) + 1, 16) or action.shape != (len(rows) + 1, 3):
        raise ValueError(f"episode tensor shape mismatch: {path}")
    td = TensorDict(
        {"obs": observation, "action": action, "reward": reward, "terminated": terminated},
        batch_size=(len(rows) + 1,),
    )
    flat = {
        "observation": torch.as_tensor([row["observation"] for row in rows], dtype=torch.float32),
        "action": torch.as_tensor([row["teacher_action"] for row in rows], dtype=torch.float32),
        "executed_action": torch.as_tensor([row["action"] for row in rows], dtype=torch.float32),
        "next_observation": torch.as_tensor([row["next_observation"] for row in rows], dtype=torch.float32),
        "reward": torch.as_tensor([row["reward"] for row in rows], dtype=torch.float32),
    }
    return td, flat


def load_collection(collection_root: Path):
    result = json.loads((collection_root / "RESULT.json").read_text(encoding="utf-8"))
    if not result.get("joint_training_permitted"):
        raise ValueError("collection gate does not permit joint training")
    train_tds, validation_tds = [], []
    train_flat, validation_flat = [], []
    for summary in result["evaluations"]:
        td, flat = load_episode(collection_root / summary["trace"])
        if int(summary["replicate"]) == 4:
            validation_tds.append(td)
            validation_flat.append(flat)
        else:
            train_tds.append(td)
            train_flat.append(flat)
    if len(train_tds) != 12 or len(validation_tds) != 3:
        raise ValueError("V14 episode split must be 12 TRAIN and 3 TRAIN-validation episodes")
    return train_tds, validation_tds, _concatenate(train_flat), _concatenate(validation_flat), result


def _concatenate(rows: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {key: torch.cat([row[key] for row in rows], dim=0) for key in rows[0]}


def populate_buffer(cfg, episodes: list[TensorDict]) -> Buffer:
    buffer = Buffer(cfg)
    for episode in episodes:
        buffer.add(episode)
    return buffer


def behavior_clone_step(agent: TDMPC2, optimizer, observation, target_action) -> dict[str, float]:
    agent.model.train()
    z = agent.model.encode(observation, task=None)
    _, info = agent.model.pi(z, task=None)
    mean = info["mean"]
    mse = torch.nn.functional.mse_loss(mean, target_action)
    force = observation[:, 0]
    target = observation[:, 1]
    contact_weight = 1.0 + 2.0 * (force >= 0.20).float() + 3.0 * (target >= 0.70).float()
    per_row = (mean - target_action).square().mean(dim=-1)
    weighted = (per_row * contact_weight).sum() / contact_weight.sum().clamp_min(1.0)
    loss = 0.25 * mse + 0.75 * weighted
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        list(agent.model._encoder.parameters()) + list(agent.model._pi.parameters()), 10.0
    )
    optimizer.step()
    agent.model.eval()
    return {"bc_loss": float(loss.detach().cpu()), "bc_mse": float(mse.detach().cpu())}


@torch.no_grad()
def audit_model(agent: TDMPC2, validation: dict[str, torch.Tensor], *, planner_states: int = 12):
    device = agent.device
    obs = validation["observation"].to(device)
    action = validation["executed_action"].to(device)
    teacher_action = validation["action"].to(device)
    next_obs = validation["next_observation"].to(device)
    model = agent.model.eval()
    predictions = []
    actor_means = []
    latent_errors = []
    batch = 512
    for start in range(0, len(obs), batch):
        end = min(start + batch, len(obs))
        z = model.encode(obs[start:end], None)
        predicted_z = model.next(z, action[start:end], None)
        target_z = model.encode(next_obs[start:end], None)
        predictions.append(model.force(z, action[start:end], None).squeeze(-1).cpu())
        _, info = model.pi(z, None)
        actor_means.append(info["mean"].cpu())
        latent_errors.append((predicted_z - target_z).square().mean(dim=-1).cpu())
    force_prediction = torch.cat(predictions)
    actor_mean = torch.cat(actor_means)
    latent_error = torch.cat(latent_errors)
    force_rmse_n = float(torch.sqrt(torch.mean((force_prediction - next_obs[:, 0].cpu()) ** 2)) * 15.0)
    actor_rmse = float(torch.sqrt(torch.mean((actor_mean - teacher_action.cpu()) ** 2)))
    latent_mse = float(latent_error.mean())

    # Stratified, deterministic offline planning probe. No environment step is taken.
    chosen = []
    for target_value in (5.0 / 15.0, 8.0 / 15.0, 12.0 / 15.0):
        index = torch.where(torch.isclose(obs[:, 1], torch.tensor(target_value, device=device), atol=1e-5))[0]
        if len(index):
            picks = index[torch.linspace(0, len(index) - 1, max(1, planner_states // 3), device=device).long()]
            chosen.extend(int(value) for value in picks.cpu())
    planner_actions = []
    for ordinal, index in enumerate(chosen[:planner_states]):
        agent._prev_mean.zero_()
        planner_actions.append(agent.act(obs[index].cpu(), t0=True, eval_mode=True).numpy())
    planned = np.asarray(planner_actions, dtype=np.float64)
    return {
        "validation_transitions": len(obs),
        "one_step_force_rmse_n": force_rmse_n,
        "actor_teacher_action_rmse": actor_rmse,
        "latent_consistency_mse": latent_mse,
        "planner_probe_states": len(planned),
        "planner_actions_finite": bool(len(planned) and np.isfinite(planned).all()),
        "planner_action_abs_max": float(np.abs(planned).max()) if len(planned) else None,
        "planner_action_mean": planned.mean(axis=0).tolist() if len(planned) else None,
    }
