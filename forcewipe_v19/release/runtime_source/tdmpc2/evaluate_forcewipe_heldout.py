import csv
import os
from pathlib import Path
import warnings

os.environ["MUJOCO_GL"] = os.getenv("MUJOCO_GL", "egl")
warnings.filterwarnings("ignore")

import hydra
import numpy as np
import torch

from common.parser import parse_cfg
from common.seed import set_seed
from envs import make_env
from tdmpc2 import TDMPC2

torch.backends.cudnn.benchmark = True


def _scalar(value, default=np.nan):
    if value is None:
        return default
    if torch.is_tensor(value):
        return float(value.detach().cpu().flatten()[0])
    return float(value)


def _failure_reason(info):
    if _scalar(info.get("success"), 0.0) >= 0.5:
        return "success"
    if _scalar(info.get("normal_force"), 0.0) > 15.0:
        return "force_safety_fail"
    if _scalar(info.get("normal_force"), 0.0) < 3.0:
        return "no_contact_or_low_force"
    return "not_success"


def _apply_force_filter(action, obs, cfg):
    """Deployment-only force boundary filter for ForceWipe held-out tests.

    This is not part of the learned-only claim. It tests the engineering idea
    of detecting no-contact / over-force states and applying a minimal safety
    correction at the action layer.
    """
    if not bool(getattr(cfg, "heldout_force_filter", False)):
        return action
    if isinstance(obs, dict):
        return action
    force_idx = int(getattr(cfg, "heldout_force_filter_force_obs_idx", getattr(cfg, "force_obs_idx", 34)))
    z_idx = int(getattr(cfg, "heldout_force_filter_z_idx", 2))
    x_idx = int(getattr(cfg, "heldout_force_filter_x_idx", 0))
    obs_flat = obs.detach().flatten()
    if obs_flat.numel() <= force_idx or action.shape[-1] <= max(z_idx, x_idx):
        return action
    force = float(obs_flat[force_idx].detach().cpu())
    low_n = float(getattr(cfg, "heldout_force_filter_low_n", 3.0))
    high_n = float(getattr(cfg, "heldout_force_filter_high_n", 12.0))
    down_z = float(getattr(cfg, "heldout_force_filter_down_z", -0.018))
    lift_z = float(getattr(cfg, "heldout_force_filter_lift_z", 0.020))
    slow_x = float(getattr(cfg, "heldout_force_filter_slow_x", 0.35))
    action = action.clone()
    if force < low_n:
        action[..., x_idx] = action[..., x_idx] * slow_x
        action[..., z_idx] = torch.minimum(
            action[..., z_idx],
            torch.tensor(down_z, dtype=action.dtype, device=action.device),
        )
    elif force > high_n:
        action[..., x_idx] = action[..., x_idx] * slow_x
        action[..., z_idx] = torch.maximum(
            action[..., z_idx],
            torch.tensor(lift_z, dtype=action.dtype, device=action.device),
        )
    return action.clamp(-1, 1)


@hydra.main(config_name="config", config_path=".")
def main(cfg):
    assert torch.cuda.is_available()
    assert cfg.eval_episodes > 0
    cfg = parse_cfg(cfg)
    set_seed(cfg.seed)

    env = make_env(cfg)
    agent = TDMPC2(cfg)
    assert Path(cfg.checkpoint).exists(), f"checkpoint not found: {cfg.checkpoint}"
    agent.load(cfg.checkpoint)
    agent.model.eval()

    output_csv = Path(cfg.get("heldout_output_csv", "heldout_eval.csv"))
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    eval_seed_start = int(getattr(cfg, "eval_seed_start", -1))
    # Per-target ±25% success band read from the runtime env (set by ForcePressWrapper).
    raw_env = env.unwrapped
    for i in range(int(cfg.eval_episodes)):
        eval_seed = eval_seed_start + i if eval_seed_start >= 0 else -1
        obs = env.reset(seed=eval_seed) if eval_seed >= 0 else env.reset()
        done = False
        ep_reward = 0.0
        t = 0
        info = {}
        episode_target_n = float(getattr(raw_env, "target_force_n", np.nan))
        episode_band_n = float(getattr(raw_env, "force_success_band_n", 2.0))
        while not done:
            action = agent.act(obs, t0=(t == 0), eval_mode=True)
            action = _apply_force_filter(action, obs, cfg)
            obs, reward, done, info = env.step(action)
            ep_reward += _scalar(reward)
            t += 1
        normal_force = _scalar(info.get("normal_force"), 0.0)
        force_error = _scalar(info.get("force_error"), np.nan)
        success = _scalar(info.get("success"), 0.0)
        # force_band uses the per-episode target band (not hardcoded 2.0N).
        # Matches the env's evaluate() success condition exactly.
        force_band_ok = (
            np.isfinite(force_error)
            and (3.0 <= normal_force <= 15.0)
            and (force_error <= episode_band_n)
        )
        rows.append(
            {
                "checkpoint": str(cfg.checkpoint),
                "eval_episode": i,
                "eval_seed": eval_seed,
                "episode_reward": ep_reward,
                "success": success,
                "episode_length": t,
                "final_wipe_progress": _scalar(info.get("wipe_progress"), np.nan),
                "final_tcp_progress": _scalar(info.get("tcp_progress"), np.nan),
                "final_force_band_progress": _scalar(info.get("force_band_progress"), np.nan),
                "final_instant_force_band_progress": _scalar(info.get("instant_force_band_progress"), np.nan),
                "final_force_error": force_error,
                "final_normal_force": normal_force,
                "final_y_error": _scalar(info.get("y_error", info.get("xy_error")), np.nan),
                "failure_reason": _failure_reason(info),
                "no_contact": float(normal_force < 3.0),
                "safe_contact": float(3.0 <= normal_force <= 15.0),
                "over_force": float(normal_force > 15.0),
                "force_band": float(force_band_ok),
                "episode_target_n": episode_target_n,
                "episode_band_n": episode_band_n,
            }
        )

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    success = np.mean([r["success"] for r in rows])
    no_contact = np.mean([r["no_contact"] for r in rows])
    over_force = np.mean([r["over_force"] for r in rows])
    force_band = np.mean([r["force_band"] for r in rows])
    print(
        f"heldout_eval success={success:.3f} "
        f"no_contact={no_contact:.3f} over_force={over_force:.3f} "
        f"force_band={force_band:.3f} output={output_csv}"
    )


if __name__ == "__main__":
    main()
