"""Frozen method constructors for the V16.30 matched final study."""

from __future__ import annotations

from pathlib import Path

import torch

from forcewipe_v16.actor_only_direct import ActorOnlyDirectTDMPC2, configure_actor_only_direct
from forcewipe_v16.adaptive_admittance_baseline import AdaptiveAdmittanceController
from forcewipe_v16.direct_ppo_baseline import DirectPPOActorCritic, DirectPPOConfig
from forcewipe_v16.force_calibrated_strong_bc_training import build_force_calibrated_strong_bc_config
from forcewipe_v16.unified_tracking_gated_actor_to_mppi import (
    UnifiedTrackingGatedActorToMPPITDMPC2, configure_unified_tracking_gated_actor_to_mppi,
)


TDMPC_SEEDS = (172, 173, 174, 175, 176)
PPO_SEEDS = (301, 302, 303, 304, 305)
METHODS = ("M0", "M1", "M2", "M3")


class MethodLoadError(ValueError):
    pass


def tdmpc_checkpoint(root: Path, seed: int) -> Path:
    if int(seed) not in TDMPC_SEEDS:
        raise MethodLoadError("invalid frozen TD-MPC2 checkpoint seed")
    run = f"v16p25_force_calibrated_strong_bc_seed{int(seed)}_gate200_bc2000_joint1800_20260828_r1"
    return root / "results" / "train" / "native_tdmpc2_v6_v16" / run / "v16p3_target_expert_tdmpc2.pt"


def ppo_checkpoint(root: Path, seed: int) -> Path:
    if int(seed) not in PPO_SEEDS:
        raise MethodLoadError("invalid frozen PPO seed")
    run = f"v16p30_direct_ppo_seed{int(seed)}_transitions82774_20260830_r2"
    return root / "results" / "train" / "v16p30_direct_ppo" / run / f"direct_ppo_seed{int(seed)}.pt"


def load_tdmpc(root: Path, *, method: str, seed: int, work_dir: Path):
    if method not in {"M0", "M1"}:
        raise MethodLoadError("TD-MPC loader supports M0/M1")
    build = configure_unified_tracking_gated_actor_to_mppi if method == "M0" else configure_actor_only_direct
    cls = UnifiedTrackingGatedActorToMPPITDMPC2 if method == "M0" else ActorOnlyDirectTDMPC2
    cfg = build(build_force_calibrated_strong_bc_config(seed=int(seed), work_dir=work_dir))
    agent = cls(cfg)
    path = tdmpc_checkpoint(root, seed)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    agent.model.load_state_dict(payload["model"])
    agent.capture_eval_ema(); agent._eval_ema_model.load_state_dict(payload["eval_ema_model"])
    agent.model.eval(); agent._eval_ema_model.eval()
    return agent


def load_ppo(root: Path, *, seed: int):
    path = ppo_checkpoint(root, seed)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "forcewipe_v16p30_direct_ppo_checkpoint_v1":
        raise MethodLoadError("unexpected PPO checkpoint format")
    if int(payload.get("seed")) != int(seed) or int(payload.get("environment_transitions")) != 82_774:
        raise MethodLoadError("PPO checkpoint identity/budget mismatch")
    config = DirectPPOConfig(**payload["config"])
    model = DirectPPOActorCritic(config)
    model.load_state_dict(payload["model"]); model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model


def load_method(root: Path, *, method: str, seed: int | None, work_dir: Path):
    if method in {"M0", "M1"}:
        if seed is None:
            raise MethodLoadError("learned method seed is required")
        return load_tdmpc(root, method=method, seed=seed, work_dir=work_dir)
    if method == "M2":
        if seed is None:
            raise MethodLoadError("PPO seed is required")
        return load_ppo(root, seed=seed)
    if method == "M3":
        if seed is not None:
            raise MethodLoadError("adaptive admittance has no training seed")
        return AdaptiveAdmittanceController()
    raise MethodLoadError("unknown V16.30 method")


def method_action(agent, *, method: str, observation, step: int):
    if method in {"M0", "M1"}:
        action = agent.act(
            torch.from_numpy(observation), t0=int(step) == 0, eval_mode=True
        ).numpy()
        mode = str(agent._last_tdmpc2_control_mode)
    elif method == "M2":
        action = agent.act(observation, deterministic=True)
        mode = str(agent._last_control_mode)
    elif method == "M3":
        action = agent.act(observation)
        mode = str(agent._last_control_mode)
    else:
        raise MethodLoadError("unknown method action request")
    return action, mode


def reset_method(agent, *, method: str) -> None:
    if method in {"M0", "M1"}:
        agent._prev_mean.zero_()
    elif method == "M3":
        agent.reset()
