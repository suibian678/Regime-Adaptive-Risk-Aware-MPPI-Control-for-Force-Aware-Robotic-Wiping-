"""Reusable actor-only physical DEV runner for versioned V14 checkpoints."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Callable

import numpy as np
import torch

from forcewipe_v14.native_tdmpc2_v6_training import build_v14_config
from forcewipe_v6.tdmpc2_direct_firstpass import DirectFirstPassConfig
import forcewipe_v6.tdmpc2_direct_sapien_env as direct_env_module
from forcewipe_v6.tdmpc2_direct_sapien_env import V6DirectFirstPassEnv
from tdmpc2.tdmpc2 import TDMPC2


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def run_actor_dev(
    *,
    checkpoint: Path,
    source_gate: Path,
    source_gate_key: str,
    output_root: Path,
    run_id: str,
    seed: int,
    scenario_factory: Callable,
    scenario_factory_name: str,
    scenario_seed_base: int,
    scenario_id_base: int,
    episodes_per_target: int,
    matched_or_fresh: str,
) -> int:
    gate = json.loads(source_gate.read_text(encoding="utf-8"))
    if gate.get(source_gate_key) is not True:
        raise RuntimeError(f"source gate does not permit DEV: {source_gate_key}")
    final = output_root / run_id
    staging = output_root / f".{run_id}.creating"
    output_root.mkdir(parents=True, exist_ok=True)
    if final.exists() or staging.exists():
        raise RuntimeError(f"run id already exists: {run_id}")
    staging.mkdir()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cfg = build_v14_config(seed=seed, work_dir=staging)
    agent = TDMPC2(cfg)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    agent.model.load_state_dict(payload["model"])
    agent.capture_eval_ema()
    agent._eval_ema_model.load_state_dict(payload["eval_ema_model"])
    model = agent._eval_ema_model.eval()
    direct_env_module.nominal_direct_scenario = scenario_factory
    config = DirectFirstPassConfig()

    atomic_json(staging / "RUN_DEFINITION.json", {
        "format": "forcewipe_v14_actor_only_v6_dev_v2",
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "source_gate": str(source_gate),
        "source_gate_sha256": sha256(source_gate),
        "source_gate_key": source_gate_key,
        "scenario_factory": scenario_factory_name,
        "targets_n": [5.0, 8.0, 12.0],
        "episodes_per_target": episodes_per_target,
        "scenario_seed_base": scenario_seed_base,
        "scenario_id_base": scenario_id_base,
        "scenario_relation": matched_or_fresh,
        "policy": "jointly trained TD-MPC2 EMA actor mean; MPPI disabled",
        "training_teacher_available": False,
        "classical_force_controller": False,
        "shield": False,
        "force_dependent_action_projection": False,
        "rewiping": False,
        "tracking_metrics": "descriptive mean and RMSE; no tolerance-band success criterion",
        "claim_boundary": "Development screen only; not CAL, TEST, a confidence bound, or deployment evidence.",
    })

    evaluations = []
    for target_index, target in enumerate((5.0, 8.0, 12.0)):
        for replicate in range(episodes_per_target):
            offset = target_index * episodes_per_target + replicate
            scenario_seed = scenario_seed_base + offset
            scenario_id = scenario_id_base + offset
            env = V6DirectFirstPassEnv(
                target_force_n=target,
                scenario_seed=scenario_seed,
                scenario_id=scenario_id,
                config=config,
            )
            observation, reset_info = env.reset(seed=scenario_seed)
            rows = []
            episode_return = 0.0
            try:
                while len(rows) < config.maximum_steps:
                    with torch.no_grad():
                        obs = torch.from_numpy(observation).float().cuda().unsqueeze(0)
                        latent = model.encode(obs, None)
                        _, info = model.pi(latent, None)
                        action = info["mean"][0].cpu().numpy().astype(np.float32)
                    next_observation, reward, terminated, truncated, info = env.step(action)
                    episode_return += float(reward)
                    rows.append({
                        "control_step": len(rows),
                        "observation": np.asarray(observation, dtype=np.float32).tolist(),
                        "action": action.tolist(),
                        "next_observation": np.asarray(next_observation, dtype=np.float32).tolist(),
                        "reward": float(reward),
                        **info,
                    })
                    observation = next_observation
                    if terminated or truncated:
                        break
            finally:
                env.close()
            trace = staging / f"eval_{scenario_id}_{int(target)}n_trace.jsonl"
            with trace.open("w", encoding="utf-8", buffering=1) as stream:
                for row in rows:
                    stream.write(json.dumps(row, sort_keys=True) + "\n")
            force = np.asarray([row["normal_force_n"] for row in rows], dtype=np.float64)
            contact = force >= config.safe_contact_min_n
            evaluation = {
                "target_force_n": target,
                "replicate": replicate,
                "scenario_seed": scenario_seed,
                "scenario_id": scenario_id,
                "native_samples": len(rows),
                "success": bool(rows[-1]["success"]),
                "return": episode_return,
                "progress": float(rows[-1]["progress"]),
                "completed_dose_bins": int(rows[-1]["completed_dose_bins"]),
                "minimum_bin_dose": int(rows[-1]["minimum_bin_dose"]),
                "force_limit_violation_samples": int(np.sum(force > config.force_limit_n)),
                "peak_force_n": float(force.max()),
                "contact_fraction": float(contact.mean()),
                "contact_mean_force_n": float(force[contact].mean()) if contact.any() else None,
                "contact_rmse_n": float(np.sqrt(np.mean((force[contact] - target) ** 2))) if contact.any() else None,
                "trace": trace.name,
                "trace_sha256": sha256(trace),
                "reset_info": reset_info,
            }
            evaluations.append(evaluation)
            atomic_json(staging / "RUN_STATE.json", {
                "status": "running",
                "completed_evaluations": len(evaluations),
                "successful_evaluations": sum(row["success"] for row in evaluations),
                "force_limit_violation_samples": sum(row["force_limit_violation_samples"] for row in evaluations),
            })
            print(json.dumps(evaluation, sort_keys=True), flush=True)

    tiers = []
    for target in (5.0, 8.0, 12.0):
        cells = [row for row in evaluations if row["target_force_n"] == target]
        tiers.append({
            "target_force_n": target,
            "evaluations": len(cells),
            "successes": sum(row["success"] for row in cells),
            "force_limit_violation_samples": sum(row["force_limit_violation_samples"] for row in cells),
            "peak_force_n": max(row["peak_force_n"] for row in cells),
            "contact_mean_force_n_mean": float(np.mean([row["contact_mean_force_n"] for row in cells])),
            "contact_mean_force_n_std": float(np.std([row["contact_mean_force_n"] for row in cells], ddof=1)) if len(cells) > 1 else None,
            "contact_rmse_n_mean": float(np.mean([row["contact_rmse_n"] for row in cells])),
            "contact_rmse_n_std": float(np.std([row["contact_rmse_n"] for row in cells], ddof=1)) if len(cells) > 1 else None,
            "progress_mean": float(np.mean([row["progress"] for row in cells])),
        })
    passed = all(row["success"] and row["force_limit_violation_samples"] == 0 for row in evaluations)
    result = {
        "format": "forcewipe_v14_actor_only_v6_dev_result_v2",
        "status": "completed_pass" if passed else "completed_scientific_fail",
        "checkpoint_sha256": sha256(checkpoint),
        "evaluations": evaluations,
        "tier_summaries": tiers,
        "task_and_sampled_force_gate_passed": passed,
        "expanded_evaluation_permitted": passed and episodes_per_target == 1 and matched_or_fresh == "fresh",
        "multiseed_training_permitted": passed and episodes_per_target >= 5 and matched_or_fresh == "fresh",
        "claim_boundary": "Development screen only; no CAL, TEST, confidence bound, or deployment-level guarantee.",
    }
    atomic_json(staging / "RESULT.json", result)
    atomic_json(staging / "RUN_STATE.json", {"status": result["status"]})
    os.replace(staging, final)
    print(json.dumps({
        "status": result["status"],
        "tier_summaries": tiers,
        "expanded_evaluation_permitted": result["expanded_evaluation_permitted"],
        "multiseed_training_permitted": result["multiseed_training_permitted"],
    }, indent=2, sort_keys=True))
    return 0 if passed else 1
