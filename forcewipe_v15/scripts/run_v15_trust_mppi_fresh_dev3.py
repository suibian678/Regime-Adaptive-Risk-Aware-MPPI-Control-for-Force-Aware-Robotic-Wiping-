#!/usr/bin/env python3
"""Fresh three-tier closed-loop DEV for V15 actor-trust TD-MPC2 MPPI."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
TEACHER_ROOT = ROOT.parent
for source in (
    ROOT / "code",
    TEACHER_ROOT / "forcewipe_v14" / "code",
    TEACHER_ROOT / "forcewipe_v6" / "code",
    TEACHER_ROOT / "forcewipe_v4" / "code",
    Path("/tmp/forcewipe_legacy_workspace/tdmpc2"),
    Path("/tmp/forcewipe_legacy_workspace"),
):
    sys.path.insert(0, str(source))

from forcewipe_v14.native_tdmpc2_v6_training import build_v14_config  # noqa: E402
from forcewipe_v14.v6_dev_scenarios import v14_development_scenario  # noqa: E402
from forcewipe_v15.actor_trust_mppi import ActorTrustRegionTDMPC2, configure_actor_trust_mppi  # noqa: E402
from forcewipe_v6.tdmpc2_direct_firstpass import DirectFirstPassConfig  # noqa: E402
import forcewipe_v6.tdmpc2_direct_sapien_env as direct_env_module  # noqa: E402
from forcewipe_v6.tdmpc2_direct_sapien_env import V6DirectFirstPassEnv  # noqa: E402


CHECKPOINT = TEACHER_ROOT / "forcewipe_v14/results/train/native_tdmpc2_v6_v14p1/v14p1_native_tdmpc2_v6_seed141_bc2000_joint1800_20260828_r1/v14_native_tdmpc2_v6.pt"
MATCHED_RESULT = ROOT / "results/dev/v15_trust_mppi_matched_prior_failure_5n_20260828_r1/RESULT.json"
OUTPUT_ROOT = ROOT / "results/dev"
RUN_ID = "v15_trust_mppi_fresh_5_8_12n_dev3_20260828_r1"
TARGETS = (5.0, 8.0, 12.0)
SCENARIO_SEEDS = (80_000_005, 80_000_008, 80_000_012)
SCENARIO_IDS = (8_149_005, 8_149_008, 8_149_012)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_agent(staging: Path) -> ActorTrustRegionTDMPC2:
    random.seed(151)
    np.random.seed(151)
    torch.manual_seed(151)
    torch.cuda.manual_seed_all(151)
    cfg = configure_actor_trust_mppi(build_v14_config(seed=141, work_dir=staging))
    agent = ActorTrustRegionTDMPC2(cfg)
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    agent.model.load_state_dict(payload["model"])
    agent.capture_eval_ema()
    agent._eval_ema_model.load_state_dict(payload["eval_ema_model"])
    agent.model.eval()
    agent._eval_ema_model.eval()
    return agent


def run_episode(
    *,
    agent: ActorTrustRegionTDMPC2,
    target: float,
    scenario_seed: int,
    scenario_id: int,
    config: DirectFirstPassConfig,
) -> tuple[list[dict], dict, float]:
    env = V6DirectFirstPassEnv(
        target_force_n=target,
        scenario_seed=scenario_seed,
        scenario_id=scenario_id,
        config=config,
    )
    observation, reset_info = env.reset(seed=scenario_seed)
    agent._prev_mean.zero_()
    rows: list[dict] = []
    episode_return = 0.0
    try:
        while len(rows) < config.maximum_steps:
            started = time.perf_counter()
            action = agent.act(
                torch.from_numpy(observation),
                t0=len(rows) == 0,
                eval_mode=True,
            ).numpy().astype(np.float32)
            torch.cuda.synchronize()
            planning_ms = (time.perf_counter() - started) * 1000.0
            center = agent._last_selected_actor_center.cpu().numpy().astype(np.float32)
            next_observation, reward, terminated, truncated, info = env.step(action)
            episode_return += float(reward)
            rows.append({
                "control_step": len(rows),
                "observation": np.asarray(observation, dtype=np.float32).tolist(),
                "action": action.tolist(),
                "actor_center": center.tolist(),
                "absolute_actor_deviation": np.abs(action - center).tolist(),
                "next_observation": np.asarray(next_observation, dtype=np.float32).tolist(),
                "reward": float(reward),
                "planning_ms": planning_ms,
                **info,
            })
            observation = next_observation
            if terminated or truncated:
                break
    finally:
        env.close()
    return rows, reset_info, episode_return


def summarize(rows: list[dict], *, target: float, episode_return: float) -> dict:
    force = np.asarray([row["normal_force_n"] for row in rows], dtype=np.float64)
    contact = force >= 3.0
    deviation = np.asarray([row["absolute_actor_deviation"] for row in rows], dtype=np.float64)
    planning = np.asarray([row["planning_ms"] for row in rows], dtype=np.float64)
    return {
        "target_force_n": target,
        "native_samples": len(rows),
        "success": bool(rows[-1]["success"]),
        "return": episode_return,
        "progress": float(rows[-1]["progress"]),
        "completed_dose_bins": int(rows[-1]["completed_dose_bins"]),
        "minimum_bin_dose": int(rows[-1]["minimum_bin_dose"]),
        "force_limit_violation_samples": int(np.sum(force > 15.0)),
        "peak_force_n": float(force.max()),
        "contact_fraction": float(contact.mean()),
        "contact_mean_force_n": float(force[contact].mean()) if contact.any() else None,
        "contact_rmse_n": float(np.sqrt(np.mean((force[contact] - target) ** 2))) if contact.any() else None,
        "maximum_actor_deviation": deviation.max(axis=0).tolist(),
        "mean_planning_ms": float(planning.mean()),
        "median_planning_ms": float(np.median(planning)),
        "maximum_planning_ms": float(planning.max()),
    }


def main() -> int:
    matched = json.loads(MATCHED_RESULT.read_text(encoding="utf-8"))
    if matched.get("fresh_three_tier_dev_permitted") is not True:
        raise SystemExit("matched prior-failure diagnostic did not permit fresh DEV")
    final = OUTPUT_ROOT / RUN_ID
    staging = OUTPUT_ROOT / f".{RUN_ID}.creating"
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if final.exists() or staging.exists():
        raise SystemExit(f"run id already exists: {RUN_ID}")
    staging.mkdir()
    atomic_json(staging / "RUN_DEFINITION.json", {
        "format": "forcewipe_v15_trust_mppi_fresh_dev3_definition_v1",
        "run_id": RUN_ID,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": sha256(CHECKPOINT),
        "matched_gate_result_sha256": sha256(MATCHED_RESULT),
        "target_force_n": list(TARGETS),
        "scenario_seeds": list(SCENARIO_SEEDS),
        "scenario_ids": list(SCENARIO_IDS),
        "planner": "native TD-MPC2 MPPI plus actor trust region and actor-deviation trajectory penalty",
        "teacher_available": False,
        "classical_force_controller": False,
        "shield": False,
        "force_dependent_action_projection": False,
        "rewiping": False,
        "gate": "all three episodes complete all 20 bins with zero native-rate F>15 N samples",
        "claim_boundary": "Three fresh development scenarios; a mechanism screen, not final independent evidence.",
    })
    agent = load_agent(staging)
    direct_env_module.nominal_direct_scenario = v14_development_scenario
    config = DirectFirstPassConfig()
    summaries = []
    for target, scenario_seed, scenario_id in zip(TARGETS, SCENARIO_SEEDS, SCENARIO_IDS):
        rows, reset_info, episode_return = run_episode(
            agent=agent,
            target=target,
            scenario_seed=scenario_seed,
            scenario_id=scenario_id,
            config=config,
        )
        trace = staging / f"target_{int(target)}n_trace.jsonl"
        with trace.open("w", encoding="utf-8", buffering=1) as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
        summary = summarize(rows, target=target, episode_return=episode_return)
        summary.update({
            "scenario_seed": scenario_seed,
            "scenario_id": scenario_id,
            "reset_info": reset_info,
            "trace": trace.name,
            "trace_sha256": sha256(trace),
        })
        summaries.append(summary)
        print(json.dumps(summary, sort_keys=True), flush=True)
    gate = all(
        item["success"]
        and item["completed_dose_bins"] == 20
        and item["force_limit_violation_samples"] == 0
        for item in summaries
    )
    result = {
        "format": "forcewipe_v15_trust_mppi_fresh_dev3_result_v1",
        "status": "completed_pass" if gate else "completed_scientific_fail",
        "episodes": summaries,
        "total_native_samples": sum(item["native_samples"] for item in summaries),
        "total_force_limit_violation_samples": sum(item["force_limit_violation_samples"] for item in summaries),
        "maximum_peak_force_n": max(item["peak_force_n"] for item in summaries),
        "expanded_dev_permitted": gate,
        "claim_boundary": "Three fresh development scenarios only; not a statistical performance estimate or safety guarantee.",
    }
    atomic_json(staging / "RESULT.json", result)
    atomic_json(staging / "RUN_STATE.json", {"status": result["status"]})
    os.replace(staging, final)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
