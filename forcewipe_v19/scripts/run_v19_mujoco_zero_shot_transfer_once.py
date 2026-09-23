#!/usr/bin/env python3
"""Execute the frozen V19 zero-shot MuJoCo transfer stress test once."""

from __future__ import annotations

from dataclasses import asdict
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
V16_ROOT = TEACHER_ROOT / "forcewipe_v16"
for source in (
    ROOT / "code",
    ROOT / "scripts",
    V16_ROOT / "code",
    V16_ROOT / "scripts",
    TEACHER_ROOT / "forcewipe_v15" / "code",
    TEACHER_ROOT / "forcewipe_v15" / "scripts",
    TEACHER_ROOT / "forcewipe_v14" / "code",
    TEACHER_ROOT / "forcewipe_v6" / "code",
    TEACHER_ROOT / "forcewipe_v4" / "code",
    ROOT / "release" / "runtime_source" / "tdmpc2",
    ROOT / "release" / "runtime_source",
):
    sys.path.insert(0, str(source))

from forcewipe_v6.tdmpc2_direct_firstpass import DirectFirstPassConfig  # noqa: E402
from forcewipe_v19.factor_separated_scenarios import (  # noqa: E402
    SCENARIO_SEED,
    factor_separated_scenario,
    scenario_digest,
)
from forcewipe_v19.mujoco_crosssim_env import (  # noqa: E402
    MuJoCoContactConfig,
    MuJoCoDirectFirstPassEnv,
)
from run_v15_trust_mppi_fresh_dev3 import atomic_json, sha256, summarize  # noqa: E402
from run_v19_seed201_method_dev12 import load_agent  # noqa: E402


PROTOCOL = ROOT / "config/V19_MUJOCO_ZERO_SHOT_TRANSFER_PROTOCOL_2026-09-21.json"
OUTPUT_ROOT = ROOT / "results/crosssim"


def _verify_frozen_file(relative: str, expected: str) -> None:
    path = ROOT / relative
    if not path.is_file() or sha256(path) != expected:
        raise RuntimeError(f"frozen identity mismatch: {relative}")


def _checkpoint(seed: int) -> Path:
    return ROOT / (
        f"results/train/v19_force_conditioned_seed{seed}_bc2_u4000/"
        "v19_force_conditioned_tdmpc2.pt"
    )


def _run_episode(*, agent, seed: int, spec, config, contact_config):
    env = MuJoCoDirectFirstPassEnv(
        scenario_spec=spec,
        config=config,
        contact_config=contact_config,
    )
    observation, reset_info = env.reset(seed=SCENARIO_SEED)
    agent._prev_mean.zero_()
    rows: list[dict] = []
    episode_return = 0.0
    try:
        while len(rows) < config.maximum_steps:
            step_seed = (
                9_190_000_000
                + seed * 10_000_000
                + spec.scenario_id * 2_000
                + len(rows)
            )
            random.seed(step_seed)
            np.random.seed(step_seed % (2**32))
            torch.manual_seed(step_seed)
            torch.cuda.manual_seed_all(step_seed)
            started = time.perf_counter()
            action_tensor = agent.act(
                torch.from_numpy(observation),
                t0=len(rows) == 0,
                eval_mode=True,
            )
            torch.cuda.synchronize()
            planning_ms = (time.perf_counter() - started) * 1_000.0
            action = action_tensor.detach().cpu().numpy().astype(np.float32)
            actor_center = (
                agent._last_selected_actor_center.detach().cpu().numpy().astype(np.float32)
            )
            diagnostics = agent.risk_diagnostics()
            next_observation, reward, terminated, truncated, info = env.step(action)
            episode_return += float(reward)
            rows.append({
                "control_step": len(rows),
                "planner_seed": step_seed,
                "observation": np.asarray(observation, dtype=np.float32).tolist(),
                "action": action.tolist(),
                "actor_center": actor_center.tolist(),
                "absolute_actor_deviation": np.abs(action - actor_center).tolist(),
                "next_observation": np.asarray(next_observation, dtype=np.float32).tolist(),
                "reward": float(reward),
                "planning_ms": planning_ms,
                "risk_diagnostics": diagnostics,
                **info,
            })
            observation = next_observation
            if terminated or truncated:
                break
    finally:
        env.close()
    return rows, reset_info, episode_return


def _tracking(row: dict) -> bool:
    return bool(
        row["mean_relative_error"] is not None
        and row["target_normalized_rmse"] is not None
        and row["mean_relative_error"] <= 0.15
        and row["target_normalized_rmse"] <= 0.20
    )


def _joint(row: dict) -> bool:
    return bool(
        row["success"]
        and _tracking(row)
        and row["force_limit_violation_samples"] == 0
    )


def _engine_summary(rows: list[dict]) -> dict:
    return {
        "evaluations": len(rows),
        "task_successes": int(sum(bool(row["success"]) for row in rows)),
        "tracking_passes": int(sum(_tracking(row) for row in rows)),
        "sampled_force_safety_passes": int(sum(
            row["force_limit_violation_samples"] == 0 for row in rows
        )),
        "joint_passes": int(sum(_joint(row) for row in rows)),
        "force_limit_violation_samples": int(sum(
            row["force_limit_violation_samples"] for row in rows
        )),
        "maximum_peak_force_n": float(max(row["peak_force_n"] for row in rows)),
    }


def _paired_analysis(physx_rows: list[dict], mujoco_rows: list[dict]) -> dict:
    def key(row: dict) -> tuple[int, str, float]:
        return int(row["training_seed"]), str(row["block_id"]), float(row["target_force_n"])

    physx = {key(row): row for row in physx_rows}
    mujoco = {key(row): row for row in mujoco_rows}
    if len(physx) != 45 or len(mujoco) != 45 or set(physx) != set(mujoco):
        raise RuntimeError("paired PhysX/MuJoCo key set is incomplete")
    pairs = []
    for item in sorted(physx):
        left, right = physx[item], mujoco[item]
        pairs.append({
            "training_seed": item[0],
            "block_id": item[1],
            "target_force_n": item[2],
            "task_difference": int(bool(right["success"])) - int(bool(left["success"])),
            "tracking_difference": int(_tracking(right)) - int(_tracking(left)),
            "joint_difference": int(_joint(right)) - int(_joint(left)),
            "peak_force_difference_n": float(right["peak_force_n"] - left["peak_force_n"]),
            "mre_difference": (
                float(right["mean_relative_error"] - left["mean_relative_error"])
                if right["mean_relative_error"] is not None
                and left["mean_relative_error"] is not None else None
            ),
            "nrmse_difference": (
                float(right["target_normalized_rmse"] - left["target_normalized_rmse"])
                if right["target_normalized_rmse"] is not None
                and left["target_normalized_rmse"] is not None else None
            ),
        })
    return {
        "pairs": len(pairs),
        "mujoco_minus_physx": {
            "task_success_rate": float(np.mean([row["task_difference"] for row in pairs])),
            "tracking_pass_rate": float(np.mean([row["tracking_difference"] for row in pairs])),
            "joint_pass_rate": float(np.mean([row["joint_difference"] for row in pairs])),
            "mean_peak_force_n": float(np.mean([
                row["peak_force_difference_n"] for row in pairs
            ])),
            "mean_mre": float(np.mean([
                row["mre_difference"] for row in pairs
                if row["mre_difference"] is not None
            ])),
            "mean_target_normalized_rmse": float(np.mean([
                row["nrmse_difference"] for row in pairs
                if row["nrmse_difference"] is not None
            ])),
        },
        "pair_rows": pairs,
    }


def _manifest(directory: Path) -> list[dict]:
    rows = []
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        rows.append({
            "path": str(path.relative_to(directory)).replace("\\", "/"),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    return rows


def main() -> int:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol.get("status") != "FROZEN_READY_TO_EXECUTE":
        raise SystemExit("cross-engine protocol is not executable")
    for relative, digest in protocol["frozen_sources"].items():
        _verify_frozen_file(relative, digest)
    _verify_frozen_file(
        protocol["calibration_interpretation"]["result_path"],
        protocol["calibration_interpretation"]["result_sha256"],
    )
    _verify_frozen_file(
        protocol["matched_physx_source"]["path"],
        protocol["matched_physx_source"]["sha256"],
    )
    for seed_text, digest in protocol["checkpoint_sha256"].items():
        path = _checkpoint(int(seed_text))
        if not path.is_file() or sha256(path) != digest:
            raise RuntimeError(f"checkpoint identity mismatch: seed {seed_text}")

    run_id = protocol["execution"]["run_id"]
    final = OUTPUT_ROOT / run_id
    staging = OUTPUT_ROOT / f".{run_id}.creating"
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if final.exists() or staging.exists():
        raise SystemExit(f"run id already exists: {run_id}")
    staging.mkdir()
    contact_config = MuJoCoContactConfig(**protocol["port_configuration"])
    config = DirectFirstPassConfig()
    definitions = []
    for block_id in protocol["execution"]["blocks"]:
        for target in protocol["execution"]["targets_n"]:
            spec = factor_separated_scenario(block_id, float(target))
            definitions.append({
                "block_id": block_id,
                "target_force_n": target,
                "scenario_id": spec.scenario_id,
                "scenario_seed": spec.scenario_seed,
                "scenario_digest": scenario_digest(spec),
            })
    atomic_json(staging / "RUN_DEFINITION.json", {
        "format": "forcewipe_v19_mujoco_zero_shot_transfer_definition_v1",
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "protocol_sha256": sha256(PROTOCOL),
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "contact_config": asdict(contact_config),
        "scenario_definitions": definitions,
        "claim_boundary": protocol["claim_boundary"],
    })

    evaluations = []
    for seed in protocol["execution"]["training_seeds"]:
        agent = load_agent(
            int(seed),
            protocol["execution"]["method_arm"],
            staging,
            float(protocol["execution"]["bc_coefficient"]),
            protocol["execution"]["risk_envelope_mode"],
        )
        for block_id in protocol["execution"]["blocks"]:
            for target in protocol["execution"]["targets_n"]:
                spec = factor_separated_scenario(block_id, float(target))
                rows, reset_info, episode_return = _run_episode(
                    agent=agent,
                    seed=int(seed),
                    spec=spec,
                    config=config,
                    contact_config=contact_config,
                )
                trace = staging / f"seed_{seed}_{block_id}_target_{int(target)}n.jsonl"
                with trace.open("x", encoding="utf-8", buffering=1) as stream:
                    for row in rows:
                        stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                summary = summarize(rows, target=float(target), episode_return=episode_return)
                mean_force = summary["contact_mean_force_n"]
                rmse = summary["contact_rmse_n"]
                summary.update({
                    "training_seed": int(seed),
                    "block_id": block_id,
                    "scenario_id": spec.scenario_id,
                    "scenario_seed": spec.scenario_seed,
                    "scenario_digest": scenario_digest(spec),
                    "reset_info": reset_info,
                    "trace": trace.name,
                    "trace_sha256": sha256(trace),
                    "mean_relative_error": (
                        abs(float(mean_force) - float(target)) / float(target)
                        if mean_force is not None else None
                    ),
                    "signed_relative_bias": (
                        (float(mean_force) - float(target)) / float(target)
                        if mean_force is not None else None
                    ),
                    "target_normalized_rmse": (
                        float(rmse) / float(target) if rmse is not None else None
                    ),
                    "mean_planning_samples": float(np.mean([
                        row["risk_diagnostics"]["budget"]["num_samples"]
                        for row in rows
                    ])),
                })
                evaluations.append(summary)
                print(json.dumps({
                    "seed": seed,
                    "block": block_id,
                    "target": target,
                    "task": summary["success"],
                    "tracking": _tracking(summary),
                    "joint": _joint(summary),
                    "mre": summary["mean_relative_error"],
                    "nrmse": summary["target_normalized_rmse"],
                    "peak": summary["peak_force_n"],
                }, sort_keys=True), flush=True)
        del agent
        torch.cuda.empty_cache()

    physx_payload = json.loads(
        (ROOT / protocol["matched_physx_source"]["path"]).read_text(encoding="utf-8")
    )
    physx_rows = [
        row for row in physx_payload["evaluations"]
        if row["arm"] == protocol["matched_physx_source"]["arm"]
        and row["block_id"] in protocol["execution"]["blocks"]
        and float(row["target_force_n"]) in protocol["execution"]["targets_n"]
    ]
    paired = _paired_analysis(physx_rows, evaluations)
    summaries = [{
        "engine": "PhysX",
        **_engine_summary(physx_rows),
    }, {
        "engine": "MuJoCo",
        **_engine_summary(evaluations),
    }]
    result_path = staging / "RESULT.json"
    atomic_json(result_path, {
        "format": "forcewipe_v19_mujoco_zero_shot_transfer_result_v1",
        "status": "completed",
        "completed_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "protocol_sha256": sha256(PROTOCOL),
        "claim_boundary": protocol["claim_boundary"],
        "evaluations": evaluations,
        "engine_summaries": summaries,
        "paired_analysis": paired,
    })
    atomic_json(staging / "RUN_STATE.json", {
        "status": "completed",
        "evaluations": len(evaluations),
        "protocol_sha256": sha256(PROTOCOL),
        "result_sha256": sha256(result_path),
    })
    atomic_json(staging / "MANIFEST.json", {
        "format": "forcewipe_v19_mujoco_zero_shot_transfer_manifest_v1",
        "files": _manifest(staging),
    })
    os.replace(staging, final)
    print(json.dumps({
        "run_id": run_id,
        "engine_summaries": summaries,
        "paired_analysis": paired["mujoco_minus_physx"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
