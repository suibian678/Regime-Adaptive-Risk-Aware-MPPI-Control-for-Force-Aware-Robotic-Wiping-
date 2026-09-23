#!/usr/bin/env python3
"""Calibrate V19 force upper bounds on the frozen TRAIN-validation split.

This script performs checkpoint inference only.  It does not create a SAPIEN
environment and it does not read DEV, qualification, CAL, or TEST traces.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
TEACHER_ROOT = ROOT.parent
V16_ROOT = TEACHER_ROOT / "forcewipe_v16"
for source in (
    ROOT / "code",
    V16_ROOT / "code",
    TEACHER_ROOT / "forcewipe_v15" / "code",
    TEACHER_ROOT / "forcewipe_v14" / "code",
    TEACHER_ROOT / "forcewipe_v6" / "code",
    TEACHER_ROOT / "forcewipe_v4" / "code",
    ROOT / "release" / "runtime_source" / "tdmpc2",
    ROOT / "release" / "runtime_source",
):
    sys.path.insert(0, str(source))

from forcewipe_v16.force_calibrated_strong_bc_training import (  # noqa: E402
    build_force_calibrated_strong_bc_config,
)
from forcewipe_v16.recovery_aware_training import (  # noqa: E402
    load_recovery_aware_collection,
)
from forcewipe_v16.unified_tracking_gated_actor_to_mppi import (  # noqa: E402
    UnifiedTrackingGatedActorToMPPITDMPC2,
    configure_unified_tracking_gated_actor_to_mppi,
)
from forcewipe_v19.calibration import fit_target_residual_radii  # noqa: E402


SEEDS = (172, 173, 174, 175, 176)
TARGETS_N = (5.0, 8.0, 12.0)
COVERAGE = 0.90
OUTPUT = ROOT / "results/calibration/V19_TRAIN_VALIDATION_UNCERTAINTY_CALIBRATION.json"
TRAIN_ROOT = V16_ROOT / "results/train/native_tdmpc2_v6_v16"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint(seed: int) -> Path:
    return (
        TRAIN_ROOT
        / f"v16p25_force_calibrated_strong_bc_seed{seed}_gate200_bc2000_joint1800_20260828_r1"
        / "v16p3_target_expert_tdmpc2.pt"
    )


def load_agent(seed: int, work_dir: Path) -> UnifiedTrackingGatedActorToMPPITDMPC2:
    initialization_seed = int(seed) + 12
    random.seed(initialization_seed)
    np.random.seed(initialization_seed)
    torch.manual_seed(initialization_seed)
    torch.cuda.manual_seed_all(initialization_seed)
    cfg = configure_unified_tracking_gated_actor_to_mppi(
        build_force_calibrated_strong_bc_config(seed=int(seed), work_dir=work_dir)
    )
    agent = UnifiedTrackingGatedActorToMPPITDMPC2(cfg)
    payload = torch.load(checkpoint(seed), map_location="cpu", weights_only=False)
    agent.model.load_state_dict(payload["model"])
    agent.capture_eval_ema()
    agent._eval_ema_model.load_state_dict(payload["eval_ema_model"])
    agent.model.eval()
    agent._eval_ema_model.eval()
    return agent


@torch.inference_mode()
def calibrate_checkpoint(agent, validation: dict[str, torch.Tensor]) -> dict:
    model = agent._eval_ema_model
    obs = validation["observation"]
    action = validation["executed_action"]
    next_obs = validation["next_observation"]
    predicted_upper_parts = []
    batch = 1024
    for start in range(0, len(obs), batch):
        stop = min(start + batch, len(obs))
        obs_batch = obs[start:stop].to(agent.device)
        action_batch = action[start:stop].to(agent.device)
        z = model.encode(obs_batch, None)
        predicted_force = model.force(z, action_batch, None).squeeze(-1)
        predicted_envelope = model.envelope(z, action_batch, None).squeeze(-1)
        predicted_upper_parts.append(torch.maximum(predicted_force, predicted_envelope).cpu())
    predicted_upper_n = torch.cat(predicted_upper_parts) * 15.0
    actual_next_force_n = next_obs[:, 0] * 15.0
    absolute_residual_n = (predicted_upper_n - actual_next_force_n).abs()

    records = []
    rows = {}
    for target_n in TARGETS_N:
        mask = torch.isclose(obs[:, 1], torch.tensor(target_n / 15.0), atol=1e-5)
        residual = absolute_residual_n[mask]
        if not len(residual):
            raise RuntimeError(f"TRAIN-validation contains no {target_n:g} N rows")
        records.extend(
            {
                "target_force_n": target_n,
                "absolute_residual_n": float(value),
            }
            for value in residual.tolist()
        )
        error = predicted_upper_n[mask] - actual_next_force_n[mask]
        rows[f"{target_n:g}"] = {
            "samples": int(mask.sum()),
            "upper_predictor_bias_n": float(error.mean()),
            "upper_predictor_rmse_n": float(torch.sqrt(error.square().mean())),
            "absolute_residual_p95_n": float(torch.quantile(residual, 0.95)),
        }

    radii = fit_target_residual_radii(records, coverage=COVERAGE)
    for target_n, radius_n in radii.items():
        rows[f"{target_n:g}"]["calibrated_radius_n"] = float(radius_n)
        rows[f"{target_n:g}"]["empirical_coverage"] = float(
            (
                absolute_residual_n[
                    torch.isclose(
                        obs[:, 1], torch.tensor(target_n / 15.0), atol=1e-5
                    )
                ]
                <= radius_n
            ).float().mean()
        )
    return rows


def main() -> int:
    if OUTPUT.exists():
        raise SystemExit(f"calibration output already exists: {OUTPUT}")
    _, validation_episodes, _, validation, metadata = load_recovery_aware_collection(
        Path("unused")
    )
    per_checkpoint = {}
    checkpoint_hashes = {}
    work_dir = OUTPUT.parent / ".v19_calibration_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    try:
        for seed in SEEDS:
            path = checkpoint(seed)
            checkpoint_hashes[str(seed)] = sha256(path)
            agent = load_agent(seed, work_dir)
            per_checkpoint[str(seed)] = calibrate_checkpoint(agent, validation)
            del agent
            torch.cuda.empty_cache()
    finally:
        if work_dir.exists() and not any(work_dir.iterdir()):
            work_dir.rmdir()

    payload = {
        "format": "forcewipe_v19_train_validation_uncertainty_calibration_v1",
        "completed_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "coverage": COVERAGE,
        "targets_n": list(TARGETS_N),
        "checkpoint_seeds": list(SEEDS),
        "checkpoint_sha256": checkpoint_hashes,
        "validation_episodes": len(validation_episodes),
        "validation_transitions": int(len(validation["observation"])),
        "data_partition": "TRAIN-validation only",
        "source_metadata": metadata,
        "per_checkpoint": per_checkpoint,
        "method": (
            "finite-sample order statistic of the absolute residual between "
            "the larger of the EMA next-force and transient-envelope predictions "
            "and the measured next-step force"
        ),
        "claim_boundary": (
            "Offline checkpoint calibration only; no simulator environment was "
            "created and no DEV, qualification, CAL, or TEST trace was read."
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, OUTPUT)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
