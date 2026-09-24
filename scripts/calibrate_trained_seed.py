#!/usr/bin/env python3
"""Calibrate one trained V19 checkpoint on TRAIN-validation data."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
TEACHER_ROOT = ROOT.parent
for source in (ROOT / "src", ROOT / "scripts", ROOT / "vendor/tdmpc2", ROOT / "vendor"):
    sys.path.insert(0, str(source))

from forcewipe.learning.recovery_aware_training import load_recovery_aware_collection  # noqa: E402
from forcewipe.calibration import fit_target_residual_radii  # noqa: E402
from forcewipe.force_conditioned_world_model import ForceConditionedTDMPC2  # noqa: E402
from forcewipe.training import build_v19_training_config  # noqa: E402


TARGETS_N = (5.0, 8.0, 12.0)
COVERAGE = 0.90


@torch.inference_mode()
def analyse(agent, validation: dict[str, torch.Tensor]) -> dict:
    model = agent._eval_ema_model
    obs = validation["observation"]
    action = validation["executed_action"]
    next_obs = validation["next_observation"]
    predictions = []
    for start in range(0, len(obs), 1024):
        stop = min(start + 1024, len(obs))
        z = model.encode(obs[start:stop].to(agent.device), None)
        act = action[start:stop].to(agent.device)
        upper = torch.maximum(
            model.force(z, act, None).squeeze(-1),
            model.envelope(z, act, None).squeeze(-1),
        )
        predictions.append(upper.cpu())
    predicted_upper_n = torch.cat(predictions) * 15.0
    actual_n = next_obs[:, 0] * 15.0
    residual_n = (predicted_upper_n - actual_n).abs()
    records, rows, condition_centres = [], {}, []
    for target_n in TARGETS_N:
        mask = torch.isclose(obs[:, 1], torch.tensor(target_n / 15.0), atol=1e-5)
        values = residual_n[mask]
        records.extend(
            {"target_force_n": target_n, "absolute_residual_n": float(value)}
            for value in values.tolist()
        )
        error = predicted_upper_n[mask] - actual_n[mask]
        rows[f"{target_n:g}"] = {
            "samples": int(mask.sum()),
            "upper_predictor_bias_n": float(error.mean()),
            "upper_predictor_rmse_n": float(torch.sqrt(error.square().mean())),
            "absolute_residual_p95_n": float(torch.quantile(values, 0.95)),
        }
        representative = obs[torch.where(mask)[0][0]].unsqueeze(0).to(agent.device)
        condition_centres.append(model.force_condition(model.encode(representative, None))[0].cpu())
    radii = fit_target_residual_radii(records, coverage=COVERAGE)
    for target_n, radius_n in radii.items():
        mask = torch.isclose(obs[:, 1], torch.tensor(target_n / 15.0), atol=1e-5)
        rows[f"{target_n:g}"].update(
            {
                "calibrated_radius_n": float(radius_n),
                "empirical_coverage": float((residual_n[mask] <= radius_n).float().mean()),
            }
        )
    pairwise_l2 = [
        float(torch.linalg.vector_norm(condition_centres[i] - condition_centres[j]))
        for i, j in ((0, 1), (0, 2), (1, 2))
    ]
    return {
        "targets": rows,
        "target_condition_pairwise_l2": pairwise_l2,
        "target_conditions_distinct": min(pairwise_l2) > 1e-6,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--bc-coef", type=float, required=True, choices=(0.0, 0.5, 2.0))
    args = parser.parse_args()
    run_id = f"v19_force_conditioned_seed{args.seed}_bc{args.bc_coef:g}_u4000"
    run = ROOT / "results/train" / run_id
    output = run / "TRAIN_VALIDATION_CALIBRATION.json"
    if output.exists():
        raise SystemExit(f"calibration already exists: {output}")
    payload = torch.load(run / "v19_force_conditioned_tdmpc2.pt", map_location="cpu", weights_only=False)
    cfg = build_v19_training_config(
        seed=args.seed,
        work_dir=run,
        bc_coefficient=args.bc_coef,
    )
    agent = ForceConditionedTDMPC2(cfg)
    agent.model.load_state_dict(payload["model"])
    agent.capture_eval_ema()
    agent._eval_ema_model.load_state_dict(payload["eval_ema_model"])
    agent.model.eval()
    agent._eval_ema_model.eval()
    _, validation_episodes, _, validation, _ = load_recovery_aware_collection(Path("unused"))
    result = {
        "format": "forcewipe_v19_trained_seed_calibration_v1",
        "completed_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "run_id": run_id,
        "coverage": COVERAGE,
        "validation_episodes": len(validation_episodes),
        "validation_transitions": int(len(validation["observation"])),
        **analyse(agent, validation),
        "claim_boundary": "TRAIN-validation inference only; no simulator evaluation.",
    }
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
