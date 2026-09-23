#!/usr/bin/env python3
"""Exercise the complete V19 planner on causal, non-physical probe states."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
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
from forcewipe_v19.risk_aware_mppi import (  # noqa: E402
    RegimeAdaptiveRiskAwareTDMPC2,
    configure_regime_adaptive_risk_mppi,
)


SEED = 172
CALIBRATION = ROOT / "results/calibration/V19_TRAIN_VALIDATION_UNCERTAINTY_CALIBRATION.json"
OUTPUT = ROOT / "results/code_only/V19_OFFLINE_PLANNING_SMOKE_AUDIT.json"
CHECKPOINT = (
    V16_ROOT
    / "results/train/native_tdmpc2_v6_v16"
    / "v16p25_force_calibrated_strong_bc_seed172_gate200_bc2000_joint1800_20260828_r1"
    / "v16p3_target_expert_tdmpc2.pt"
)


def load_agent(work_dir: Path) -> RegimeAdaptiveRiskAwareTDMPC2:
    calibration = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    rows = calibration["per_checkpoint"][str(SEED)]
    radii = {
        float(target): float(detail["calibrated_radius_n"])
        for target, detail in rows.items()
    }
    initialization_seed = SEED + 12
    random.seed(initialization_seed)
    np.random.seed(initialization_seed)
    torch.manual_seed(initialization_seed)
    torch.cuda.manual_seed_all(initialization_seed)
    cfg = configure_regime_adaptive_risk_mppi(
        build_force_calibrated_strong_bc_config(seed=SEED, work_dir=work_dir),
        calibrated_radii_n=radii,
    )
    agent = RegimeAdaptiveRiskAwareTDMPC2(cfg)
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    agent.model.load_state_dict(payload["model"])
    agent.capture_eval_ema()
    agent._eval_ema_model.load_state_dict(payload["eval_ema_model"])
    agent.model.eval()
    agent._eval_ema_model.eval()
    return agent


def probe_observation(template: torch.Tensor, *, target_n: float, phase: str) -> torch.Tensor:
    obs = template.clone()
    obs[1] = float(target_n) / 15.0
    if phase == "acquisition":
        obs[0], obs[2] = 0.0, 0.0
    elif phase == "tracking":
        obs[0], obs[2] = float(target_n) / 15.0, 0.0
    elif phase == "recovery":
        obs[0], obs[2] = 0.35 * float(target_n) / 15.0, -0.10
    elif phase == "transient":
        obs[0], obs[2] = 0.90 * float(target_n) / 15.0, 0.50
    else:
        raise ValueError(phase)
    return obs


def main() -> int:
    if OUTPUT.exists():
        raise SystemExit(f"offline smoke output already exists: {OUTPUT}")
    _, _, _, validation, _ = load_recovery_aware_collection(Path("unused"))
    template = validation["observation"][0]
    work_dir = OUTPUT.parent / ".v19_smoke_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    agent = load_agent(work_dir)
    rows = []
    for target_n in (5.0, 8.0, 12.0):
        for phase in ("acquisition", "tracking", "recovery", "transient"):
            obs = probe_observation(template, target_n=target_n, phase=phase)
            agent._prev_mean.zero_()
            agent._had_contact = phase != "acquisition"
            started = time.perf_counter()
            action = agent.act(obs, t0=(phase == "acquisition"), eval_mode=True)
            elapsed_ms = 1_000.0 * (time.perf_counter() - started)
            diagnostics = agent.risk_diagnostics()
            action_array = np.asarray(action.detach().cpu(), dtype=float)
            finite = bool(np.all(np.isfinite(action_array)))
            bounded = bool(np.max(np.abs(action_array)) <= 1.0 + 1e-6)
            if not finite or not bounded:
                raise RuntimeError(f"invalid offline action for {target_n:g} N {phase}")
            rows.append(
                {
                    "target_force_n": target_n,
                    "probe_phase": phase,
                    "action": action_array.tolist(),
                    "finite_action": finite,
                    "bounded_action": bounded,
                    "planning_time_ms": elapsed_ms,
                    "diagnostics": diagnostics,
                }
            )

    observed_budgets = sorted(
        {row["diagnostics"]["budget"]["regime"] for row in rows}
    )
    payload = {
        "format": "forcewipe_v19_offline_planning_smoke_audit_v1",
        "completed_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "checkpoint_seed": SEED,
        "probes": rows,
        "checks": {
            "probe_count_12": len(rows) == 12,
            "all_actions_finite": all(row["finite_action"] for row in rows),
            "all_actions_bounded": all(row["bounded_action"] for row in rows),
            "all_risk_scores_finite": all(
                math.isfinite(row["diagnostics"]["weights"]["risk_score"])
                for row in rows
            ),
            "multiple_compute_regimes_exercised": len(observed_budgets) >= 3,
        },
        "observed_budget_regimes": observed_budgets,
        "claim_boundary": (
            "Code-only planner smoke test on constructed causal observations; "
            "no simulator environment or physical evaluation was run."
        ),
    }
    if not all(payload["checks"].values()):
        raise RuntimeError(f"offline planner smoke failed: {payload['checks']}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, OUTPUT)
    if work_dir.exists() and not any(work_dir.iterdir()):
        work_dir.rmdir()
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
