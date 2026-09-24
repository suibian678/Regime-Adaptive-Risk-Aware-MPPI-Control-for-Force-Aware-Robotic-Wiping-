#!/usr/bin/env python3
"""Exercise every deployment-gap factor through a real SAPIEN environment step."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TEACHER_ROOT = ROOT.parent
for source in (ROOT / "src", ROOT / "scripts", ROOT / "vendor/tdmpc2", ROOT / "vendor"):
    sys.path.insert(0, str(source))

import forcewipe.direct.tdmpc2_direct_sapien_env as sapien_module  # noqa: E402
from forcewipe.deployment_gap import DeploymentGapConfig  # noqa: E402
from forcewipe.deployment_gap_sapien_env import V19DeploymentGapEnv  # noqa: E402
from forcewipe.factor_separated_scenarios import (  # noqa: E402
    SCENARIO_ID_BASE,
    SCENARIO_SEED,
    v19_factor_separated_scenario,
)


PROTOCOL = ROOT / "config/DEPLOYMENT_GAP_STRESS_PROTOCOL_2026-09-20.json"
OUTPUT = ROOT / "results/code_only/DEPLOYMENT_GAP_BINDING_AUDIT_2026-09-20.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(config: DeploymentGapConfig) -> dict:
    env = V19DeploymentGapEnv(
        gap_config=config,
        target_force_n=8.0,
        scenario_seed=SCENARIO_SEED,
        scenario_id=SCENARIO_ID_BASE + 1,
    )
    try:
        initial, reset_info = env.reset(seed=SCENARIO_SEED)
        current, _, _, _, info = env.step(np.array([0.0, 0.0, 1.0], dtype=np.float32))
        second, _, _, _, second_info = env.step(
            np.array([0.0, 0.0, 1.0], dtype=np.float32)
        )
        return {
            "config": asdict(config),
            "factor": config.factor,
            "initial": initial.tolist(),
            "current": current.tolist(),
            "second": second.tolist(),
            "reset_log": reset_info["deployment_gap"],
            "step_log": info["deployment_gap"],
            "second_step_log": second_info["deployment_gap"],
        }
    finally:
        env.close()


def main() -> int:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    rows = [
        run(DeploymentGapConfig(random_seed=201)),
        run(DeploymentGapConfig(force_noise_rms_n=0.035, random_seed=201)),
        run(DeploymentGapConfig(force_bias_n=0.8, random_seed=201)),
        run(DeploymentGapConfig(observation_delay_samples=1, random_seed=201)),
        run(DeploymentGapConfig(tcp_pose_error_magnitude_m=0.0001, random_seed=201)),
        run(DeploymentGapConfig(surface_normal_error_magnitude_deg=5.0, random_seed=201)),
    ]
    checks = []
    nominal = rows[0]
    checks.append(("nominal_identity", nominal["step_log"]["native_observation"] == nominal["step_log"]["learner_visible_observation"]))
    noise = rows[1]
    checks.append(("force_noise_nonzero", any(
        abs(row["sampled_force_noise_n"]) > 0.0
        for row in (noise["step_log"], noise["second_step_log"])
    )))
    bias = rows[2]
    checks.append(("force_bias_exact", bias["step_log"]["force_bias_n"] == 0.8))
    delay = rows[3]
    checks.append(("one_step_delay_bound", delay["step_log"]["learner_visible_observation"] == delay["reset_log"]["perturbed_current_observation"]))
    tcp = rows[4]
    checks.append(("tcp_error_exact_magnitude", abs(
        np.linalg.norm(tcp["step_log"]["tcp_bias_cross_normal_m"]) - 0.0001
    ) < 1e-12))
    normal = rows[5]
    checks.append(("normal_error_exact_magnitude", abs(
        abs(normal["step_log"]["signed_surface_normal_error_deg"]) - 5.0
    ) < 1e-12))
    checks.append(("normal_error_reaches_actuator", abs(
        normal["step_log"]["issued_physical_action"][0]
    ) > 0.0))
    if not all(value for _, value in checks):
        raise RuntimeError(f"deployment-gap binding failed: {checks}")
    payload = {
        "format": "forcewipe_v19_deployment_gap_binding_audit_v1",
        "status": "pass",
        "completed_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "protocol_sha256": sha256(PROTOCOL),
        "checks": [{"name": name, "pass": bool(value)} for name, value in checks],
        "rows": rows,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"PASS {len(checks)}/{len(checks)} {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
