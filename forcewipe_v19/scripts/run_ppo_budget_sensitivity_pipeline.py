#!/usr/bin/env python3
"""Resumable sequential launcher for the frozen PPO sensitivity study."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT.parent / "forcewipe_v16" / "code"))

from forcewipe_v19.ppo_budget_sensitivity import (  # noqa: E402
    TRAINING_SEEDS,
    configuration_endpoints,
)


TRAIN_ROOT = ROOT / "results" / "train" / "ppo_budget_sensitivity"
DEV_ROOT = ROOT / "results" / "dev" / "ppo_budget_sensitivity"


def completed(path: Path, *, evaluations: int | None = None) -> bool:
    result = path / "RESULT.json"
    if not result.is_file():
        return False
    payload = json.loads(result.read_text(encoding="utf-8"))
    if payload.get("status") != "completed":
        return False
    return evaluations is None or len(payload.get("evaluations", [])) == evaluations


def invoke(*arguments: str) -> None:
    subprocess.run([sys.executable, *arguments], cwd=ROOT, check=True)


def invoke_parallel(commands: list[tuple[str, ...]]) -> None:
    if not commands:
        return
    with ThreadPoolExecutor(max_workers=min(5, len(commands))) as executor:
        futures = [executor.submit(invoke, *command) for command in commands]
        for future in futures:
            future.result()


def main() -> int:
    # Train each profile once per seed.  The default run yields all three
    # budget checkpoints; non-default runs yield the 2x checkpoint.
    profiles = []
    for profile_id, _multiplier in configuration_endpoints():
        if profile_id not in profiles:
            profiles.append(profile_id)
    for profile_id in profiles:
        commands = []
        for seed in TRAINING_SEEDS:
            output = TRAIN_ROOT / f"ppo_sensitivity_{profile_id}_seed{seed}"
            if completed(output):
                print(f"SKIP completed training {profile_id} seed {seed}", flush=True)
                continue
            staging = TRAIN_ROOT / f".ppo_sensitivity_{profile_id}_seed{seed}.creating"
            if staging.exists():
                print(f"SKIP active/staged training {profile_id} seed {seed}", flush=True)
                continue
            commands.append((
                "scripts/train_ppo_budget_sensitivity.py",
                "--profile", profile_id,
                "--seed", str(seed),
            ))
        invoke_parallel(commands)

    # Evaluate every prespecified checkpoint on the independent DEV roster.
    for profile_id, multiplier in configuration_endpoints():
        commands = []
        for seed in TRAINING_SEEDS:
            output = DEV_ROOT / f"dev_{profile_id}_{multiplier}x_seed{seed}"
            if completed(output, evaluations=18):
                print(f"SKIP completed DEV {profile_id} {multiplier}x seed {seed}", flush=True)
                continue
            commands.append((
                "scripts/evaluate_ppo_sensitivity_dev.py",
                "--profile", profile_id,
                "--budget", str(multiplier),
                "--seed", str(seed),
            ))
        invoke_parallel(commands)
    invoke("scripts/analyze_ppo_sensitivity_dev.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
