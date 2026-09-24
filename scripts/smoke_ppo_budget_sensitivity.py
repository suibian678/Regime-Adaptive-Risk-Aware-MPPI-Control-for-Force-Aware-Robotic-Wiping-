#!/usr/bin/env python3
"""Short runtime smoke for the extended PPO training path; no result artifact."""

from __future__ import annotations

from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
TEACHER_ROOT = ROOT.parent
for source in (ROOT / "src", ROOT / "scripts", ROOT / "vendor/tdmpc2", ROOT / "vendor"):
    sys.path.insert(0, str(source))

import forcewipe.direct.tdmpc2_direct_sapien_env as direct_env_module  # noqa: E402
from forcewipe.learning.direct_ppo_baseline import DirectPPOActorCritic, DirectPPOOptimizer  # noqa: E402
from forcewipe.learning.ppo_scenarios import v16p30_ppo_scenario  # noqa: E402
from forcewipe.ppo_budget_sensitivity import config_for_profile  # noqa: E402
from ppo_collector import OnPolicyCollector  # noqa: E402


def main() -> int:
    direct_env_module.nominal_direct_scenario = v16p30_ppo_scenario
    config = config_for_profile("default")
    model = DirectPPOActorCritic(config)
    optimiser = DirectPPOOptimizer(model)
    collector = OnPolicyCollector(
        model,
        device=torch.device("cpu"),
        action_generator=torch.Generator().manual_seed(1),
    )
    try:
        batch = collector.collect(32)
        update = optimiser.update(batch, generator=torch.Generator().manual_seed(2))
    finally:
        collector.close()
    finite = all(torch.isfinite(parameter).all().item() for parameter in model.parameters())
    print({"samples": len(batch["action"]), "updates": update["optimizer_updates"], "finite": finite})
    if len(batch["action"]) != 32 or update["optimizer_updates"] != 80 or not finite:
        raise RuntimeError("PPO sensitivity smoke test failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
