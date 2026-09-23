"""Recovery-aware TD-MPC2 training with the intended V16 force objective."""

from __future__ import annotations

from pathlib import Path

from forcewipe_v16.recovery_aware_training import build_recovery_aware_config


def build_force_calibrated_recovery_config(
    *, seed: int, work_dir: Path, batch_size: int = 64
):
    cfg = build_recovery_aware_config(
        seed=seed, work_dir=work_dir, batch_size=batch_size
    )
    # Restore the exact-force V16 settings that were unintentionally lost when
    # the target-expert branch inherited directly from the V14 configuration.
    cfg.force_plan_deadband = 0.0
    cfg.force_plan_coef = 6.0
    cfg.force_coef = 10.0
    cfg.demo_bc_online_coef = 0.5
    cfg.exp_name = "v16p22-v6-force-calibrated-recovery-aware-target-experts"
    return cfg


__all__ = ["build_force_calibrated_recovery_config"]
