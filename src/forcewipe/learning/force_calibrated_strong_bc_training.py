"""Exact-force world model with the recovery actor's original BC strength."""

from __future__ import annotations

from pathlib import Path

from forcewipe.learning.recovery_aware_training import build_recovery_aware_config


def build_force_calibrated_strong_bc_config(
    *, seed: int, work_dir: Path, batch_size: int = 64
):
    cfg = build_recovery_aware_config(
        seed=seed, work_dir=work_dir, batch_size=batch_size
    )
    cfg.force_plan_deadband = 0.0
    cfg.force_plan_coef = 6.0
    cfg.force_coef = 10.0
    # Preserve the V16.17/V14 teacher-retention strength.  V16.22 changed this
    # simultaneously with the force objective and degraded low-force contact.
    cfg.demo_bc_online_coef = 2.0
    cfg.exp_name = "v16p25-v6-force-calibrated-recovery-strong-bc"
    return cfg


__all__ = ["build_force_calibrated_strong_bc_config"]
