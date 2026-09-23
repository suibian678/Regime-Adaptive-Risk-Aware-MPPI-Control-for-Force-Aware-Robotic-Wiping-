"""Matched V19 training configuration."""

from __future__ import annotations

from pathlib import Path

from forcewipe_v16.force_calibrated_strong_bc_training import (
    build_force_calibrated_strong_bc_config,
)
from forcewipe_v19.force_conditioned_world_model import (
    configure_force_conditioned_world_model,
)


def build_v19_training_config(
    *,
    seed: int,
    work_dir: Path,
    bc_coefficient: float,
    batch_size: int = 64,
):
    coefficient = float(bc_coefficient)
    if coefficient not in (0.0, 0.5, 2.0):
        raise ValueError("V19 BC coefficient must be one of 0, 0.5, or 2")
    cfg = build_force_calibrated_strong_bc_config(
        seed=int(seed),
        work_dir=work_dir,
        batch_size=int(batch_size),
    )
    cfg.demo_bc_online_coef = coefficient
    cfg = configure_force_conditioned_world_model(cfg)
    cfg.exp_name = f"v19-force-conditioned-bc{coefficient:g}"
    return cfg


__all__ = ["build_v19_training_config"]
