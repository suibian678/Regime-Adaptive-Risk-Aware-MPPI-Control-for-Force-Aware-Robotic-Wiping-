"""Recovery-aware V16 training source without DEV/CAL/TEST leakage."""

from __future__ import annotations

import json
from pathlib import Path
from forcewipe.paths import TRAINING_DATA_ROOT

import torch

from forcewipe.training_data.native_tdmpc2 import load_episode
from forcewipe.training_data.batch_training import concatenate
from forcewipe.learning.ood_aware_training import (
    build_ood_aware_config,
    load_ood_aware_target_expert_collection,
)


TEACHER_ROOT = TRAINING_DATA_ROOT
RECOVERY_COLLECTION = (
    TEACHER_ROOT / "forcewipe_v16/results/train/ood_aware_collection"
    / "v16p16_low_mid_forced_contact_loss_train24_20260828_r1"
)
INTERVENTION_ACTION = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32)


def build_recovery_aware_config(*, seed: int, work_dir: Path, batch_size: int = 64):
    cfg = build_ood_aware_config(seed=seed, work_dir=work_dir, batch_size=batch_size)
    cfg.exp_name = "v16p17-v6-recovery-aware-target-conditioned-experts"
    return cfg


def _without_intervention_rows(flat: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], int]:
    mask = ~torch.isclose(flat["action"], INTERVENTION_ACTION, atol=1e-8).all(dim=-1)
    removed = int((~mask).sum())
    return {key: value[mask] for key, value in flat.items()}, removed


def load_recovery_aware_collection(_collection_root: Path):
    base_train, base_validation, base_demo, base_validation_flat, base_metadata = (
        load_ood_aware_target_expert_collection(Path("unused"))
    )
    result = json.loads((RECOVERY_COLLECTION / "RESULT.json").read_text(encoding="utf-8"))
    if not result.get("joint_training_permitted"):
        raise ValueError("V16.16 recovery collection is not training-permitted")

    train_tds = list(base_train)
    validation_tds = list(base_validation)
    demo_rows = [base_demo]
    validation_rows = [base_validation_flat]
    recovery_train_episodes = 0
    recovery_validation_episodes = 0
    recovery_demo_rows = 0
    excluded_intervention_rows = 0
    for summary in result["evaluations"]:
        td, flat = load_episode(RECOVERY_COLLECTION / summary["trace"])
        if int(summary["replicate"]) == 11:
            validation_tds.append(td)
            validation_rows.append(flat)
            recovery_validation_episodes += 1
            continue
        train_tds.append(td)
        recovery_train_episodes += 1
        if not (
            bool(summary["success"])
            and bool(summary["tracking_gate_passed"])
            and bool(summary["contact_loss_then_recovery"])
            and int(summary["force_limit_violation_samples"]) == 0
        ):
            raise ValueError("V16.16 TRAIN actor source violates the frozen recovery gate")
        filtered, removed = _without_intervention_rows(flat)
        if removed != int(summary["intervention_rows"]):
            raise ValueError("V16.16 intervention-row identity mismatch")
        excluded_intervention_rows += removed
        recovery_demo_rows += len(filtered["action"])
        demo_rows.append(filtered)

    if recovery_train_episodes != 22 or recovery_validation_episodes != 2:
        raise ValueError("V16.17 recovery split must contain 22 TRAIN and 2 validation episodes")
    if len(train_tds) != 94 or len(validation_tds) != 8:
        raise ValueError("V16.17 combined world-model split is inconsistent")
    metadata = {
        "collection_gate_passed": True,
        "joint_training_permitted": True,
        "base_collection_metadata": base_metadata,
        "recovery_collection": str(RECOVERY_COLLECTION),
        "world_model_train_episodes": len(train_tds),
        "train_validation_episodes": len(validation_tds),
        "recovery_world_model_train_episodes": recovery_train_episodes,
        "recovery_validation_episodes": recovery_validation_episodes,
        "recovery_actor_demo_rows": recovery_demo_rows,
        "excluded_intervention_rows": excluded_intervention_rows,
        "actor_demo_rule": (
            "exclude exact TRAIN-only outward intervention actions; retain subsequent "
            "safe teacher contact-recovery actions"
        ),
        "claim_boundary": "TRAIN and TRAIN-validation only; no DEV, CAL, or TEST trajectory used.",
    }
    return (
        train_tds,
        validation_tds,
        concatenate(demo_rows),
        concatenate(validation_rows),
        metadata,
    )


__all__ = ["build_recovery_aware_config", "load_recovery_aware_collection"]

