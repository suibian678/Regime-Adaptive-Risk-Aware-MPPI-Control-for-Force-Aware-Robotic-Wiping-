"""V16.8 domain-randomized world-model and target-expert training source."""

from __future__ import annotations

import json
from pathlib import Path

from forcewipe_v14.native_tdmpc2_v6_training import load_episode
from forcewipe_v14.native_tdmpc2_v6_training_v2 import concatenate
from forcewipe_v16.native_tdmpc2_v6_training import load_target_expert_collection
from forcewipe_v16.target_expert_training import build_target_expert_config


TEACHER_ROOT = Path(__file__).resolve().parents[3]
OOD_COLLECTION = (
    TEACHER_ROOT
    / "forcewipe_v16/results/train/ood_aware_collection"
    / "v16p8_ood_train45_teacher_v3_20260828_r1"
)


def build_ood_aware_config(*, seed: int, work_dir: Path, batch_size: int = 64):
    cfg = build_target_expert_config(seed=seed, work_dir=work_dir, batch_size=batch_size)
    # Buffer capacity is min(steps, buffer_size) in the upstream implementation.
    # Both fields must cover the combined nominal + OOD transition source.
    cfg.steps = 100_000
    cfg.buffer_size = 100_000
    cfg.exp_name = "v16p8-v6-ood-aware-target-conditioned-experts"
    return cfg


def load_ood_aware_target_expert_collection(_collection_root: Path):
    base_train, base_validation, base_demo, base_validation_flat, base_metadata = (
        load_target_expert_collection(Path("unused"))
    )
    result = json.loads((OOD_COLLECTION / "RESULT.json").read_text(encoding="utf-8"))
    if not result.get("joint_training_permitted"):
        raise ValueError("V16.8 OOD-aware collection is not training-permitted")

    train_tds = list(base_train)
    validation_tds = list(base_validation)
    demo_rows = [base_demo]
    validation_rows = [base_validation_flat]
    ood_train_episodes = 0
    ood_validation_episodes = 0
    ood_demo_episodes = 0
    excluded_demo_episodes = []
    for summary in result["evaluations"]:
        td, flat = load_episode(OOD_COLLECTION / summary["trace"])
        if int(summary["replicate"]) == 14:
            validation_tds.append(td)
            validation_rows.append(flat)
            ood_validation_episodes += 1
            continue
        train_tds.append(td)
        ood_train_episodes += 1
        if (
            bool(summary["success"])
            and bool(summary["tracking_gate_passed"])
            and int(summary["force_limit_violation_samples"]) == 0
        ):
            demo_rows.append(flat)
            ood_demo_episodes += 1
        else:
            excluded_demo_episodes.append(int(summary["episode_id"]))

    if ood_train_episodes != 42 or ood_validation_episodes != 3:
        raise ValueError("V16.8 OOD-aware TRAIN/validation split is incomplete")
    if len(train_tds) != 72 or len(validation_tds) != 6:
        raise ValueError("V16.8 combined episode split is inconsistent")
    metadata = {
        "collection_gate_passed": True,
        "joint_training_permitted": True,
        "base_collection_metadata": base_metadata,
        "ood_collection": str(OOD_COLLECTION),
        "ood_collection_result_format": result["format"],
        "world_model_train_episodes": len(train_tds),
        "train_validation_episodes": len(validation_tds),
        "ood_world_model_train_episodes": ood_train_episodes,
        "ood_actor_demo_episodes": ood_demo_episodes,
        "excluded_ood_actor_demo_episode_ids": excluded_demo_episodes,
        "actor_demo_rule": "safe tracking-gate OOD teacher rows plus the frozen V16.4 target-expert demonstrations",
        "claim_boundary": "TRAIN and TRAIN-validation only; no DEV, CAL, or TEST trajectory is used.",
    }
    return (
        train_tds,
        validation_tds,
        concatenate(demo_rows),
        concatenate(validation_rows),
        metadata,
    )


__all__ = ["build_ood_aware_config", "load_ood_aware_target_expert_collection"]
