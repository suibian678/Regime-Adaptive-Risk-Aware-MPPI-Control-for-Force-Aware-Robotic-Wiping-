"""V16 mixed-data training utilities for direct TD-MPC2 force tracking."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from forcewipe_v14.native_tdmpc2_v6_training import (
    audit_model,
    build_v14_config,
    config_dict,
    load_episode,
    populate_buffer,
)
from forcewipe_v14.native_tdmpc2_v6_training_v2 import behavior_clone_step, concatenate


TEACHER_ROOT = Path(__file__).resolve().parents[3]
SAFE_COLLECTION = (
    TEACHER_ROOT
    / "forcewipe_v14/results/train/v6_closed_loop_collection"
    / "v14p1_v6_closed_loop_train30_teacher_v3_20260828_r1"
)
HIGH_TRACKING_COLLECTION = (
    TEACHER_ROOT
    / "forcewipe_v14/results/train/v6_closed_loop_collection"
    / "v14_v6_closed_loop_train15_20260828_r1"
)
HIGH_TRACKING_DEMO_MULTIPLIER = 2
TARGET_EXPERT_HIGH_DEMO_MULTIPLIER = 3


def build_v16_config(*, seed: int, work_dir: Path, batch_size: int = 64):
    cfg = build_v14_config(seed=seed, work_dir=work_dir, batch_size=batch_size)
    # Exact tracking objective: no planner deadband.  The weaker online BC term
    # lets TD-MPC2 improve beyond the conservative teacher while retaining the
    # initial policy support.
    cfg.force_plan_deadband = 0.0
    cfg.force_plan_coef = 6.0
    cfg.force_coef = 10.0
    cfg.demo_bc_online_coef = 0.5
    cfg.exp_name = "v16-v6-mixed-force-joint"
    return cfg


def load_mixed_collection(_collection_root: Path):
    safe_result = json.loads((SAFE_COLLECTION / "RESULT.json").read_text())
    high_result = json.loads((HIGH_TRACKING_COLLECTION / "RESULT.json").read_text())
    if not safe_result.get("joint_training_permitted"):
        raise ValueError("safe V3 collection is not training-permitted")

    train_tds, validation_tds = [], []
    safe_train_flat, validation_flat = [], []
    for summary in safe_result["evaluations"]:
        td, flat = load_episode(SAFE_COLLECTION / summary["trace"])
        if int(summary["replicate"]) == 9:
            validation_tds.append(td)
            validation_flat.append(flat)
        else:
            train_tds.append(td)
            safe_train_flat.append(flat)

    selected_high = [
        summary
        for summary in high_result["evaluations"]
        if float(summary["target_force_n"]) == 12.0
        and bool(summary["success"])
        and int(summary["force_limit_violation_samples"]) == 0
    ]
    if len(selected_high) != 3:
        raise ValueError("expected exactly three safe high-tracking 12 N TRAIN episodes")
    high_flat = []
    for summary in selected_high:
        td, flat = load_episode(HIGH_TRACKING_COLLECTION / summary["trace"])
        train_tds.append(td)
        high_flat.append(flat)

    if len(train_tds) != 30 or len(validation_tds) != 3:
        raise ValueError("V16 split must contain 30 TRAIN and 3 validation episodes")
    demo_rows = safe_train_flat + high_flat * HIGH_TRACKING_DEMO_MULTIPLIER
    metadata = {
        "collection_gate_passed": True,
        "joint_training_permitted": True,
        "safe_collection": str(SAFE_COLLECTION),
        "high_tracking_collection": str(HIGH_TRACKING_COLLECTION),
        "selected_high_tracking_12n_episodes": len(selected_high),
        "selected_high_tracking_trace_sha256": [row["trace_sha256"] for row in selected_high],
        "high_tracking_demo_multiplier": HIGH_TRACKING_DEMO_MULTIPLIER,
        "claim_boundary": "TRAIN-only mixed source; no DEV, CAL, or TEST trajectory is used.",
    }
    return (
        train_tds,
        validation_tds,
        concatenate(demo_rows),
        concatenate(validation_flat),
        metadata,
    )


def load_target_expert_collection(_collection_root: Path):
    """Keep all safe transitions for the world model, but deconflict actor labels.

    Low and mid experts imitate their V3 trajectories.  The high-force expert
    imitates only the safe high-tracking 12 N subset; conservative V3 12 N
    transitions remain in the model buffer for tail-dynamics learning.
    """
    safe_result = json.loads((SAFE_COLLECTION / "RESULT.json").read_text())
    high_result = json.loads((HIGH_TRACKING_COLLECTION / "RESULT.json").read_text())
    train_tds, validation_tds = [], []
    lower_demo_flat, validation_flat = [], []
    for summary in safe_result["evaluations"]:
        td, flat = load_episode(SAFE_COLLECTION / summary["trace"])
        if int(summary["replicate"]) == 9:
            validation_tds.append(td)
            validation_flat.append(flat)
        else:
            train_tds.append(td)
            if float(summary["target_force_n"]) < 10.0:
                lower_demo_flat.append(flat)
    selected_high = [
        summary
        for summary in high_result["evaluations"]
        if float(summary["target_force_n"]) == 12.0
        and bool(summary["success"])
        and int(summary["force_limit_violation_samples"]) == 0
    ]
    high_flat = []
    for summary in selected_high:
        td, flat = load_episode(HIGH_TRACKING_COLLECTION / summary["trace"])
        train_tds.append(td)
        high_flat.append(flat)
    if len(train_tds) != 30 or len(validation_tds) != 3 or len(selected_high) != 3:
        raise ValueError("V16.4 target-expert source split is incomplete")
    demo_rows = lower_demo_flat + high_flat * TARGET_EXPERT_HIGH_DEMO_MULTIPLIER
    metadata = {
        "collection_gate_passed": True,
        "joint_training_permitted": True,
        "safe_collection": str(SAFE_COLLECTION),
        "high_tracking_collection": str(HIGH_TRACKING_COLLECTION),
        "world_model_train_episodes": len(train_tds),
        "policy_demo_rule": "5/8 N V3 rows plus only safe high-tracking 12 N rows",
        "selected_high_tracking_12n_episodes": len(selected_high),
        "target_expert_high_demo_multiplier": TARGET_EXPERT_HIGH_DEMO_MULTIPLIER,
        "claim_boundary": "TRAIN-only target-expert source separation; no DEV, CAL, or TEST trajectory is used.",
    }
    return (
        train_tds,
        validation_tds,
        concatenate(demo_rows),
        concatenate(validation_flat),
        metadata,
    )


__all__ = [
    "audit_model",
    "behavior_clone_step",
    "build_v16_config",
    "config_dict",
    "load_mixed_collection",
    "load_target_expert_collection",
    "populate_buffer",
]
