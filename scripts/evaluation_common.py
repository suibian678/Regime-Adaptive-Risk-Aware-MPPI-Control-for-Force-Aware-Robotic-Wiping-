"""Shared frozen evaluation metrics and file helpers."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def summarize(rows: list[dict], *, target: float, episode_return: float) -> dict:
    force = np.asarray([row["normal_force_n"] for row in rows], dtype=np.float64)
    contact = force >= 3.0
    deviation = np.asarray([row["absolute_actor_deviation"] for row in rows], dtype=np.float64)
    planning = np.asarray([row["planning_ms"] for row in rows], dtype=np.float64)
    return {
        "target_force_n": target,
        "native_samples": len(rows),
        "success": bool(rows[-1]["success"]),
        "return": episode_return,
        "progress": float(rows[-1]["progress"]),
        "completed_dose_bins": int(rows[-1]["completed_dose_bins"]),
        "minimum_bin_dose": int(rows[-1]["minimum_bin_dose"]),
        "force_limit_violation_samples": int(np.sum(force > 15.0)),
        "peak_force_n": float(force.max()),
        "contact_fraction": float(contact.mean()),
        "contact_mean_force_n": float(force[contact].mean()) if contact.any() else None,
        "contact_rmse_n": float(np.sqrt(np.mean((force[contact] - target) ** 2))) if contact.any() else None,
        "maximum_actor_deviation": deviation.max(axis=0).tolist(),
        "mean_planning_ms": float(planning.mean()),
        "median_planning_ms": float(np.median(planning)),
        "maximum_planning_ms": float(planning.max()),
    }
