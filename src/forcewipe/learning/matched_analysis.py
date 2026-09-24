"""Method-neutral evaluation metrics for the V16.30 final matched study."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


CONTACT_THRESHOLD_N = 3.0
FORCE_LIMIT_N = 15.0
DESCRIPTIVE_HEADROOM_N = 14.5
SAMPLE_RATE_HZ = 100.0
MRE_LIMIT = 0.15
NRMSE_LIMIT = 0.20


class FinalAnalysisError(ValueError):
    pass


def _segments(mask: Iterable[bool]) -> list[tuple[int, int]]:
    values = list(bool(value) for value in mask)
    output = []
    start = None
    for index, active in enumerate(values + [False]):
        if active and start is None:
            start = index
        elif not active and start is not None:
            output.append((start, index)); start = None
    return output


def analyze_evaluation(rows: list[dict], *, target_force_n: float) -> dict:
    target = float(target_force_n)
    if target not in {5.0, 8.0, 12.0}:
        raise FinalAnalysisError("target must be 5, 8, or 12 N")
    if not rows:
        return {
            "native_samples": 0, "task_success": False, "tracking_pass": False,
            "safety_pass": False, "authority_pass": False,
            "scientific_failure": "empty_or_reset_failure",
            "contact_mean_force_n": None, "signed_bias_n": None,
            "mre": None, "rmse_n": None, "nrmse": None,
            "peak_force_n": None, "peak_overshoot_n": None,
            "samples_gt_14p5n": 0, "samples_gt_15n": 0,
            "contact_fraction": 0.0, "contact_loss_count": 0,
            "contact_loss_total_s": 0.0, "contact_loss_max_s": 0.0,
            "actor_samples": 0, "mppi_samples": 0,
            "actor_fraction": None, "mppi_fraction": None,
            "planning_mean_ms": None, "planning_p50_ms": None,
            "planning_p95_ms": None, "planning_p99_ms": None,
        }
    if any(row.get("scientific_failure_event") for row in rows):
        failure = next(row["scientific_failure_event"] for row in rows if row.get("scientific_failure_event"))
    else:
        failure = None
    finite_rows = [row for row in rows if row.get("normal_force_n") is not None]
    force = np.asarray([float(row["normal_force_n"]) for row in finite_rows], dtype=np.float64)
    if force.size and not np.isfinite(force).all():
        raise FinalAnalysisError("nonfinite force escaped runner sanitization")
    contact = force >= CONTACT_THRESHOLD_N
    contact_force = force[contact]
    if contact_force.size:
        mean = float(contact_force.mean())
        signed_bias = mean - target
        rmse = float(np.sqrt(np.mean((contact_force - target) ** 2)))
        mre = abs(signed_bias) / target
        nrmse = rmse / target
        tracking = bool(mre <= MRE_LIMIT and nrmse <= NRMSE_LIMIT)
        first_contact = int(np.flatnonzero(contact)[0])
        loss_mask = (~contact).copy(); loss_mask[: first_contact + 1] = False
        loss_segments = _segments(loss_mask.tolist())
    else:
        mean = signed_bias = rmse = mre = nrmse = None
        tracking = False; loss_segments = []
    peak = float(force.max()) if force.size else None
    gt14 = int(np.sum(force > DESCRIPTIVE_HEADROOM_N)) if force.size else 0
    gt15 = int(np.sum(force > FORCE_LIMIT_N)) if force.size else 0
    last = rows[-1]
    task = bool(
        failure is None and last.get("success") is True
        and int(last.get("completed_dose_bins", 0)) == 20
        and float(last.get("progress", 0.0)) >= 0.98
    )
    authority = bool(failure is None and all(
        row.get("method_action_authority", row.get("direct_tdmpc2_action_authority")) is True
        and row.get("force_dependent_action_projection") is False
        and row.get("rewiping_enabled") is False
        for row in finite_rows
    ))
    modes = [str(row.get("control_mode", row.get("tdmpc2_control_mode", "unknown"))) for row in finite_rows]
    actor = sum(mode in {"recovery_aware_ema_actor", "ema_actor_only"} for mode in modes)
    mppi = sum(mode == "recovery_aware_world_model_mppi" for mode in modes)
    planning = np.asarray([
        float(row["planning_ms"]) for row in rows
        if row.get("planning_ms") is not None and math.isfinite(float(row["planning_ms"]))
    ], dtype=np.float64)
    durations = [(end - start) / SAMPLE_RATE_HZ for start, end in loss_segments]
    return {
        "native_samples": len(rows), "task_success": task,
        "tracking_pass": bool(tracking and failure is None),
        "safety_pass": bool(gt15 == 0 and failure is None),
        "authority_pass": authority, "scientific_failure": failure,
        "contact_mean_force_n": mean, "signed_bias_n": signed_bias,
        "mre": mre, "rmse_n": rmse, "nrmse": nrmse,
        "peak_force_n": peak,
        "peak_overshoot_n": (max(peak - target, 0.0) if peak is not None else None),
        "samples_gt_14p5n": gt14, "samples_gt_15n": gt15,
        "contact_fraction": float(contact.mean()) if force.size else 0.0,
        "contact_loss_count": len(loss_segments),
        "contact_loss_total_s": float(sum(durations)),
        "contact_loss_max_s": float(max(durations, default=0.0)),
        "actor_samples": actor, "mppi_samples": mppi,
        "actor_fraction": (actor / len(modes) if modes else None),
        "mppi_fraction": (mppi / len(modes) if modes else None),
        "planning_mean_ms": float(planning.mean()) if planning.size else None,
        "planning_p50_ms": float(np.percentile(planning, 50)) if planning.size else None,
        "planning_p95_ms": float(np.percentile(planning, 95)) if planning.size else None,
        "planning_p99_ms": float(np.percentile(planning, 99)) if planning.size else None,
    }


def paired_block_checkpoint_bootstrap(
    rows: list[dict], *, method_a: str = "M0", method_b: str = "M1",
    resamples: int = 10_000, seed: int = 616_330_900,
) -> list[dict]:
    """Two-way paired bootstrap over common blocks and checkpoint seeds."""

    metrics = ("task_success", "tracking_pass", "peak_force_n", "mre", "nrmse", "planning_p50_ms")
    selected = [row for row in rows if row["method"] in {method_a, method_b}]
    blocks = sorted({row["block_id"] for row in selected})
    seeds = sorted({int(row["method_seed"]) for row in selected})
    targets = (5.0, 8.0, 12.0)
    lookup = {
        (row["method"], row["block_id"], int(row["method_seed"]), float(row["target_force_n"])): row
        for row in selected
    }
    expected = 2 * len(blocks) * len(seeds) * len(targets)
    if len(lookup) != expected:
        raise FinalAnalysisError("M0/M1 paired matrix is incomplete")
    rng = np.random.default_rng(seed)
    block_draws = rng.integers(0, len(blocks), size=(resamples, len(blocks)))
    seed_draws = rng.integers(0, len(seeds), size=(resamples, len(seeds)))
    output = []
    for target_label in ("all", "5", "8", "12"):
        active_targets = targets if target_label == "all" else (float(target_label),)
        for metric in metrics:
            matrix = np.empty((len(blocks), len(seeds)), dtype=np.float64)
            for i, block in enumerate(blocks):
                for j, checkpoint in enumerate(seeds):
                    differences = []
                    for target in active_targets:
                        a = lookup[(method_a, block, checkpoint, target)][metric]
                        b = lookup[(method_b, block, checkpoint, target)][metric]
                        if a is None or b is None:
                            differences.append(np.nan)
                        else:
                            differences.append(float(a) - float(b))
                    matrix[i, j] = float(np.mean(differences))
            if not np.isfinite(matrix).all():
                output.append({
                    "target_force_n": target_label, "metric": metric,
                    "point_estimate_m0_minus_m1": None, "ci95_low": None,
                    "ci95_high": None, "status": "not_estimable_due_to_scientific_failure",
                })
                continue
            draws = matrix[block_draws[:, :, None], seed_draws[:, None, :]].mean(axis=(1, 2))
            output.append({
                "target_force_n": target_label, "metric": metric,
                "point_estimate_m0_minus_m1": float(matrix.mean()),
                "ci95_low": float(np.percentile(draws, 2.5)),
                "ci95_high": float(np.percentile(draws, 97.5)),
                "status": "pointwise_unadjusted_two_way_paired_bootstrap",
            })
    return output
