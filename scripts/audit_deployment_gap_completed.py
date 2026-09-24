#!/usr/bin/env python3
"""Independently recompute the completed deployment-gap evaluation artefacts."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/final/v19_deployment_gap_stress_20260920_r1"
STAGING = RUN.parent / f".{RUN.name}.creating"
PROTOCOL = ROOT / "config/DEPLOYMENT_GAP_STRESS_PROTOCOL_2026-09-20.json"
ANALYSIS = ROOT / "results/analysis/deployment_gap_stress_20260920/RESULT.json"
OUT = ROOT / "results/analysis/deployment_gap_stress_20260920/INDEPENDENT_AUDIT.json"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a: float, b: float, tolerance: float = 1e-9) -> bool:
    return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=tolerance)


def check(condition: bool, label: str, checks: list[dict]) -> None:
    checks.append({"check": label, "pass": bool(condition)})
    if not condition:
        raise AssertionError(label)


def main() -> None:
    checks: list[dict] = []
    check(RUN.is_dir(), "completed run directory exists", checks)
    check(not STAGING.exists(), "no staging directory remains", checks)
    result = load(RUN / "RESULT.json")
    definition = load(RUN / "RUN_DEFINITION.json")
    state = load(RUN / "RUN_STATE.json")
    protocol = load(PROTOCOL)
    check(result["status"] == "completed" and state["status"] == "completed",
          "result and run state are completed", checks)
    check(result["new_evaluations"] == 375
          and len(result["evaluations"]) == 375,
          "result contains 375 evaluations", checks)
    check(definition["new_evaluations"] == 375
          and len(definition["definitions"]) == 375,
          "run definition contains 375 evaluations", checks)
    check(definition["protocol_sha256"] == digest(PROTOCOL),
          "run definition binds the protocol bytes", checks)
    check(protocol["design"]["new_evaluations"] == 375,
          "protocol denominator is 375", checks)

    identities = {
        (row["condition_id"], row["block_id"], int(row["training_seed"]),
         float(row["target_force_n"]))
        for row in result["evaluations"]
    }
    definition_identities = {
        (row["condition_id"], row["block_id"], int(row["training_seed"]),
         float(row["target_force_n"]))
        for row in definition["definitions"]
    }
    check(len(identities) == 375 and identities == definition_identities,
          "evaluation identities are unique and match the run definition", checks)

    total_samples = 0
    total_task = 0
    total_tracking = 0
    total_compound = 0
    total_violations = 0
    global_peak = -math.inf
    trace_names: set[str] = set()
    planner_seed_errors = 0
    authority_errors = 0
    binding_errors = 0
    metric_errors: list[str] = []

    for evaluation in result["evaluations"]:
        trace_name = str(evaluation["trace"])
        trace = RUN / trace_name
        if trace_name in trace_names or not trace.is_file():
            raise AssertionError(f"missing or duplicate trace {trace_name}")
        trace_names.add(trace_name)
        if digest(trace) != evaluation["trace_sha256"]:
            raise AssertionError(f"trace digest mismatch {trace_name}")

        rows = []
        with trace.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                row = json.loads(line)
                rows.append(row)
                expected_seed = (
                    9_190_000_000
                    + int(evaluation["training_seed"]) * 10_000_000
                    + int(evaluation["scenario_id"]) * 2_000
                    + int(row["control_step"])
                )
                planner_seed_errors += int(int(row["planner_seed"]) != expected_seed)
                authority_errors += int(
                    not bool(row["direct_tdmpc2_action_authority"])
                    or bool(row["force_dependent_action_projection"])
                    or bool(row["rewiping_enabled"])
                    or row["tdmpc2_action"] != row["action"]
                    or row["deployment_gap"]["policy_action"] != row["action"]
                    or row["deployment_gap"]["issued_physical_action"]
                    != row["physical_action"]
                )
                binding_errors += int(
                    row["deployment_gap"]["factor"] != row["deployment_gap_factor"]
                    or not close(row["deployment_gap"]["native_force_n"],
                                 row["normal_force_n"])
                    or len(row["deployment_gap"]["native_observation"]) != 16
                    or len(row["deployment_gap"]["perturbed_current_observation"]) != 16
                    or len(row["deployment_gap"]["learner_visible_observation"]) != 16
                )
        if not rows:
            raise AssertionError(f"empty trace {trace_name}")

        forces = [float(row["normal_force_n"]) for row in rows]
        rewards = [float(row["reward"]) for row in rows]
        planning = [float(row["planning_ms"]) for row in rows]
        deviations = [row["absolute_actor_deviation"] for row in rows]
        contact = [value for value in forces if value >= 3.0]
        target = float(evaluation["target_force_n"])
        peak = max(forces)
        violation_samples = sum(value > 15.0 for value in forces)
        mean_force = statistics.fmean(contact) if contact else None
        rmse = (
            math.sqrt(statistics.fmean((value - target) ** 2 for value in contact))
            if contact else None
        )
        contact_fraction = len(contact) / len(forces)
        success = bool(rows[-1]["success"])
        tracking = bool(
            mean_force is not None and rmse is not None
            and abs(mean_force - target) / target <= 0.15
            and rmse / target <= 0.20
        )
        compound = bool(success and tracking and violation_samples == 0)
        maximum_deviation = [
            max(float(row[index]) for row in deviations) for index in range(3)
        ]

        local_ok = (
            int(evaluation["native_samples"]) == len(rows)
            and bool(evaluation["success"]) == success
            and bool(evaluation["tracking_pass"]) == tracking
            and bool(evaluation["compound_pass"]) == compound
            and int(evaluation["force_limit_violation_samples"]) == violation_samples
            and close(evaluation["peak_force_n"], peak)
            and close(evaluation["contact_fraction"], contact_fraction)
            and int(evaluation["completed_dose_bins"])
            == int(rows[-1]["completed_dose_bins"])
            and int(evaluation["minimum_bin_dose"])
            == int(rows[-1]["minimum_bin_dose"])
            and close(evaluation["progress"], rows[-1]["progress"])
            and close(evaluation["return"], sum(rewards))
            and all(close(a, b) for a, b in zip(
                evaluation["maximum_actor_deviation"], maximum_deviation
            ))
            and close(evaluation["mean_planning_ms"], statistics.fmean(planning))
            and close(evaluation["median_planning_ms"], statistics.median(planning))
            and close(evaluation["maximum_planning_ms"], max(planning))
        )
        if mean_force is None:
            local_ok = local_ok and evaluation["contact_mean_force_n"] is None
            local_ok = local_ok and evaluation["contact_rmse_n"] is None
        else:
            local_ok = local_ok and close(evaluation["contact_mean_force_n"], mean_force)
            local_ok = local_ok and close(evaluation["contact_rmse_n"], rmse)
        if not local_ok:
            metric_errors.append(trace_name)

        total_samples += len(rows)
        total_task += int(success)
        total_tracking += int(tracking)
        total_compound += int(compound)
        total_violations += violation_samples
        global_peak = max(global_peak, peak)

    on_disk_traces = {path.name for path in RUN.glob("*.jsonl")}
    check(len(trace_names) == 375 and on_disk_traces == trace_names,
          "375 unique traces are present with no unindexed trace", checks)
    check(not metric_errors, "all trace-level scientific metrics recompute", checks)
    check(planner_seed_errors == 0, "all per-step planner seeds recompute", checks)
    check(authority_errors == 0,
          "all steps preserve direct action authority without shield or rewiping", checks)
    check(binding_errors == 0,
          "all steps contain closed native, perturbed, and learner-visible evidence", checks)
    check(int(result["total_native_samples"]) == total_samples,
          "aggregate native sample count recomputes", checks)
    check(int(result["task_successes"]) == total_task
          and int(result["tracking_passes"]) == total_tracking
          and int(result["compound_passes"]) == total_compound,
          "aggregate task, tracking, and compound counts recompute", checks)
    check(int(result["force_limit_violation_samples"]) == total_violations
          and close(result["maximum_peak_force_n"], global_peak),
          "aggregate force-tail metrics recompute", checks)

    if ANALYSIS.is_file():
        analysis = load(ANALYSIS)
        check(analysis["status"] == "completed"
              and analysis["new_stress_evaluations"] == 375
              and analysis["reused_nominal_evaluations"] == 45
              and analysis["reported_evaluations"] == 420,
              "completed analysis reports 375 stress and 45 reused nominal evaluations",
              checks)

    payload = {
        "format": "forcewipe_v19_deployment_gap_independent_audit_v1",
        "status": "PASS",
        "checks_passed": sum(row["pass"] for row in checks),
        "checks_total": len(checks),
        "checks": checks,
        "recomputed": {
            "evaluations": 375,
            "native_samples": total_samples,
            "task_successes": total_task,
            "tracking_passes": total_tracking,
            "compound_passes": total_compound,
            "force_limit_violation_samples": total_violations,
            "maximum_peak_force_n": global_peak,
            "planner_seed_errors": planner_seed_errors,
            "authority_errors": authority_errors,
            "binding_errors": binding_errors,
        },
        "result_sha256": digest(RUN / "RESULT.json"),
        "run_definition_sha256": digest(RUN / "RUN_DEFINITION.json"),
        "protocol_sha256": digest(PROTOCOL),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"PASS {payload['checks_passed']}/{payload['checks_total']} {OUT}")


if __name__ == "__main__":
    main()
