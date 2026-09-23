#!/usr/bin/env python3
"""Audit the principal experimental numbers reported by the V19 manuscript.

The audit reads frozen result artifacts rather than reusing manuscript tables.
It checks the experimental Cartesian products, unique evaluation keys, primary
outcomes, confidence intervals, controlled-suite totals, the BC screen, and
the numerical-sensitivity panel.  It also checks that the corresponding
rounded literals occur in the LaTeX sources.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STUDY_ROOT = ROOT.parent
V16 = STUDY_ROOT / "forcewipe_v16"
V17 = STUDY_ROOT / "forcewipe_v17"
PAPER = ROOT / "paper"
OUT = ROOT / "results" / "analysis" / "manuscript_experiment_design_audit"


def load(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def require(condition: bool, label: str, checks: list[dict]) -> None:
    checks.append({"check": label, "pass": bool(condition)})
    if not condition:
        raise AssertionError(label)


def main() -> None:
    checks: list[dict] = []
    paper_text = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(PAPER.rglob("*.tex"))
    )

    protocol = load(ROOT / "config" / "V19_FACTOR_SEPARATED_EVALUATION_PROTOCOL_DRAFT.json")
    result = load(
        ROOT
        / "results/final/v19_factor_separated_m1_m4_evaluation_20260920_r1/RESULT.json"
    )
    analysis = load(
        ROOT
        / "results/analysis/v19_factor_separated_m1_m4/FACTOR_SEPARATED_ANALYSIS.json"
    )

    require(protocol["evaluation_count"] == 9 * 3 * 5 * 2 == 270,
            "factor-separated Cartesian product is 270", checks)
    evaluations = result["evaluations"]
    require(len(evaluations) == 270, "factor-separated result has 270 rows", checks)
    keys = {
        (row["arm"], row["training_seed"], row["block_id"], row["target_force_n"])
        for row in evaluations
    }
    require(len(keys) == 270, "factor-separated evaluation keys are unique", checks)
    require({row["training_seed"] for row in evaluations} == {201, 202, 203, 204, 205},
            "factor-separated study contains all five frozen seeds", checks)
    require({row["target_force_n"] for row in evaluations} == {5.0, 8.0, 12.0},
            "factor-separated study contains all three targets", checks)

    arms = {row["arm"]: row for row in result["arm_summaries"]}
    fixed = arms["M1_fixed_objective_fixed_compute"]
    adaptive = arms["M4_full_adaptive"]
    require((fixed["joint_passes"], adaptive["joint_passes"]) == (85, 100),
            "compound-pass counts are 85/135 and 100/135", checks)
    require((fixed["task_successes"], adaptive["task_successes"]) == (133, 132),
            "task-success counts are 133/135 and 132/135", checks)
    require((fixed["tracking_passes"], adaptive["tracking_passes"]) == (85, 100),
            "tracking-pass counts are 85/135 and 100/135", checks)
    require((fixed["force_limit_violation_samples"], adaptive["force_limit_violation_samples"]) == (2, 2),
            "both final arms contain two samples above 15 N", checks)

    intervals = {row["metric"]: row for row in analysis["intervals"]}
    joint = intervals["joint_pass"]
    require(abs(joint["difference_full_minus_fixed"] - 0.1111111111111111) < 1e-12,
            "compound-pass difference is 11.11 percentage points", checks)
    require(abs(joint["interval_low"] - 0.037037037037037035) < 1e-12
            and abs(joint["interval_high"] - 0.18518518518518517) < 1e-12,
            "compound-pass 95% interval is [3.70, 18.52] percentage points", checks)
    require(intervals["mean_relative_error"]["difference_full_minus_fixed"] < 0,
            "adaptive planner reduces mean MRE", checks)
    require(intervals["median_planning_ms"]["difference_full_minus_fixed"] < 0,
            "adaptive planner reduces median planning time", checks)
    require(intervals["peak_force_n"]["difference_full_minus_fixed"] > 0,
            "adaptive planner does not improve observed peak force", checks)

    qualification = load(
        V16
        / "results/qualification/v16p28_fully_crossed_paired_qualification_20260828_r4/RESULT.json"
    )
    calibration = load(
        V16 / "results/cal/v16p29_independent_cal_20260829_r1/RESULT.json"
    )
    require(qualification["total_evaluations"] == 75
            and calibration["total_evaluations"] == 150,
            "controlled suites contain 75 and 150 evaluations", checks)
    controlled = qualification["evaluations"] + calibration["evaluations"]
    require(len(controlled) == 225, "controlled evidence contains 225 evaluations", checks)
    require(sum(bool(row["task_gate_passed"]) for row in controlled) == 225,
            "controlled task completion is 225/225", checks)
    require(sum(bool(row["tracking_gate_passed"]) for row in controlled) == 223,
            "controlled tracking pass is 223/225", checks)

    ood = load(
        V16 / "results/final/v16p30_final_matched_simulation_20260830_r1/RESULT.json"
    )
    m0 = next(
        row for row in ood["method_target_summary"]
        if row["method"] == "M0" and row["target_force_n"] == "all"
    )
    require((m0["task_successes"], m0["tracking_passes"], m0["evaluations"])
            == (39, 37, 90),
            "full TD-MPC2 OOD result is 39/90 task and 37/90 tracking", checks)

    bc = load(ROOT / "results/dev/V19_BC_ABLATION_DEVELOPMENT_RESULT.json")
    bc_rows = {float(row["bc_coefficient"]): row for row in bc["group_summaries"]}
    require(set(bc_rows) == {0.0, 0.5, 2.0}, "BC grid is exactly {0, 0.5, 2}", checks)
    require(bc_rows[0.0]["no_contact_evaluations"] == 15,
            "BC=0 produces 15/15 no-contact development evaluations", checks)
    require(bc_rows[2.0]["task_successes"] == 15
            and bc_rows[2.0]["force_limit_violation_samples"] == 0,
            "BC=2 produces 15/15 task success and zero sampled-limit violations", checks)

    numerical = load(
        V17 / "V17_STAGE0A_R3_REV4_ARTIFACT_RECOVERY_INDEPENDENT_AUDIT_2026-09-05.json"
    )
    counts = numerical["counts"]
    require((counts["sensitivity_cells"], counts["triggered_cells"]) == (32, 11),
            "numerical panel triggers in 11/32 intervention cells", checks)
    require((counts["reduction_cells"], counts["increase_cells"]) == (10, 1),
            "numerical responses comprise ten reductions and one increase", checks)

    ppo_protocol = load(
        ROOT / "config/PPO_BUDGET_HYPERPARAMETER_STUDY_PROTOCOL_2026-09-20.json"
    )
    require(ppo_protocol["budget_design"]["transitions"]
            == [82_774, 165_548, 331_096],
            "PPO 1x/2x/4x budgets are exact geometric multiples", checks)
    require(len(ppo_protocol["hyperparameter_design"]["profiles"]) == 7
            and len(ppo_protocol["configuration_budget_endpoints"]) == 9,
            "PPO design has seven profiles and nine profile-budget endpoints", checks)
    require(9 * 5 == 45
            and ppo_protocol["development_selection"]["total_evaluations"]
            == 9 * 5 * 18 == 810,
            "PPO design contains 45 checkpoints and 810 DEV evaluations", checks)

    ppo_selection = load(
        ROOT / "results/analysis/ppo_budget_sensitivity/DEV_SELECTION_RESULT.json"
    )
    require(ppo_selection["status"] == "completed"
            and ppo_selection["evaluations"] == 810
            and len(ppo_selection["endpoints"]) == 9,
            "PPO DEV selection contains all 810 evaluations and nine endpoints", checks)
    require(ppo_selection["selected_profile_id"] == "clip_0p1"
            and ppo_selection["selected_budget_multiplier"] == 2
            and ppo_selection["final_sets_used_for_selection"] is False,
            "PPO selection chooses clip 0.1 at 2x without final-set selection", checks)
    selected_endpoint = next(
        row for row in ppo_selection["endpoints"]
        if row["profile_id"] == "clip_0p1" and row["budget_multiplier"] == 2
    )
    require((selected_endpoint["compound_passes"], selected_endpoint["task_successes"],
             selected_endpoint["tracking_passes"], selected_endpoint["safety_passes"],
             selected_endpoint["authority_passes"]) == (3, 11, 6, 79, 90),
            "selected PPO DEV endpoint is 3/11/6/79/90", checks)

    ppo_confirmation = load(
        ROOT
        / "results/final/ppo_budget_sensitivity_selected_confirmation_20260920_r1/RESULT.json"
    )
    ppo_rows = ppo_confirmation["evaluations"]
    require(ppo_confirmation["status"] == "completed" and len(ppo_rows) == 225,
            "PPO frozen confirmation contains 225 evaluations", checks)
    require(len({
        (row["suite"], row["block_id"], row["target_force_n"], row["method_seed"])
        for row in ppo_rows
    }) == 225, "PPO confirmation identities are unique", checks)
    ppo_suites = {row["suite"]: row for row in ppo_confirmation["suite_summaries"]}
    weak = ppo_suites["matched_weak_curvature"]
    factor = ppo_suites["factor_separated"]
    require((weak["evaluations"], weak["task_successes"], weak["tracking_passes"],
             weak["compound_passes"]) == (90, 0, 0, 0),
            "selected PPO is 0/90 task, tracking, and compound on weak curvature", checks)
    require((factor["evaluations"], factor["task_successes"], factor["tracking_passes"],
             factor["compound_passes"]) == (135, 19, 8, 6),
            "selected PPO is 19/135 task, 8/135 tracking, and 6/135 compound", checks)
    require(sum(row["samples_gt_15n"] for row in ppo_rows) == 64
            and abs(max(row["peak_force_n"] for row in ppo_rows) - 22.112783432006836) < 1e-12,
            "PPO confirmation has 64 samples above 15 N and 22.113 N maximum", checks)
    compound_by_seed = {
        seed: sum(
            row["task_success"] and row["tracking_pass"]
            and row["safety_pass"] and row["authority_pass"]
            for row in ppo_rows if row["method_seed"] == seed
        )
        for seed in (301, 302, 303, 304, 305)
    }
    require(compound_by_seed == {301: 0, 302: 0, 303: 0, 304: 0, 305: 6},
            "all six PPO confirmation compound passes occur in seed 305", checks)

    crosssim = load(
        ROOT / "results/calibration/mujoco_crosssim_calibration_r3_v1/RESULT.json"
    )
    require(crosssim["policy_checkpoint_used"] is False
            and crosssim["actuator_fit"]["within_tolerance"] is True,
            "cross-simulator R3 actuator calibration is policy-independent", checks)
    require(crosssim["status"] == "completed_validation_fail"
            and crosssim["held_out_validation"]["within_all_tolerances"] is False,
            "cross-simulator R3 records the residual held-out mismatch", checks)
    crosssim_validation = {
        row["motion_id"]: row for row in crosssim["contact_metrics"]
        if row["role"] == "validation"
    }
    normal_cycle = crosssim_validation["validation_multilevel_normal_cycle"]
    tangent = crosssim_validation["validation_bidirectional_tangent_release"]
    require(abs(normal_cycle["force_nrmse"] - 0.07155892293058787) < 1e-12
            and abs(tangent["force_nrmse"] - 0.09368441368881938) < 1e-12,
            "cross-simulator R3 held-out force NRMSE values are exact", checks)
    require(abs(normal_cycle["position_rmse_m"] - 0.00033727601375180055) < 1e-12
            and abs(tangent["position_rmse_m"] - 0.0008332098069093114) < 1e-12,
            "cross-simulator R3 held-out position RMSE values are exact", checks)
    require(normal_cycle["within_all_tolerances"] is True
            and tangent["within_all_tolerances"] is False,
            "R3 passes the normal cycle but retains tangential mismatch", checks)

    transfer = load(
        ROOT / "results/crosssim/v19_mujoco_zero_shot_transfer_20260921_r1/RESULT.json"
    )
    require(transfer["status"] == "completed" and len(transfer["evaluations"]) == 45,
            "zero-shot MuJoCo stress test completes all 45 evaluations", checks)
    transfer_rows = transfer["evaluations"]
    require(len({
        (row["training_seed"], row["block_id"], row["target_force_n"])
        for row in transfer_rows
    }) == 45, "zero-shot MuJoCo evaluation identities are unique", checks)
    mu_summary = next(row for row in transfer["engine_summaries"]
                      if row["engine"] == "MuJoCo")
    px_summary = next(row for row in transfer["engine_summaries"]
                      if row["engine"] == "PhysX")
    require((mu_summary["task_successes"], mu_summary["tracking_passes"],
             mu_summary["sampled_force_safety_passes"], mu_summary["joint_passes"])
            == (30, 30, 30, 30),
            "MuJoCo aggregate is 30/45 on all four gates", checks)
    require((px_summary["task_successes"], px_summary["tracking_passes"],
             px_summary["sampled_force_safety_passes"], px_summary["joint_passes"])
            == (42, 24, 43, 24),
            "matched PhysX aggregate is 42/24/43/24 out of 45", checks)
    for block_id, expected in (("B0", 15), ("S1", 15), ("S2", 0)):
        rows = [row for row in transfer_rows if row["block_id"] == block_id]
        require(len(rows) == 15
                and sum(bool(row["success"] and row["mean_relative_error"] <= 0.15
                             and row["target_normalized_rmse"] <= 0.20
                             and row["force_limit_violation_samples"] == 0)
                        for row in rows) == expected,
                f"MuJoCo {block_id} compound count is {expected}/15", checks)
    require(mu_summary["force_limit_violation_samples"] == 15
            and abs(mu_summary["maximum_peak_force_n"] - 20.24368426070928) < 1e-12,
            "MuJoCo stress test records 15 sampled violations and 20.244 N maximum", checks)
    transfer_audit = load(
        ROOT / "results/analysis/v19_mujoco_zero_shot_transfer_independent_audit.json"
    )
    require(transfer_audit["status"] == "PASS"
            and transfer_audit["checks_passed"] == transfer_audit["checks_total"] == 12,
            "zero-shot MuJoCo independent audit passes 12/12", checks)

    gap_protocol = load(
        ROOT / "config/DEPLOYMENT_GAP_STRESS_PROTOCOL_2026-09-20.json"
    )
    require(gap_protocol["design"]["new_evaluations"] == 375
            and gap_protocol["design"]["existing_nominal_evaluations_reused"] == 45
            and gap_protocol["design"]["reported_evaluations"] == 420,
            "deployment-gap design adds 375 and reuses 45 nominal evaluations", checks)
    binding = load(ROOT / "results/code_only/DEPLOYMENT_GAP_BINDING_AUDIT_2026-09-20.json")
    require(binding["status"] == "pass" and len(binding["checks"]) == 7
            and all(row["pass"] for row in binding["checks"]),
            "deployment-gap real SAPIEN binding passes 7/7 checks", checks)

    gap_result = load(
        ROOT / "results/final/v19_deployment_gap_stress_20260920_r1/RESULT.json"
    )
    gap_rows = gap_result["evaluations"]
    require(gap_result["status"] == "completed" and len(gap_rows) == 375,
            "deployment-gap run completes all 375 new evaluations", checks)
    require(len({
        (row["condition_id"], row["block_id"], row["training_seed"],
         row["target_force_n"])
        for row in gap_rows
    }) == 375, "deployment-gap completed identities are unique", checks)
    require(gap_result["task_successes"]
            == sum(bool(row["success"]) for row in gap_rows)
            and gap_result["tracking_passes"]
            == sum(bool(row["tracking_pass"]) for row in gap_rows)
            and gap_result["compound_passes"]
            == sum(bool(row["compound_pass"]) for row in gap_rows),
            "deployment-gap aggregate task, tracking, and compound counts close",
            checks)
    require(gap_result["force_limit_violation_samples"]
            == sum(int(row["force_limit_violation_samples"]) for row in gap_rows)
            and abs(gap_result["maximum_peak_force_n"]
                    - max(float(row["peak_force_n"]) for row in gap_rows)) < 1e-12,
            "deployment-gap aggregate force-tail metrics close", checks)

    gap_analysis = load(
        ROOT / "results/analysis/deployment_gap_stress_20260920/RESULT.json"
    )
    require(gap_analysis["status"] == "completed"
            and gap_analysis["new_stress_evaluations"] == 375
            and gap_analysis["reused_nominal_evaluations"] == 45
            and gap_analysis["reported_evaluations"] == 420,
            "deployment-gap analysis reports 375 new and 45 reused evaluations",
            checks)
    require(len(gap_analysis["condition_summaries"]) == 28
            and len(gap_analysis["paired_intervals"]) == 250,
            "deployment-gap condition and interval panels are complete", checks)
    gap_audit = load(
        ROOT
        / "results/analysis/deployment_gap_stress_20260920/INDEPENDENT_AUDIT.json"
    )
    require(gap_audit["status"] == "PASS"
            and gap_audit["checks_passed"] == gap_audit["checks_total"]
            and gap_audit["recomputed"]["evaluations"] == 375,
            "deployment-gap independent trace audit passes in full", checks)
    require("\\input{tables/deployment_gap_endpoints}" in paper_text,
            "manuscript includes the audited deployment-gap endpoint table", checks)

    expected_literals = [
        "225/225", "223/225", "39/90", "37/90", "85", "100",
        "11.11", "3.70", "18.52", "11/32", "82,774", "4,000",
        "1600", "3400", "0.3", "0.7", "1.2", "15\\%", "20\\%",
        "165,548", "3/90", "11/90", "0/90", "19/135", "6/135",
        "7.16", "9.37", "0.337", "0.833", "30/30", "0/15",
        "20.24",
        "375+45", "420",
    ]
    for literal in expected_literals:
        require(literal in paper_text, f"manuscript contains audited literal {literal}", checks)

    payload = {
        "format": "forcewipe_v19_manuscript_experiment_number_audit_v1",
        "status": "PASS",
        "checks_passed": sum(row["pass"] for row in checks),
        "checks_total": len(checks),
        "checks": checks,
        "scope": (
            "Core frozen design and result claims currently present in the manuscript. "
            "It additionally validates the completed PPO budget and sensitivity "
            "outcomes, the R3 cross-simulator calibration, the completed zero-shot "
            "MuJoCo stress test, and the deployment-gap panel with independent audits."
        ),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / "MANUSCRIPT_EXPERIMENT_NUMBER_AUDIT.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"PASS {payload['checks_passed']}/{payload['checks_total']} {out_path}")


if __name__ == "__main__":
    main()
