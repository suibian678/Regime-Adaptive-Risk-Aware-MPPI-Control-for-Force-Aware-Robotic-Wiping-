"""Build the upload-ready anonymous artifact for the V19 method paper."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path


HERE = Path(__file__).resolve()
V19 = HERE.parents[1]
TEACHER_ROOT = V19.parent
PAPER = V19 / "paper"
RELEASE = V19 / "release"
ARCHIVE = RELEASE / "ForceWipe_anonymous_method_artifact_v3.zip"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def add_tree(entries: dict[str, Path], root: Path, prefix: str, patterns: tuple[str, ...]) -> None:
    for pattern in patterns:
        for path in sorted(root.rglob(pattern)):
            if path.is_file() and "__pycache__" not in path.parts:
                entries[f"{prefix}/{path.relative_to(root).as_posix()}"] = path


def collect() -> dict[str, Path]:
    entries: dict[str, Path] = {}
    add_tree(entries, V19 / "code", "code", ("*.py",))
    add_tree(entries, V19 / "config", "config", ("*.json", "*.csv"))
    add_tree(entries, V19 / "tests", "tests", ("*.py",))
    add_tree(entries, V19 / "scripts", "scripts", ("*.py",))
    add_tree(entries, RELEASE / "runtime_source", "runtime_source", ("*.py", "*.yaml"))
    for version in ("forcewipe_v16", "forcewipe_v15", "forcewipe_v14", "forcewipe_v6", "forcewipe_v4"):
        add_tree(entries, TEACHER_ROOT / version / "code", f"transitive_code/{version}", ("*.py",))

    final_run = V19 / "results" / "final" / "v19_factor_separated_m1_m4_evaluation_20260920_r1"
    add_tree(entries, final_run, "results/factor_separated_final", ("*.json", "*.jsonl", "*.csv"))
    ppo_confirmation = (
        V19 / "results" / "final"
        / "ppo_budget_sensitivity_selected_confirmation_20260920_r1"
    )
    add_tree(
        entries, ppo_confirmation, "results/ppo_selected_confirmation",
        ("*.json", "*.jsonl", "*.csv"),
    )
    deployment_gap = (
        V19 / "results" / "final" / "v19_deployment_gap_stress_20260920_r1"
    )
    add_tree(
        entries, deployment_gap, "results/deployment_gap_stress",
        ("*.json", "*.jsonl", "*.csv"),
    )
    crosssim = V19 / "results" / "calibration" / "mujoco_crosssim_calibration_r3_v1"
    add_tree(
        entries, crosssim, "results/crosssim_calibration",
        ("*.json", "*.jsonl", "*.csv", "*.xml"),
    )
    crosssim_transfer = (
        V19 / "results" / "crosssim"
        / "v19_mujoco_zero_shot_transfer_20260921_r1"
    )
    add_tree(
        entries, crosssim_transfer, "results/crosssim_zero_shot_transfer",
        ("*.json", "*.jsonl", "*.csv", "*.xml"),
    )
    entries["analysis/crosssim_r3_independent_audit.json"] = (
        V19 / "results" / "analysis" / "crosssim_r3_independent_audit.json"
    )
    entries["analysis/v19_mujoco_zero_shot_transfer_independent_audit.json"] = (
        V19 / "results" / "analysis"
        / "v19_mujoco_zero_shot_transfer_independent_audit.json"
    )

    for name in (
        "v19_factor_separated_m1_m4",
        "v19_historical_threshold_sensitivity",
        "v19_envelope_ablation",
        "v19_ppo_training_fairness",
        "ppo_budget_sensitivity",
        "deployment_gap_stress_20260920",
        "manuscript_experiment_design_audit",
    ):
        source = V19 / "results" / "analysis" / name
        add_tree(entries, source, f"analysis/{name}", ("*.json", "*.csv", "*.pdf", "*.png"))

    for seed in range(201, 206):
        source = V19 / "results" / "train" / f"v19_force_conditioned_seed{seed}_bc2_u4000"
        add_tree(
            entries,
            source,
            f"checkpoints/seed_{seed}",
            ("*.pt", "RESULT.json", "RUN_DEFINITION.json", "RUN_FILE_MANIFEST.json"),
        )

    for name in ("V19_MAIN_ROSTER_TRAINING_SUMMARY.json", "V19_BC_ABLATION_DEVELOPMENT_RESULT.json"):
        for path in (V19 / "results").rglob(name):
            entries[f"summaries/{name}"] = path
    for path in sorted(V19.glob("V19_*2026-09-20.md")):
        if path.name.startswith("V19_TEACHER_REQUIREMENTS_FINAL_RECHECK"):
            continue
        entries[f"reports/{path.name}"] = path
    for name in (
        "PPO_BUDGET_SENSITIVITY_CONFIRMATION_RESULT_2026-09-20.md",
        "CROSS_SIMULATOR_CALIBRATION_RESULT_2026-09-20.md",
        "DEPLOYMENT_GAP_STRESS_RESULT_2026-09-21.md",
        "V19_MUJOCO_ZERO_SHOT_TRANSFER_RESULT_2026-09-21.md",
        "EXPERIMENT_DESIGN_AND_NUMERIC_EVIDENCE_AUDIT_2026-09-20.md",
    ):
        entries[f"reports/{name}"] = V19 / name

    for name in (
        "root_method_draft.tex",
        "01_abstract_method_draft.tex",
        "02_introduction_method_draft.tex",
        "03_related_work_method_draft.tex",
        "04_simulation_task_method_draft.tex",
        "05_risk_aware_method_draft.tex",
        "06_experimental_design_method_draft.tex",
        "07_development_results_draft.tex",
        "07a_factor_separated_results.tex",
        "07b_provenance_numerical_audit_method_draft.tex",
        "07c_external_validity_results.tex",
        "08_discussion_method_draft.tex",
        "09_conclusion_method_draft.tex",
        "11_appendix_method_draft.tex",
        "PAPER_REFERENCES.bib",
        "ForceWipe_method_paper_draft_no_title_2026-09-21.pdf",
    ):
        entries[f"paper/{name}"] = PAPER / name
    add_tree(entries, PAPER / "figures", "paper/figures", ("*.pdf", "*.png"))
    add_tree(entries, PAPER / "tables", "paper/tables", ("*.tex", "*.csv"))

    entries["environment/RUNTIME_ENVIRONMENT_BASELINE.json"] = (
        TEACHER_ROOT
        / "forcewipe_v16"
        / "v16_protocol"
        / "v16p30_final_matched_r1"
        / "RUNTIME_ENVIRONMENT_V16P30_R1.json"
    )
    v17 = TEACHER_ROOT / "forcewipe_v17"
    for name in (
        "V17_STAGE0A_R3_REV4_ARTIFACT_RECOVERY_FINAL_RESULT_2026-09-05.md",
        "V17_STAGE0A_R3_REV4_ARTIFACT_RECOVERY_INDEPENDENT_AUDIT_2026-09-05.json",
    ):
        entries[f"numerical_audit/{name}"] = v17 / name
    entries["numerical_audit/numerical_audit_cells.tex"] = (
        TEACHER_ROOT
        / "latex_direct_tdmpc2_revision_2026-08-30"
        / "tables"
        / "numerical_audit_cells.tex"
    )

    missing = [name for name, source in entries.items() if not source.is_file()]
    if missing:
        raise FileNotFoundError(f"missing artifact inputs: {missing}")
    return dict(sorted(entries.items()))


README = """# Anonymous ForceWipe method-paper artifact

This package contains the frozen method implementation, task and factor-block
definitions, five reported checkpoints, all 270 factor-separated evaluation
traces, PPO development evidence and 225 confirmation traces, the 375
deployment-gap stress traces, policy-free cross-simulator calibration, 45
zero-shot MuJoCo stress-test traces, analysis outputs, numerical-sensitivity
records, manuscript sources, and verification tests.

`MANIFEST_SHA256.csv` lists every payload file with its byte count and SHA-256
digest. The archive is upload-ready; a stable anonymous URL is assigned by the
external deposit service.

The direct PPO comparison is conditioned on its documented frozen
hyperparameters and transition budget. The revised MuJoCo calibration retains
a held-out tangential-position mismatch; the subsequent checkpoint deployment
is therefore reported as a zero-shot cross-engine stress test rather than
matched-engine replication. The numerical, deployment-gap, and cross-engine
studies are simulation evidence rather than hardware validation.
"""


def main() -> None:
    deployment_result = (
        V19 / "results" / "final" / "v19_deployment_gap_stress_20260920_r1"
        / "RESULT.json"
    )
    deployment_audit = (
        V19 / "results" / "analysis" / "deployment_gap_stress_20260920"
        / "INDEPENDENT_AUDIT.json"
    )
    number_audit = (
        V19 / "results" / "analysis" / "manuscript_experiment_design_audit"
        / "MANUSCRIPT_EXPERIMENT_NUMBER_AUDIT.json"
    )
    transfer_result = (
        V19 / "results" / "crosssim"
        / "v19_mujoco_zero_shot_transfer_20260921_r1" / "RESULT.json"
    )
    transfer_audit = (
        V19 / "results" / "analysis"
        / "v19_mujoco_zero_shot_transfer_independent_audit.json"
    )
    if (V19 / "results" / "final" / ".v19_deployment_gap_stress_20260920_r1.creating").exists():
        raise RuntimeError("deployment-gap staging still exists")
    if not deployment_result.is_file() or not deployment_audit.is_file():
        raise FileNotFoundError("completed deployment-gap result and audit are required")
    if json.loads(deployment_result.read_text(encoding="utf-8")).get("new_evaluations") != 375:
        raise RuntimeError("deployment-gap result is incomplete")
    if json.loads(deployment_audit.read_text(encoding="utf-8")).get("status") != "PASS":
        raise RuntimeError("deployment-gap independent audit has not passed")
    if not number_audit.is_file() or json.loads(number_audit.read_text(encoding="utf-8")).get("status") != "PASS":
        raise RuntimeError("manuscript number audit has not passed")
    if not transfer_result.is_file() or not transfer_audit.is_file():
        raise FileNotFoundError("completed zero-shot cross-engine result and audit are required")
    transfer_payload = json.loads(transfer_result.read_text(encoding="utf-8"))
    if transfer_payload.get("status") != "completed" or len(transfer_payload.get("evaluations", [])) != 45:
        raise RuntimeError("zero-shot cross-engine result is incomplete")
    transfer_audit_payload = json.loads(transfer_audit.read_text(encoding="utf-8"))
    if transfer_audit_payload.get("status") != "PASS":
        raise RuntimeError("zero-shot cross-engine independent audit has not passed")
    entries = collect()
    RELEASE.mkdir(parents=True, exist_ok=True)
    rows = [(name, source.stat().st_size, sha256(source)) for name, source in entries.items()]

    manifest = io.StringIO(newline="")
    writer = csv.writer(manifest, lineterminator="\n")
    writer.writerow(("path", "bytes", "sha256"))
    writer.writerows(rows)

    with zipfile.ZipFile(ARCHIVE, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.writestr("README.md", README)
        archive.writestr("MANIFEST_SHA256.csv", manifest.getvalue())
        for name, source in entries.items():
            archive.write(source, name)

    summary = {
        "archive": str(ARCHIVE),
        "archive_bytes": ARCHIVE.stat().st_size,
        "archive_sha256": sha256(ARCHIVE),
        "payload_files": len(entries),
        "payload_bytes": sum(row[1] for row in rows),
    }
    (RELEASE / "ANONYMOUS_ARTIFACT_BUILD_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
