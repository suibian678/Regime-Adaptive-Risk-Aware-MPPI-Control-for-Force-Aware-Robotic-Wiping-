#!/usr/bin/env python3
"""Generate the deployment-gap paper table and a concise result report."""

from __future__ import annotations

from collections import defaultdict
from datetime import date
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "results/analysis/deployment_gap_stress_20260920/RESULT.json"
AUDIT = ROOT / "results/analysis/deployment_gap_stress_20260920/INDEPENDENT_AUDIT.json"
TABLE = ROOT / "paper/tables/deployment_gap_endpoints.tex"
REPORT = ROOT / "DEPLOYMENT_GAP_STRESS_RESULT_2026-09-21.md"

CONDITIONS = (
    ("nominal", "Nominal"),
    ("force_noise_rms_0p035n", "Force noise 0.035~N RMS"),
    ("force_bias_m0p8n", "Force bias $-0.8$~N"),
    ("force_bias_p0p8n", "Force bias $+0.8$~N"),
    ("delay_5samples", "Delay 50~ms"),
    ("tcp_pose_error_0p1mm", "TCP error 0.10~mm"),
    ("normal_error_5deg", "Normal error 5$^{\\circ}$"),
)
BLOCKS = ("B0", "S1", "S2")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    if not ANALYSIS.is_file() or not AUDIT.is_file():
        raise SystemExit("completed deployment-gap analysis and audit are required")
    analysis = load(ANALYSIS)
    audit = load(AUDIT)
    if analysis.get("status") != "completed" or audit.get("status") != "PASS":
        raise SystemExit("deployment-gap artefacts are not complete")
    lookup = {
        (row["block_id"], row["condition_id"]): row
        for row in analysis["condition_summaries"]
    }

    table_rows = []
    report_rows = []
    for condition_id, label in CONDITIONS:
        cells = [lookup[(block, condition_id)] for block in BLOCKS]
        evaluations = sum(int(row["evaluations"]) for row in cells)
        if evaluations != 45:
            raise RuntimeError(f"{condition_id} must aggregate to 45 evaluations")
        task = sum(int(row["task_successes"]) for row in cells)
        tracking = sum(int(row["tracking_passes"]) for row in cells)
        safety = sum(int(row["safety_passes"]) for row in cells)
        compound = sum(int(row["compound_passes"]) for row in cells)
        maximum_peak = max(float(row["maximum_peak_force_n"]) for row in cells)
        mean_mre = sum(float(row["mean_mre"]) * int(row["evaluations"]) for row in cells) / 45
        table_rows.append(
            f"{label} & "
            + " & ".join(f"{int(row['compound_passes'])}/15" for row in cells)
            + f" & {task}/45 & {tracking}/45 & {safety}/45 & {maximum_peak:.3f} \\\\"
        )
        report_rows.append({
            "condition_id": condition_id,
            "label": label.replace("~", " ").replace("$", "").replace("^{\\circ}", "°"),
            "compound": compound,
            "task": task,
            "tracking": tracking,
            "safety": safety,
            "mean_mre_percent": 100 * mean_mre,
            "maximum_peak_force_n": maximum_peak,
            "block_compound": {
                row["block_id"]: int(row["compound_passes"]) for row in cells
            },
        })

    TABLE.parent.mkdir(parents=True, exist_ok=True)
    TABLE.write_text(
        "\\begin{table*}[t]\n"
        "\\centering\n"
        "\\caption{Severe deployment-gap endpoints. Compound counts are per 15 evaluations in each fixed block; task, tracking, and sampled-limit safety counts are over 45 evaluations. Peak force uses the native simulator signal.}\n"
        "\\label{tab:deployment_gap_endpoints}\n"
        "\\scriptsize\n"
        "\\begin{tabular}{lrrrrrrr}\n"
        "\\toprule\n"
        "Condition & B0 comp. & S1 comp. & S2 comp. & Task & Tracking & Safety & Max peak (N) \\\\\n"
        "\\midrule\n"
        + "\n".join(table_rows)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n",
        encoding="utf-8",
    )

    worst = min(report_rows[1:], key=lambda row: (row["compound"], -row["mean_mre_percent"]))
    audit_counts = f"{audit['checks_passed']}/{audit['checks_total']}"
    lines = [
        "# ForceWipe deployment-gap 压力评估结果",
        "",
        f"日期：{date.today().isoformat()}",
        "",
        "## 结论",
        "",
        "冻结的直接 TD-MPC2 方法完成了 375 次新增单因素压力评估，并复用 45 次 nominal 评估形成 420 次报告面板。该面板测量仿真中的 learner-visible 误差敏感性，不代表真机成功率或联合硬件误差分布。",
        "",
        f"独立 trace 复算：{audit_counts} PASS。规划随机流、原生力指标、tracking gate、action authority 和工件哈希均无 mismatch。",
        "",
        "## 最大端点汇总",
        "",
        "| 条件 | Compound | Task | Tracking | Safety | Mean MRE | Max peak |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report_rows:
        lines.append(
            f"| {row['label']} | {row['compound']}/45 | {row['task']}/45 | "
            f"{row['tracking']}/45 | {row['safety']}/45 | "
            f"{row['mean_mre_percent']:.2f}% | {row['maximum_peak_force_n']:.3f} N |"
        )
    lines.extend([
        "",
        "## 主要边界",
        "",
        f"最大端点中最弱条件为 {worst['label']}：{worst['compound']}/45 compound pass，mean MRE {worst['mean_mre_percent']:.2f}%，最大峰值 {worst['maximum_peak_force_n']:.3f} N。",
        "",
        "完整 B0 中间档位、每个固定块的 15-run 汇总及逐点 paired intervals 保存在 analysis 目录。论文只使用完成态工件中的数字，不根据该结果更改压力档位或控制器。",
        "",
        "## 工件",
        "",
        "- `results/final/v19_deployment_gap_stress_20260920_r1/RESULT.json`",
        "- `results/analysis/deployment_gap_stress_20260920/RESULT.json`",
        "- `results/analysis/deployment_gap_stress_20260920/INDEPENDENT_AUDIT.json`",
        "- `results/analysis/deployment_gap_stress_20260920/CONDITION_SUMMARY.csv`",
        "- `results/analysis/deployment_gap_stress_20260920/PAIRED_INTERVALS.csv`",
        "",
    ])
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(TABLE)
    print(REPORT)


if __name__ == "__main__":
    main()
