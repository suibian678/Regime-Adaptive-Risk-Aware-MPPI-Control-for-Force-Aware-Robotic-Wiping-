#!/usr/bin/env python3
"""Plot the frozen factor-separated M1--M4 comparison."""

from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/analysis/v19_factor_separated_m1_m4/FACTOR_LEVEL_SUMMARY.csv"
OUT = ROOT / "paper/figures"
FIXED = "M1_fixed_objective_fixed_compute"
ADAPTIVE = "M4_full_adaptive"
ORDER = [
    ("reference", "reference", "Reference"),
    ("surface", "incline_10deg", "Incline"),
    ("surface", "cylinder_r1.2m", "Cylinder"),
    ("path", "arc", "Arc"),
    ("path", "s_curve", "S-curve"),
    ("friction", "mu_0.30", r"$\mu=0.30$"),
    ("friction", "mu_0.70", r"$\mu=0.70$"),
    ("support_stiffness", "1600_N_per_m", r"$k=1600$"),
    ("support_stiffness", "3400_N_per_m", r"$k=3400$"),
]


def main() -> None:
    with SOURCE.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    lookup = {(row["arm"], row["factor"], row["level"]): row for row in rows}
    x = np.arange(len(ORDER), dtype=float)
    labels = [item[2] for item in ORDER]
    width = 0.36
    fixed_joint = [float(lookup[(FIXED, f, level)]["joint_passes"]) for f, level, _ in ORDER]
    adaptive_joint = [float(lookup[(ADAPTIVE, f, level)]["joint_passes"]) for f, level, _ in ORDER]
    fixed_mre = [100 * float(lookup[(FIXED, f, level)]["mean_mre"]) for f, level, _ in ORDER]
    adaptive_mre = [100 * float(lookup[(ADAPTIVE, f, level)]["mean_mre"]) for f, level, _ in ORDER]

    plt.rcParams.update({"font.size": 8.5, "axes.labelsize": 9.5, "legend.fontsize": 8.5})
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.75), constrained_layout=False)
    fig.subplots_adjust(left=0.08, right=0.995, bottom=0.32, top=0.77, wspace=0.30)
    colours = ("#6B7280", "#167D5A")

    fixed_bars = axes[0].bar(x - width / 2, fixed_joint, width, color=colours[0], label="Fixed")
    adaptive_bars = axes[0].bar(
        x + width / 2, adaptive_joint, width, color=colours[1],
        edgecolor="#145A44", hatch="//", linewidth=0.4, label="Adaptive",
    )
    axes[0].set_ylabel("Compound passes (of 15)")
    axes[0].set_ylim(0, 16.5)
    axes[0].set_yticks([0, 5, 10, 15])

    fixed_line, = axes[1].plot(x, fixed_mre, color=colours[0], marker="o", lw=1.6, label="Fixed")
    adaptive_line, = axes[1].plot(x, adaptive_mre, color=colours[1], marker="s", lw=1.6, label="Adaptive")
    criterion_line = axes[1].axhline(15, color="#B42318", lw=1.0, ls="--", label="15% criterion")
    axes[1].set_ylabel("Mean relative error (%)")
    axes[1].set_ylim(0, 22)

    # A single figure-level legend keeps both panels free of labels that can
    # cover bars, markers, or the 15% reference line.
    fig.legend(
        [fixed_bars[0], adaptive_bars[0], criterion_line],
        ["Fixed", "Adaptive", "15% criterion"],
        loc="upper center", bbox_to_anchor=(0.54, 0.985), ncol=3,
        frameon=False, handlelength=1.7, columnspacing=1.5,
    )

    for ax, panel in zip(axes, ("(a)", "(b)")):
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.grid(axis="y", alpha=0.25, lw=0.6)
        ax.set_axisbelow(True)
        ax.text(-0.10, 1.04, panel, transform=ax.transAxes, fontweight="bold")

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "factor_separated_results.pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(OUT / "factor_separated_results.png", dpi=260, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


if __name__ == "__main__":
    main()
