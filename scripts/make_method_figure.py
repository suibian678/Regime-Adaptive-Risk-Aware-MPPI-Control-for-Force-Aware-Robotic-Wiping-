#!/usr/bin/env python3
"""Create the compact two-column RA-RMPPI architecture figure."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper/figures"


def box(ax, x, y, w, h, text, *, face, edge, fontsize=8.2):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.025,rounding_size=0.09",
        linewidth=1.25, edgecolor=edge, facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)
    return patch


def arrow(ax, start, end, *, colour="#303030", label=None, label_offset=(0, 0)):
    patch = FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=10,
        linewidth=1.25, color=colour, shrinkA=3, shrinkB=3,
        connectionstyle="arc3,rad=0",
    )
    ax.add_patch(patch)
    if label:
        ax.text(
            (start[0] + end[0]) / 2 + label_offset[0],
            (start[1] + end[1]) / 2 + label_offset[1],
            label, ha="center", va="center", fontsize=7.1, color=colour,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.4),
        )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.15, 2.85))
    ax.set_xlim(0, 18); ax.set_ylim(0, 6.8); ax.axis("off")

    blue = (0.88, 0.94, 0.98); blue_edge = "#2B6F9E"
    purple = (0.94, 0.90, 0.97); purple_edge = "#7B4FA1"
    orange = (1.00, 0.94, 0.84); orange_edge = "#C87800"
    green = (0.88, 0.96, 0.90); green_edge = "#2E7D4F"
    grey = (0.94, 0.94, 0.94); grey_edge = "#666666"

    box(ax, 0.15, 4.45, 2.90, 1.10, "Causal inputs\n$F_t,\;\dot F_t,\;F^\star,\;\mathbf{s}_t$", face=grey, edge=grey_edge, fontsize=7.5)
    box(ax, 3.60, 4.45, 3.15, 1.10, "Force-conditioned\nlatent dynamics", face=blue, edge=blue_edge, fontsize=7.5)
    box(ax, 7.30, 4.45, 3.00, 1.10, "Learned next-force\nprediction", face=purple, edge=purple_edge, fontsize=7.0)
    box(ax, 10.85, 4.45, 3.15, 1.10, "Causal risk signal\n$U_{t+1}=\\widehat F_{t+1}$", face=purple, edge=purple_edge, fontsize=7.4)
    box(ax, 10.75, 2.40, 3.35, 1.10, "Regime + transient\nmemberships", face=orange, edge=orange_edge, fontsize=7.5)
    box(ax, 3.60, 0.35, 3.15, 1.10, "EMA actor prior", face=grey, edge=grey_edge, fontsize=7.6)
    box(ax, 7.30, 0.35, 3.45, 1.10, "Regime-adaptive\nrisk-aware MPPI", face=green, edge=green_edge, fontsize=7.6)
    box(ax, 14.70, 0.35, 3.10, 1.10, "Cartesian action\n$\Delta\mathbf{x}_t$", face=green, edge=green_edge, fontsize=7.6)

    # The top-row data flow uses generous gaps so arrowheads remain distinct
    # from both the box borders and the text inside the boxes.
    arrow(ax, (3.05, 5.00), (3.60, 5.00))
    arrow(ax, (6.75, 5.00), (7.30, 5.00))
    arrow(ax, (10.30, 5.00), (10.85, 5.00))
    arrow(ax, (12.43, 4.45), (12.43, 3.50))
    arrow(ax, (6.75, 0.90), (7.30, 0.90), colour=grey_edge)
    arrow(ax, (10.75, 0.90), (14.70, 0.90), colour=green_edge)

    # Route the two explanatory connections orthogonally.  Labels sit above
    # their own horizontal segments rather than masking the lines.
    ax.plot([5.18, 5.18, 8.15], [4.45, 3.55, 3.55], color=blue_edge, lw=1.25)
    arrow(ax, (8.15, 3.55), (8.15, 1.45), colour=blue_edge)
    ax.text(6.66, 3.82, "latent rollouts", ha="center", va="center", fontsize=7.0, color=blue_edge)

    ax.plot([12.43, 12.43, 9.85], [2.40, 2.00, 2.00], color=orange_edge, lw=1.25)
    arrow(ax, (9.85, 2.00), (9.85, 1.45), colour=orange_edge)
    ax.text(12.00, 1.70, "weights + budget", ha="center", va="center", fontsize=6.9, color=orange_edge)

    ax.text(5.18, 6.30, "persistent target-force context", ha="center", va="center", fontsize=7.1, color=blue_edge)
    arrow(ax, (5.18, 6.02), (5.18, 5.55), colour=blue_edge)

    fig.savefig(OUT / "ra_rmppi_architecture.pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(OUT / "ra_rmppi_architecture.png", dpi=260, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


if __name__ == "__main__":
    main()
