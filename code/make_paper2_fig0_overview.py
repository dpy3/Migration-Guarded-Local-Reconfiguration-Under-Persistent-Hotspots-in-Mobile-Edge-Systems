"""Redraw the Paper 2 method overview with an explicit budget-guard contract."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
        "font.size": 7.5,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.linewidth": 0.75,
        "savefig.facecolor": "white",
    }
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"


def box(ax, x, y, w, h, title, body, face, edge="#4D4D4D"):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor=face, edgecolor=edge, linewidth=0.8,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h * 0.67, title, ha="center", va="center",
            fontsize=7.5, fontweight="bold", color="#222222")
    ax.text(x + w / 2, y + h * 0.32, body, ha="center", va="center",
            fontsize=6.2, linespacing=1.24, color="#333333")


def arrow(ax, start, end, color="#52606D", ls="-"):
    ax.add_patch(
        FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=8.5,
            linewidth=0.8, color=color, linestyle=ls,
            connectionstyle="arc3,rad=0.0",
        )
    )


def draw() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.4, 3.02))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.02, 0.965, "a", ha="left", va="top", fontsize=9, fontweight="bold")
    ax.text(0.065, 0.965, "State-preserving local reconfiguration with a recorded migration guard",
            ha="left", va="top", fontsize=8.6, fontweight="bold")

    # Inputs and main processing lane.
    box(ax, 0.03, 0.56, 0.145, 0.23, "Previous state", "repaired placement\n$\\theta_{t-1}$", "#F2F2F2")
    box(ax, 0.03, 0.23, 0.145, 0.21, "Current state", "persistent hotspot\nmobility state", "#F8ECE9", "#B66055")
    box(ax, 0.225, 0.485, 0.145, 0.28, "Pressure localize", "rank node pressure\nselect top two nodes", "#E8F0F7", "#0F4D92")
    box(ax, 0.425, 0.485, 0.145, 0.28, "Risk-bounded pool", "assigned tasks\nretain at most $3B$", "#E8F0F7", "#0F4D92")
    box(ax, 0.625, 0.485, 0.145, 0.28, "Trial + repair", "reassign candidate\napply shared repair", "#E8F0F7", "#0F4D92")
    box(ax, 0.825, 0.485, 0.145, 0.28, "Next state", "repaired placement\n$\\theta_t$", "#E4F2E0", "#3A7D44")

    arrow(ax, (0.175, 0.675), (0.225, 0.63))
    arrow(ax, (0.175, 0.335), (0.225, 0.56))
    arrow(ax, (0.37, 0.625), (0.425, 0.625))
    arrow(ax, (0.57, 0.625), (0.625, 0.625))
    arrow(ax, (0.77, 0.625), (0.825, 0.625))

    # Explicit guard lane: this is deliberately not labelled a hard budget.
    box(ax, 0.255, 0.065, 0.49, 0.25, "Local-trial migration guard",
        "$B=\\max(1,\\lceil \\rho N \\rceil)$\nreject repaired trials with migrations $>B$\nrecord final-repair excess for audit",
        "#FFF6DD", "#A67C00")
    arrow(ax, (0.50, 0.485), (0.50, 0.315), "#A67C00", "--")
    arrow(ax, (0.70, 0.315), (0.70, 0.485), "#A67C00", "--")

    # Small footer clarifies the scientific contract without becoming a caption.
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    stem = OUT / "Fig0_BudgetedLNS_Overview_PMC"
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {stem}.[svg|pdf|tiff|png]")


if __name__ == "__main__":
    draw()
