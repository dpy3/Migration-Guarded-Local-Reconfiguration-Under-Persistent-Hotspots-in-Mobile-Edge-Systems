"""Publication-grade figures for the versioned Paper 2 revision experiments.

The script reads only frozen CSV outputs and writes source-data tables plus
SVG/PDF/TIFF/PNG exports. It intentionally keeps the scientific content of the
original figures unchanged while improving the visual hierarchy for submission.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7.6,
        "axes.labelsize": 7.8,
        "axes.titlesize": 8.0,
        "xtick.labelsize": 7.1,
        "ytick.labelsize": 7.1,
        "legend.fontsize": 7.0,
        "axes.linewidth": 0.75,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "legend.frameon": False,
        "savefig.facecolor": "white",
    }
)

PAPER = Path(__file__).resolve().parent
# The packaged runner keeps frozen CSV outputs and final figures at the
# submission-package root, next to this code directory.
RESULTS = PAPER.parent / "data"
OUT = PAPER.parent / "figures"

SCENARIOS = ["compute_hotspot", "link_hotspot", "coupled_hotspot"]
S_LABEL = {
    "compute_hotspot": "Compute",
    "link_hotspot": "Link",
    "coupled_hotspot": "Coupled",
}

METHODS = ["FullResched", "HIVE", "BigMECGreedy", "BudgetedLNS"]
M_LABEL = {
    "FullResched": "FullResched",
    "HIVE": "HIVE comparator",
    "BigMECGreedy": "BigMEC adapter",
    "BudgetedLNS": "BudgetedLNS",
}
M_SHORT = {
    "FullResched": "Full",
    "HIVE": "HIVE",
    "BigMECGreedy": "BigMEC",
    "BudgetedLNS": "BudgetedLNS",
}
M_COLOR = {
    "FullResched": "#6F6F6F",
    "HIVE": "#B66055",
    "BigMECGreedy": "#5B9A78",
    "BudgetedLNS": "#0F4D92",
}
M_MARKER = {
    "FullResched": "o",
    "HIVE": "s",
    "BigMECGreedy": "^",
    "BudgetedLNS": "D",
}

VARIANTS = ["BudgetedLNS", "LNSColdStart", "LNSGlobalPool", "LNSNoGuard"]
V_LABEL = {
    "BudgetedLNS": "Full method",
    "LNSColdStart": "Cold start",
    "LNSGlobalPool": "Global pool",
    "LNSNoGuard": "No guard",
}
V_COLOR = {
    "BudgetedLNS": "#0F4D92",
    "LNSColdStart": "#9A4D8E",
    "LNSGlobalPool": "#42949E",
    "LNSNoGuard": "#B64342",
}


def save(fig: plt.Figure, name: str) -> None:
    base = OUT / name
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def source(frame: pd.DataFrame, name: str) -> None:
    frame.to_csv(OUT / f"SourceData_{name}.csv", index=False)


def panel(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=9.2,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def clean(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.grid(axis=grid_axis, color="#E8E8E8", linewidth=0.45)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", colors="#303030")


def bootstrap_ci(values: np.ndarray, seed: int, n: int = 5000) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    sample = rng.choice(values, (n, len(values)), replace=True).mean(axis=1)
    lo, hi = np.quantile(sample, [0.025, 0.975])
    return float(values.mean()), float(lo), float(hi)


def line_ci(block: pd.DataFrame, metric: str, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pivot = block.pivot(index="Run", columns="Slot", values=metric).sort_index(axis=1)
    values = pivot.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(1200, len(values)))
    means = values[idx].mean(axis=1)
    return (
        pivot.columns.to_numpy(),
        values.mean(axis=0),
        np.quantile(means, 0.025, axis=0),
        np.quantile(means, 0.975, axis=0),
    )


def mean_ci_points(ax: plt.Axes, frame: pd.DataFrame, metric: str, log: bool = False) -> None:
    rng = np.random.default_rng(20260802)
    for i, method in enumerate(METHODS):
        vals = frame.loc[frame.Algorithm.eq(method), metric].to_numpy(dtype=float)
        mean, lo, hi = bootstrap_ci(vals, 1000 + i)
        color = M_COLOR[method]
        jitter = rng.uniform(-0.055, 0.055, size=len(vals))
        ax.scatter(
            np.full(len(vals), i) + jitter,
            vals,
            s=9 if method != "BudgetedLNS" else 12,
            color=color,
            alpha=0.18 if method != "BudgetedLNS" else 0.26,
            linewidths=0,
            zorder=2,
        )
        ax.plot([i, i], [lo, hi], color=color, lw=1.05, zorder=3)
        ax.scatter(
            i,
            mean,
            marker=M_MARKER[method],
            s=28 if method == "BudgetedLNS" else 23,
            color=color,
            edgecolor="white",
            linewidth=0.45,
            zorder=4,
        )
    ax.set_xticks(range(len(METHODS)), [M_SHORT[m] for m in METHODS])
    if log:
        ax.set_yscale("log")
    clean(ax)


def fig_primary() -> None:
    data = pd.read_csv(RESULTS / "FORMAL_V2" / "Paper2_Revision_RunLevel.csv")
    source(data, "Fig2_PrimaryRunLevel")

    metrics = [
        ("MeanSatisfaction", "Satisfaction (%)", False),
        ("TotalMigrations", "Total migrations", True),
        ("MigrationMB", "Migration-volume proxy (MB)", True),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(7.35, 6.25), sharex="col")

    for row, (metric, ylabel, log) in enumerate(metrics):
        for col, scenario in enumerate(SCENARIOS):
            ax = axes[row, col]
            mean_ci_points(ax, data[data.Scenario.eq(scenario)], metric, log=log)
            if row == 0:
                ax.set_title(S_LABEL[scenario], fontweight="bold", pad=4)
            if col == 0:
                ax.set_ylabel(ylabel)
            else:
                ax.set_yticklabels([])
            if row < 2:
                ax.set_xticklabels([])
            else:
                ax.tick_params(axis="x", labelrotation=25)
                for tick in ax.get_xticklabels():
                    tick.set_ha("right")

    axes[0, 0].set_ylim(30, 72)
    axes[0, 1].set_ylim(30, 72)
    axes[0, 2].set_ylim(30, 72)
    for ax in axes[1, :]:
        ax.set_ylim(90, 20000)
    for ax in axes[2, :]:
        ax.set_ylim(120, 30000)

    for row in range(3):
        panel(axes[row, 0], chr(ord("a") + row))

    handles = [
        Line2D(
            [0],
            [0],
            marker=M_MARKER[m],
            color="none",
            markerfacecolor=M_COLOR[m],
            markeredgecolor="white",
            markeredgewidth=0.5,
            markersize=5.5,
            label=M_LABEL[m],
        )
        for m in METHODS
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.53, 0.002),
        handletextpad=0.45,
        columnspacing=1.5,
    )
    fig.suptitle("Primary outcomes", x=0.085, ha="left", y=0.995, fontsize=8.5, fontweight="bold")
    fig.subplots_adjust(left=0.095, right=0.995, top=0.955, bottom=0.14, hspace=0.20, wspace=0.10)
    save(fig, "Fig2_Primary_Comparison")


def fig_effects() -> None:
    data = pd.read_csv(RESULTS / "FORMAL_V2" / "Paper2_Formal_Analysis.csv")
    source(data, "Fig3_PairedEffects")

    baselines = ["FullResched", "HIVE", "BigMECGreedy"]
    y_base = np.arange(len(SCENARIOS))[::-1]
    offsets = np.array([0.20, 0.00, -0.20])
    fig, axes = plt.subplots(1, 3, figsize=(7.35, 2.85))

    for bi, baseline in enumerate(baselines):
        block = data[data.Baseline.eq(baseline)].set_index("Scenario").loc[SCENARIOS]
        color = M_COLOR[baseline]
        for si, scenario in enumerate(SCENARIOS):
            y = y_base[si] + offsets[bi]
            row = block.loc[scenario]
            axes[0].plot([row.SatCI95Low, row.SatCI95High], [y, y], color=color, lw=1.0)
            axes[0].scatter(row.PrimaryMinusBaselineSatPP, y, color=color, s=22, marker=M_MARKER[baseline], zorder=3)
            axes[1].scatter(row.MigrationReductionPct, y, color=color, s=22, marker=M_MARKER[baseline], zorder=3)
            axes[2].scatter(row.MigrationMBReductionPct, y, color=color, s=22, marker=M_MARKER[baseline], zorder=3)

    axes[0].axvline(0, color="#777777", linestyle="--", lw=0.75)
    axes[1].axvline(90, color="#D0D0D0", linestyle="--", lw=0.65)
    axes[2].axvline(90, color="#D0D0D0", linestyle="--", lw=0.65)
    axes[0].set_xlim(5, 22)
    axes[1].set_xlim(88, 98)
    axes[2].set_xlim(72, 98)
    axes[0].set_xlabel("Satisfaction gain (pp)")
    axes[1].set_xlabel("Migration-count reduction (%)")
    axes[2].set_xlabel("Migration-volume reduction (%)")
    for ax in axes:
        ax.set_yticks(y_base, [S_LABEL[s] for s in SCENARIOS])
        clean(ax, grid_axis="x")
    handles = [
        Line2D([0], [0], marker=M_MARKER[m], color="none", markerfacecolor=M_COLOR[m], markersize=5.5, label=M_LABEL[m])
        for m in baselines
    ]
    for i, ax in enumerate(axes):
        panel(ax, chr(ord("a") + i))
    # Keep the conclusion title in its own row; place the shared method legend
    # below the panels so it cannot collide with the title at single-column size.
    fig.suptitle("Paired effects by hotspot", x=0.085, ha="left", y=0.99, fontsize=8.5, fontweight="bold")
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.55, 0.005),
               handletextpad=0.40, columnspacing=1.0, borderaxespad=0.0)
    fig.subplots_adjust(left=0.095, right=0.995, bottom=0.24, top=0.86, wspace=0.40)
    save(fig, "Fig3_Paired_Effects")


def fig_onset() -> None:
    data = pd.read_csv(RESULTS / "ONSET_V2" / "Paper2_Revision_SlotLevel.csv")
    source(data, "Fig4_OnsetSlotLevel")
    fig, axes = plt.subplots(2, 3, figsize=(7.35, 4.62), sharex=True)

    for col, scenario in enumerate(SCENARIOS):
        block = data[data.Scenario.eq(scenario)]
        for method in METHODS:
            part = block[block.Algorithm.eq(method)]
            color = M_COLOR[method]
            slots, mean, lo, hi = line_ci(part, "Satisfaction", 2000 + col * 10 + METHODS.index(method))
            axes[0, col].plot(slots, mean, color=color, lw=1.12 if method == "BudgetedLNS" else 0.92)
            axes[0, col].fill_between(slots, lo, hi, color=color, alpha=0.075, linewidth=0)
            slots, mean, lo, hi = line_ci(part, "Migrations", 3000 + col * 10 + METHODS.index(method))
            axes[1, col].plot(slots, mean, color=color, lw=1.05 if method == "BudgetedLNS" else 0.88)
            axes[1, col].fill_between(slots, lo, hi, color=color, alpha=0.055, linewidth=0)

        for row in range(2):
            ax = axes[row, col]
            ax.axvspan(51, 200, color="#F5F7FA", zorder=-10)
            ax.axvline(51, color="#B64342", linestyle="--", lw=0.85)
            ax.set_xlim(1, 214)
            clean(ax)
            if col:
                ax.set_yticklabels([])
        axes[0, col].set_title(S_LABEL[scenario], fontweight="bold", pad=4)
        axes[1, col].set_xlabel("Slot")

    axes[0, 0].set_ylabel("Satisfaction (%)")
    axes[1, 0].set_ylabel("Migrations / slot")
    axes[0, 0].text(53, axes[0, 0].get_ylim()[1] - 1.0, "hotspot on", color="#B64342", fontsize=6.4, va="top")
    label_y = {
        0: {
            "BudgetedLNS": 65.0,
            "FullResched": 55.0,
            "HIVE": 49.5,
            "BigMECGreedy": 41.5,
        },
        1: {
            "HIVE": 55.0,
            "FullResched": 47.0,
            "BigMECGreedy": 20.0,
            "BudgetedLNS": 2.0,
        },
    }
    for method in METHODS:
        for row, metric in [(0, "Satisfaction"), (1, "Migrations")]:
            ax = axes[row, 2]
            part = data[(data.Scenario.eq("coupled_hotspot")) & (data.Algorithm.eq(method))]
            slots, mean, _, _ = line_ci(part, metric, 4000 + row * 10 + METHODS.index(method))
            y_text = label_y[row][method]
            ax.plot([199, 203], [mean[-1], y_text], color=M_COLOR[method], lw=0.55, clip_on=False)
            ax.text(
                204,
                y_text,
                M_SHORT[method],
                color=M_COLOR[method],
                fontsize=5.8,
                va="center",
                ha="left",
                clip_on=False,
            )
    panel(axes[0, 0], "a")
    panel(axes[1, 0], "b")
    fig.suptitle("Response after hotspot onset", x=0.085, ha="left", y=0.995, fontsize=8.5, fontweight="bold")
    fig.subplots_adjust(left=0.095, right=0.955, bottom=0.13, top=0.93, hspace=0.16, wspace=0.10)
    save(fig, "Fig4_Hotspot_Onset")


def fig_ablation() -> None:
    analysis = pd.read_csv(RESULTS / "ABLATION_V2" / "Paper2_Ablation_Analysis.csv")
    raw = pd.read_csv(RESULTS / "ABLATION_V2" / "Paper2_Revision_RunLevel.csv")
    source(analysis, "Fig5_AblationEffects")
    source(raw, "Fig5_AblationRunLevel")

    fig, axes = plt.subplots(1, 3, figsize=(7.35, 2.95))
    variants = ["LNSColdStart", "LNSGlobalPool", "LNSNoGuard"]
    y_base = np.arange(len(SCENARIOS))[::-1]
    offsets = np.array([0.20, 0.00, -0.20])
    for vi, variant in enumerate(variants):
        block = analysis[analysis.Variant.eq(variant)].set_index("Scenario").loc[SCENARIOS]
        color = V_COLOR[variant]
        for si, scenario in enumerate(SCENARIOS):
            y = y_base[si] + offsets[vi]
            row = block.loc[scenario]
            axes[0].plot([row.SatCI95Low, row.SatCI95High], [y, y], color=color, lw=1.0)
            axes[0].scatter(row.FullMinusVariantPostSatPP, y, color=color, s=24, zorder=3)

    x = np.arange(len(SCENARIOS))
    width = 0.17
    for vi, variant in enumerate(VARIANTS):
        offset = (vi - 1.5) * width
        means_mig = [
            raw[(raw.Scenario.eq(s)) & (raw.Algorithm.eq(variant))].PostOnsetMigrations.mean()
            for s in SCENARIOS
        ]
        means_mb = [
            raw[(raw.Scenario.eq(s)) & (raw.Algorithm.eq(variant))].PostOnsetMigrationMB.mean()
            for s in SCENARIOS
        ]
        axes[1].bar(x + offset, means_mig, width=width, color=V_COLOR[variant], alpha=0.86, linewidth=0)
        axes[2].bar(x + offset, means_mb, width=width, color=V_COLOR[variant], alpha=0.86, linewidth=0)

    axes[0].axvline(0, color="#777777", linestyle="--", lw=0.75)
    axes[0].set_yticks(y_base, [S_LABEL[s] for s in SCENARIOS])
    axes[0].set_xlabel("Full method minus ablated variant (pp)")
    axes[0].set_ylabel("Scenario")
    axes[1].set_xticks(x, [S_LABEL[s] for s in SCENARIOS])
    axes[2].set_xticks(x, [S_LABEL[s] for s in SCENARIOS])
    axes[1].set_yscale("log")
    axes[2].set_yscale("log")
    axes[1].set_ylabel("Post-onset migrations")
    axes[2].set_ylabel("Post-onset migration-volume (MB)")
    for ax in axes:
        clean(ax, grid_axis="x" if ax is axes[0] else "y")
    handles = [Patch(facecolor=V_COLOR[v], edgecolor="none", label=V_LABEL[v]) for v in VARIANTS]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.56, 0.005),
        ncol=4,
        columnspacing=1.15,
        handlelength=1.2,
    )
    for i, ax in enumerate(axes):
        panel(ax, chr(ord("a") + i))
    fig.suptitle("Component ablation", x=0.10, ha="left", y=0.995, fontsize=8.5, fontweight="bold")
    fig.subplots_adjust(left=0.105, right=0.995, bottom=0.28, top=0.86, wspace=0.42)
    save(fig, "Fig5_Component_Ablation")


def fig_sensitivity() -> None:
    frames = []
    for path in sorted((RESULTS / "SENSITIVITY_V2").glob("B*/Paper2_Revision_RunLevel.csv")):
        frames.append(pd.read_csv(path))
    data = pd.concat(frames, ignore_index=True)
    source(data, "Fig6_GuardSensitivity")

    fig, axes = plt.subplots(1, 3, figsize=(7.35, 2.75))
    budgets = [0.01, 0.05, 0.10]
    x = np.arange(len(budgets))
    scenario_colors = {
        "compute_hotspot": "#3775BA",
        "link_hotspot": "#42949E",
        "coupled_hotspot": "#9A4D8E",
    }
    metric_specs = [
        ("MeanSatisfaction", "Satisfaction (%)"),
        ("TotalMigrations", "Total migrations"),
        ("RepairOverrides", "Repair overrides / run"),
    ]
    for ax, (metric, ylabel) in zip(axes, metric_specs):
        ax.axvspan(0.78, 1.22, color="#EFEFEF", zorder=-10)
        for scenario in SCENARIOS:
            means, lows, highs = [], [], []
            for b in budgets:
                vals = data[(data.Scenario.eq(scenario)) & (np.isclose(data.BudgetFraction, b))][metric].to_numpy(dtype=float)
                mean, lo, hi = bootstrap_ci(vals, 5000 + int(b * 1000) + SCENARIOS.index(scenario))
                means.append(mean)
                lows.append(lo)
                highs.append(hi)
            yerr = [np.array(means) - np.array(lows), np.array(highs) - np.array(means)]
            ax.errorbar(
                x,
                means,
                yerr=yerr,
                marker="o",
                ms=3.8,
                capsize=2,
                lw=1.05,
                color=scenario_colors[scenario],
                label=S_LABEL[scenario],
            )
        ax.set_xticks(x, ["1%", "5%", "10%"])
        ax.set_xlabel("Migration guard")
        ax.set_ylabel(ylabel)
        clean(ax)
    axes[0].text(1, axes[0].get_ylim()[1], "primary", ha="center", va="bottom", fontsize=6, color="#555555")
    axes[0].legend(loc="lower right")
    for i, ax in enumerate(axes):
        panel(ax, chr(ord("a") + i))
    fig.suptitle("Budget sensitivity", x=0.085, ha="left", y=0.995, fontsize=8.5, fontweight="bold")
    fig.subplots_adjust(left=0.095, right=0.995, bottom=0.20, top=0.86, wspace=0.40)
    save(fig, "Fig6_Guard_Sensitivity")


def fig_diagnostics() -> None:
    data = pd.read_csv(RESULTS / "FORMAL_V2" / "Paper2_Revision_RunLevel.csv")
    primary = data[data.Algorithm.eq("BudgetedLNS")].copy()
    source(primary, "Fig7_MechanismDiagnostics")
    metrics = [
        ("CandidateEvaluations", "Candidate evaluations / run"),
        ("AcceptedMoves", "Accepted reconfiguration moves / run"),
        ("BudgetActiveSlots", "Budget-active slots / run"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.35, 2.75))
    rng = np.random.default_rng(20260802)

    for idx, (metric, ylabel) in enumerate(metrics):
        ax = axes[idx]
        for si, scenario in enumerate(SCENARIOS):
            vals = primary.loc[primary.Scenario.eq(scenario), metric].to_numpy(dtype=float)
            mean, lo, hi = bootstrap_ci(vals, 7000 + idx * 10 + si)
            ax.scatter(
                si + rng.uniform(-0.07, 0.07, size=len(vals)),
                vals,
                s=10,
                color="#8FB1D3",
                alpha=0.34,
                linewidths=0,
            )
            ax.plot([si, si], [lo, hi], color="#0F4D92", lw=1.05, zorder=3)
            ax.scatter(si, mean, marker="D", s=25, color="#0F4D92", edgecolor="white", linewidth=0.45, zorder=4)
        ax.set_xticks(range(3), [S_LABEL[s] for s in SCENARIOS], rotation=18, ha="right")
        ax.set_ylabel(ylabel)
        clean(ax)
        panel(ax, chr(ord("a") + idx))
    fig.suptitle("Search diagnostics", x=0.09, ha="left", y=0.995, fontsize=8.5, fontweight="bold")
    fig.subplots_adjust(left=0.10, right=0.995, bottom=0.22, top=0.86, wspace=0.40)
    save(fig, "Fig7_Mechanism_Diagnostics")


def main() -> None:
    global RESULTS, OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    RESULTS = args.results.resolve()
    OUT = args.out.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    fig_primary()
    fig_effects()
    fig_onset()
    fig_ablation()
    fig_sensitivity()
    fig_diagnostics()
    print(f"Wrote publication figures and source data to {OUT}")


if __name__ == "__main__":
    main()
