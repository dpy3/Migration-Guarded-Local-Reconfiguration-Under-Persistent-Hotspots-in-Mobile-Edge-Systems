"""Validate and analyze the versioned Paper 2 revision experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent / "Results_Paper2_Revision"
SCENARIOS = ["compute_hotspot", "link_hotspot", "coupled_hotspot"]
PRIMARY = "BudgetedLNS"
EXPECTED_SLOTS = 200
EXPECTED_METHODS = {
    "FORMAL_V2": {"FullResched", "HIVE", "BigMECGreedy", "BudgetedLNS"},
    "ONSET_V2": {"FullResched", "HIVE", "BigMECGreedy", "BudgetedLNS"},
    "ABLATION_V2": {"BudgetedLNS", "LNSColdStart", "LNSGlobalPool", "LNSNoGuard"},
}


def mean_ci(values: pd.Series, seed: int) -> tuple[float, float, float]:
    """Return an arithmetic mean and deterministic run-level bootstrap CI."""
    return paired_ci(values.to_numpy(dtype=float), seed)


def paired_ci(values: np.ndarray, seed: int, n_resamples: int = 20_000) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(values, (n_resamples, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def signflip_p(values: np.ndarray, seed: int, n_samples: int = 100_000) -> float:
    values = np.asarray(values, dtype=float)
    observed = abs(values.mean())
    rng = np.random.default_rng(seed)
    batch = 5_000
    exceed = 0
    done = 0
    while done < n_samples:
        size = min(batch, n_samples - done)
        signs = rng.choice((-1.0, 1.0), (size, len(values)))
        exceed += int((np.abs((signs * values).mean(axis=1)) >= observed - 1e-15).sum())
        done += size
    return (exceed + 1) / (n_samples + 1)


def holm(values: pd.Series) -> pd.Series:
    order = np.argsort(values.to_numpy())
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(values) - rank) * float(values.iloc[idx]))
        adjusted[idx] = min(running, 1.0)
    return pd.Series(adjusted, index=values.index)


def load_cell(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    runs = pd.read_csv(path / "Paper2_Revision_RunLevel.csv")
    slots = pd.read_csv(path / "Paper2_Revision_SlotLevel.csv")
    run_keys = ["Scenario", "Run", "Algorithm"]
    slot_keys = [*run_keys, "Slot"]
    if runs.duplicated(run_keys).any():
        raise ValueError(f"Duplicate run keys in {path}")
    if slots.duplicated(slot_keys).any():
        raise ValueError(f"Duplicate slot keys in {path}")
    if not (slots.AdmittedTasks + slots.RejectedTasks == slots.N).all():
        raise ValueError(f"Admission accounting failed in {path}")
    if not ((runs.MeanAdmittedTasks + runs.MeanRejectedTasks - runs.N).abs() < 1e-8).all():
        raise ValueError(f"Run admission accounting failed in {path}")
    if not (slots.ResourceViolationRate.abs() < 1e-12).all():
        raise ValueError(f"Resource feasibility failed in {path}")
    expected_slot_set = set(range(1, EXPECTED_SLOTS + 1))
    for key, group in slots.groupby(run_keys, sort=False):
        if set(group.Slot.astype(int)) != expected_slot_set:
            raise ValueError(f"Missing or unexpected slots for {key} in {path}")
        run_row = runs[
            runs.Scenario.eq(key[0]) & runs.Run.eq(key[1]) & runs.Algorithm.eq(key[2])
        ]
        if len(run_row) != 1:
            raise ValueError(f"Missing run-level row for {key} in {path}")
        admitted = group.AdmittedTasks.mean()
        rejected = group.RejectedTasks.mean()
        if abs(admitted - run_row.MeanAdmittedTasks.iloc[0]) > 1e-8:
            raise ValueError(f"Run/slot admitted-task mismatch for {key} in {path}")
        if abs(rejected - run_row.MeanRejectedTasks.iloc[0]) > 1e-8:
            raise ValueError(f"Run/slot rejected-task mismatch for {key} in {path}")
    return runs, slots


def assert_protocol(path: Path, runs: pd.DataFrame, slots: pd.DataFrame) -> None:
    cell = path.name
    methods = EXPECTED_METHODS.get(cell, {"BudgetedLNS"} if path.parent.name == "SENSITIVITY_V2" else None)
    if methods is None:
        raise ValueError(f"Unknown result cell {path}")
    expected_runs = len(SCENARIOS) * 30 * len(methods)
    expected_rows = expected_runs * EXPECTED_SLOTS
    if len(runs) != expected_runs or len(slots) != expected_rows:
        raise ValueError(
            f"Protocol size mismatch in {path}: runs={len(runs)}/{expected_runs}, "
            f"slots={len(slots)}/{expected_rows}"
        )
    if set(runs.Scenario) != set(SCENARIOS):
        raise ValueError(f"Scenario set mismatch in {path}")
    if set(runs.Algorithm) != methods:
        raise ValueError(f"Method set mismatch in {path}")
    if set(runs.Run.astype(int)) != set(range(1, 31)):
        raise ValueError(f"Run set mismatch in {path}")


def main_results_table(path: Path) -> pd.DataFrame:
    """Create the manuscript-facing table with service, churn, and admission accounting."""
    runs, slots = load_cell(path)
    assert_protocol(path, runs, slots)
    metrics = [
        "MeanSatisfaction", "TotalMigrations", "MigrationMB",
        "MeanAdmittedTasks", "MeanRejectedTasks",
    ]
    rows = []
    for si, scenario in enumerate(SCENARIOS):
        for mi, algorithm in enumerate(runs.Algorithm.drop_duplicates()):
            block = runs[runs.Scenario.eq(scenario) & runs.Algorithm.eq(algorithm)]
            row = {"Scenario": scenario, "Algorithm": algorithm, "Runs": len(block)}
            for xi, metric in enumerate(metrics):
                mean, low, high = mean_ci(block[metric], 2026076000 + 100 * si + 10 * mi + xi)
                row[f"{metric}Mean"] = mean
                row[f"{metric}CI95Low"] = low
                row[f"{metric}CI95High"] = high
            rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(path / "Paper2_Main_Results_Table.csv", index=False)
    return result


def paired_blocks(frame: pd.DataFrame, reference: str, metric: str) -> list[tuple[str, str, np.ndarray, pd.DataFrame, pd.DataFrame]]:
    blocks = []
    for scenario in SCENARIOS:
        block = frame[frame.Scenario.eq(scenario)]
        primary = block[block.Algorithm.eq(PRIMARY)].sort_values("Run")
        other = block[block.Algorithm.eq(reference)].sort_values("Run")
        if not primary.Run.reset_index(drop=True).equals(other.Run.reset_index(drop=True)):
            raise ValueError(f"Unpaired {scenario} {reference}")
        blocks.append((scenario, reference, primary[metric].to_numpy() - other[metric].to_numpy(), primary, other))
    return blocks


def main_analysis(path: Path, mode: str, satisfaction_metric: str, migration_metric: str, mb_metric: str) -> pd.DataFrame:
    runs, slots = load_cell(path)
    assert_protocol(path, runs, slots)
    baselines = [a for a in runs.Algorithm.unique() if a != PRIMARY]
    rows = []
    for baseline_index, baseline in enumerate(baselines):
        for scenario_index, (scenario, _, sat_diff, primary, other) in enumerate(
            paired_blocks(runs, baseline, satisfaction_metric)
        ):
            seed = 2026073100 + 100 * baseline_index + scenario_index
            mean, low, high = paired_ci(sat_diff, seed)
            migration_delta = other[migration_metric].to_numpy() - primary[migration_metric].to_numpy()
            mb_delta = other[mb_metric].to_numpy() - primary[mb_metric].to_numpy()
            rows.append({
                "Mode": mode, "Scenario": scenario, "Baseline": baseline, "Pairs": len(primary),
                "MeanSatisfactionPrimary": primary[satisfaction_metric].mean(),
                "MeanSatisfactionBaseline": other[satisfaction_metric].mean(),
                "PrimaryMinusBaselineSatPP": mean, "SatCI95Low": low, "SatCI95High": high,
                "PSignFlipSatisfaction": signflip_p(sat_diff, seed + 10_000),
                "MeanMigrationsPrimary": primary[migration_metric].mean(),
                "MeanMigrationsBaseline": other[migration_metric].mean(),
                "MigrationReductionPct": 100 * migration_delta.sum() / max(other[migration_metric].sum(), 1),
                "PSignFlipMigration": signflip_p(migration_delta, seed + 20_000),
                "MeanMigrationMBPrimary": primary[mb_metric].mean(),
                "MeanMigrationMBBaseline": other[mb_metric].mean(),
                "MigrationMBReductionPct": 100 * mb_delta.sum() / max(other[mb_metric].sum(), 1e-12),
                "MeanAdmittedPrimary": primary.MeanAdmittedTasks.mean(),
                "MeanRejectedPrimary": primary.MeanRejectedTasks.mean(),
                "MeanAdmittedBaseline": other.MeanAdmittedTasks.mean(),
                "MeanRejectedBaseline": other.MeanRejectedTasks.mean(),
                "MeanRepairOverridesPrimary": primary.RepairOverrides.mean(),
                "MeanRuntimePrimary": primary.RuntimeSec.mean(),
                "MeanRuntimeBaseline": other.RuntimeSec.mean(),
            })
    result = pd.DataFrame(rows)
    result["PHolmSatisfaction"] = result.groupby("Baseline", group_keys=False)[
        "PSignFlipSatisfaction"
    ].transform(holm)
    result["PHolmMigration"] = result.groupby("Baseline", group_keys=False)[
        "PSignFlipMigration"
    ].transform(holm)
    result.to_csv(path / f"Paper2_{mode}_Analysis.csv", index=False)
    return result


def ablation_analysis(path: Path) -> pd.DataFrame:
    runs, slots = load_cell(path)
    assert_protocol(path, runs, slots)
    variants = [a for a in runs.Algorithm.unique() if a != PRIMARY]
    rows = []
    for vi, variant in enumerate(variants):
        for si, scenario in enumerate(SCENARIOS):
            block = runs[runs.Scenario.eq(scenario)]
            full = block[block.Algorithm.eq(PRIMARY)].sort_values("Run")
            ablated = block[block.Algorithm.eq(variant)].sort_values("Run")
            diff = full.PostOnsetMeanSatisfaction.to_numpy() - ablated.PostOnsetMeanSatisfaction.to_numpy()
            mean, low, high = paired_ci(diff, 2026073500 + 100 * vi + si)
            rows.append({"Scenario": scenario, "Variant": variant, "Pairs": len(full),
                "FullMinusVariantPostSatPP": mean, "SatCI95Low": low, "SatCI95High": high,
                "PSignFlipSatisfaction": signflip_p(diff, 2026074500 + 100 * vi + si),
                "MeanPostMigrationsFull": full.PostOnsetMigrations.mean(),
                "MeanPostMigrationsVariant": ablated.PostOnsetMigrations.mean(),
                "MeanRepairOverridesFull": full.RepairOverrides.mean(),
                "MeanRepairOverridesVariant": ablated.RepairOverrides.mean()})
    result = pd.DataFrame(rows)
    result["PHolmSatisfaction"] = holm(result.PSignFlipSatisfaction)
    result.to_csv(path / "Paper2_Ablation_Analysis.csv", index=False)
    return result


def sensitivity_analysis(root: Path) -> pd.DataFrame:
    frames = []
    for budget_dir in sorted(root.glob("B*")):
        runs, slots = load_cell(budget_dir)
        assert_protocol(budget_dir, runs, slots)
        frames.append(runs)
    data = pd.concat(frames, ignore_index=True)
    result = data.groupby(["Scenario", "BudgetFraction"], as_index=False).agg(
        Pairs=("Run", "count"), MeanSatisfaction=("MeanSatisfaction", "mean"),
        MeanMigrations=("TotalMigrations", "mean"), MeanMigrationMB=("MigrationMB", "mean"),
        MeanAdmittedTasks=("MeanAdmittedTasks", "mean"), MeanRejectedTasks=("MeanRejectedTasks", "mean"),
        MeanRepairOverrides=("RepairOverrides", "mean"), MaxBudgetExcess=("MaxBudgetExcess", "max"))
    result.to_csv(root / "Paper2_Sensitivity_Analysis.csv", index=False)
    return result


def validation_table() -> pd.DataFrame:
    rows = []
    for path in [ROOT / "FORMAL_V2", ROOT / "ONSET_V2", ROOT / "ABLATION_V2", *sorted((ROOT / "SENSITIVITY_V2").glob("B*"))]:
        runs, slots = load_cell(path)
        assert_protocol(path, runs, slots)
        for row in runs.itertuples(index=False):
            block = slots[(slots.Scenario == row.Scenario) & (slots.Run == row.Run) & (slots.Algorithm == row.Algorithm)]
            deviations = block.AdmittedTasks + block.RejectedTasks - block.N
            expected = set(range(1, int(row.Slots) + 1))
            observed = set(block.Slot.astype(int))
            passed = (
                len(block) == int(row.Slots)
                and observed == expected
                and bool((deviations == 0).all())
                and abs(row.MeanAdmittedTasks + row.MeanRejectedTasks - row.N) < 1e-8
                and bool((block.ResourceViolationRate.abs() < 1e-12).all())
            )
            rows.append({"ResultCell": str(path.relative_to(ROOT)), "Scenario": row.Scenario,
                "Run": row.Run, "Algorithm": row.Algorithm, "N": row.N,
                "MeanAdmittedTasks": row.MeanAdmittedTasks, "MeanRejectedTasks": row.MeanRejectedTasks,
                "AdmissionDeviation": row.MeanAdmittedTasks + row.MeanRejectedTasks - row.N,
                "SlotRows": len(block), "ExpectedSlotRows": int(row.Slots),
                "MaxAbsoluteSlotAdmissionDeviation": float(deviations.abs().max()),
                "ContiguousSlots": observed == expected,
                "ResourceFeasibleAllSlots": bool((block.ResourceViolationRate.abs() < 1e-12).all()),
                "Pass": passed})
    result = pd.DataFrame(rows)
    if len(result) != 1350:
        raise ValueError(f"Expected 1350 validated method-runs, found {len(result)}")
    result.to_csv(ROOT / "Paper2_Admission_Validation.csv", index=False)
    return result


def main() -> None:
    global ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    ROOT = args.root.resolve()
    main_results_table(ROOT / "FORMAL_V2")
    main_analysis(ROOT / "FORMAL_V2", "Formal", "MeanSatisfaction", "TotalMigrations", "MigrationMB")
    main_analysis(ROOT / "ONSET_V2", "Onset", "PostOnsetMeanSatisfaction", "PostOnsetMigrations", "PostOnsetMigrationMB")
    ablation_analysis(ROOT / "ABLATION_V2")
    sensitivity_analysis(ROOT / "SENSITIVITY_V2")
    validation = validation_table()
    if not validation.Pass.all() or not (validation.AdmissionDeviation.abs() < 1e-8).all():
        raise ValueError("Admission validation failed")
    print(f"Paper 2 analysis complete: {len(validation)} run cells validated")


if __name__ == "__main__":
    main()
