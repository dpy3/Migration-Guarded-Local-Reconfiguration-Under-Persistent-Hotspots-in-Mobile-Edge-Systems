"""Merge versioned Paper 2 MATLAB shards without accepting partial cells."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


FORMAL_SHARD = re.compile(r"^r\d{2}_\d{2}$", re.IGNORECASE)


def merge_file(root: Path, name: str, keys: list[str]) -> pd.DataFrame:
    files = sorted(
        path for path in (root / "SHARDS").glob(f"*/{name}")
        if FORMAL_SHARD.fullmatch(path.parent.name)
    )
    if not files:
        raise FileNotFoundError(f"No shard files found for {name} under {root}")
    frame = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    duplicates = frame.duplicated(keys, keep=False)
    if duplicates.any():
        repeated = frame.loc[duplicates].sort_values(keys)
        if not repeated.groupby(keys, dropna=False).apply(
            lambda g: len(g.drop_duplicates()) == 1, include_groups=False
        ).all():
            raise ValueError(f"Conflicting duplicate rows in {name}")
        frame = frame.drop_duplicates(keys, keep="first")
    return frame.sort_values(keys).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--expected-runs", type=int, required=True)
    parser.add_argument("--expected-slots", type=int, required=True)
    args = parser.parse_args()
    root = args.result_dir.resolve()
    runs = merge_file(root, "Paper2_Revision_RunLevel.csv", ["Scenario", "Run", "Algorithm"])
    slots = merge_file(root, "Paper2_Revision_SlotLevel.csv", ["Scenario", "Run", "Algorithm", "Slot"])
    if len(runs) != args.expected_runs:
        raise ValueError(f"Expected {args.expected_runs} run rows, found {len(runs)}")
    if len(slots) != args.expected_slots:
        raise ValueError(f"Expected {args.expected_slots} slot rows, found {len(slots)}")
    if not (slots["AdmittedTasks"] + slots["RejectedTasks"] == slots["N"]).all():
        raise ValueError("Slot-level admitted + rejected != N")
    if not ((runs["MeanAdmittedTasks"] + runs["MeanRejectedTasks"] - runs["N"]).abs() < 1e-8).all():
        raise ValueError("Run-level mean admitted + rejected != N")
    if not (slots["ResourceViolationRate"].abs() < 1e-12).all():
        raise ValueError("Nonzero resource violation in merged data")
    runs.to_csv(root / "Paper2_Revision_RunLevel.csv", index=False)
    slots.to_csv(root / "Paper2_Revision_SlotLevel.csv", index=False)
    shards = sorted(
        path.name for path in (root / "SHARDS").iterdir()
        if path.is_dir() and FORMAL_SHARD.fullmatch(path.name)
    )
    print(f"Merged shards {shards}: {len(runs)} run rows and {len(slots)} slot rows in {root}")


if __name__ == "__main__":
    main()
