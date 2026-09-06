# Paper 2 revision reproducibility package

## Scope

This package reproduces the versioned `paper2_revision_v2_20260731`
simulator study. The fixed scope is five edge nodes, `N=180` persistent
tasks, 200 slots, 30 paired seeds, and compute, link, and coupled persistent
hotspots. It does not establish generalization across topology, population,
load, disturbance duration, simulator, or deployment.

## Compared methods

- `FullResched`: slot-wise MPC rollout followed by the common feasibility repair.
- `HIVE`: an internally implemented hierarchical comparator, not an external published baseline or bibliography item.
- `BigMECGreedy`: a code-derived, equal-priority, non-displacing adapter of
  the public BigMEC greedy contract. It is not a reproduction of the complete
  BigMEC learning system or San Francisco experiment.
- `BudgetedLNS`: the implementation identifier for the migration-guarded,
  state-preserving local reconfiguration method reported in the manuscript.

BigMEC source: <https://github.com/flbrandh/MEC-Simulator-2-BigMEC>, audited
commit `a586ae3b98119f83bb7883bd94ea95b1fbed7a8c`, GPL-3.0, software DOI
<https://doi.org/10.5281/zenodo.10810301>.

## Accounting definitions

A migration is counted only when a task changes from one nonzero edge-node
assignment to a different nonzero edge-node assignment. Admission of a
previously rejected task and rejection of a previously admitted task are not
migrations. `MigrationMB` sums the simulator's scaled DNN profile data-vector
footprint for counted migrations; it is a profile-derived volume proxy, not a
packet trace or a full container-state measurement.

Every method-run must contain slots 1--200 exactly once. Every slot must satisfy
`AdmittedTasks + RejectedTasks = N` and have zero resource-violation rate. The
analysis recomputes run-level admission and rejection means from the slot rows.

## Experiment cells

| Cell | Methods | Hotspot onset | Expected run rows | Expected slot rows |
|---|---:|---:|---:|---:|
| `FORMAL_V2` | 4 | slot 1 | 360 | 72,000 |
| `ONSET_V2` | 4 | slot 51 after 50 normal slots | 360 | 72,000 |
| `ABLATION_V2` | 4 | slot 51 after 50 normal slots | 360 | 72,000 |
| `SENSITIVITY_V2/B010` | 1 | slot 1 | 90 | 18,000 |
| `SENSITIVITY_V2/B050` | 1 | slot 1 | 90 | 18,000 |
| `SENSITIVITY_V2/B100` | 1 | slot 1 | 90 | 18,000 |

The ablation variants are cold-start initialization, a full admitted-task
candidate pool, and no local-trial guard. The primary method uses the previous
repaired state, pressure localization with an at-most-`3B` pool, and a 5% guard.
Final feasibility repair can exceed the local-trial guard; every such event and
its excess are recorded as a repair override. The method is not described as a
universal absolute-budget controller; the override events remain visible in
the audit CSV and table.

## Replay

Run MATLAB from `code/` (the solver's dependent folders are included):

```matlab
Run_Paper2_Revision('formal');
Run_Paper2_Revision('onset');
Run_Paper2_Revision('ablation');
Run_Paper2_Revision('sensitivity', 1:30, [], '', 0.01);
Run_Paper2_Revision('sensitivity', 1:30, [], '', 0.05);
Run_Paper2_Revision('sensitivity', 1:30, [], '', 0.10);
```

Sharded runs must be merged with `merge_paper2_revision.py` using the expected
row counts above. Then run:

```powershell
python code/analyze_paper2_revision.py --root data
python code/make_paper2_revision_figures.py --results data --out figures
```

The plotting backend is Python/matplotlib only. Quantitative figures are
exported as editable SVG/PDF, 600 dpi TIFF, PNG preview, and panel source CSV.

## Integrity

`CHECKSUMS_SHA256.txt` covers every packaged file except the manifest itself.
Regenerate it only after the code, CSV, analysis, and figure bundle is frozen.
