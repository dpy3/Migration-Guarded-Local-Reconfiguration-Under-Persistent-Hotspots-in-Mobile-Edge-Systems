# Paper 2 final claim--evidence audit (2026-08-02)

## Quantitative claims

| Claim in manuscript | Evidence | Status and boundary |
|---|---|---|
| BudgetedLNS improves satisfaction over FullResched by 7.82--8.55 pp in the formal cells. | `Results_Paper2_Revision/FORMAL_V2/Paper2_Formal_Analysis.csv`; 30 paired runs per scenario; all paired 95% CIs are positive. | Supported only for N=180, 200 slots, three persistent-hotspot mechanisms, and the implemented simulator. |
| BudgetedLNS reduces formal migration count by 96.39--96.55% versus FullResched. | Same formal analysis CSV; Holm-adjusted sign-flip p=0.000030 in all three scenarios. | Supported within the frozen protocol; not a deployment-wide or cross-topology claim. |
| BudgetedLNS improves satisfaction over the implemented HIVE comparator by 9.65--13.82 pp and reduces migrations by 96.74--96.88%. | Same formal analysis CSV. | HIVE is explicitly an internal comparator, not a published method or a claim about every HIVE implementation. |
| BigMEC adapter comparison shows 15.49--19.92 pp satisfaction gain and 91.21--92.36% fewer migrations. | Same formal analysis CSV. | BigMEC is a code-derived greedy adapter, not a reproduction of the complete published system. |
| Hotspot onset after a 50-slot feasible warm-up preserves the advantage. | `Results_Paper2_Revision/ONSET_V2/Paper2_Onset_Analysis.csv`; 30 paired onset runs per scenario. | Post-onset evidence for the specified slot-51 intervention only. |
| State preservation and pressure localization matter. | `Results_Paper2_Revision/ABLATION_V2/Paper2_Ablation_Analysis.csv`; cold-start and global-pool effects are positive and paired. | Mechanism evidence is operational, not a universal causal decomposition. |
| Guard sensitivity exposes repair overrides. | `Results_Paper2_Revision/SENSITIVITY_V2/Paper2_Sensitivity_Analysis.csv`; 1%, 5%, and 10% cells retain override counts and maximum excess. | The method is described as a budget guard with recorded repair overrides, not a hard budget. |
| The primary guard audit is numerically reproducible. | `data/FORMAL_V2/Paper2_Revision_RunLevel.csv`; BudgetedLNS means are 0.90, 0.60, and 1.33 recorded override events per 200-slot run for compute, link, and coupled hotspots; maximum excesses are 3, 6, and 5. | Supported as an audit of the final-repair contract at $N=180$, $B=9$; not a guarantee of zero excess. |

## Accounting and validation

- `Paper2_Admission_Validation.csv` contains 1,350 validated method-runs.
- Every method-run has exactly 200 contiguous slots.
- Every slot satisfies `AdmittedTasks + RejectedTasks = N` and zero resource-violation rate.
- Admission/rejection transitions are excluded from both `TotalMigrations` and `MigrationMB`; `MigrationMB` is a profile-derived volume proxy.

## Method and citation boundaries

The implementation supports state-preserving initialization, pressure localization to
the two most pressured nodes, an at-most-`3B` candidate pool, lexicographic
acceptance, and common final feasibility repair. The manuscript does not claim
cross-topology, cross-load, cross-scale, live-deployment, universal-controller, or
causal generalization.

## Reproducibility integrity

The submission package contains the solver/dependency folders, Python audit and
plotting scripts, all formal CSV files, analysis and validation tables, figure source
data, and a SHA-256 manifest. Re-run the analysis before regenerating figures; do not
edit frozen CSV files after the checksum manifest is produced.

**Decision: pass for evidence alignment and submission assembly, with author
metadata, public repository URL/DOI, and declarations still to be completed.**

## Revision status

The manuscript now reports the exact budget equation, trial-level migration
definition, final-repair excess equation, admission/rejection accounting,
profile-derived MigrationMB semantics, implemented HIVE and code-derived
BigMEC boundaries, and a standalone formal guard-audit table.
