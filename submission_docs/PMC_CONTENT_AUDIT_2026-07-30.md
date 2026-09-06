# PMC Content Audit

## Decision

The manuscript is content-complete for author review and technically compilable in the Elsevier elsarticle format. It is not yet externally submittable because author identity, affiliations, declarations, funding, environment metadata, and the final public repository identifier remain author-controlled fields.

## Passed Checks

- The research question remains limited to persistent local hotspots and state-preserving reconfiguration.
- The method description matches the implementation: repaired local trials above the budget are rejected, while final-repair excess is recorded rather than hidden.
- The abstract, introduction, results, and conclusion use the same bounded evidence scope.
- The primary protocol is fixed at N=180, 200 slots, 30 paired seeds, three hotspot types, and a 5% budget.
- The onset cell introduces the hotspot at slot 51 after a 50-slot feasible warm-up, and the ablation cell isolates cold-start, global-pool, and no-guard variants.
- The 1% sensitivity result is described as an infeasible boundary, not as a successful hard-cap setting.
- Runtime is described as measured solver runtime and explicitly excludes runner-level final repair.
- Table numbering and figure placement are internally consistent.
- Figure captions state sample size, scenario scope, interval meaning, and comparison boundaries.
- All 22 bibliography entries are cited in the manuscript, resolve against the bundled BibTeX file, and had DOI-title matches against Crossref on 2026-07-30.
- The 22-page LaTeX manuscript compiles with BibTeX and has no undefined references, missing figures, overfull boxes, or LaTeX errors.
- The latest rendered PDF was visually checked page by page; figures, tables, captions, equations, pseudocode, and references have no clipping or overlap.
- The revision reproducibility package includes code, frozen CSV files, analysis/validation tables, figure source data, and a 93-entry checksum manifest.

## Remaining Scientific Risks

- Evidence is from one simulator and does not establish cross-topology, cross-load, or deployment generalization.
- HIVE is an internally implemented comparator, not a published baseline or an exhaustive comparison with every recent controller family.
- The satisfaction metric and latency model are simulator-specific.
- The runtime comparison is not a deployment latency guarantee.
- The 10% setting is sensitivity evidence and must not be presented as a newly tuned primary configuration.

## Author-Controlled Blockers

See `AUTHOR_TO_COMPLETE_BEFORE_SUBMISSION.md`. These items are not scientific defects, but the submission must not be finalized until they are completed.
