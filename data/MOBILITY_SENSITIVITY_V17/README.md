# v17 mobility-intensity sensitivity evidence

This folder contains the formal data and audit materials supporting the 15/35/70 m mobility-intensity sensitivity added to the revised manuscript. It is a simulator-level, one-factor sensitivity study, not a deployment or hardware measurement.

## Protocol and raw outputs

- Mobility is independent per-slot isotropic Gaussian jitter around each task's initial position, clipped to the 1,000 m by 1,000 m area.
- Sigma levels are 15 m, 35 m, and 70 m; the 35 m level is the existing manuscript setting.
- Each level includes compute-hotspot, link-hotspot, and coupled-hotspot profiles, 30 paired seeds, two methods, and 200 slots.
- For each sigma, the code/Results_Paper2_Revision/MOBILITY_SENSITIVITY_FORMAL_V1/sigma_* directory contains run-level, slot-level, and profile-slot-level CSV outputs.
- Per sigma, the run-level CSV has 180 data rows and the slot-level CSV has 36,000 data rows. The three levels combine to 540 run-level rows and 108,000 slot-level rows.

## Audit and analysis

qa/audit_mobility_formal_all.py verifies keys, complete slot sequences, admission accounting, finite metrics, zero resource violations, run-to-slot reconciliation, and paired-seed consistency. It passed on the source outputs and again on the packaged copies; the largest run-to-slot reconciliation difference was 1.82e-11.

qa/analyze_mobility_sensitivity.py and its CSV/Markdown outputs contain the paired analysis used for the manuscript. It uses 20,000 paired bootstrap resamples for intervals and makes no p-value or universal-trend claim. qa/RUN_MANIFEST_V17.md records the finalized protocol and audit results; the formal console log is included for provenance.

## Excluded artifacts

Large MATLAB checkpoint files and smoke-run outputs are excluded. The package retains the final formal CSV records, the runner and analysis/audit scripts, and the formal log; the excluded files are not used as manuscript result data.

The study keeps task composition, node count, hotspot protocol, and budget fixed. It does not independently test task-composition or hotspot-severity sensitivity.
