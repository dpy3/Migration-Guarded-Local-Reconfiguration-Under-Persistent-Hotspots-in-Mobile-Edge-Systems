# Mobility-intensity sensitivity run manifest — v17

## Frozen question and protocol

This is a one-factor sensitivity analysis of the standard deviation of the simulator's existing independent, per-slot isotropic Gaussian task-position jitter. Each task position is sampled around its initial position and clipped to the 1,000 m by 1,000 m region. The tested standard deviations are 15 m, 35 m, and 70 m; 35 m is the existing manuscript setting.

The design fixes N=180 tasks, five nodes, the frozen task mix, three persistent-hotspot profiles (compute, link, coupled), a 5% local-search budget, 200 slots, and 30 paired seeds per profile. FullResched and BudgetedLNS use the same evaluator and common final repair. The sensitivity varies mobility intensity only.

## Formal outputs

Each sigma has 180 run-level rows and 36,000 slot-level rows: three profiles × 30 seeds × two methods, with 200 slots for every run. Across all three sigma levels this is 540 run-level rows and 108,000 slot-level rows. Profile-slot-level CSVs are also retained as derived outputs.

## Audit results

The independent audit passed for both the source experiment files and the packaged copies:

- unique run and slot keys; complete slot sequences 1–200;
- all expected sigma/profile/seed/method cells present exactly once;
- admitted plus rejected tasks equals 180 in every slot;
- zero resource-violation rate and finite required metrics;
- identical paired seed construction across methods and sigma levels;
- run-level metrics recoverable from slot-level rows, with maximum absolute difference 1.82e-11.

The paired analysis uses 20,000 bootstrap resamples for intervals and does not report p-values. Results support only the tested simulator conditions; the experiment does not vary task composition or hotspot severity independently and does not measure deployment hardware, network traffic, or physical migration latency.

Large MATLAB checkpoints and smoke-run outputs are not part of this evidence package. The formal CSV outputs, exact runner and source files, analysis, audit script, and formal console log are included.
