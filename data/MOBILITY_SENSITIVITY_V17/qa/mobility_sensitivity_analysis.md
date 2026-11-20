# Mobility-intensity sensitivity: paired formal analysis

Data: 30 paired run seeds per scenario and sigma; 200 slots per run; paired methods share scenario/seed inputs.
Primary contrasts: BudgetedLNS minus FullResched satisfaction (percentage points) and total migrations per 200-slot run.
Intervals: paired percentile bootstrap, 20,000 resamples, fixed seeds recorded in the analysis script. No p-values or universal-trend claim are made.

| Sigma (m) | Scenario | Δ satisfaction, pp (95% CI) | Δ migrations (95% CI) | Migration reduction from group means | Better satisfaction seeds / 30 |
|---:|---|---:|---:|---:|---:|
| 15 | compute_hotspot | 8.07 (6.74, 9.39) | -8926.43 (-9139.30, -8709.27) | 97.37% (96.97, 97.77) | 30 / 30 |
| 15 | link_hotspot | 7.39 (6.01, 8.82) | -7966.73 (-8248.13, -7706.30) | 96.98% (96.38, 97.55) | 30 / 30 |
| 15 | coupled_hotspot | 8.25 (7.11, 9.46) | -8806.83 (-9003.13, -8610.36) | 97.27% (96.78, 97.73) | 30 / 30 |
| 35 | compute_hotspot | 8.55 (7.31, 9.75) | -9397.67 (-9608.23, -9185.63) | 96.55% (96.06, 97.01) | 30 / 30 |
| 35 | link_hotspot | 7.82 (6.27, 9.37) | -8576.43 (-8863.67, -8309.53) | 96.39% (95.77, 96.99) | 28 / 30 |
| 35 | coupled_hotspot | 8.39 (7.28, 9.57) | -9283.23 (-9471.27, -9102.16) | 96.50% (95.91, 97.03) | 30 / 30 |
| 70 | compute_hotspot | 10.94 (9.70, 11.98) | -10732.67 (-10951.00, -10515.30) | 96.89% (96.33, 97.40) | 29 / 30 |
| 70 | link_hotspot | 9.05 (7.31, 10.81) | -10041.10 (-10328.18, -9770.56) | 95.87% (95.06, 96.64) | 30 / 30 |
| 70 | coupled_hotspot | 8.01 (6.84, 9.25) | -10509.30 (-10704.20, -10314.97) | 95.30% (94.76, 95.82) | 30 / 30 |

## Change relative to the 35 m setting

| Scenario | Sigma vs 35 m | Change in satisfaction advantage, pp (95% CI) | Change in migration difference (95% CI) |
|---|---:|---:|---:|
| compute_hotspot | 15 m | -0.47 (-1.46, 0.65) | 471.23 (419.90, 522.67) |
| compute_hotspot | 70 m | 2.39 (1.14, 3.74) | -1335.00 (-1414.70, -1256.87) |
| link_hotspot | 15 m | -0.43 (-1.18, 0.32) | 609.70 (536.63, 684.00) |
| link_hotspot | 70 m | 1.23 (0.33, 2.12) | -1464.67 (-1551.47, -1383.33) |
| coupled_hotspot | 15 m | -0.15 (-0.96, 0.49) | 476.40 (414.83, 532.60) |
| coupled_hotspot | 70 m | -0.39 (-0.90, 0.13) | -1226.07 (-1292.37, -1164.23) |

## Scope note

These are simulator-level results for the tested task mix, five-node environment, three hotspot scenarios, and independent per-slot Gaussian location jitter. The analysis tests mobility intensity only; it does not test task-composition or hotspot-severity sensitivity and does not establish behavior under physical user trajectories.
