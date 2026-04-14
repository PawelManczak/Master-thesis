# Experiment 2: Universal Rules Shuffling (Self-Annotated)

**Date:** 2026-04-14 22:45:36

## Overview
Focuses on the 4 self-annotated datasets at a lowered threshold (**minsup**: 0.1). Computes the intersection of rules across all 4 datasets to find 'universal' rules, and compares this against the universal rules found in randomly shuffled data to ensure the universality is genuine.

## Dimensional Model Intersections (Across all 4 datasets)

- **Universal Original Rules discovered:** 14
- **Universal Shuffled Rules discovered:** 0
- **Original Universal Rules present in Shuffled Universal:** 0

### Original Universal Rules:
| Rule Signature | Avg Conf | Avg Sup | Avg Lift |
| --- | --- | --- | --- |
| `hrv_cvnn_medium equals hrv_rmssd_low => hrv_cvnn_medium equals hrv_rmssd_low AND hrv_cvnn_medium meets valence_medium AND hrv_rmssd_low meets valence_medium` | 0.8319 | 0.1847 | 0.8319 |
| `hrv_cvnn_medium equals hrv_cvsd_low AND hrv_cvnn_medium equals hrv_rmssd_low AND hrv_cvsd_low equals hrv_rmssd_low => hrv_cvnn_medium equals hrv_cvsd_low AND hrv_cvnn_medium equals hrv_rmssd_low AND hrv_cvnn_medium meets valence_medium AND hrv_cvsd_low equals hrv_rmssd_low AND hrv_cvsd_low meets valence_medium AND hrv_rmssd_low meets valence_medium` | 0.8161 | 0.1253 | 0.8161 |
| `hrv_cvnn_medium equals hrv_cvsd_low => hrv_cvnn_medium equals hrv_cvsd_low AND hrv_cvnn_medium meets valence_medium AND hrv_cvsd_low meets valence_medium` | 0.7744 | 0.1420 | 0.7744 |
| `hrv_hfn_low meets arousal_medium => hrv_hfn_low meets arousal_medium AND hrv_hfn_low meets valence_medium AND arousal_medium equals valence_medium` | 0.7272 | 0.3091 | 0.7272 |
| `hrv_cvnn_medium equals hrv_cvsd_low => hrv_cvnn_medium equals hrv_cvsd_low AND hrv_cvnn_medium equals hrv_rmssd_low AND hrv_cvnn_medium meets valence_medium AND hrv_cvsd_low equals hrv_rmssd_low AND hrv_cvsd_low meets valence_medium AND hrv_rmssd_low meets valence_medium` | 0.6911 | 0.1253 | 2.4056 |
| `hrv_hfn_low equals hrv_pnn50_high => hrv_hfn_low equals hrv_pnn50_high AND hrv_hfn_low meets valence_medium AND hrv_pnn50_high meets valence_medium` | 0.6798 | 0.1411 | 0.6798 |
| `hrv_hfn_low equals hrv_rmssd_low => hrv_hfn_low equals hrv_rmssd_low AND hrv_hfn_low meets arousal_medium AND hrv_rmssd_low meets arousal_medium` | 0.6771 | 0.1828 | 0.6823 |
| `hrv_cvnn_high equals hrv_sdnn_high AND hrv_cvnn_high meets arousal_medium AND hrv_sdnn_high meets arousal_medium => hrv_cvnn_high equals hrv_sdnn_high AND hrv_cvnn_high meets arousal_medium AND hrv_cvnn_high meets valence_medium AND hrv_sdnn_high meets arousal_medium AND hrv_sdnn_high meets valence_medium AND arousal_medium equals valence_medium` | 0.6598 | 0.3496 | 0.6598 |
| `hrv_cvnn_medium equals hrv_hfn_low => hrv_cvnn_medium equals hrv_hfn_low AND hrv_cvnn_medium meets valence_medium AND hrv_hfn_low meets valence_medium` | 0.6488 | 0.1426 | 0.6488 |
| `hrv_cvsd_low equals hrv_hfn_low => hrv_cvsd_low equals hrv_hfn_low AND hrv_cvsd_low meets arousal_medium AND hrv_hfn_low meets arousal_medium` | 0.6429 | 0.1583 | 0.6470 |
| `hrv_sdnn_high meets arousal_medium => hrv_sdnn_high meets arousal_medium AND hrv_sdnn_high meets valence_medium AND arousal_medium equals valence_medium` | 0.6416 | 0.3580 | 0.6416 |
| `hrv_cvnn_high meets arousal_medium => hrv_cvnn_high meets arousal_medium AND hrv_cvnn_high meets valence_medium AND arousal_medium equals valence_medium` | 0.6408 | 0.3658 | 0.6408 |
| `hrv_cvsd_low equals hrv_hfn_low AND hrv_cvsd_low equals hrv_rmssd_low AND hrv_hfn_low equals hrv_rmssd_low => hrv_cvsd_low equals hrv_hfn_low AND hrv_cvsd_low equals hrv_rmssd_low AND hrv_cvsd_low meets arousal_medium AND hrv_hfn_low equals hrv_rmssd_low AND hrv_hfn_low meets arousal_medium AND hrv_rmssd_low meets arousal_medium` | 0.6401 | 0.1500 | 0.6442 |
| `hrv_cvsd_low equals hrv_hfn_low => hrv_cvsd_low equals hrv_hfn_low AND hrv_cvsd_low equals hrv_rmssd_low AND hrv_cvsd_low meets arousal_medium AND hrv_hfn_low equals hrv_rmssd_low AND hrv_hfn_low meets arousal_medium AND hrv_rmssd_low meets arousal_medium` | 0.6179 | 0.1500 | 2.0888 |

---

## Discrete Model Intersections (Across all 4 datasets)

- **Universal Original Rules discovered:** 0
- **Universal Shuffled Rules discovered:** 0
- **Original Universal Rules present in Shuffled Universal:** 0


---

