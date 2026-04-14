# Experiment: Shuffling Baseline (Random Permutation)

**Date:** 2026-04-14 09:21:44

## Overview
To validate that the discovered association rules capture genuine physiological-emotional mechanisms rather than artifacts of event frequencies, this experiment compares the rule yield of the original data against a randomized baseline.

**Shuffling strategy:** For each participant, the physiological and emotional labels (`state` column) were randomly reassigned across their existing time intervals. This completely destroys real temporal relationships but preserves the exact frequency of every event for every participant.

### ARMADA Parameters
- **minsup**: 0.3 (30%)
- **minconf**: 0.5
- **maxgap**: 5s
- **max_pattern_size**: 4

## Results

| Dataset | Original Rules | Shuffled Rules | Reduction (%) |
|---------|----------------|----------------|---------------|
| CASE | 2432 | 82 | 96.6% |
| K-emoCon | 1208 | 44 | 96.4% |
| CEAP-360VR | 3032 | 498 | 83.6% |
| EmoWorker_v2 | 929 | 50 | 94.6% |
| K-emoCon (ext) | 1633 | 114 | 93.0% |
| EMBOA | 2972 | 808 | 72.8% |

## Conclusion
A drastic reduction in the number of rules discovered in the shuffled datasets indicates that the ARMADA algorithm is capturing genuine, structurally significant temporal relationships rather than random noise.

# Experiment 2: Universal Rules Shuffling (Self-Annotated)

## Overview
Focuses on the 4 self-annotated datasets at a lowered threshold (**minsup**: 0.1 / 10%). Computes the intersection of rules across all 4 datasets to find 'universal' rules, and compares this against the universal rules found in randomly shuffled data.

## Universal Intersections (Across all 4 datasets)

- **Universal Original Rules discovered:** 14
- **Universal Shuffled Rules discovered:** 0
- **Original Universal Rules present in Shuffled Universal:** 0

### Original Universal Rules:
- `hrv_cvnn_high equals hrv_sdnn_high AND hrv_cvnn_high meets arousal_medium AND hrv_sdnn_high meets arousal_medium => hrv_cvnn_high equals hrv_sdnn_high AND hrv_cvnn_high meets arousal_medium AND hrv_cvnn_high meets valence_medium AND hrv_sdnn_high meets arousal_medium AND hrv_sdnn_high meets valence_medium AND arousal_medium equals valence_medium`
- `hrv_cvnn_high meets arousal_medium => hrv_cvnn_high meets arousal_medium AND hrv_cvnn_high meets valence_medium AND arousal_medium equals valence_medium`
- `hrv_cvnn_medium equals hrv_cvsd_low => hrv_cvnn_medium equals hrv_cvsd_low AND hrv_cvnn_medium equals hrv_rmssd_low AND hrv_cvnn_medium meets valence_medium AND hrv_cvsd_low equals hrv_rmssd_low AND hrv_cvsd_low meets valence_medium AND hrv_rmssd_low meets valence_medium`
- `hrv_cvnn_medium equals hrv_cvsd_low => hrv_cvnn_medium equals hrv_cvsd_low AND hrv_cvnn_medium meets valence_medium AND hrv_cvsd_low meets valence_medium`
- `hrv_cvnn_medium equals hrv_cvsd_low AND hrv_cvnn_medium equals hrv_rmssd_low AND hrv_cvsd_low equals hrv_rmssd_low => hrv_cvnn_medium equals hrv_cvsd_low AND hrv_cvnn_medium equals hrv_rmssd_low AND hrv_cvnn_medium meets valence_medium AND hrv_cvsd_low equals hrv_rmssd_low AND hrv_cvsd_low meets valence_medium AND hrv_rmssd_low meets valence_medium`
- `hrv_cvnn_medium equals hrv_hfn_low => hrv_cvnn_medium equals hrv_hfn_low AND hrv_cvnn_medium meets valence_medium AND hrv_hfn_low meets valence_medium`
- `hrv_cvnn_medium equals hrv_rmssd_low => hrv_cvnn_medium equals hrv_rmssd_low AND hrv_cvnn_medium meets valence_medium AND hrv_rmssd_low meets valence_medium`
- `hrv_cvsd_low equals hrv_hfn_low => hrv_cvsd_low equals hrv_hfn_low AND hrv_cvsd_low equals hrv_rmssd_low AND hrv_cvsd_low meets arousal_medium AND hrv_hfn_low equals hrv_rmssd_low AND hrv_hfn_low meets arousal_medium AND hrv_rmssd_low meets arousal_medium`
- `hrv_cvsd_low equals hrv_hfn_low => hrv_cvsd_low equals hrv_hfn_low AND hrv_cvsd_low meets arousal_medium AND hrv_hfn_low meets arousal_medium`
- `hrv_cvsd_low equals hrv_hfn_low AND hrv_cvsd_low equals hrv_rmssd_low AND hrv_hfn_low equals hrv_rmssd_low => hrv_cvsd_low equals hrv_hfn_low AND hrv_cvsd_low equals hrv_rmssd_low AND hrv_cvsd_low meets arousal_medium AND hrv_hfn_low equals hrv_rmssd_low AND hrv_hfn_low meets arousal_medium AND hrv_rmssd_low meets arousal_medium`
- `hrv_hfn_low equals hrv_pnn50_high => hrv_hfn_low equals hrv_pnn50_high AND hrv_hfn_low meets valence_medium AND hrv_pnn50_high meets valence_medium`
- `hrv_hfn_low equals hrv_rmssd_low => hrv_hfn_low equals hrv_rmssd_low AND hrv_hfn_low meets arousal_medium AND hrv_rmssd_low meets arousal_medium`
- `hrv_hfn_low meets arousal_medium => hrv_hfn_low meets arousal_medium AND hrv_hfn_low meets valence_medium AND arousal_medium equals valence_medium`
- `hrv_sdnn_high meets arousal_medium => hrv_sdnn_high meets arousal_medium AND hrv_sdnn_high meets valence_medium AND arousal_medium equals valence_medium`
