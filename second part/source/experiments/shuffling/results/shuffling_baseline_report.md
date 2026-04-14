# Experiment: Shuffling Baseline (Random Permutation)

**Date:** 2026-04-14 13:03:56

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
| CASE (Discrete) | 643 | 38 | 94.1% |
| K-emoCon (Discrete) | 144 | 23 | 84.0% |
| CEAP-360VR (Discrete) | 485 | 91 | 81.2% |
| EmoWorker_v2 (Discrete) | 171 | 20 | 88.3% |
| K-emoCon (ext) (Discrete) | 324 | 34 | 89.5% |
| EMBOA (Discrete) | 1184 | 492 | 58.4% |

## Conclusion
A drastic reduction in the number of rules discovered in the shuffled datasets indicates that the ARMADA algorithm is capturing genuine, structurally significant temporal relationships rather than random noise.
