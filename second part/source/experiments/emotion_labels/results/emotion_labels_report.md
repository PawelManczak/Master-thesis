# Dimensional vs Discrete Emotional Model — Comparison

## Parameters

| Parameter | Value |
|-----------|-------|
| minsup | 0.3 |
| minconf | 0.5 |
| maxgap | 5s |
| max_pattern_size | 4 |

## Per-Dataset Results

| Dataset | Ann. | Dim Filt | Disc Filt | Ratio |
|---------|------|----------|-----------|-------|
| CASE | self | 2432 | 643 | 0.26 |
| K-emoCon | self | 1208 | 144 | 0.12 |
| CEAP | self | 3032 | 485 | 0.16 |
| EmoWorker_v2 | self | 929 | 171 | 0.18 |
| K-emoCon_ext | external | 1633 | 324 | 0.20 |
| EMBOA | external | 2972 | 1184 | 0.40 |

## Cross-Dataset Comparison

| Metric | Dimensional | Discrete |
|--------|-------------|----------|
| Union (unique rules) | 11261 | 2775 |
| Intersection (in ALL datasets) | 0 | 0 |

## Pairwise Jaccard Similarity — Dimensional

| | CASE | K-emoCon | CEAP | EmoWorker_v2 | K-emoCon_ext | EMBOA |
|---|---|---|---|---|---|---|
| **CASE** | 1.000 | 0.027 | 0.007 | 0.100 | 0.078 | 0.006 |
| **K-emoCon** | 0.027 | 1.000 | 0.002 | 0.054 | 0.033 | 0.002 |
| **CEAP** | 0.007 | 0.002 | 1.000 | 0.010 | 0.005 | 0.007 |
| **EmoWorker_v2** | 0.100 | 0.054 | 0.010 | 1.000 | 0.078 | 0.002 |
| **K-emoCon_ext** | 0.078 | 0.033 | 0.005 | 0.078 | 1.000 | 0.002 |
| **EMBOA** | 0.006 | 0.002 | 0.007 | 0.002 | 0.002 | 1.000 |

## Pairwise Jaccard Similarity — Discrete

| | CASE | K-emoCon | CEAP | EmoWorker_v2 | K-emoCon_ext | EMBOA |
|---|---|---|---|---|---|---|
| **CASE** | 1.000 | 0.008 | 0.014 | 0.065 | 0.071 | 0.003 |
| **K-emoCon** | 0.008 | 1.000 | 0.011 | 0.006 | 0.013 | 0.000 |
| **CEAP** | 0.014 | 0.011 | 1.000 | 0.008 | 0.006 | 0.005 |
| **EmoWorker_v2** | 0.065 | 0.006 | 0.008 | 1.000 | 0.067 | 0.001 |
| **K-emoCon_ext** | 0.071 | 0.013 | 0.006 | 0.067 | 1.000 | 0.001 |
| **EMBOA** | 0.003 | 0.000 | 0.005 | 0.001 | 0.001 | 1.000 |
