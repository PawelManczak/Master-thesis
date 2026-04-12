# Dimensional vs Discrete Emotional Model — Comparison

## Parameters

| Parameter | Value |
|-----------|-------|
| minsup | 0.5 |
| minconf | 0.5 |
| maxgap | 5s |
| max_pattern_size | 2 |

## Per-Dataset Results

| Dataset | Ann. | Dim Filt | Disc Filt | Ratio |
|---------|------|----------|-----------|-------|
| CASE | self | 140 | 72 | 0.51 |
| K-emoCon | self | 82 | 18 | 0.22 |
| CEAP | self | 114 | 28 | 0.25 |
| EmoWorker_v2 | self | 108 | 35 | 0.32 |
| K-emoCon_ext | external | 115 | 46 | 0.40 |
| EMBOA | external | 80 | 36 | 0.45 |

## Cross-Dataset Comparison

| Metric | Dimensional | Discrete |
|--------|-------------|----------|
| Union (unique rules) | 445 | 193 |
| Intersection (in ALL datasets) | 0 | 0 |

## Pairwise Jaccard Similarity — Dimensional

| | CASE | K-emoCon | CEAP | EmoWorker_v2 | K-emoCon_ext | EMBOA |
|---|---|---|---|---|---|---|
| **CASE** | 1.000 | 0.062 | 0.050 | 0.246 | 0.214 | 0.078 |
| **K-emoCon** | 0.062 | 1.000 | 0.037 | 0.073 | 0.107 | 0.032 |
| **CEAP** | 0.050 | 0.037 | 1.000 | 0.083 | 0.050 | 0.078 |
| **EmoWorker_v2** | 0.246 | 0.073 | 0.083 | 1.000 | 0.205 | 0.022 |
| **K-emoCon_ext** | 0.214 | 0.107 | 0.050 | 0.205 | 1.000 | 0.016 |
| **EMBOA** | 0.078 | 0.032 | 0.078 | 0.022 | 0.016 | 1.000 |

## Pairwise Jaccard Similarity — Discrete

| | CASE | K-emoCon | CEAP | EmoWorker_v2 | K-emoCon_ext | EMBOA |
|---|---|---|---|---|---|---|
| **CASE** | 1.000 | 0.011 | 0.042 | 0.126 | 0.103 | 0.029 |
| **K-emoCon** | 0.011 | 1.000 | 0.070 | 0.000 | 0.049 | 0.000 |
| **CEAP** | 0.042 | 0.070 | 1.000 | 0.033 | 0.042 | 0.000 |
| **EmoWorker_v2** | 0.126 | 0.000 | 0.033 | 1.000 | 0.095 | 0.000 |
| **K-emoCon_ext** | 0.103 | 0.049 | 0.042 | 0.095 | 1.000 | 0.000 |
| **EMBOA** | 0.029 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |
