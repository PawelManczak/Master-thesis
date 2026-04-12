# Universal Rules Experiment

This experiment aims to identify rules that are universally true across all evaluated datasets.

## Experiment Parameters

- **minsup**: 0.1 (10% participants)
- **minconf**: 0.5 (50% confidence)
- **maxgap**: 5 seconds
- **max_pattern_size**: 2

## Rule Filters

- **FILTER_BVP_ONLY**: True
- **FILTER_EDA_ONLY**: True
- **FILTER_PHYSIO_CROSS**: True
- **FILTER_SINGLE_FEATURE**: True

## Dataset Processing

Evaluated on 2 datasets: K-emo_ext, EMBOA

| Dataset | Participants | Rules (Before Filters) | Rules (Filtered) |
|-------|-------------|---------|-------|
| **K-emo_ext** | 28 | 520 | 137 |
| **EMBOA** | 16 | 271 | 96 |

## Universal Rules Identification

TOTAL Universal rules found across all 2 datasets: **4**

### Universal Rules Details (sorted by avg_confidence)

| Rule | Avg Conf | Avg Sup | Min Conf | Min Sup | K-emo_ext | EMBOA |
|---|---|---|---|---|---|---|
| `(eda_std_high) => eda_std_high starts valence_medium` | **0.871** | 0.871 | 0.812 | 0.812 | c:0.93 s:0.93 n:26 | c:0.81 s:0.81 n:13 |
| `(arousal_high) => arousal_high meets valence_medium` | **0.746** | 0.696 | 0.692 | 0.643 | c:0.69 s:0.64 n:18 | c:0.80 s:0.75 n:12 |
| `(eda_max_low) => eda_max_low meets valence_medium` | **0.701** | 0.701 | 0.688 | 0.688 | c:0.71 s:0.71 n:20 | c:0.69 s:0.69 n:11 |
| `(valence_high) => valence_high meets eda_medium` | **0.640** | 0.536 | 0.529 | 0.321 | c:0.53 s:0.32 n:9 | c:0.75 s:0.75 n:12 |
