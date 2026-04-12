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

Evaluated on 4 datasets: CASE, K-emoCon, CEAP, EmoWorker_v2

| Dataset | Participants | Rules (Before Filters) | Rules (Filtered) |
|-------|-------------|---------|-------|
| **CASE** | 30 | 526 | 142 |
| **K-emoCon** | 28 | 550 | 108 |
| **CEAP** | 32 | 509 | 158 |
| **EmoWorker_v2** | 31 | 471 | 130 |

## Universal Rules Identification

TOTAL Universal rules found across all 4 datasets: **0**

