# Universal Rules Experiment

This experiment aims to identify rules that are universally true across all evaluated datasets.

## Experiment Parameters

- **minsup**: 0.1 (10% participants)
- **minconf**: 0.5 (50% confidence)
- **maxgap**: 5 seconds
- **max_pattern_size**: 4

## Rule Filters

- **FILTER_BVP_ONLY**: True
- **FILTER_EDA_ONLY**: True
- **FILTER_PHYSIO_CROSS**: True
- **FILTER_SINGLE_FEATURE**: True

## Dataset Processing

Evaluated on 6 datasets: CASE, K-emoCon, CEAP, EmoWorker_v2, K-emo_ext, EMBOA

| Dataset | Participants | Rules (Before Filters) | Rules (Filtered) |
|-------|-------------|---------|-------|
| **CASE** | 30 | 43453 | 19069 |
| **K-emoCon** | 28 | 63947 | 25327 |
| **CEAP** | 32 | 46746 | 27441 |
| **EmoWorker_v2** | 31 | 25983 | 8915 |
| **K-emo_ext** | 28 | 50619 | 22379 |
| **EMBOA** | 16 | 35112 | 14787 |

## Universal Rules Identification

TOTAL Universal rules found across all 6 datasets: **0**

