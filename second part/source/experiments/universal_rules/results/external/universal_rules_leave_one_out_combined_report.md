# Leave-One-Out Aggregated Universal Rules

This version computes rules using N-1 merged datasets (global support threshold)
and checks for their survival exclusively on the held-out test set.

## Parameters
| Parameter | Value |
|-----------|-------|
| minsup (Train Pool) | 0.1 |
| minsup (Test Set) | 0.1 |
| minconf | 0.5 |

## Results Summary

| Test Set | Train Pool Rules | Test Set Rules | Generalized Rules | Failed Rules |
|----------|------------------|----------------|-------------------|--------------|
| **EMBOA** | 137 | 96 | **4** | 133 |
| **K-emo_ext** | 96 | 137 | **4** | 92 |

## Validated Generalized Rules (by Test Set)

### Trained on others, tested on EMBOA
| Rule | Train Confidence | Test Confidence |
|------|------------------|-----------------|
| `(eda_std_high) => eda_std_high starts valence_medium` | 0.9286 | 0.8125 |
| `(eda_max_low) => eda_max_low meets valence_medium` | 0.7143 | 0.6875 |
| `(arousal_high) => arousal_high meets valence_medium` | 0.6923 | 0.8000 |
| `(valence_high) => valence_high meets eda_medium` | 0.5294 | 0.7500 |

### Trained on others, tested on K-emo_ext
| Rule | Train Confidence | Test Confidence |
|------|------------------|-----------------|
| `(eda_std_high) => eda_std_high starts valence_medium` | 0.8125 | 0.9286 |
| `(arousal_high) => arousal_high meets valence_medium` | 0.8000 | 0.6923 |
| `(valence_high) => valence_high meets eda_medium` | 0.7500 | 0.5294 |
| `(eda_max_low) => eda_max_low meets valence_medium` | 0.6875 | 0.7143 |
