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
| **CASE** | 109 | 142 | **36** | 73 |
| **CEAP** | 120 | 158 | **15** | 105 |
| **EmoWorker_v2** | 138 | 130 | **58** | 80 |
| **K-emoCon** | 132 | 108 | **26** | 106 |

## Validated Generalized Rules (by Test Set)

### Trained on others, tested on CASE
| Rule | Train Confidence | Test Confidence |
|------|------------------|-----------------|
| `(valence_medium) => valence_medium is-finished-by eda_scr_auc_high` | 0.7143 | 0.6333 |
| `(arousal_low) => arousal_low meets eda_scr_auc_medium` | 0.7013 | 0.5926 |
| `(eda_medium) => eda_medium meets valence_medium` | 0.6914 | 0.8000 |
| `(arousal_medium) => arousal_medium is-finished-by eda_peaks_medium` | 0.6778 | 0.9333 |
| `(arousal_medium) => arousal_medium meets eda_scr_auc_medium` | 0.6778 | 0.8333 |
| `(valence_medium) => valence_medium meets eda_scr_auc_medium` | 0.6484 | 0.6667 |
| `(hrv_hfn_medium) => hrv_hfn_medium meets valence_medium` | 0.6333 | 0.8667 |
| `(valence_medium) => valence_medium meets arousal_medium` | 0.6154 | 0.8667 |
| `(eda_peaks_medium) => eda_peaks_medium meets valence_medium` | 0.6092 | 0.5000 |
| `(hrv_sdnn_medium) => hrv_sdnn_medium meets arousal_medium` | 0.5690 | 0.9000 |
| `(hrv_hfn_high) => hrv_hfn_high meets valence_medium` | 0.5682 | 0.9667 |
| `(hrv_rmssd_low) => hrv_rmssd_low meets valence_medium` | 0.5667 | 0.8000 |
| `(hrv_pnn20_low) => hrv_pnn20_low meets arousal_medium` | 0.5574 | 0.8000 |
| `(hrv_pnn20_high) => hrv_pnn20_high meets arousal_medium` | 0.5571 | 0.8000 |
| `(eda_std_high) => eda_std_high starts arousal_medium` | 0.5568 | 0.8667 |
| `(hrv_cvnn_high) => hrv_cvnn_high meets valence_medium` | 0.5556 | 0.8276 |
| `(hrv_cvnn_high) => hrv_cvnn_high meets arousal_medium` | 0.5556 | 0.7931 |
| `(hrv_pnn50_high) => hrv_pnn50_high meets valence_medium` | 0.5538 | 0.9000 |
| `(hrv_pnn50_high) => hrv_pnn50_high meets arousal_medium` | 0.5538 | 0.8667 |
| `(hrv_pnn50_low) => hrv_pnn50_low meets arousal_medium` | 0.5410 | 0.8667 |
| `(hrv_cvnn_medium) => hrv_cvnn_medium meets arousal_medium` | 0.5410 | 0.9000 |
| `(hrv_cvsd_medium) => hrv_cvsd_medium meets arousal_medium` | 0.5397 | 0.8333 |
| `(hr_high) => hr_high starts arousal_medium` | 0.5333 | 0.7000 |
| `(hrv_rmssd_medium) => hrv_rmssd_medium meets arousal_medium` | 0.5333 | 0.8333 |
| `(hrv_hfn_medium) => hrv_hfn_medium meets arousal_medium` | 0.5333 | 0.8333 |
| `(eda_scr_auc_high) => eda_scr_auc_high starts arousal_medium` | 0.5333 | 0.6333 |
| `(hrv_hfn_low) => hrv_hfn_low meets valence_medium` | 0.5254 | 0.7241 |
| `(arousal_high) => arousal_high before eda_std_high` | 0.5244 | 0.5862 |
| `(hrv_sdnn_high) => hrv_sdnn_high meets valence_medium` | 0.5205 | 0.8000 |
| `(hrv_pnn20_medium) => hrv_pnn20_medium meets valence_medium` | 0.5185 | 0.9000 |
| `(hrv_cvnn_low) => hrv_cvnn_low meets arousal_medium` | 0.5172 | 0.8333 |
| `(hrv_rmssd_low) => hrv_rmssd_low meets arousal_medium` | 0.5167 | 0.7667 |
| `(hrv_sdnn_low) => hrv_sdnn_low meets arousal_medium` | 0.5161 | 0.8000 |
| `(hrv_cvsd_low) => hrv_cvsd_low meets valence_medium` | 0.5085 | 0.8333 |
| `(hrv_pnn50_low) => hrv_pnn50_low meets valence_medium` | 0.5082 | 0.8667 |
| `(hrv_sdnn_high) => hrv_sdnn_high meets arousal_medium` | 0.5068 | 0.8000 |

### Trained on others, tested on CEAP
| Rule | Train Confidence | Test Confidence |
|------|------------------|-----------------|
| `(arousal_medium) => arousal_medium meets eda_scr_auc_medium` | 0.7614 | 0.5938 |
| `(hrv_sdnn_medium) => hrv_sdnn_medium meets valence_medium` | 0.7284 | 0.7143 |
| `(hrv_cvsd_medium) => hrv_cvsd_medium meets valence_medium` | 0.7229 | 0.6000 |
| `(hrv_cvnn_medium) => hrv_cvnn_medium meets valence_medium` | 0.7160 | 0.6000 |
| `(eda_medium) => eda_medium meets valence_medium` | 0.6977 | 0.8000 |
| `(arousal_medium) => arousal_medium equals valence_medium` | 0.6591 | 0.9375 |
| `(eda_medium) => eda_medium meets arousal_medium` | 0.6395 | 0.8000 |
| `(eda_max_medium) => eda_max_medium meets arousal_medium` | 0.6364 | 0.7692 |
| `(valence_high) => valence_high meets arousal_medium` | 0.6364 | 0.7000 |
| `(eda_max_medium) => eda_max_medium before valence_medium` | 0.6136 | 0.8846 |
| `(eda_max_low) => eda_max_low before arousal_medium` | 0.6067 | 0.7000 |
| `(valence_high) => valence_high meets eda_peaks_high` | 0.6061 | 0.8333 |
| `(valence_medium) => valence_medium meets eda_peaks_high` | 0.5618 | 0.8750 |
| `(eda_std_medium) => eda_std_medium meets arousal_medium` | 0.5281 | 0.8077 |
| `(eda_max_high) => eda_max_high meets arousal_medium` | 0.5056 | 0.6667 |

### Trained on others, tested on EmoWorker_v2
| Rule | Train Confidence | Test Confidence |
|------|------------------|-----------------|
| `(hrv_hfn_medium) => hrv_hfn_medium meets valence_medium` | 0.8372 | 0.5294 |
| `(hrv_hfn_high) => hrv_hfn_high meets valence_medium` | 0.7818 | 0.5789 |
| `(hrv_sdnn_medium) => hrv_sdnn_medium meets valence_medium` | 0.7667 | 0.6429 |
| `(hrv_cvsd_medium) => hrv_cvsd_medium meets valence_medium` | 0.7576 | 0.5926 |
| `(eda_max_medium) => eda_max_medium before valence_medium` | 0.7381 | 0.5000 |
| `(hrv_cvnn_medium) => hrv_cvnn_medium meets valence_medium` | 0.7344 | 0.6296 |
| `(hrv_pnn20_medium) => hrv_pnn20_medium meets valence_medium` | 0.7241 | 0.5000 |
| `(hrv_rmssd_medium) => hrv_rmssd_medium meets valence_medium` | 0.7097 | 0.5714 |
| `(arousal_medium) => arousal_medium equals valence_medium` | 0.7000 | 0.8333 |
| `(arousal_medium) => arousal_medium is-finished-by eda_std_medium` | 0.7000 | 0.8667 |
| `(hrv_pnn50_high) => hrv_pnn50_high meets valence_medium` | 0.6866 | 0.6071 |
| `(hrv_pnn20_high) => hrv_pnn20_high meets valence_medium` | 0.6806 | 0.6071 |
| `(eda_medium) => eda_medium meets arousal_medium` | 0.6790 | 0.6667 |
| `(hrv_hfn_medium) => hrv_hfn_medium meets arousal_medium` | 0.6744 | 0.7059 |
| `(hrv_pnn50_medium) => hrv_pnn50_medium meets valence_medium` | 0.6716 | 0.6429 |
| `(hrv_pnn50_low) => hrv_pnn50_low meets valence_medium` | 0.6667 | 0.5357 |
| `(hrv_sdnn_medium) => hrv_sdnn_medium meets arousal_medium` | 0.6667 | 0.7143 |
| `(hrv_rmssd_low) => hrv_rmssd_low meets valence_medium` | 0.6613 | 0.6071 |
| `(arousal_low) => arousal_low meets eda_scr_auc_medium` | 0.6579 | 0.7143 |
| `(hrv_pnn50_high) => hrv_pnn50_high meets arousal_medium` | 0.6567 | 0.6429 |
| `(valence_high) => valence_high meets arousal_medium` | 0.6471 | 0.7273 |
| `(hrv_cvnn_medium) => hrv_cvnn_medium meets arousal_medium` | 0.6406 | 0.7037 |
| `(hrv_cvsd_low) => hrv_cvsd_low meets valence_medium` | 0.6349 | 0.5769 |
| `(eda_scr_amp_medium) => eda_scr_amp_medium starts arousal_medium` | 0.6333 | 0.5517 |
| `(hrv_cvnn_low) => hrv_cvnn_low meets valence_medium` | 0.6333 | 0.5357 |
| `(eda_scr_amp_low) => eda_scr_amp_low meets valence_medium` | 0.6203 | 0.6667 |
| `(hrv_pnn20_low) => hrv_pnn20_low meets arousal_medium` | 0.6190 | 0.6786 |
| `(hrv_pnn50_low) => hrv_pnn50_low meets arousal_medium` | 0.6190 | 0.7143 |
| `(hrv_cvnn_low) => hrv_cvnn_low meets arousal_medium` | 0.6167 | 0.6429 |
| `(hrv_hfn_low) => hrv_hfn_low meets valence_medium` | 0.6154 | 0.5217 |
| `(hrv_cvnn_high) => hrv_cvnn_high meets valence_medium` | 0.6111 | 0.6897 |
| `(eda_std_high) => eda_std_high starts arousal_medium` | 0.6092 | 0.7097 |
| `(hrv_sdnn_high) => hrv_sdnn_high meets valence_medium` | 0.6081 | 0.5862 |
| `(hrv_pnn20_medium) => hrv_pnn20_medium meets arousal_medium` | 0.6034 | 0.5000 |
| `(hrv_cvsd_low) => hrv_cvsd_low meets arousal_medium` | 0.6032 | 0.6154 |
| `(hrv_cvnn_high) => hrv_cvnn_high meets arousal_medium` | 0.5972 | 0.6897 |
| `(hrv_pnn50_medium) => hrv_pnn50_medium meets arousal_medium` | 0.5970 | 0.6786 |
| `(hrv_rmssd_low) => hrv_rmssd_low meets arousal_medium` | 0.5968 | 0.6071 |
| `(hrv_rmssd_high) => hrv_rmssd_high meets valence_medium` | 0.5909 | 0.5517 |
| `(hrv_pnn20_low) => hrv_pnn20_low meets valence_medium` | 0.5873 | 0.5714 |
| `(hrv_sdnn_low) => hrv_sdnn_low meets valence_medium` | 0.5873 | 0.5517 |
| `(hrv_sdnn_low) => hrv_sdnn_low meets arousal_medium` | 0.5873 | 0.6552 |
| `(hrv_hfn_low) => hrv_hfn_low meets arousal_medium` | 0.5846 | 0.5652 |
| `(eda_std_medium) => eda_std_medium meets arousal_medium` | 0.5833 | 0.6129 |
| `(hrv_sdnn_high) => hrv_sdnn_high meets arousal_medium` | 0.5811 | 0.6207 |
| `(arousal_low) => arousal_low meets eda_std_medium` | 0.5789 | 0.5714 |
| `(hrv_rmssd_high) => hrv_rmssd_high meets arousal_medium` | 0.5758 | 0.6207 |
| `(hrv_cvsd_medium) => hrv_cvsd_medium meets arousal_medium` | 0.5758 | 0.7778 |
| `(eda_low) => eda_low meets arousal_medium` | 0.5747 | 0.8065 |
| `(hrv_pnn20_high) => hrv_pnn20_high meets arousal_medium` | 0.5694 | 0.7857 |
| `(hrv_cvsd_high) => hrv_cvsd_high meets arousal_medium` | 0.5672 | 0.6207 |
| `(hrv_rmssd_medium) => hrv_rmssd_medium meets arousal_medium` | 0.5645 | 0.7857 |
| `(eda_scr_auc_high) => eda_scr_auc_high starts arousal_medium` | 0.5618 | 0.5484 |
| `(valence_high) => valence_high is-finished-by eda_scr_auc_low` | 0.5412 | 0.5455 |
| `(hrv_cvsd_high) => hrv_cvsd_high meets valence_medium` | 0.5373 | 0.6207 |
| `(eda_peaks_low) => eda_peaks_low starts arousal_medium` | 0.5349 | 0.5484 |
| `(eda_peaks_high) => eda_peaks_high meets arousal_medium` | 0.5000 | 0.7097 |
| `(arousal_medium) => arousal_medium contains eda_scr_amp_high` | 0.5000 | 0.6333 |

### Trained on others, tested on K-emoCon
| Rule | Train Confidence | Test Confidence |
|------|------------------|-----------------|
| `(arousal_medium) => arousal_medium is-finished-by eda_peaks_medium` | 0.7826 | 0.6071 |
| `(eda_medium) => eda_medium meets valence_medium` | 0.7647 | 0.5769 |
| `(hrv_hfn_high) => hrv_hfn_high meets valence_medium` | 0.7586 | 0.6250 |
| `(hrv_hfn_medium) => hrv_hfn_medium meets valence_medium` | 0.7500 | 0.7500 |
| `(arousal_medium) => arousal_medium meets eda_scr_auc_medium` | 0.7283 | 0.6786 |
| `(hrv_pnn50_high) => hrv_pnn50_high meets valence_medium` | 0.7101 | 0.5385 |
| `(hrv_pnn50_high) => hrv_pnn50_high meets arousal_medium` | 0.6957 | 0.5385 |
| `(hrv_pnn20_high) => hrv_pnn20_high meets arousal_medium` | 0.6757 | 0.5000 |
| `(hrv_rmssd_low) => hrv_rmssd_low meets valence_medium` | 0.6716 | 0.5652 |
| `(hrv_cvnn_high) => hrv_cvnn_high meets valence_medium` | 0.6622 | 0.5556 |
| `(hrv_cvnn_high) => hrv_cvnn_high meets arousal_medium` | 0.6622 | 0.5185 |
| `(hrv_pnn50_low) => hrv_pnn50_low meets valence_medium` | 0.6618 | 0.5217 |
| `(hrv_pnn20_medium) => hrv_pnn20_medium meets valence_medium` | 0.6562 | 0.6500 |
| `(hrv_cvsd_low) => hrv_cvsd_low meets valence_medium` | 0.6418 | 0.5455 |
| `(hrv_cvnn_low) => hrv_cvnn_low meets valence_medium` | 0.6308 | 0.5217 |
| `(arousal_low) => arousal_low meets eda_scr_auc_medium` | 0.6250 | 0.8333 |
| `(valence_medium) => valence_medium meets eda_scr_auc_medium` | 0.6237 | 0.7500 |
| `(hrv_hfn_low) => hrv_hfn_low meets arousal_medium` | 0.6061 | 0.5000 |
| `(hrv_sdnn_high) => hrv_sdnn_high meets valence_medium` | 0.6053 | 0.5926 |
| `(hrv_rmssd_high) => hrv_rmssd_high meets valence_medium` | 0.6029 | 0.5185 |
| `(hrv_sdnn_low) => hrv_sdnn_low meets valence_medium` | 0.6029 | 0.5000 |
| `(eda_peaks_medium) => eda_peaks_medium meets valence_medium` | 0.5889 | 0.5556 |
| `(hr_medium) => hr_medium meets valence_medium` | 0.5714 | 0.6296 |
| `(hrv_hfn_low) => hrv_hfn_low meets valence_medium` | 0.5606 | 0.6818 |
| `(arousal_high) => arousal_high is-finished-by eda_std_medium` | 0.5465 | 0.5600 |
| `(eda_scr_amp_high) => eda_scr_amp_high starts valence_medium` | 0.5455 | 0.6429 |
