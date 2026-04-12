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
| **CASE** | 112 | 142 | **48** | 64 |
| **CEAP** | 113 | 158 | **15** | 98 |
| **EMBOA** | 120 | 96 | **11** | 109 |
| **EmoWorker_v2** | 124 | 130 | **56** | 68 |
| **K-emoCon** | 116 | 108 | **26** | 90 |
| **K-emo_ext** | 114 | 137 | **45** | 69 |

## Validated Generalized Rules (by Test Set)

### Trained on others, tested on CASE
| Rule | Train Confidence | Test Confidence |
|------|------------------|-----------------|
| `(eda_medium) => eda_medium meets valence_medium` | 0.7355 | 0.8000 |
| `(hrv_rmssd_low) => hrv_rmssd_low meets valence_medium` | 0.6905 | 0.8000 |
| `(hrv_pnn20_medium) => hrv_pnn20_medium meets valence_medium` | 0.6667 | 0.9000 |
| `(valence_medium) => valence_medium meets arousal_medium` | 0.6667 | 0.8667 |
| `(hrv_cvsd_low) => hrv_cvsd_low meets valence_medium` | 0.6506 | 0.8333 |
| `(hrv_cvnn_low) => hrv_cvnn_low meets valence_medium` | 0.6341 | 0.8333 |
| `(hrv_hfn_medium) => hrv_hfn_medium meets valence_medium` | 0.6333 | 0.8667 |
| `(hrv_pnn50_high) => hrv_pnn50_high meets valence_medium` | 0.6292 | 0.9000 |
| `(hrv_pnn50_low) => hrv_pnn50_low meets valence_medium` | 0.6265 | 0.8667 |
| `(arousal_low) => arousal_low meets eda_scr_auc_medium` | 0.6250 | 0.5926 |
| `(eda_peaks_medium) => eda_peaks_medium meets valence_medium` | 0.6250 | 0.5000 |
| `(hrv_cvnn_high) => hrv_cvnn_high meets valence_medium` | 0.6237 | 0.8276 |
| `(hrv_sdnn_low) => hrv_sdnn_low meets valence_medium` | 0.6235 | 0.7667 |
| `(hr_high) => hr_high starts arousal_medium` | 0.6214 | 0.7000 |
| `(hrv_pnn20_low) => hrv_pnn20_low meets valence_medium` | 0.6145 | 0.7667 |
| `(hrv_sdnn_medium) => hrv_sdnn_medium meets arousal_medium` | 0.6125 | 0.9000 |
| `(eda_std_high) => eda_std_high starts arousal_medium` | 0.5985 | 0.8667 |
| `(hrv_sdnn_high) => hrv_sdnn_high meets valence_medium` | 0.5938 | 0.8000 |
| `(valence_medium) => valence_medium meets eda_scr_auc_medium` | 0.5926 | 0.6667 |
| `(hrv_cvnn_high) => hrv_cvnn_high meets arousal_medium` | 0.5914 | 0.7931 |
| `(arousal_medium) => arousal_medium is-finished-by eda_peaks_medium` | 0.5896 | 0.9333 |
| `(arousal_medium) => arousal_medium meets eda_scr_auc_medium` | 0.5746 | 0.8333 |
| `(hrv_pnn50_high) => hrv_pnn50_high meets arousal_medium` | 0.5730 | 0.8667 |
| `(hrv_rmssd_high) => hrv_rmssd_high meets valence_medium` | 0.5714 | 0.8148 |
| `(hrv_hfn_high) => hrv_hfn_high meets valence_medium` | 0.5682 | 0.9667 |
| `(hrv_rmssd_medium) => hrv_rmssd_medium meets arousal_medium` | 0.5663 | 0.8333 |
| `(hrv_pnn20_high) => hrv_pnn20_high meets arousal_medium` | 0.5638 | 0.8000 |
| `(valence_medium) => valence_medium is-finished-by eda_scr_auc_high` | 0.5630 | 0.6333 |
| `(hrv_cvnn_medium) => hrv_cvnn_medium meets arousal_medium` | 0.5581 | 0.9000 |
| `(eda_max_high) => eda_max_high meets arousal_medium` | 0.5564 | 0.7000 |
| `(hrv_cvsd_high) => hrv_cvsd_high meets arousal_medium` | 0.5543 | 0.8519 |
| `(eda_scr_amp_high) => eda_scr_amp_high starts arousal_medium` | 0.5526 | 0.5000 |
| `(eda_scr_auc_high) => eda_scr_auc_high starts arousal_medium` | 0.5508 | 0.6333 |
| `(hrv_cvsd_medium) => hrv_cvsd_medium meets arousal_medium` | 0.5465 | 0.8333 |
| `(arousal_high) => arousal_high before eda_std_high` | 0.5447 | 0.5862 |
| `(hrv_cvsd_high) => hrv_cvsd_high meets valence_medium` | 0.5435 | 0.8519 |
| `(hrv_pnn20_low) => hrv_pnn20_low meets arousal_medium` | 0.5422 | 0.8000 |
| `(hrv_pnn50_low) => hrv_pnn50_low meets arousal_medium` | 0.5422 | 0.8667 |
| `(hrv_sdnn_high) => hrv_sdnn_high meets arousal_medium` | 0.5417 | 0.8000 |
| `(hrv_rmssd_high) => hrv_rmssd_high meets arousal_medium` | 0.5385 | 0.8519 |
| `(hrv_rmssd_low) => hrv_rmssd_low meets arousal_medium` | 0.5357 | 0.7667 |
| `(hrv_hfn_medium) => hrv_hfn_medium meets arousal_medium` | 0.5333 | 0.8333 |
| `(hrv_sdnn_low) => hrv_sdnn_low meets arousal_medium` | 0.5294 | 0.8000 |
| `(hrv_hfn_low) => hrv_hfn_low meets valence_medium` | 0.5254 | 0.7241 |
| `(hrv_cvsd_low) => hrv_cvsd_low meets arousal_medium` | 0.5181 | 0.8667 |
| `(hrv_cvnn_low) => hrv_cvnn_low meets arousal_medium` | 0.5122 | 0.8333 |
| `(hrv_pnn20_medium) => hrv_pnn20_medium meets arousal_medium` | 0.5000 | 0.8667 |
| `(hrv_pnn50_medium) => hrv_pnn50_medium meets arousal_medium` | 0.5000 | 0.9000 |

### Trained on others, tested on CEAP
| Rule | Train Confidence | Test Confidence |
|------|------------------|-----------------|
| `(hrv_sdnn_medium) => hrv_sdnn_medium meets valence_medium` | 0.7670 | 0.7143 |
| `(hrv_cvsd_medium) => hrv_cvsd_medium meets valence_medium` | 0.7642 | 0.6000 |
| `(hrv_cvnn_medium) => hrv_cvnn_medium meets valence_medium` | 0.7642 | 0.6000 |
| `(eda_medium) => eda_medium meets valence_medium` | 0.7381 | 0.8000 |
| `(arousal_medium) => arousal_medium equals valence_medium` | 0.7197 | 0.9375 |
| `(eda_max_medium) => eda_max_medium before valence_medium` | 0.7000 | 0.8846 |
| `(eda_max_medium) => eda_max_medium meets arousal_medium` | 0.6692 | 0.7692 |
| `(valence_high) => valence_high meets arousal_medium` | 0.6465 | 0.7000 |
| `(eda_medium) => eda_medium meets arousal_medium` | 0.6429 | 0.8000 |
| `(arousal_medium) => arousal_medium meets eda_scr_auc_medium` | 0.6288 | 0.5938 |
| `(eda_max_low) => eda_max_low before arousal_medium` | 0.6015 | 0.7000 |
| `(eda_std_medium) => eda_std_medium meets arousal_medium` | 0.5940 | 0.8077 |
| `(eda_max_high) => eda_max_high meets arousal_medium` | 0.5639 | 0.6667 |
| `(valence_high) => valence_high meets eda_peaks_high` | 0.5253 | 0.8333 |
| `(valence_medium) => valence_medium meets eda_peaks_high` | 0.5038 | 0.8750 |

### Trained on others, tested on EMBOA
| Rule | Train Confidence | Test Confidence |
|------|------------------|-----------------|
| `(eda_low) => eda_low meets valence_medium` | 0.7740 | 0.6875 |
| `(arousal_medium) => arousal_medium equals valence_medium` | 0.7432 | 0.9375 |
| `(eda_std_high) => eda_std_high starts valence_medium` | 0.7329 | 0.8125 |
| `(hr_low) => hr_low starts valence_medium` | 0.6789 | 0.7500 |
| `(eda_std_medium) => eda_std_medium meets valence_medium` | 0.6643 | 0.8750 |
| `(eda_high) => eda_high starts valence_medium` | 0.6486 | 0.9375 |
| `(eda_high) => eda_high starts arousal_medium` | 0.6486 | 0.7500 |
| `(eda_std_high) => eda_std_high starts arousal_medium` | 0.6370 | 0.7500 |
| `(valence_high) => valence_high meets arousal_medium` | 0.6283 | 0.8750 |
| `(eda_std_medium) => eda_std_medium meets arousal_medium` | 0.6084 | 0.8125 |
| `(hr_low) => hr_low starts arousal_medium` | 0.5688 | 0.9375 |

### Trained on others, tested on EmoWorker_v2
| Rule | Train Confidence | Test Confidence |
|------|------------------|-----------------|
| `(hrv_hfn_medium) => hrv_hfn_medium meets valence_medium` | 0.8372 | 0.5294 |
| `(hrv_sdnn_medium) => hrv_sdnn_medium meets valence_medium` | 0.8049 | 0.6429 |
| `(hrv_pnn20_medium) => hrv_pnn20_medium meets valence_medium` | 0.8049 | 0.5000 |
| `(hrv_cvsd_medium) => hrv_cvsd_medium meets valence_medium` | 0.7978 | 0.5926 |
| `(hrv_cvnn_medium) => hrv_cvnn_medium meets valence_medium` | 0.7865 | 0.6296 |
| `(eda_max_medium) => eda_max_medium before valence_medium` | 0.7857 | 0.5000 |
| `(hrv_hfn_high) => hrv_hfn_high meets valence_medium` | 0.7818 | 0.5789 |
| `(hrv_rmssd_medium) => hrv_rmssd_medium meets valence_medium` | 0.7647 | 0.5714 |
| `(hrv_rmssd_low) => hrv_rmssd_low meets valence_medium` | 0.7558 | 0.6071 |
| `(arousal_medium) => arousal_medium equals valence_medium` | 0.7463 | 0.8333 |
| `(hrv_pnn50_medium) => hrv_pnn50_medium meets valence_medium` | 0.7444 | 0.6429 |
| `(hrv_pnn50_low) => hrv_pnn50_low meets valence_medium` | 0.7412 | 0.5357 |
| `(hrv_cvnn_low) => hrv_cvnn_low meets valence_medium` | 0.7381 | 0.5357 |
| `(hrv_cvsd_low) => hrv_cvsd_low meets valence_medium` | 0.7356 | 0.5769 |
| `(hrv_pnn50_high) => hrv_pnn50_high meets valence_medium` | 0.7253 | 0.6071 |
| `(arousal_medium) => arousal_medium is-finished-by eda_std_medium` | 0.7239 | 0.8667 |
| `(hrv_pnn20_high) => hrv_pnn20_high meets valence_medium` | 0.7188 | 0.6071 |
| `(hrv_sdnn_low) => hrv_sdnn_low meets valence_medium` | 0.6977 | 0.5517 |
| `(hrv_sdnn_medium) => hrv_sdnn_medium meets arousal_medium` | 0.6829 | 0.7143 |
| `(hrv_pnn20_low) => hrv_pnn20_low meets valence_medium` | 0.6824 | 0.5714 |
| `(hrv_hfn_medium) => hrv_hfn_medium meets arousal_medium` | 0.6744 | 0.7059 |
| `(eda_medium) => eda_medium meets arousal_medium` | 0.6694 | 0.6667 |
| `(hrv_cvnn_high) => hrv_cvnn_high meets valence_medium` | 0.6667 | 0.6897 |
| `(hrv_sdnn_high) => hrv_sdnn_high meets valence_medium` | 0.6598 | 0.5862 |
| `(eda_scr_amp_low) => eda_scr_amp_low meets valence_medium` | 0.6542 | 0.6667 |
| `(valence_high) => valence_high meets arousal_medium` | 0.6525 | 0.7273 |
| `(hrv_rmssd_high) => hrv_rmssd_high meets valence_medium` | 0.6517 | 0.5517 |
| `(hrv_pnn50_high) => hrv_pnn50_high meets arousal_medium` | 0.6484 | 0.6429 |
| `(hrv_pnn20_medium) => hrv_pnn20_medium meets arousal_medium` | 0.6341 | 0.5000 |
| `(eda_std_high) => eda_std_high starts arousal_medium` | 0.6336 | 0.7097 |
| `(eda_std_medium) => eda_std_medium meets arousal_medium` | 0.6328 | 0.6129 |
| `(arousal_low) => arousal_low meets eda_std_medium` | 0.6306 | 0.5714 |
| `(hrv_cvnn_medium) => hrv_cvnn_medium meets arousal_medium` | 0.6292 | 0.7037 |
| `(hrv_cvnn_high) => hrv_cvnn_high meets arousal_medium` | 0.6237 | 0.6897 |
| `(hrv_cvsd_high) => hrv_cvsd_high meets arousal_medium` | 0.6222 | 0.6207 |
| `(hrv_hfn_low) => hrv_hfn_low meets valence_medium` | 0.6154 | 0.5217 |
| `(hrv_cvsd_high) => hrv_cvsd_high meets valence_medium` | 0.6111 | 0.6207 |
| `(eda_low) => eda_low meets arousal_medium` | 0.6107 | 0.8065 |
| `(hrv_cvsd_low) => hrv_cvsd_low meets arousal_medium` | 0.6092 | 0.6154 |
| `(hrv_rmssd_high) => hrv_rmssd_high meets arousal_medium` | 0.6067 | 0.6207 |
| `(hrv_pnn50_low) => hrv_pnn50_low meets arousal_medium` | 0.6000 | 0.7143 |
| `(hrv_sdnn_high) => hrv_sdnn_high meets arousal_medium` | 0.5979 | 0.6207 |
| `(arousal_low) => arousal_low meets eda_scr_auc_medium` | 0.5946 | 0.7143 |
| `(hrv_rmssd_low) => hrv_rmssd_low meets arousal_medium` | 0.5930 | 0.6071 |
| `(hrv_pnn20_low) => hrv_pnn20_low meets arousal_medium` | 0.5882 | 0.6786 |
| `(hrv_rmssd_medium) => hrv_rmssd_medium meets arousal_medium` | 0.5882 | 0.7857 |
| `(hrv_hfn_low) => hrv_hfn_low meets arousal_medium` | 0.5846 | 0.5652 |
| `(hrv_cvnn_low) => hrv_cvnn_low meets arousal_medium` | 0.5833 | 0.6429 |
| `(hrv_sdnn_low) => hrv_sdnn_low meets arousal_medium` | 0.5814 | 0.6552 |
| `(hrv_pnn50_medium) => hrv_pnn50_medium meets arousal_medium` | 0.5778 | 0.6786 |
| `(hrv_cvsd_medium) => hrv_cvsd_medium meets arousal_medium` | 0.5730 | 0.7778 |
| `(hrv_pnn20_high) => hrv_pnn20_high meets arousal_medium` | 0.5729 | 0.7857 |
| `(eda_scr_auc_high) => eda_scr_auc_high starts arousal_medium` | 0.5726 | 0.5484 |
| `(eda_scr_amp_medium) => eda_scr_amp_medium starts arousal_medium` | 0.5698 | 0.5517 |
| `(eda_peaks_low) => eda_peaks_low starts arousal_medium` | 0.5614 | 0.5484 |
| `(eda_peaks_high) => eda_peaks_high meets arousal_medium` | 0.5508 | 0.7097 |

### Trained on others, tested on K-emoCon
| Rule | Train Confidence | Test Confidence |
|------|------------------|-----------------|
| `(eda_medium) => eda_medium meets valence_medium` | 0.7840 | 0.5769 |
| `(hrv_hfn_high) => hrv_hfn_high meets valence_medium` | 0.7586 | 0.6250 |
| `(hrv_rmssd_low) => hrv_rmssd_low meets valence_medium` | 0.7582 | 0.5652 |
| `(hrv_pnn20_medium) => hrv_pnn20_medium meets valence_medium` | 0.7500 | 0.6500 |
| `(hrv_hfn_medium) => hrv_hfn_medium meets valence_medium` | 0.7500 | 0.7500 |
| `(hrv_pnn50_high) => hrv_pnn50_high meets valence_medium` | 0.7419 | 0.5385 |
| `(hrv_cvsd_low) => hrv_cvsd_low meets valence_medium` | 0.7363 | 0.5455 |
| `(hrv_pnn50_low) => hrv_pnn50_low meets valence_medium` | 0.7333 | 0.5217 |
| `(hrv_cvnn_low) => hrv_cvnn_low meets valence_medium` | 0.7303 | 0.5217 |
| `(hrv_cvnn_high) => hrv_cvnn_high meets valence_medium` | 0.7053 | 0.5556 |
| `(hrv_sdnn_low) => hrv_sdnn_low meets valence_medium` | 0.7033 | 0.5000 |
| `(hrv_pnn50_high) => hrv_pnn50_high meets arousal_medium` | 0.6774 | 0.5385 |
| `(hrv_cvnn_high) => hrv_cvnn_high meets arousal_medium` | 0.6737 | 0.5185 |
| `(arousal_medium) => arousal_medium is-finished-by eda_peaks_medium` | 0.6618 | 0.6071 |
| `(hrv_rmssd_high) => hrv_rmssd_high meets valence_medium` | 0.6593 | 0.5185 |
| `(hrv_sdnn_high) => hrv_sdnn_high meets valence_medium` | 0.6566 | 0.5926 |
| `(hrv_pnn20_high) => hrv_pnn20_high meets arousal_medium` | 0.6531 | 0.5000 |
| `(eda_scr_amp_high) => eda_scr_amp_high starts valence_medium` | 0.6379 | 0.6429 |
| `(hr_medium) => hr_medium meets valence_medium` | 0.6226 | 0.6296 |
| `(arousal_medium) => arousal_medium meets eda_scr_auc_medium` | 0.6103 | 0.6786 |
| `(eda_peaks_medium) => eda_peaks_medium meets valence_medium` | 0.6087 | 0.5556 |
| `(hrv_hfn_low) => hrv_hfn_low meets arousal_medium` | 0.6061 | 0.5000 |
| `(valence_medium) => valence_medium meets eda_scr_auc_medium` | 0.5766 | 0.7500 |
| `(arousal_low) => arousal_low meets eda_scr_auc_medium` | 0.5739 | 0.8333 |
| `(hrv_hfn_low) => hrv_hfn_low meets valence_medium` | 0.5606 | 0.6818 |
| `(arousal_high) => arousal_high is-finished-by eda_std_medium` | 0.5591 | 0.5600 |

### Trained on others, tested on K-emo_ext
| Rule | Train Confidence | Test Confidence |
|------|------------------|-----------------|
| `(hrv_sdnn_medium) => hrv_sdnn_medium meets valence_medium` | 0.7273 | 0.9091 |
| `(eda_medium) => eda_medium meets valence_medium` | 0.7244 | 0.8750 |
| `(hrv_cvsd_medium) => hrv_cvsd_medium meets valence_medium` | 0.7097 | 0.9130 |
| `(hrv_cvnn_medium) => hrv_cvnn_medium meets valence_medium` | 0.7033 | 0.9200 |
| `(eda_std_high) => eda_std_high starts valence_medium` | 0.7015 | 0.9286 |
| `(valence_medium) => valence_medium meets arousal_medium` | 0.6861 | 0.7857 |
| `(hrv_sdnn_medium) => hrv_sdnn_medium meets arousal_medium` | 0.6818 | 0.7273 |
| `(eda_medium) => eda_medium meets arousal_medium` | 0.6772 | 0.6250 |
| `(eda_max_medium) => eda_max_medium meets arousal_medium` | 0.6769 | 0.7308 |
| `(hrv_rmssd_medium) => hrv_rmssd_medium meets valence_medium` | 0.6667 | 0.9130 |
| `(hrv_pnn50_medium) => hrv_pnn50_medium meets valence_medium` | 0.6632 | 0.9565 |
| `(hrv_pnn50_high) => hrv_pnn50_high meets valence_medium` | 0.6632 | 0.8333 |
| `(hrv_pnn20_high) => hrv_pnn20_high meets valence_medium` | 0.6600 | 0.8333 |
| `(hrv_cvnn_medium) => hrv_cvnn_medium meets arousal_medium` | 0.6593 | 0.6000 |
| `(hrv_pnn20_medium) => hrv_pnn20_medium meets valence_medium` | 0.6548 | 1.0000 |
| `(hrv_pnn50_high) => hrv_pnn50_high meets arousal_medium` | 0.6526 | 0.6250 |
| `(hrv_pnn50_low) => hrv_pnn50_low meets arousal_medium` | 0.6484 | 0.5455 |
| `(hrv_rmssd_low) => hrv_rmssd_low meets valence_medium` | 0.6444 | 1.0000 |
| `(hrv_pnn20_low) => hrv_pnn20_low meets arousal_medium` | 0.6374 | 0.5000 |
| `(hrv_cvsd_medium) => hrv_cvsd_medium meets arousal_medium` | 0.6344 | 0.5652 |
| `(hrv_cvnn_high) => hrv_cvnn_high meets valence_medium` | 0.6337 | 0.8571 |
| `(hrv_rmssd_medium) => hrv_rmssd_medium meets arousal_medium` | 0.6333 | 0.6522 |
| `(arousal_low) => arousal_low meets eda_scr_auc_medium` | 0.6306 | 0.5714 |
| `(hrv_pnn20_high) => hrv_pnn20_high meets arousal_medium` | 0.6300 | 0.5833 |
| `(hrv_pnn50_low) => hrv_pnn50_low meets valence_medium` | 0.6264 | 0.9545 |
| `(eda_scr_auc_high) => eda_scr_auc_high starts valence_medium` | 0.6250 | 0.9286 |
| `(hrv_cvnn_high) => hrv_cvnn_high meets arousal_medium` | 0.6238 | 0.7143 |
| `(hrv_cvsd_low) => hrv_cvsd_low meets valence_medium` | 0.6180 | 1.0000 |
| `(eda_scr_auc_medium) => eda_scr_auc_medium starts valence_medium` | 0.6091 | 0.7308 |
| `(hrv_cvsd_low) => hrv_cvsd_low meets arousal_medium` | 0.6067 | 0.6250 |
| `(hrv_cvnn_low) => hrv_cvnn_low meets valence_medium` | 0.6023 | 1.0000 |
| `(hrv_sdnn_high) => hrv_sdnn_high meets valence_medium` | 0.6019 | 0.8261 |
| `(hrv_rmssd_low) => hrv_rmssd_low meets arousal_medium` | 0.6000 | 0.5833 |
| `(hrv_sdnn_high) => hrv_sdnn_high meets arousal_medium` | 0.5922 | 0.6522 |
| `(hrv_rmssd_high) => hrv_rmssd_high meets arousal_medium` | 0.5895 | 0.6957 |
| `(hrv_cvsd_high) => hrv_cvsd_high meets arousal_medium` | 0.5833 | 0.7826 |
| `(hrv_pnn20_low) => hrv_pnn20_low meets valence_medium` | 0.5824 | 0.9545 |
| `(hrv_rmssd_high) => hrv_rmssd_high meets valence_medium` | 0.5789 | 0.8261 |
| `(valence_medium) => valence_medium meets eda_scr_auc_medium` | 0.5766 | 0.7500 |
| `(hrv_sdnn_low) => hrv_sdnn_low meets valence_medium` | 0.5761 | 1.0000 |
| `(eda_peaks_low) => eda_peaks_low starts valence_medium` | 0.5726 | 0.8214 |
| `(hrv_pnn20_medium) => hrv_pnn20_medium meets arousal_medium` | 0.5714 | 0.7083 |
| `(eda_scr_amp_high) => eda_scr_amp_high starts valence_medium` | 0.5690 | 0.9286 |
| `(hrv_cvsd_high) => hrv_cvsd_high meets valence_medium` | 0.5625 | 0.8261 |
| `(eda_peaks_high) => eda_peaks_high meets arousal_medium` | 0.5537 | 0.7143 |
