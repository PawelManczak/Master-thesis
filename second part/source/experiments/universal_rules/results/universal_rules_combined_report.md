# Aggregated Universal Rules (Combined Global Minimum Support)

This experiment evaluates universal rules by completely mixing the datasets 
into a single, massive dataframe before applying ARMADA.

## Parameters
| Parameter | Value |
|-----------|-------|
| minsup | 0.1 (Global) |
| minconf | 0.5 |
| maxgap | 5s |
| max_pattern_size | 2 |

## Discovered Rules

| Antecedent | Consequent | Confidence | Lift | Support |
|------------|------------|------------|------|---------|
| `(eda_max_low)` | `eda_max_low before valence_medium` | 0.7899 | 0.7899 | 0.7769 |
| `(arousal_low)` | `arousal_low before valence_medium` | 0.7692 | 0.7692 | 0.6612 |
| `(eda_low)` | `eda_low meets valence_medium` | 0.7627 | 0.7627 | 0.7438 |
| `(arousal_low)` | `arousal_low is-finished-by eda_peaks_medium` | 0.7500 | 0.7756 | 0.6446 |
| `(hrv_hfn_medium)` | `hrv_hfn_medium meets valence_medium` | 0.7500 | 0.7500 | 0.3719 |
| `(arousal_medium)` | `arousal_medium is-finished-by eda_peaks_medium` | 0.7417 | 0.7670 | 0.7355 |
| `(arousal_medium)` | `arousal_medium is-finished-by eda_std_medium` | 0.7417 | 0.7804 | 0.7355 |
| `(arousal_medium)` | `arousal_medium equals valence_medium` | 0.7333 | 0.7333 | 0.7273 |
| `(hrv_hfn_high)` | `hrv_hfn_high meets valence_medium` | 0.7297 | 0.7297 | 0.4463 |
| `(hrv_sdnn_medium)` | `hrv_sdnn_medium meets valence_medium` | 0.7273 | 0.7273 | 0.5289 |
| `(arousal_medium)` | `arousal_medium meets eda_peaks_high` | 0.7250 | 0.7250 | 0.7190 |
| `(arousal_low)` | `arousal_low contains eda_peaks_high` | 0.7212 | 0.7212 | 0.6198 |
| `(eda_medium)` | `eda_medium meets valence_medium` | 0.7207 | 0.7207 | 0.6612 |
| `(valence_medium)` | `valence_medium is-finished-by eda_std_medium` | 0.7190 | 0.7565 | 0.7190 |
| `(arousal_medium)` | `arousal_medium meets eda_scr_auc_medium` | 0.7167 | 0.7883 | 0.7107 |
| `(hrv_cvsd_medium)` | `hrv_cvsd_medium meets valence_medium` | 0.7097 | 0.7097 | 0.5455 |
| `(hrv_cvnn_medium)` | `hrv_cvnn_medium meets valence_medium` | 0.7033 | 0.7033 | 0.5289 |
| `(valence_medium)` | `valence_medium is-finished-by eda_peaks_medium` | 0.7025 | 0.7265 | 0.7025 |
| `(arousal_medium)` | `arousal_medium contains eda_scr_auc_high` | 0.7000 | 0.7058 | 0.6942 |
| `(valence_medium)` | `valence_medium is-finished-by eda_scr_auc_high` | 0.6942 | 0.7000 | 0.6942 |
| `(eda_std_high)` | `eda_std_high starts valence_medium` | 0.6864 | 0.6864 | 0.6694 |
| `(hrv_hfn_medium)` | `hrv_hfn_medium meets arousal_medium` | 0.6833 | 0.6890 | 0.3388 |
| `(hrv_sdnn_medium)` | `hrv_sdnn_medium meets arousal_medium` | 0.6818 | 0.6875 | 0.4959 |
| `(valence_medium)` | `valence_medium meets arousal_medium` | 0.6777 | 0.6833 | 0.6777 |
| `(valence_high)` | `valence_high meets eda_peaks_high` | 0.6771 | 0.6771 | 0.5372 |
| `(arousal_high)` | `arousal_high contains eda_peaks_medium` | 0.6757 | 0.6988 | 0.6198 |
| `(eda_medium)` | `eda_medium meets arousal_medium` | 0.6757 | 0.6813 | 0.6198 |
| `(eda_max_medium)` | `eda_max_medium before valence_medium` | 0.6754 | 0.6754 | 0.6364 |
| `(arousal_low)` | `arousal_low meets eda_scr_auc_medium` | 0.6731 | 0.7404 | 0.5785 |
| `(eda_max_medium)` | `eda_max_medium meets arousal_medium` | 0.6667 | 0.6722 | 0.6281 |
| `(hrv_rmssd_medium)` | `hrv_rmssd_medium meets valence_medium` | 0.6667 | 0.6667 | 0.4959 |
| `(hrv_pnn50_high)` | `hrv_pnn50_high meets valence_medium` | 0.6632 | 0.6632 | 0.5207 |
| `(hrv_pnn50_medium)` | `hrv_pnn50_medium meets valence_medium` | 0.6632 | 0.6632 | 0.5207 |
| `(hrv_pnn20_high)` | `hrv_pnn20_high meets valence_medium` | 0.6600 | 0.6600 | 0.5455 |
| `(hrv_cvnn_medium)` | `hrv_cvnn_medium meets arousal_medium` | 0.6593 | 0.6648 | 0.4959 |
| `(valence_high)` | `valence_high meets arousal_medium` | 0.6562 | 0.6617 | 0.5207 |
| `(hrv_pnn20_medium)` | `hrv_pnn20_medium meets valence_medium` | 0.6548 | 0.6548 | 0.4545 |
| `(valence_medium)` | `valence_medium meets eda_scr_auc_medium` | 0.6529 | 0.7182 | 0.6529 |
| `(hrv_pnn50_high)` | `hrv_pnn50_high meets arousal_medium` | 0.6526 | 0.6581 | 0.5124 |
| `(hrv_pnn50_low)` | `hrv_pnn50_low meets arousal_medium` | 0.6484 | 0.6538 | 0.4876 |
| `(valence_high)` | `valence_high meets eda_peaks_medium` | 0.6458 | 0.6679 | 0.5124 |
| `(valence_medium)` | `valence_medium meets eda_peaks_high` | 0.6446 | 0.6446 | 0.6446 |
| `(hrv_rmssd_low)` | `hrv_rmssd_low meets valence_medium` | 0.6444 | 0.6444 | 0.4793 |
| `(eda_std_medium)` | `eda_std_medium meets valence_medium` | 0.6435 | 0.6435 | 0.6116 |
| `(hrv_pnn20_low)` | `hrv_pnn20_low meets arousal_medium` | 0.6374 | 0.6427 | 0.4793 |
| `(eda_low)` | `eda_low meets arousal_medium` | 0.6356 | 0.6409 | 0.6198 |
| `(eda_std_high)` | `eda_std_high starts arousal_medium` | 0.6356 | 0.6409 | 0.6198 |
| `(hrv_cvsd_medium)` | `hrv_cvsd_medium meets arousal_medium` | 0.6344 | 0.6397 | 0.4876 |
| `(hrv_cvnn_high)` | `hrv_cvnn_high meets valence_medium` | 0.6337 | 0.6337 | 0.5289 |
| `(hrv_rmssd_medium)` | `hrv_rmssd_medium meets arousal_medium` | 0.6333 | 0.6386 | 0.4711 |
| `(eda_scr_amp_low)` | `eda_scr_amp_low meets valence_medium` | 0.6330 | 0.6330 | 0.5702 |
| `(eda_max_low)` | `eda_max_low before arousal_medium` | 0.6303 | 0.6355 | 0.6198 |
| `(hrv_pnn20_high)` | `hrv_pnn20_high meets arousal_medium` | 0.6300 | 0.6352 | 0.5207 |
| `(hrv_pnn50_low)` | `hrv_pnn50_low meets valence_medium` | 0.6264 | 0.6264 | 0.4711 |
| `(eda_scr_auc_high)` | `eda_scr_auc_high starts valence_medium` | 0.6250 | 0.6250 | 0.6198 |
| `(hrv_cvnn_low)` | `hrv_cvnn_low meets arousal_medium` | 0.6250 | 0.6302 | 0.4545 |
| `(hrv_cvnn_high)` | `hrv_cvnn_high meets arousal_medium` | 0.6238 | 0.6290 | 0.5207 |
| `(hr_low)` | `hr_low starts valence_medium` | 0.6220 | 0.6220 | 0.4215 |
| `(hrv_pnn50_medium)` | `hrv_pnn50_medium meets arousal_medium` | 0.6211 | 0.6262 | 0.4876 |
| `(hrv_cvsd_low)` | `hrv_cvsd_low meets valence_medium` | 0.6180 | 0.6180 | 0.4545 |
| `(arousal_high)` | `arousal_high is-finished-by eda_scr_auc_medium` | 0.6126 | 0.6739 | 0.5620 |
| `(eda_scr_auc_medium)` | `eda_scr_auc_medium starts valence_medium` | 0.6091 | 0.6091 | 0.5537 |
| `(hrv_sdnn_low)` | `hrv_sdnn_low meets arousal_medium` | 0.6087 | 0.6138 | 0.4628 |
| `(eda_high)` | `eda_high starts arousal_medium` | 0.6083 | 0.6134 | 0.6033 |
| `(valence_low)` | `valence_low is-finished-by eda_peaks_high` | 0.6075 | 0.6075 | 0.5372 |
| `(eda_scr_amp_medium)` | `eda_scr_amp_medium starts arousal_medium` | 0.6067 | 0.6118 | 0.4463 |
| `(hrv_cvsd_low)` | `hrv_cvsd_low meets arousal_medium` | 0.6067 | 0.6118 | 0.4463 |
| `(arousal_high)` | `arousal_high is-finished-by eda_peaks_high` | 0.6036 | 0.6036 | 0.5537 |
| `(eda_peaks_high)` | `eda_peaks_high starts valence_medium` | 0.6033 | 0.6033 | 0.6033 |
| `(hrv_cvnn_low)` | `hrv_cvnn_low meets valence_medium` | 0.6023 | 0.6023 | 0.4380 |
| `(hrv_sdnn_high)` | `hrv_sdnn_high meets valence_medium` | 0.6019 | 0.6019 | 0.5124 |
| `(hrv_rmssd_low)` | `hrv_rmssd_low meets arousal_medium` | 0.6000 | 0.6050 | 0.4463 |
| `(eda_high)` | `eda_high starts valence_medium` | 0.6000 | 0.6000 | 0.5950 |
| `(valence_high)` | `valence_high is-finished-by eda_scr_auc_medium` | 0.5938 | 0.6531 | 0.4711 |
| `(hrv_sdnn_high)` | `hrv_sdnn_high meets arousal_medium` | 0.5922 | 0.5972 | 0.5041 |
| `(eda_std_medium)` | `eda_std_medium meets arousal_medium` | 0.5913 | 0.5962 | 0.5620 |
| `(hrv_hfn_low)` | `hrv_hfn_low meets valence_medium` | 0.5909 | 0.5909 | 0.4298 |
| `(hrv_rmssd_high)` | `hrv_rmssd_high meets arousal_medium` | 0.5895 | 0.5944 | 0.4628 |
| `(hr_high)` | `hr_high equals valence_medium` | 0.5889 | 0.5889 | 0.4380 |
| `(hr_high)` | `hr_high starts arousal_medium` | 0.5889 | 0.5938 | 0.4380 |
| `(hr_medium)` | `hr_medium meets valence_medium` | 0.5889 | 0.5889 | 0.4380 |
| `(eda_scr_amp_medium)` | `eda_scr_amp_medium starts valence_medium` | 0.5843 | 0.5843 | 0.4298 |
| `(hrv_cvsd_high)` | `hrv_cvsd_high meets arousal_medium` | 0.5833 | 0.5882 | 0.4628 |
| `(hrv_pnn20_low)` | `hrv_pnn20_low meets valence_medium` | 0.5824 | 0.5824 | 0.4380 |
| `(eda_peaks_medium)` | `eda_peaks_medium meets valence_medium` | 0.5812 | 0.5812 | 0.5620 |
| `(hrv_hfn_high)` | `hrv_hfn_high meets arousal_medium` | 0.5811 | 0.5859 | 0.3554 |
| `(hrv_hfn_low)` | `hrv_hfn_low meets arousal_medium` | 0.5795 | 0.5844 | 0.4215 |
| `(hrv_rmssd_high)` | `hrv_rmssd_high meets valence_medium` | 0.5789 | 0.5789 | 0.4545 |
| `(arousal_low)` | `arousal_low meets eda_std_medium` | 0.5769 | 0.6070 | 0.4959 |
| `(hrv_sdnn_low)` | `hrv_sdnn_low meets valence_medium` | 0.5761 | 0.5761 | 0.4380 |
| `(eda_peaks_low)` | `eda_peaks_low starts valence_medium` | 0.5726 | 0.5726 | 0.5537 |
| `(hrv_pnn20_medium)` | `hrv_pnn20_medium meets arousal_medium` | 0.5714 | 0.5762 | 0.3967 |
| `(eda_scr_amp_high)` | `eda_scr_amp_high starts valence_medium` | 0.5690 | 0.5690 | 0.5455 |
| `(eda_max_high)` | `eda_max_high equals valence_medium` | 0.5630 | 0.5630 | 0.5537 |
| `(hrv_cvsd_high)` | `hrv_cvsd_high meets valence_medium` | 0.5625 | 0.5625 | 0.4463 |
| `(valence_medium)` | `valence_medium starts eda_std_high` | 0.5620 | 0.5763 | 0.5620 |
| `(valence_low)` | `valence_low is-finished-by eda_scr_auc_high` | 0.5607 | 0.5654 | 0.4959 |
| `(arousal_medium)` | `arousal_medium starts eda_std_high` | 0.5583 | 0.5725 | 0.5537 |
| `(eda_scr_auc_high)` | `eda_scr_auc_high starts arousal_medium` | 0.5583 | 0.5630 | 0.5537 |
| `(eda_peaks_high)` | `eda_peaks_high meets arousal_medium` | 0.5537 | 0.5583 | 0.5537 |
| `(valence_medium)` | `valence_medium is-finished-by eda_scr_amp_low` | 0.5537 | 0.6147 | 0.5537 |
| `(arousal_medium)` | `arousal_medium is-finished-by eda_scr_amp_low` | 0.5500 | 0.6106 | 0.5455 |
| `(arousal_high)` | `arousal_high equals valence_medium` | 0.5495 | 0.5495 | 0.5041 |
| `(arousal_high)` | `arousal_high is-finished-by eda_std_medium` | 0.5495 | 0.5782 | 0.5041 |
| `(eda_max_high)` | `eda_max_high meets arousal_medium` | 0.5462 | 0.5508 | 0.5372 |
| `(eda_scr_auc_medium)` | `eda_scr_auc_medium starts arousal_medium` | 0.5455 | 0.5500 | 0.4959 |
| `(valence_high)` | `valence_high is-finished-by eda_scr_auc_low` | 0.5417 | 0.6554 | 0.4298 |
| `(arousal_high)` | `arousal_high before eda_std_high` | 0.5405 | 0.5543 | 0.4959 |
| `(eda_peaks_low)` | `eda_peaks_low starts arousal_medium` | 0.5385 | 0.5429 | 0.5207 |
| `(eda_peaks_medium)` | `eda_peaks_medium meets arousal_medium` | 0.5385 | 0.5429 | 0.5207 |
| `(arousal_medium)` | `arousal_medium contains eda_scr_amp_high` | 0.5333 | 0.5563 | 0.5289 |
| `(arousal_medium)` | `arousal_medium is-finished-by eda_scr_auc_low` | 0.5333 | 0.6453 | 0.5289 |
| `(arousal_low)` | `arousal_low before eda_std_low` | 0.5288 | 0.5925 | 0.4545 |
| `(hr_low)` | `hr_low starts arousal_medium` | 0.5244 | 0.5288 | 0.3554 |
| `(valence_low)` | `valence_low is-finished-by eda_std_high` | 0.5234 | 0.5367 | 0.4628 |
| `(valence_low)` | `valence_low before eda_scr_auc_medium` | 0.5140 | 0.5654 | 0.4545 |
| `(valence_medium)` | `valence_medium contains eda_scr_amp_high` | 0.5124 | 0.5345 | 0.5124 |
| `(arousal_medium)` | `arousal_medium before eda_max_low` | 0.5083 | 0.5169 | 0.5041 |
| `(arousal_high)` | `arousal_high equals eda_scr_auc_high` | 0.5045 | 0.5087 | 0.4628 |
| `(arousal_low)` | `arousal_low before eda_scr_auc_low` | 0.5000 | 0.6050 | 0.4298 |
| `(arousal_low)` | `arousal_low contains eda_std_high` | 0.5000 | 0.5127 | 0.4298 |
| `(arousal_low)` | `arousal_low overlaps eda_scr_amp_low` | 0.5000 | 0.5550 | 0.4298 |
| `(eda_std_low)` | `eda_std_low meets arousal_medium` | 0.5000 | 0.5042 | 0.4463 |
| `(eda_std_low)` | `eda_std_low overlaps valence_medium` | 0.5000 | 0.5000 | 0.4463 |