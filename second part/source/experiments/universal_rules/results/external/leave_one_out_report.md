# Leave-One-Out Universal Rules Report

This report lists the temporal patterns (rules) that are **almost universal** – they occur with high confidence in exactly 3 out of 4 datasets, but are completely absent in the 4th dataset.

## Summary of Missing Rules per Dataset

We analyzed 2 datasets. A total of **225** rules were identified as present in exactly 3 datasets.

- **EMBOA** is the outlier for **133** rules.
- **K-emo_ext** is the outlier for **92** rules.

## Top Rules Missing per Dataset

The following tables show the top rules missing in each dataset, sorted by their average confidence in the 3 datasets where they *do* appear.

### Missing in: EMBOA

| Rule | Avg Conf | Avg Lift | Avg Sup | Formed By |
|---|---|---|---|---|
| `(hrv_cvsd_low) => hrv_cvsd_low meets valence_medium` | 1.0000 | 1.0000 | 0.8571 | K-emo_ext |
| `(hrv_pnn20_medium) => hrv_pnn20_medium meets valence_medium` | 1.0000 | 1.0000 | 0.8571 | K-emo_ext |
| `(hrv_sdnn_low) => hrv_sdnn_low meets valence_medium` | 1.0000 | 1.0000 | 0.8214 | K-emo_ext |
| `(hrv_cvnn_low) => hrv_cvnn_low meets valence_medium` | 1.0000 | 1.0000 | 0.8571 | K-emo_ext |
| `(hrv_rmssd_low) => hrv_rmssd_low meets valence_medium` | 1.0000 | 1.0000 | 0.8571 | K-emo_ext |
| `(hrv_pnn50_medium) => hrv_pnn50_medium meets valence_medium` | 0.9565 | 0.9565 | 0.7857 | K-emo_ext |
| `(hrv_pnn50_low) => hrv_pnn50_low meets valence_medium` | 0.9545 | 0.9545 | 0.7500 | K-emo_ext |
| `(hrv_pnn20_low) => hrv_pnn20_low meets valence_medium` | 0.9545 | 0.9545 | 0.7500 | K-emo_ext |
| `(eda_scr_amp_high) => eda_scr_amp_high starts valence_medium` | 0.9286 | 0.9286 | 0.9286 | K-emo_ext |
| `(eda_scr_auc_high) => eda_scr_auc_high starts valence_medium` | 0.9286 | 0.9286 | 0.9286 | K-emo_ext |
| `(hrv_cvnn_medium) => hrv_cvnn_medium meets valence_medium` | 0.9200 | 0.9200 | 0.8214 | K-emo_ext |
| `(hrv_rmssd_medium) => hrv_rmssd_medium meets valence_medium` | 0.9130 | 0.9130 | 0.7500 | K-emo_ext |
| `(hrv_cvsd_medium) => hrv_cvsd_medium meets valence_medium` | 0.9130 | 0.9130 | 0.7500 | K-emo_ext |
| `(hrv_sdnn_medium) => hrv_sdnn_medium meets valence_medium` | 0.9091 | 0.9091 | 0.7143 | K-emo_ext |
| `(arousal_medium) => arousal_medium meets eda_scr_auc_low` | 0.8929 | 0.8929 | 0.8929 | K-emo_ext |
| `(valence_medium) => valence_medium contains eda_std_medium` | 0.8929 | 0.8929 | 0.8929 | K-emo_ext |
| `(valence_medium) => valence_medium is-finished-by eda_scr_auc_low` | 0.8929 | 0.8929 | 0.8929 | K-emo_ext |
| `(eda_max_medium) => eda_max_medium equals valence_medium` | 0.8846 | 0.8846 | 0.8214 | K-emo_ext |
| `(eda_medium) => eda_medium meets valence_medium` | 0.8750 | 0.8750 | 0.7500 | K-emo_ext |
| `(arousal_medium) => arousal_medium starts eda_std_medium` | 0.8571 | 0.8571 | 0.8571 | K-emo_ext |

### Missing in: K-emo_ext

| Rule | Avg Conf | Avg Lift | Avg Sup | Formed By |
|---|---|---|---|---|
| `(arousal_low) => arousal_low equals valence_low` | 1.0000 | 1.3333 | 0.4375 | EMBOA |
| `(hr_low) => hr_low starts arousal_medium` | 0.9375 | 0.9375 | 0.9375 | EMBOA |
| `(arousal_medium) => arousal_medium contains hr_low` | 0.9375 | 0.9375 | 0.9375 | EMBOA |
| `(eda_high) => eda_high starts valence_medium` | 0.9375 | 0.9375 | 0.9375 | EMBOA |
| `(arousal_medium) => arousal_medium equals valence_medium` | 0.9375 | 0.9375 | 0.9375 | EMBOA |
| `(arousal_medium) => arousal_medium contains eda_low` | 0.8750 | 0.8750 | 0.8750 | EMBOA |
| `(eda_std_medium) => eda_std_medium meets valence_medium` | 0.8750 | 0.8750 | 0.8750 | EMBOA |
| `(arousal_medium) => arousal_medium is-finished-by eda_std_high` | 0.8750 | 0.8750 | 0.8750 | EMBOA |
| `(eda_max_medium) => eda_max_medium starts valence_medium` | 0.8750 | 0.8750 | 0.8750 | EMBOA |
| `(valence_high) => valence_high meets arousal_medium` | 0.8750 | 0.8750 | 0.8750 | EMBOA |
| `(arousal_low) => arousal_low meets valence_medium` | 0.8571 | 0.8571 | 0.3750 | EMBOA |
| `(arousal_low) => arousal_low equals hr_medium` | 0.8571 | 0.8571 | 0.3750 | EMBOA |
| `(valence_low) => valence_low meets arousal_medium` | 0.8333 | 0.8333 | 0.6250 | EMBOA |
| `(hr_medium) => hr_medium equals valence_medium` | 0.8125 | 0.8125 | 0.8125 | EMBOA |
| `(hr_high) => hr_high before arousal_medium` | 0.8125 | 0.8125 | 0.8125 | EMBOA |
| `(eda_std_medium) => eda_std_medium meets arousal_medium` | 0.8125 | 0.8125 | 0.8125 | EMBOA |
| `(eda_std_low) => eda_std_low starts arousal_medium` | 0.8125 | 0.8125 | 0.8125 | EMBOA |
| `(arousal_medium) => arousal_medium meets valence_high` | 0.8125 | 0.8125 | 0.8125 | EMBOA |
| `(hr_high) => hr_high before valence_medium` | 0.8125 | 0.8125 | 0.8125 | EMBOA |
| `(arousal_medium) => arousal_medium contains eda_max_low` | 0.8125 | 0.8125 | 0.8125 | EMBOA |
