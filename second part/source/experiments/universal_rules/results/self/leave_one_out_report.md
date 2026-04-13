# Leave-One-Out Universal Rules Report

This report lists the temporal patterns (rules) that are **almost universal** – they occur with high confidence in exactly 3 out of 4 datasets, but are completely absent in the 4th dataset.

## Summary of Missing Rules per Dataset

We analyzed 4 datasets. A total of **30** rules were identified as present in exactly 3 datasets.

- **CEAP** is the outlier for **18** rules.
- **K-emoCon** is the outlier for **9** rules.
- **EmoWorker_v2** is the outlier for **2** rules.
- **CASE** is the outlier for **1** rules.

## Top Rules Missing per Dataset

The following tables show the top rules missing in each dataset, sorted by their average confidence in the 3 datasets where they *do* appear.

### Missing in: CASE

| Rule | Avg Conf | Avg Lift | Avg Sup | Formed By |
|---|---|---|---|---|
| `(valence_low) => valence_low before eda_peaks_high` | 0.7469 | 0.7469 | 0.6449 | EmoWorker_v2, K-emoCon, CEAP |

### Missing in: CEAP

| Rule | Avg Conf | Avg Lift | Avg Sup | Formed By |
|---|---|---|---|---|
| `(hrv_hfn_high) => hrv_hfn_high meets valence_medium` | 0.7235 | 0.7235 | 0.5595 | CASE, EmoWorker_v2, K-emoCon |
| `(hrv_hfn_medium) => hrv_hfn_medium meets valence_medium` | 0.7154 | 0.7154 | 0.4571 | CASE, EmoWorker_v2, K-emoCon |
| `(arousal_low) => arousal_low meets eda_scr_auc_medium` | 0.7134 | 0.7316 | 0.6309 | CASE, EmoWorker_v2, K-emoCon |
| `(hrv_pnn20_high) => hrv_pnn20_high meets arousal_medium` | 0.6952 | 0.7040 | 0.6580 | CASE, EmoWorker_v2, K-emoCon |
| `(hrv_cvnn_high) => hrv_cvnn_high meets valence_medium` | 0.6909 | 0.6909 | 0.6603 | CASE, EmoWorker_v2, K-emoCon |
| `(hrv_pnn20_medium) => hrv_pnn20_medium meets valence_medium` | 0.6833 | 0.6833 | 0.5945 | CASE, EmoWorker_v2, K-emoCon |
| `(hrv_pnn50_high) => hrv_pnn50_high meets arousal_medium` | 0.6827 | 0.6898 | 0.6491 | CASE, EmoWorker_v2, K-emoCon |
| `(hrv_pnn50_high) => hrv_pnn50_high meets valence_medium` | 0.6819 | 0.6819 | 0.6495 | CASE, EmoWorker_v2, K-emoCon |
| `(hrv_cvnn_high) => hrv_cvnn_high meets arousal_medium` | 0.6671 | 0.6748 | 0.6373 | CASE, EmoWorker_v2, K-emoCon |
| `(hrv_sdnn_high) => hrv_sdnn_high meets valence_medium` | 0.6596 | 0.6596 | 0.6399 | CASE, EmoWorker_v2, K-emoCon |
| `(hrv_rmssd_low) => hrv_rmssd_low meets valence_medium` | 0.6575 | 0.6575 | 0.6042 | CASE, EmoWorker_v2, K-emoCon |
| `(hrv_cvsd_low) => hrv_cvsd_low meets valence_medium` | 0.6519 | 0.6519 | 0.5819 | CASE, EmoWorker_v2, K-emoCon |
| `(hrv_hfn_low) => hrv_hfn_low meets valence_medium` | 0.6426 | 0.6426 | 0.5409 | CASE, EmoWorker_v2, K-emoCon |
| `(hrv_pnn50_low) => hrv_pnn50_low meets valence_medium` | 0.6414 | 0.6414 | 0.5930 | CASE, EmoWorker_v2, K-emoCon |
| `(hrv_cvnn_low) => hrv_cvnn_low meets valence_medium` | 0.6303 | 0.6303 | 0.5819 | CASE, EmoWorker_v2, K-emoCon |
| `(hrv_rmssd_high) => hrv_rmssd_high meets valence_medium` | 0.6284 | 0.6284 | 0.5832 | CASE, EmoWorker_v2, K-emoCon |
| `(hrv_hfn_low) => hrv_hfn_low meets arousal_medium` | 0.6194 | 0.6257 | 0.5263 | CASE, EmoWorker_v2, K-emoCon |
| `(hrv_sdnn_low) => hrv_sdnn_low meets valence_medium` | 0.6061 | 0.6061 | 0.5705 | CASE, EmoWorker_v2, K-emoCon |

### Missing in: EmoWorker_v2

| Rule | Avg Conf | Avg Lift | Avg Sup | Formed By |
|---|---|---|---|---|
| `(eda_medium) => eda_medium meets valence_medium` | 0.7256 | 0.7256 | 0.6536 | CASE, K-emoCon, CEAP |
| `(arousal_medium) => arousal_medium meets eda_scr_auc_medium` | 0.7019 | 0.7877 | 0.7019 | CASE, K-emoCon, CEAP |

### Missing in: K-emoCon

| Rule | Avg Conf | Avg Lift | Avg Sup | Formed By |
|---|---|---|---|---|
| `(arousal_medium) => arousal_medium equals valence_medium` | 0.8014 | 0.8014 | 0.7924 | CASE, EmoWorker_v2, CEAP |
| `(eda_medium) => eda_medium meets arousal_medium` | 0.7889 | 0.7963 | 0.7234 | CASE, EmoWorker_v2, CEAP |
| `(valence_high) => valence_high meets arousal_medium` | 0.7535 | 0.7616 | 0.5825 | CASE, EmoWorker_v2, CEAP |
| `(eda_std_medium) => eda_std_medium meets arousal_medium` | 0.7291 | 0.7359 | 0.6786 | CASE, EmoWorker_v2, CEAP |
| `(hrv_sdnn_medium) => hrv_sdnn_medium meets valence_medium` | 0.7190 | 0.7190 | 0.5123 | CASE, EmoWorker_v2, CEAP |
| `(eda_max_medium) => eda_max_medium before valence_medium` | 0.6838 | 0.6838 | 0.6231 | CASE, EmoWorker_v2, CEAP |
| `(hrv_cvnn_medium) => hrv_cvnn_medium meets valence_medium` | 0.6765 | 0.6765 | 0.5120 | CASE, EmoWorker_v2, CEAP |
| `(hrv_cvsd_medium) => hrv_cvsd_medium meets valence_medium` | 0.6753 | 0.6753 | 0.5123 | CASE, EmoWorker_v2, CEAP |
| `(arousal_high) => arousal_high meets eda_scr_amp_high` | 0.6051 | 0.6448 | 0.5599 | CASE, EmoWorker_v2, CEAP |
