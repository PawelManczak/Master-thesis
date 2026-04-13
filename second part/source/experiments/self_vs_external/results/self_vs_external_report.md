# Self vs External Annotations — Rule Comparison

Comparison of ARMADA rules discovered across **self-annotated** and
**externally-annotated** physiological datasets.

**Self-annotated datasets**: CASE, K-emoCon, CEAP, EmoWorker_v2

**External-annotated datasets**: K-emoCon (ext), EMBOA

## Parameters

| Parameter | Value |
|-----------|-------|
| minsup | 0.5 (50%) |
| minconf | 0.5 (50%) |
| maxgap | 5s |
| max_pattern_size | 2 |

## Per-Dataset Statistics

### Self-Annotated Datasets

| Dataset | N | Raw Rules | Filtered Rules |
|---------|---|-----------|----------------|
| CASE | 30 | 516 | 140 |
| K-emoCon | 28 | 383 | 82 |
| CEAP | 32 | 254 | 114 |
| EmoWorker_v2 | 31 | 412 | 108 |

### External-Annotated Datasets

| Dataset | N | Raw Rules | Filtered Rules |
|---------|---|-----------|----------------|
| K-emoCon (ext) | 28 | 382 | 115 |
| EMBOA | 16 | 249 | 80 |

## Overall Comparison (Union of All Rules)

| Metric | Value |
|--------|-------|
| Total unique self rules | 348 |
| Total unique external rules | 192 |
| Shared rules | **95** |
| Self-only rules | 253 |
| External-only rules | 97 |
| Jaccard similarity | 0.213 |

## K-emoCon Controlled Comparison

Same physiological data (Empatica E4), different annotation source.

| Metric | Self | External |
|--------|------|----------|
| Filtered rules | 82 | 115 |
| Shared | \multicolumn{2}{c|}{19} |
| Self-only | 63 | — |
| External-only | — | 96 |
| Jaccard similarity | \multicolumn{2}{c|}{0.107} |

### K-emoCon Shared Rules

| Rule | Self Conf | Self Lift | Ext Conf | Ext Lift | Δ Conf |
|------|----------|-----------|---------|----------|---|
| `(hrv_rmssd_high) => hrv_rmssd_high meets valence_medium` | 0.519 | 0.519 | 0.826 | 0.826 | +0.308 |
| `(hrv_cvnn_high) => hrv_cvnn_high meets valence_medium` | 0.556 | 0.556 | 0.857 | 0.857 | +0.302 |
| `(eda_medium) => eda_medium meets valence_medium` | 0.577 | 0.577 | 0.875 | 0.875 | +0.298 |
| `(hrv_pnn50_high) => hrv_pnn50_high meets valence_medium` | 0.538 | 0.538 | 0.833 | 0.833 | +0.295 |
| `(eda_high) => eda_high before valence_medium` | 0.571 | 0.571 | 0.857 | 0.857 | +0.286 |
| `(eda_scr_amp_high) => eda_scr_amp_high starts valence_medium` | 0.643 | 0.643 | 0.929 | 0.929 | +0.286 |
| `(arousal_low) => arousal_low meets eda_scr_auc_medium` | 0.833 | 0.864 | 0.571 | 0.615 | -0.262 |
| `(hrv_sdnn_high) => hrv_sdnn_high meets valence_medium` | 0.593 | 0.593 | 0.826 | 0.826 | +0.233 |
| `(arousal_low) => arousal_low is-finished-by eda_peaks_high` | 0.833 | 0.833 | 0.607 | 0.607 | -0.226 |
| `(eda_peaks_high) => eda_peaks_high meets valence_medium` | 0.571 | 0.571 | 0.786 | 0.786 | +0.214 |
| `(eda_max_medium) => eda_max_medium equals valence_medium` | 0.679 | 0.679 | 0.885 | 0.885 | +0.206 |
| `(hrv_cvnn_high) => hrv_cvnn_high meets arousal_medium` | 0.519 | 0.519 | 0.714 | 0.714 | +0.196 |
| `(valence_medium) => valence_medium is-finished-by eda_peaks_high` | 0.750 | 0.750 | 0.607 | 0.607 | -0.143 |
| `(hrv_pnn50_high) => hrv_pnn50_high meets arousal_medium` | 0.538 | 0.538 | 0.625 | 0.625 | +0.087 |
| `(arousal_low) => arousal_low contains eda_std_medium` | 0.833 | 0.833 | 0.750 | 0.750 | -0.083 |
| `(arousal_low) => arousal_low is-finished-by eda_scr_auc_low` | 0.667 | 0.667 | 0.607 | 0.607 | -0.060 |
| `(valence_medium) => valence_medium meets eda_peaks_medium` | 0.786 | 0.815 | 0.750 | 0.840 | -0.036 |
| `(valence_medium) => valence_medium is-finished-by eda_std_low` | 0.536 | 0.536 | 0.571 | 0.571 | +0.036 |
| `(valence_medium) => valence_medium meets eda_scr_auc_medium` | 0.750 | 0.778 | 0.750 | 0.808 | +0.000 |

- Mean Δ confidence (ext − self): **+0.102**
- Rules with higher confidence in external: 11
- Rules with higher confidence in self: 5
- Rules with similar confidence (|Δ| ≤ 0.05): 3

## Pairwise Jaccard Similarity Matrix

| | CASE (self) | K-emoCon (self) | CEAP (self) | EmoWorker_v2 (self) | K-emoCon (ext) | EMBOA |
|---|---|---|---|---|---|---|
| **CASE (self)** | 1.000 | 0.062 | 0.050 | 0.246 | 0.214 | 0.078 |
| **K-emoCon (self)** | 0.062 | 1.000 | 0.037 | 0.073 | 0.107 | 0.032 |
| **CEAP (self)** | 0.050 | 0.037 | 1.000 | 0.083 | 0.050 | 0.078 |
| **EmoWorker_v2 (self)** | 0.246 | 0.073 | 0.083 | 1.000 | 0.205 | 0.022 |
| **K-emoCon (ext)** | 0.214 | 0.107 | 0.050 | 0.205 | 1.000 | 0.016 |
| **EMBOA** | 0.078 | 0.032 | 0.078 | 0.022 | 0.016 | 1.000 |

## Shared Rules (present in both self and external unions)

Found **95** rules in common.

| Rule | Present in (self) | Present in (ext) |
|------|-------------------|------------------|
| `(arousal_high) => arousal_high before eda_max_low` | K-emoCon | EMBOA |
| `(arousal_high) => arousal_high equals eda_peaks_medium` | CEAP, EmoWorker_v2 | K-emoCon (ext) |
| `(arousal_high) => arousal_high equals hr_medium` | K-emoCon | EMBOA |
| `(arousal_high) => arousal_high equals valence_high` | K-emoCon, CEAP | EMBOA |
| `(arousal_high) => arousal_high meets eda_std_high` | CEAP, EmoWorker_v2 | EMBOA |
| `(arousal_high) => arousal_high meets valence_medium` | CEAP | K-emoCon (ext), EMBOA |
| `(arousal_low) => arousal_low contains eda_std_medium` | K-emoCon | K-emoCon (ext) |
| `(arousal_low) => arousal_low is-finished-by eda_peaks_high` | K-emoCon | K-emoCon (ext) |
| `(arousal_low) => arousal_low is-finished-by eda_scr_auc_low` | K-emoCon | K-emoCon (ext) |
| `(arousal_low) => arousal_low meets eda_peaks_medium` | EmoWorker_v2 | K-emoCon (ext) |
| `(arousal_low) => arousal_low meets eda_scr_auc_medium` | CASE, K-emoCon, EmoWorker_v2 | K-emoCon (ext) |
| `(arousal_medium) => arousal_medium contains hr_low` | CASE | EMBOA |
| `(arousal_medium) => arousal_medium contains hr_medium` | CASE | EMBOA |
| `(arousal_medium) => arousal_medium equals valence_medium` | CASE, CEAP, EmoWorker_v2 | EMBOA |
| `(arousal_medium) => arousal_medium is-finished-by eda_high` | CEAP | K-emoCon (ext) |
| `(arousal_medium) => arousal_medium is-finished-by eda_scr_amp_low` | CASE | K-emoCon (ext) |
| `(arousal_medium) => arousal_medium meets eda_std_low` | CASE | EMBOA |
| `(arousal_medium) => arousal_medium meets valence_high` | CEAP | EMBOA |
| `(eda_high) => eda_high before arousal_medium` | EmoWorker_v2 | K-emoCon (ext) |
| `(eda_high) => eda_high before valence_medium` | K-emoCon | K-emoCon (ext) |
| `(eda_high) => eda_high starts arousal_medium` | CASE | EMBOA |
| `(eda_high) => eda_high starts valence_medium` | CASE | EMBOA |
| `(eda_low) => eda_low before arousal_medium` | CEAP | K-emoCon (ext) |
| `(eda_low) => eda_low meets valence_medium` | CASE | EMBOA |
| `(eda_max_high) => eda_max_high before valence_medium` | CEAP | K-emoCon (ext) |
| `(eda_max_medium) => eda_max_medium equals valence_medium` | K-emoCon | K-emoCon (ext) |
| `(eda_max_medium) => eda_max_medium meets arousal_medium` | CASE, CEAP | K-emoCon (ext) |
| `(eda_medium) => eda_medium meets arousal_medium` | CASE, CEAP, EmoWorker_v2 | K-emoCon (ext) |
| `(eda_medium) => eda_medium meets valence_medium` | CASE, K-emoCon, CEAP | K-emoCon (ext) |
| `(eda_peaks_high) => eda_peaks_high meets arousal_medium` | CASE, EmoWorker_v2 | K-emoCon (ext) |
| `(eda_peaks_high) => eda_peaks_high meets valence_medium` | K-emoCon | K-emoCon (ext) |
| `(eda_peaks_low) => eda_peaks_low starts valence_medium` | CASE | K-emoCon (ext) |
| `(eda_peaks_medium) => eda_peaks_medium starts arousal_medium` | EmoWorker_v2 | K-emoCon (ext) |
| `(eda_peaks_medium) => eda_peaks_medium starts valence_medium` | EmoWorker_v2 | K-emoCon (ext) |
| `(eda_scr_amp_high) => eda_scr_amp_high starts valence_medium` | K-emoCon | K-emoCon (ext) |
| `(eda_scr_amp_low) => eda_scr_amp_low meets arousal_medium` | CEAP, EmoWorker_v2 | K-emoCon (ext) |
| `(eda_scr_auc_high) => eda_scr_auc_high starts valence_medium` | CASE | K-emoCon (ext) |
| `(eda_scr_auc_low) => eda_scr_auc_low meets arousal_medium` | CASE | K-emoCon (ext) |
| `(eda_scr_auc_medium) => eda_scr_auc_medium before arousal_medium` | EmoWorker_v2 | K-emoCon (ext) |
| `(eda_scr_auc_medium) => eda_scr_auc_medium starts valence_medium` | CASE | K-emoCon (ext) |
| `(eda_std_high) => eda_std_high meets arousal_medium` | CEAP | K-emoCon (ext) |
| `(eda_std_high) => eda_std_high starts arousal_medium` | CASE, EmoWorker_v2 | EMBOA |
| `(eda_std_high) => eda_std_high starts valence_medium` | CASE | K-emoCon (ext), EMBOA |
| `(eda_std_medium) => eda_std_medium meets arousal_medium` | CASE, CEAP, EmoWorker_v2 | EMBOA |
| `(eda_std_medium) => eda_std_medium meets valence_medium` | CASE | EMBOA |
| `(hr_high) => hr_high before valence_medium` | CEAP | EMBOA |
| `(hr_high) => hr_high meets arousal_medium` | CEAP | K-emoCon (ext) |
| `(hr_low) => hr_low starts arousal_medium` | CASE | EMBOA |
| `(hr_low) => hr_low starts valence_medium` | CASE | EMBOA |
| `(hr_medium) => hr_medium equals valence_medium` | CEAP | EMBOA |
| `(hrv_cvnn_high) => hrv_cvnn_high meets arousal_medium` | CASE, K-emoCon, EmoWorker_v2 | K-emoCon (ext) |
| `(hrv_cvnn_high) => hrv_cvnn_high meets valence_medium` | CASE, K-emoCon, EmoWorker_v2 | K-emoCon (ext) |
| `(hrv_cvnn_low) => hrv_cvnn_low meets valence_medium` | CASE, EmoWorker_v2 | K-emoCon (ext) |
| `(hrv_cvnn_medium) => hrv_cvnn_medium meets arousal_medium` | CASE, EmoWorker_v2 | K-emoCon (ext) |
| `(hrv_cvnn_medium) => hrv_cvnn_medium meets valence_medium` | CASE, EmoWorker_v2 | K-emoCon (ext) |
| `(hrv_cvsd_high) => hrv_cvsd_high meets arousal_medium` | CASE, EmoWorker_v2 | K-emoCon (ext) |
| `(hrv_cvsd_high) => hrv_cvsd_high meets valence_medium` | CASE, EmoWorker_v2 | K-emoCon (ext) |
| `(hrv_cvsd_low) => hrv_cvsd_low meets arousal_medium` | CASE, EmoWorker_v2 | K-emoCon (ext) |
| `(hrv_cvsd_low) => hrv_cvsd_low meets valence_medium` | CASE, EmoWorker_v2 | K-emoCon (ext) |
| `(hrv_cvsd_medium) => hrv_cvsd_medium meets valence_medium` | CASE, EmoWorker_v2 | K-emoCon (ext) |
| `(hrv_pnn20_high) => hrv_pnn20_high meets arousal_medium` | CASE, EmoWorker_v2 | K-emoCon (ext) |
| `(hrv_pnn20_high) => hrv_pnn20_high meets valence_medium` | CASE, EmoWorker_v2 | K-emoCon (ext) |
| `(hrv_pnn20_low) => hrv_pnn20_low meets valence_medium` | CASE, EmoWorker_v2 | K-emoCon (ext) |
| `(hrv_pnn20_medium) => hrv_pnn20_medium meets arousal_medium` | CASE | K-emoCon (ext) |
| `(hrv_pnn20_medium) => hrv_pnn20_medium meets valence_medium` | CASE | K-emoCon (ext) |
| `(hrv_pnn50_high) => hrv_pnn50_high meets arousal_medium` | CASE, K-emoCon, EmoWorker_v2 | K-emoCon (ext) |
| `(hrv_pnn50_high) => hrv_pnn50_high meets valence_medium` | CASE, K-emoCon, EmoWorker_v2 | K-emoCon (ext) |
| `(hrv_pnn50_low) => hrv_pnn50_low meets valence_medium` | CASE, EmoWorker_v2 | K-emoCon (ext) |
| `(hrv_pnn50_medium) => hrv_pnn50_medium meets valence_medium` | CASE, EmoWorker_v2 | K-emoCon (ext) |
| `(hrv_rmssd_high) => hrv_rmssd_high meets arousal_medium` | CASE, EmoWorker_v2 | K-emoCon (ext) |
| `(hrv_rmssd_high) => hrv_rmssd_high meets valence_medium` | CASE, K-emoCon, EmoWorker_v2 | K-emoCon (ext) |
| `(hrv_rmssd_low) => hrv_rmssd_low meets arousal_medium` | CASE, EmoWorker_v2 | K-emoCon (ext) |
| `(hrv_rmssd_low) => hrv_rmssd_low meets valence_medium` | CASE, EmoWorker_v2 | K-emoCon (ext) |
| `(hrv_rmssd_medium) => hrv_rmssd_medium meets arousal_medium` | CASE, EmoWorker_v2 | K-emoCon (ext) |
| `(hrv_rmssd_medium) => hrv_rmssd_medium meets valence_medium` | CASE, EmoWorker_v2 | K-emoCon (ext) |
| `(hrv_sdnn_high) => hrv_sdnn_high meets arousal_medium` | CASE, EmoWorker_v2 | K-emoCon (ext) |
| `(hrv_sdnn_high) => hrv_sdnn_high meets valence_medium` | CASE, K-emoCon, EmoWorker_v2 | K-emoCon (ext) |
| `(hrv_sdnn_low) => hrv_sdnn_low meets valence_medium` | CASE, EmoWorker_v2 | K-emoCon (ext) |
| `(hrv_sdnn_medium) => hrv_sdnn_medium meets arousal_medium` | CASE, EmoWorker_v2 | K-emoCon (ext) |
| `(hrv_sdnn_medium) => hrv_sdnn_medium meets valence_medium` | CASE, EmoWorker_v2 | K-emoCon (ext) |
| `(valence_high) => valence_high before eda_max_low` | K-emoCon, CEAP | EMBOA |
| `(valence_high) => valence_high meets arousal_medium` | CASE, CEAP | EMBOA |
| `(valence_high) => valence_high meets hr_medium` | CASE | EMBOA |
| `(valence_medium) => valence_medium before arousal_medium` | CEAP | EMBOA |
| `(valence_medium) => valence_medium contains hr_medium` | CASE | EMBOA |
| `(valence_medium) => valence_medium is-finished-by eda_peaks_high` | K-emoCon | K-emoCon (ext) |
| `(valence_medium) => valence_medium is-finished-by eda_scr_amp_low` | CASE | K-emoCon (ext) |
| `(valence_medium) => valence_medium is-finished-by eda_scr_auc_low` | CASE | K-emoCon (ext) |
| `(valence_medium) => valence_medium is-finished-by eda_std_low` | K-emoCon | K-emoCon (ext) |
| `(valence_medium) => valence_medium meets arousal_medium` | CASE | K-emoCon (ext) |
| `(valence_medium) => valence_medium meets eda_low` | CEAP | EMBOA |
| `(valence_medium) => valence_medium meets eda_max_low` | K-emoCon, CEAP | EMBOA |
| `(valence_medium) => valence_medium meets eda_peaks_medium` | K-emoCon | K-emoCon (ext) |
| `(valence_medium) => valence_medium meets eda_scr_auc_medium` | CASE, K-emoCon | K-emoCon (ext) |
| `(valence_medium) => valence_medium meets eda_std_medium` | CEAP | EMBOA |

## Self-Only Rules

**253** rules found only in self-annotated datasets.

- `(arousal_high) => arousal_high before eda_peaks_medium` (in: K-emoCon)
- `(arousal_high) => arousal_high before eda_scr_auc_low` (in: CASE)
- `(arousal_high) => arousal_high before eda_std_high` (in: CASE)
- `(arousal_high) => arousal_high before hr_medium` (in: CASE)
- `(arousal_high) => arousal_high before valence_medium` (in: K-emoCon)
- `(arousal_high) => arousal_high contains eda_peaks_medium` (in: CASE)
- `(arousal_high) => arousal_high contains eda_scr_auc_low` (in: K-emoCon)
- `(arousal_high) => arousal_high equals eda_high` (in: CEAP)
- `(arousal_high) => arousal_high equals eda_low` (in: CEAP)
- `(arousal_high) => arousal_high equals eda_max_low` (in: CEAP)
- `(arousal_high) => arousal_high equals eda_scr_amp_low` (in: K-emoCon)
- `(arousal_high) => arousal_high equals eda_scr_auc_high` (in: CASE)
- `(arousal_high) => arousal_high equals eda_scr_auc_medium` (in: K-emoCon, EmoWorker_v2)
- `(arousal_high) => arousal_high equals eda_std_medium` (in: CEAP, EmoWorker_v2)
- `(arousal_high) => arousal_high equals hr_high` (in: CEAP)
- `(arousal_high) => arousal_high equals valence_low` (in: EmoWorker_v2)
- `(arousal_high) => arousal_high equals valence_medium` (in: CASE)
- `(arousal_high) => arousal_high is-finished-by eda_scr_auc_medium` (in: CASE)
- `(arousal_high) => arousal_high is-finished-by eda_std_medium` (in: K-emoCon)
- `(arousal_high) => arousal_high meets eda_max_high` (in: CEAP)
- `(arousal_high) => arousal_high meets eda_peaks_high` (in: K-emoCon, EmoWorker_v2)
- `(arousal_high) => arousal_high meets eda_scr_amp_high` (in: CASE, CEAP, EmoWorker_v2)
- `(arousal_high) => arousal_high meets eda_scr_auc_high` (in: CEAP, EmoWorker_v2)
- `(arousal_high) => arousal_high meets valence_low` (in: CEAP)
- `(arousal_high) => arousal_high starts eda_peaks_high` (in: CEAP)
- `(arousal_low) => arousal_low before eda_high` (in: CEAP)
- `(arousal_low) => arousal_low before eda_max_high` (in: CEAP)
- `(arousal_low) => arousal_low before eda_peaks_high` (in: EmoWorker_v2)
- `(arousal_low) => arousal_low before eda_scr_auc_low` (in: CASE)
- `(arousal_low) => arousal_low before eda_std_low` (in: CASE)
- `(arousal_low) => arousal_low before valence_medium` (in: CASE)
- `(arousal_low) => arousal_low contains eda_peaks_high` (in: CASE)
- `(arousal_low) => arousal_low contains eda_peaks_medium` (in: K-emoCon)
- `(arousal_low) => arousal_low equals eda_max_low` (in: K-emoCon)
- `(arousal_low) => arousal_low equals eda_scr_auc_high` (in: EmoWorker_v2)
- `(arousal_low) => arousal_low equals hr_high` (in: CEAP)
- `(arousal_low) => arousal_low equals valence_high` (in: CEAP)
- `(arousal_low) => arousal_low equals valence_low` (in: CEAP)
- `(arousal_low) => arousal_low is-finished-by eda_peaks_medium` (in: CASE)
- `(arousal_low) => arousal_low is-finished-by eda_scr_amp_high` (in: EmoWorker_v2)
- `(arousal_low) => arousal_low is-finished-by eda_scr_amp_low` (in: K-emoCon)
- `(arousal_low) => arousal_low is-finished-by valence_medium` (in: K-emoCon)
- `(arousal_low) => arousal_low meets eda_peaks_high` (in: CEAP)
- `(arousal_low) => arousal_low meets eda_peaks_low` (in: CASE)
- `(arousal_low) => arousal_low meets eda_std_high` (in: EmoWorker_v2)
- `(arousal_low) => arousal_low meets eda_std_medium` (in: CASE, EmoWorker_v2)
- `(arousal_low) => arousal_low meets hr_medium` (in: CEAP)
- `(arousal_low) => arousal_low meets valence_medium` (in: CEAP, EmoWorker_v2)
- `(arousal_low) => arousal_low starts hr_low` (in: CASE)
- `(arousal_medium) => arousal_medium before eda_max_high` (in: CEAP)
- `(arousal_medium) => arousal_medium before eda_max_low` (in: K-emoCon)
- `(arousal_medium) => arousal_medium before eda_scr_amp_high` (in: CEAP)
- `(arousal_medium) => arousal_medium before eda_scr_amp_low` (in: K-emoCon)
- `(arousal_medium) => arousal_medium before eda_scr_auc_high` (in: CEAP)
- `(arousal_medium) => arousal_medium before eda_scr_auc_low` (in: K-emoCon)
- `(arousal_medium) => arousal_medium before eda_std_high` (in: CEAP)
- `(arousal_medium) => arousal_medium before eda_std_medium` (in: K-emoCon)
- `(arousal_medium) => arousal_medium contains eda_peaks_medium` (in: EmoWorker_v2)
- `(arousal_medium) => arousal_medium contains eda_scr_amp_high` (in: CASE, EmoWorker_v2)
- `(arousal_medium) => arousal_medium contains eda_scr_auc_high` (in: CASE)
- `(arousal_medium) => arousal_medium contains eda_scr_auc_medium` (in: EmoWorker_v2)
- `(arousal_medium) => arousal_medium equals eda_high` (in: CASE)
- `(arousal_medium) => arousal_medium equals eda_low` (in: CEAP, EmoWorker_v2)
- `(arousal_medium) => arousal_medium equals eda_max_high` (in: CASE)
- `(arousal_medium) => arousal_medium equals eda_max_low` (in: CEAP, EmoWorker_v2)
- `(arousal_medium) => arousal_medium equals eda_peaks_high` (in: CEAP)
- `(arousal_medium) => arousal_medium equals eda_peaks_medium` (in: CEAP)
- `(arousal_medium) => arousal_medium equals eda_std_medium` (in: CEAP)
- `(arousal_medium) => arousal_medium equals hr_medium` (in: CEAP)
- `(arousal_medium) => arousal_medium is-finished-by eda_peaks_high` (in: EmoWorker_v2)
- `(arousal_medium) => arousal_medium is-finished-by eda_peaks_medium` (in: CASE, K-emoCon)
- `(arousal_medium) => arousal_medium is-finished-by eda_scr_amp_medium` (in: EmoWorker_v2)
- `(arousal_medium) => arousal_medium is-finished-by eda_scr_auc_low` (in: CASE)
- `(arousal_medium) => arousal_medium is-finished-by eda_std_medium` (in: CASE, EmoWorker_v2)
- `(arousal_medium) => arousal_medium is-finished-by hr_medium` (in: K-emoCon)
- `(arousal_medium) => arousal_medium meets eda_peaks_high` (in: CASE)
- `(arousal_medium) => arousal_medium meets eda_scr_amp_medium` (in: CASE)
- `(arousal_medium) => arousal_medium meets eda_scr_auc_high` (in: EmoWorker_v2)
- `(arousal_medium) => arousal_medium meets eda_scr_auc_medium` (in: CASE, K-emoCon, CEAP)
- `(arousal_medium) => arousal_medium meets eda_std_high` (in: EmoWorker_v2)
- `(arousal_medium) => arousal_medium meets hr_high` (in: CEAP)
- `(arousal_medium) => arousal_medium meets valence_medium` (in: K-emoCon)
- `(arousal_medium) => arousal_medium overlaps eda_peaks_high` (in: K-emoCon)
- `(arousal_medium) => arousal_medium overlaps hr_low` (in: K-emoCon)
- `(eda_high) => eda_high equals valence_medium` (in: CEAP)
- `(eda_high) => eda_high meets arousal_medium` (in: CEAP)
- `(eda_high) => eda_high meets valence_high` (in: CEAP)
- `(eda_low) => eda_low before valence_low` (in: CEAP)
- `(eda_low) => eda_low equals valence_medium` (in: CEAP, EmoWorker_v2)
- `(eda_low) => eda_low is-finished-by valence_medium` (in: K-emoCon)
- `(eda_low) => eda_low meets arousal_low` (in: K-emoCon)
- `(eda_low) => eda_low meets arousal_medium` (in: CASE, EmoWorker_v2)
- `(eda_max_high) => eda_max_high equals valence_medium` (in: CASE)
- `(eda_max_high) => eda_max_high meets arousal_medium` (in: CASE, CEAP)
- `(eda_max_high) => eda_max_high meets valence_high` (in: CEAP)
- `(eda_max_low) => eda_max_low before arousal_medium` (in: CASE, CEAP)
- `(eda_max_low) => eda_max_low before valence_low` (in: CEAP)
- `(eda_max_low) => eda_max_low before valence_medium` (in: CASE)
- `(eda_max_low) => eda_max_low equals valence_medium` (in: CEAP, EmoWorker_v2)
- `(eda_max_low) => eda_max_low is-finished-by valence_medium` (in: K-emoCon)
- `(eda_max_low) => eda_max_low meets arousal_high` (in: CEAP)
- `(eda_max_low) => eda_max_low meets arousal_low` (in: K-emoCon)
- `(eda_max_low) => eda_max_low meets arousal_medium` (in: EmoWorker_v2)
- `(eda_max_low) => eda_max_low meets valence_high` (in: CEAP)
- `(eda_max_medium) => eda_max_medium before arousal_medium` (in: K-emoCon, EmoWorker_v2)
- `(eda_max_medium) => eda_max_medium before valence_medium` (in: CASE, CEAP, EmoWorker_v2)
- `(eda_medium) => eda_medium equals valence_high` (in: CEAP)
- `(eda_medium) => eda_medium equals valence_medium` (in: EmoWorker_v2)
- `(eda_peaks_high) => eda_peaks_high before arousal_medium` (in: CEAP)
- `(eda_peaks_high) => eda_peaks_high before valence_high` (in: CEAP)
- `(eda_peaks_high) => eda_peaks_high before valence_medium` (in: EmoWorker_v2)
- `(eda_peaks_high) => eda_peaks_high equals valence_medium` (in: CEAP)
- `(eda_peaks_high) => eda_peaks_high meets arousal_low` (in: K-emoCon)
- `(eda_peaks_low) => eda_peaks_low before valence_medium` (in: K-emoCon)
- `(eda_peaks_low) => eda_peaks_low meets valence_medium` (in: EmoWorker_v2)
- `(eda_peaks_low) => eda_peaks_low starts arousal_medium` (in: CASE, EmoWorker_v2)
- `(eda_peaks_medium) => eda_peaks_medium before arousal_medium` (in: CEAP)
- `(eda_peaks_medium) => eda_peaks_medium before valence_low` (in: CEAP)
- `(eda_peaks_medium) => eda_peaks_medium equals valence_medium` (in: CEAP)
- `(eda_peaks_medium) => eda_peaks_medium meets arousal_medium` (in: CASE)
- `(eda_peaks_medium) => eda_peaks_medium meets valence_high` (in: CEAP)
- `(eda_peaks_medium) => eda_peaks_medium meets valence_medium` (in: CASE, K-emoCon)
- `(eda_scr_amp_high) => eda_scr_amp_high before valence_medium` (in: CEAP)
- `(eda_scr_amp_high) => eda_scr_amp_high equals valence_medium` (in: EmoWorker_v2)
- `(eda_scr_amp_high) => eda_scr_amp_high meets arousal_medium` (in: CEAP, EmoWorker_v2)
- `(eda_scr_amp_high) => eda_scr_amp_high starts arousal_low` (in: K-emoCon)
- `(eda_scr_amp_high) => eda_scr_amp_high starts arousal_medium` (in: CASE)
- `(eda_scr_amp_low) => eda_scr_amp_low equals valence_medium` (in: CEAP)
- `(eda_scr_amp_low) => eda_scr_amp_low is-finished-by valence_medium` (in: K-emoCon)
- `(eda_scr_amp_low) => eda_scr_amp_low meets arousal_low` (in: K-emoCon)
- `(eda_scr_amp_low) => eda_scr_amp_low meets valence_medium` (in: EmoWorker_v2)
- `(eda_scr_amp_medium) => eda_scr_amp_medium starts arousal_medium` (in: CASE, EmoWorker_v2)
- `(eda_scr_amp_medium) => eda_scr_amp_medium starts valence_medium` (in: CASE)
- `(eda_scr_auc_high) => eda_scr_auc_high before arousal_high` (in: CEAP)
- `(eda_scr_auc_high) => eda_scr_auc_high before valence_medium` (in: K-emoCon, CEAP)
- `(eda_scr_auc_high) => eda_scr_auc_high contains arousal_medium` (in: CEAP)
- `(eda_scr_auc_high) => eda_scr_auc_high contains valence_high` (in: CEAP)
- `(eda_scr_auc_high) => eda_scr_auc_high equals valence_low` (in: CEAP)
- `(eda_scr_auc_high) => eda_scr_auc_high starts arousal_low` (in: K-emoCon)
- `(eda_scr_auc_high) => eda_scr_auc_high starts arousal_medium` (in: CASE, EmoWorker_v2)
- `(eda_scr_auc_low) => eda_scr_auc_low before valence_medium` (in: EmoWorker_v2)
- `(eda_scr_auc_medium) => eda_scr_auc_medium equals valence_medium` (in: K-emoCon)
- `(eda_scr_auc_medium) => eda_scr_auc_medium meets arousal_medium` (in: CEAP)
- `(eda_scr_auc_medium) => eda_scr_auc_medium meets valence_low` (in: CEAP)
- `(eda_scr_auc_medium) => eda_scr_auc_medium meets valence_medium` (in: CEAP)
- `(eda_scr_auc_medium) => eda_scr_auc_medium starts arousal_medium` (in: CASE)
- `(eda_std_high) => eda_std_high before valence_medium` (in: CEAP)
- `(eda_std_high) => eda_std_high equals valence_medium` (in: EmoWorker_v2)
- `(eda_std_high) => eda_std_high is-finished-by valence_high` (in: CEAP)
- `(eda_std_high) => eda_std_high meets valence_medium` (in: K-emoCon)
- `(eda_std_high) => eda_std_high starts arousal_low` (in: K-emoCon)
- `(eda_std_low) => eda_std_low meets arousal_medium` (in: CASE)
- `(eda_std_low) => eda_std_low overlaps valence_medium` (in: CASE)
- `(eda_std_medium) => eda_std_medium equals valence_medium` (in: CEAP)
- `(eda_std_medium) => eda_std_medium starts valence_medium` (in: K-emoCon, EmoWorker_v2)
- `(hr_high) => hr_high equals valence_high` (in: CEAP)
- `(hr_high) => hr_high equals valence_medium` (in: CASE)
- `(hr_high) => hr_high starts arousal_medium` (in: CASE)
- `(hr_low) => hr_low before valence_medium` (in: CEAP)
- `(hr_low) => hr_low is-finished-by valence_medium` (in: K-emoCon)
- `(hr_low) => hr_low meets valence_high` (in: CEAP)
- `(hr_medium) => hr_medium before arousal_medium` (in: CEAP)
- `(hr_medium) => hr_medium meets valence_high` (in: CEAP)
- `(hr_medium) => hr_medium meets valence_medium` (in: K-emoCon)
- `(hrv_cvnn_low) => hrv_cvnn_low meets arousal_medium` (in: CASE, EmoWorker_v2)
- `(hrv_cvnn_medium) => hrv_cvnn_medium before valence_medium` (in: K-emoCon)
- `(hrv_cvsd_medium) => hrv_cvsd_medium before valence_medium` (in: K-emoCon)
- `(hrv_cvsd_medium) => hrv_cvsd_medium meets arousal_medium` (in: CASE, EmoWorker_v2)
- `(hrv_hfn_high) => hrv_hfn_high meets arousal_medium` (in: CASE)
- `(hrv_hfn_high) => hrv_hfn_high meets valence_medium` (in: CASE)
- `(hrv_hfn_low) => hrv_hfn_low meets arousal_medium` (in: CASE)
- `(hrv_hfn_low) => hrv_hfn_low meets valence_medium` (in: CASE, K-emoCon)
- `(hrv_hfn_medium) => hrv_hfn_medium meets arousal_medium` (in: CASE)
- `(hrv_hfn_medium) => hrv_hfn_medium meets valence_medium` (in: CASE)
- `(hrv_pnn20_high) => hrv_pnn20_high before valence_medium` (in: K-emoCon)
- `(hrv_pnn20_low) => hrv_pnn20_low meets arousal_medium` (in: CASE, EmoWorker_v2)
- `(hrv_pnn50_low) => hrv_pnn50_low meets arousal_medium` (in: CASE, EmoWorker_v2)
- `(hrv_pnn50_medium) => hrv_pnn50_medium before valence_medium` (in: K-emoCon)
- `(hrv_pnn50_medium) => hrv_pnn50_medium meets arousal_medium` (in: CASE, EmoWorker_v2)
- `(hrv_sdnn_low) => hrv_sdnn_low meets arousal_medium` (in: CASE, EmoWorker_v2)
- `(hrv_sdnn_medium) => hrv_sdnn_medium before valence_medium` (in: K-emoCon)
- `(valence_high) => valence_high before eda_high` (in: CEAP)
- `(valence_high) => valence_high before eda_peaks_high` (in: K-emoCon)
- `(valence_high) => valence_high before eda_scr_amp_low` (in: K-emoCon)
- `(valence_high) => valence_high contains eda_scr_auc_medium` (in: K-emoCon)
- `(valence_high) => valence_high is-finished-by eda_peaks_medium` (in: K-emoCon)
- `(valence_high) => valence_high is-finished-by eda_scr_auc_low` (in: CASE)
- `(valence_high) => valence_high is-finished-by eda_scr_auc_medium` (in: CASE)
- `(valence_high) => valence_high meets eda_max_high` (in: CEAP)
- `(valence_high) => valence_high meets eda_peaks_high` (in: CASE, CEAP)
- `(valence_high) => valence_high meets eda_peaks_medium` (in: CASE)
- `(valence_high) => valence_high meets eda_scr_amp_medium` (in: CASE)
- `(valence_high) => valence_high meets eda_std_low` (in: CASE)
- `(valence_high) => valence_high meets eda_std_medium` (in: CASE)
- `(valence_high) => valence_high meets hr_high` (in: CEAP)
- `(valence_low) => valence_low before arousal_high` (in: CASE)
- `(valence_low) => valence_low before arousal_medium` (in: CEAP)
- `(valence_low) => valence_low before eda_high` (in: CEAP)
- `(valence_low) => valence_low before eda_max_high` (in: CEAP)
- `(valence_low) => valence_low before eda_peaks_high` (in: K-emoCon, CEAP, EmoWorker_v2)
- `(valence_low) => valence_low before eda_scr_amp_medium` (in: EmoWorker_v2)
- `(valence_low) => valence_low before eda_scr_auc_high` (in: EmoWorker_v2)
- `(valence_low) => valence_low before eda_std_high` (in: EmoWorker_v2)
- `(valence_low) => valence_low before eda_std_medium` (in: EmoWorker_v2)
- `(valence_low) => valence_low before hr_medium` (in: CASE)
- `(valence_low) => valence_low contains eda_peaks_low` (in: CASE)
- `(valence_low) => valence_low contains eda_scr_auc_low` (in: CASE)
- `(valence_low) => valence_low contains eda_scr_auc_medium` (in: EmoWorker_v2)
- `(valence_low) => valence_low is-finished-by eda_peaks_medium` (in: EmoWorker_v2)
- `(valence_low) => valence_low is-finished-by eda_scr_amp_high` (in: CASE)
- `(valence_low) => valence_low is-finished-by eda_scr_amp_low` (in: EmoWorker_v2)
- `(valence_low) => valence_low is-finished-by eda_scr_auc_high` (in: CASE)
- `(valence_low) => valence_low is-finished-by eda_std_high` (in: CASE)
- `(valence_low) => valence_low meets eda_peaks_low` (in: EmoWorker_v2)
- `(valence_low) => valence_low meets eda_scr_auc_medium` (in: K-emoCon)
- `(valence_low) => valence_low meets eda_std_high` (in: CEAP)
- `(valence_low) => valence_low meets eda_std_medium` (in: K-emoCon)
- `(valence_low) => valence_low meets hr_high` (in: CEAP)
- `(valence_low) => valence_low overlaps eda_scr_amp_low` (in: K-emoCon)
- `(valence_low) => valence_low starts eda_low` (in: CEAP)
- `(valence_low) => valence_low starts eda_scr_auc_high` (in: CEAP)
- `(valence_medium) => valence_medium before eda_max_high` (in: CEAP)
- `(valence_medium) => valence_medium before eda_peaks_high` (in: EmoWorker_v2)
- `(valence_medium) => valence_medium before eda_scr_amp_high` (in: CEAP)
- `(valence_medium) => valence_medium before eda_scr_amp_medium` (in: EmoWorker_v2)
- `(valence_medium) => valence_medium before eda_scr_auc_high` (in: CEAP)
- `(valence_medium) => valence_medium before eda_scr_auc_medium` (in: EmoWorker_v2)
- `(valence_medium) => valence_medium before eda_std_high` (in: CEAP)
- `(valence_medium) => valence_medium before eda_std_medium` (in: K-emoCon, EmoWorker_v2)
- `(valence_medium) => valence_medium before hr_low` (in: CASE)
- `(valence_medium) => valence_medium before hr_medium` (in: K-emoCon)
- `(valence_medium) => valence_medium contains eda_scr_amp_high` (in: CASE)
- `(valence_medium) => valence_medium contains eda_scr_auc_low` (in: K-emoCon)
- `(valence_medium) => valence_medium is-finished-by eda_peaks_medium` (in: CASE)
- `(valence_medium) => valence_medium is-finished-by eda_scr_auc_high` (in: CASE)
- `(valence_medium) => valence_medium is-finished-by eda_std_medium` (in: CASE)
- `(valence_medium) => valence_medium meets arousal_high` (in: CEAP)
- `(valence_medium) => valence_medium meets eda_high` (in: CEAP)
- `(valence_medium) => valence_medium meets eda_max_high` (in: CASE)
- `(valence_medium) => valence_medium meets eda_peaks_high` (in: CEAP)
- `(valence_medium) => valence_medium meets eda_scr_amp_high` (in: EmoWorker_v2)
- `(valence_medium) => valence_medium meets eda_scr_amp_low` (in: K-emoCon)
- `(valence_medium) => valence_medium meets eda_scr_amp_medium` (in: CASE)
- `(valence_medium) => valence_medium meets eda_std_low` (in: CASE)
- `(valence_medium) => valence_medium meets hr_high` (in: CEAP)
- `(valence_medium) => valence_medium starts arousal_medium` (in: EmoWorker_v2)
- `(valence_medium) => valence_medium starts eda_high` (in: CASE)
- `(valence_medium) => valence_medium starts eda_low` (in: EmoWorker_v2)
- `(valence_medium) => valence_medium starts eda_max_low` (in: EmoWorker_v2)
- `(valence_medium) => valence_medium starts eda_peaks_low` (in: CASE)
- `(valence_medium) => valence_medium starts eda_peaks_medium` (in: EmoWorker_v2)
- `(valence_medium) => valence_medium starts eda_scr_auc_high` (in: EmoWorker_v2)
- `(valence_medium) => valence_medium starts eda_std_high` (in: EmoWorker_v2)

## External-Only Rules

**97** rules found only in externally-annotated datasets.

- `(arousal_high) => arousal_high before eda_low` (in: EMBOA)
- `(arousal_high) => arousal_high before eda_peaks_high` (in: K-emoCon (ext))
- `(arousal_high) => arousal_high equals eda_std_low` (in: EMBOA)
- `(arousal_high) => arousal_high equals hr_low` (in: EMBOA)
- `(arousal_high) => arousal_high meets eda_high` (in: K-emoCon (ext))
- `(arousal_high) => arousal_high meets eda_max_medium` (in: EMBOA)
- `(arousal_high) => arousal_high meets eda_medium` (in: EMBOA)
- `(arousal_high) => arousal_high meets eda_scr_amp_medium` (in: K-emoCon (ext))
- `(arousal_high) => arousal_high meets eda_scr_auc_medium` (in: K-emoCon (ext))
- `(arousal_high) => arousal_high meets eda_std_medium` (in: K-emoCon (ext))
- `(arousal_high) => arousal_high meets hr_high` (in: EMBOA)
- `(arousal_high) => arousal_high starts eda_scr_auc_low` (in: K-emoCon (ext))
- `(arousal_low) => arousal_low contains eda_scr_amp_medium` (in: K-emoCon (ext))
- `(arousal_low) => arousal_low contains eda_std_high` (in: K-emoCon (ext))
- `(arousal_low) => arousal_low equals eda_max_medium` (in: K-emoCon (ext))
- `(arousal_low) => arousal_low equals valence_medium` (in: K-emoCon (ext))
- `(arousal_medium) => arousal_medium before eda_max_medium` (in: EMBOA)
- `(arousal_medium) => arousal_medium before eda_medium` (in: EMBOA)
- `(arousal_medium) => arousal_medium before eda_peaks_high` (in: K-emoCon (ext))
- `(arousal_medium) => arousal_medium contains eda_high` (in: EMBOA)
- `(arousal_medium) => arousal_medium contains eda_low` (in: EMBOA)
- `(arousal_medium) => arousal_medium contains eda_max_high` (in: EMBOA)
- `(arousal_medium) => arousal_medium contains eda_max_low` (in: EMBOA)
- `(arousal_medium) => arousal_medium equals eda_scr_auc_medium` (in: K-emoCon (ext))
- `(arousal_medium) => arousal_medium is-finished-by eda_std_high` (in: EMBOA)
- `(arousal_medium) => arousal_medium meets eda_peaks_medium` (in: K-emoCon (ext))
- `(arousal_medium) => arousal_medium meets eda_scr_auc_low` (in: K-emoCon (ext))
- `(arousal_medium) => arousal_medium meets eda_std_medium` (in: EMBOA)
- `(arousal_medium) => arousal_medium starts eda_std_medium` (in: K-emoCon (ext))
- `(arousal_medium) => arousal_medium starts hr_low` (in: K-emoCon (ext))
- `(arousal_medium) => arousal_medium starts valence_medium` (in: K-emoCon (ext))
- `(eda_high) => eda_high equals valence_high` (in: EMBOA)
- `(eda_low) => eda_low starts arousal_medium` (in: EMBOA)
- `(eda_low) => eda_low starts valence_medium` (in: K-emoCon (ext))
- `(eda_max_high) => eda_max_high before arousal_medium` (in: K-emoCon (ext))
- `(eda_max_high) => eda_max_high contains valence_high` (in: EMBOA)
- `(eda_max_high) => eda_max_high starts arousal_medium` (in: EMBOA)
- `(eda_max_high) => eda_max_high starts valence_medium` (in: EMBOA)
- `(eda_max_low) => eda_max_low equals valence_high` (in: EMBOA)
- `(eda_max_low) => eda_max_low meets valence_medium` (in: K-emoCon (ext), EMBOA)
- `(eda_max_low) => eda_max_low overlaps arousal_medium` (in: K-emoCon (ext))
- `(eda_max_low) => eda_max_low starts arousal_medium` (in: EMBOA)
- `(eda_max_medium) => eda_max_medium before valence_high` (in: EMBOA)
- `(eda_max_medium) => eda_max_medium starts arousal_low` (in: K-emoCon (ext))
- `(eda_max_medium) => eda_max_medium starts arousal_medium` (in: EMBOA)
- `(eda_max_medium) => eda_max_medium starts valence_medium` (in: EMBOA)
- `(eda_medium) => eda_medium meets valence_high` (in: EMBOA)
- `(eda_medium) => eda_medium starts arousal_medium` (in: EMBOA)
- `(eda_medium) => eda_medium starts valence_medium` (in: EMBOA)
- `(eda_peaks_high) => eda_peaks_high before arousal_low` (in: K-emoCon (ext))
- `(eda_peaks_low) => eda_peaks_low before arousal_medium` (in: K-emoCon (ext))
- `(eda_scr_amp_high) => eda_scr_amp_high before arousal_medium` (in: K-emoCon (ext))
- `(eda_scr_amp_low) => eda_scr_amp_low before arousal_low` (in: K-emoCon (ext))
- `(eda_scr_amp_low) => eda_scr_amp_low starts valence_medium` (in: K-emoCon (ext))
- `(eda_scr_amp_medium) => eda_scr_amp_medium meets valence_medium` (in: K-emoCon (ext))
- `(eda_scr_auc_high) => eda_scr_auc_high meets arousal_medium` (in: K-emoCon (ext))
- `(eda_scr_auc_low) => eda_scr_auc_low meets valence_medium` (in: K-emoCon (ext))
- `(eda_std_high) => eda_std_high contains valence_high` (in: EMBOA)
- `(eda_std_low) => eda_std_low meets valence_high` (in: EMBOA)
- `(eda_std_low) => eda_std_low meets valence_medium` (in: K-emoCon (ext))
- `(eda_std_low) => eda_std_low starts arousal_medium` (in: EMBOA)
- `(eda_std_low) => eda_std_low starts valence_medium` (in: EMBOA)
- `(eda_std_medium) => eda_std_medium before arousal_medium` (in: K-emoCon (ext))
- `(eda_std_medium) => eda_std_medium before valence_medium` (in: K-emoCon (ext))
- `(eda_std_medium) => eda_std_medium equals valence_high` (in: EMBOA)
- `(hr_high) => hr_high before arousal_medium` (in: EMBOA)
- `(hr_high) => hr_high meets valence_high` (in: EMBOA)
- `(hr_high) => hr_high meets valence_medium` (in: K-emoCon (ext))
- `(hr_low) => hr_low before valence_high` (in: EMBOA)
- `(hr_low) => hr_low meets arousal_medium` (in: K-emoCon (ext))
- `(hr_low) => hr_low meets valence_medium` (in: K-emoCon (ext))
- `(hr_medium) => hr_medium before valence_high` (in: EMBOA)
- `(hr_medium) => hr_medium contains arousal_medium` (in: K-emoCon (ext))
- `(hr_medium) => hr_medium overlaps arousal_low` (in: K-emoCon (ext))
- `(hr_medium) => hr_medium starts arousal_medium` (in: EMBOA)
- `(hr_medium) => hr_medium starts valence_medium` (in: K-emoCon (ext))
- `(hrv_cvnn_low) => hrv_cvnn_low meets arousal_low` (in: K-emoCon (ext))
- `(valence_high) => valence_high before eda_low` (in: EMBOA)
- `(valence_high) => valence_high meets eda_max_medium` (in: EMBOA)
- `(valence_high) => valence_high meets eda_medium` (in: EMBOA)
- `(valence_high) => valence_high meets eda_std_high` (in: EMBOA)
- `(valence_high) => valence_high meets hr_low` (in: EMBOA)
- `(valence_low) => valence_low meets arousal_medium` (in: EMBOA)
- `(valence_low) => valence_low meets eda_low` (in: EMBOA)
- `(valence_medium) => valence_medium before eda_max_medium` (in: EMBOA)
- `(valence_medium) => valence_medium before eda_medium` (in: EMBOA)
- `(valence_medium) => valence_medium contains eda_max_high` (in: EMBOA)
- `(valence_medium) => valence_medium contains eda_scr_amp_medium` (in: K-emoCon (ext))
- `(valence_medium) => valence_medium contains eda_std_low` (in: EMBOA)
- `(valence_medium) => valence_medium contains eda_std_medium` (in: K-emoCon (ext))
- `(valence_medium) => valence_medium contains hr_low` (in: EMBOA)
- `(valence_medium) => valence_medium is-finished-by eda_high` (in: K-emoCon (ext))
- `(valence_medium) => valence_medium is-finished-by eda_medium` (in: K-emoCon (ext))
- `(valence_medium) => valence_medium is-finished-by eda_std_high` (in: EMBOA)
- `(valence_medium) => valence_medium is-finished-by hr_low` (in: K-emoCon (ext))
- `(valence_medium) => valence_medium meets arousal_low` (in: K-emoCon (ext))
- `(valence_medium) => valence_medium meets eda_max_medium` (in: K-emoCon (ext))
