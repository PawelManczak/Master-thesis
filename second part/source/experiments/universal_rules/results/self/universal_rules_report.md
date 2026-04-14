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

Evaluated on 4 datasets: CASE, K-emoCon, CEAP, EmoWorker_v2

| Dataset | Participants | Rules (Before Filters) | Rules (Filtered) |
|-------|-------------|---------|-------|
| **CASE** | 30 | 43453 | 19069 |
| **K-emoCon** | 28 | 63947 | 25327 |
| **CEAP** | 32 | 46746 | 27441 |
| **EmoWorker_v2** | 31 | 25983 | 8915 |

## Universal Rules Identification

TOTAL Universal rules found across all 4 datasets: **14**

### Universal Rules Details (sorted by avg_confidence)

| Rule | Avg Conf | Avg Lift | Avg Sup | Min Conf | Min Lift | Min Sup | CASE | K-emoCon | CEAP | EmoWorker_v2 |
|---|---|---|---|---|---|---|---|---|---|---|
| `hrv_cvnn_medium equals hrv_rmssd_low => hrv_cvnn_medium equals hrv_rmssd_low AND hrv_cvnn_medium meets valence_medium AND hrv_rmssd_low meets valence_medium` | **0.832** | 0.832 | 0.185 | 0.750 | 0.750 | 0.094 | c:1.00 l:1.00 s:0.13 n:4 | c:0.80 l:0.80 s:0.29 n:8 | c:0.75 l:0.75 s:0.09 n:3 | c:0.78 l:0.78 s:0.23 n:7 |
| `hrv_cvnn_medium equals hrv_cvsd_low AND hrv_cvnn_medium equals hrv_rmssd_low AND hrv_cvsd_low equals hrv_rmssd_low => hrv_cvnn_medium equals hrv_cvsd_low AND hrv_cvnn_medium equals hrv_rmssd_low AND hrv_cvnn_medium meets valence_medium AND hrv_cvsd_low equals hrv_rmssd_low AND hrv_cvsd_low meets valence_medium AND hrv_rmssd_low meets valence_medium` | **0.816** | 0.816 | 0.125 | 0.714 | 0.714 | 0.094 | c:1.00 l:1.00 s:0.10 n:3 | c:0.71 l:0.71 s:0.18 n:5 | c:0.75 l:0.75 s:0.09 n:3 | c:0.80 l:0.80 s:0.13 n:4 |
| `hrv_cvnn_medium equals hrv_cvsd_low => hrv_cvnn_medium equals hrv_cvsd_low AND hrv_cvnn_medium meets valence_medium AND hrv_cvsd_low meets valence_medium` | **0.774** | 0.774 | 0.142 | 0.714 | 0.714 | 0.094 | c:0.83 l:0.83 s:0.17 n:5 | c:0.71 l:0.71 s:0.18 n:5 | c:0.75 l:0.75 s:0.09 n:3 | c:0.80 l:0.80 s:0.13 n:4 |
| `hrv_hfn_low meets arousal_medium => hrv_hfn_low meets arousal_medium AND hrv_hfn_low meets valence_medium AND arousal_medium equals valence_medium` | **0.727** | 0.727 | 0.309 | 0.692 | 0.692 | 0.094 | c:0.74 l:0.74 s:0.57 n:17 | c:0.73 l:0.73 s:0.29 n:8 | c:0.75 l:0.75 s:0.09 n:3 | c:0.69 l:0.69 s:0.29 n:9 |
| `hrv_cvnn_medium equals hrv_cvsd_low => hrv_cvnn_medium equals hrv_cvsd_low AND hrv_cvnn_medium equals hrv_rmssd_low AND hrv_cvnn_medium meets valence_medium AND hrv_cvsd_low equals hrv_rmssd_low AND hrv_cvsd_low meets valence_medium AND hrv_rmssd_low meets valence_medium` | **0.691** | 2.406 | 0.125 | 0.500 | 0.625 | 0.094 | c:0.50 l:0.62 s:0.10 n:3 | c:0.71 l:1.54 s:0.18 n:5 | c:0.75 l:6.00 s:0.09 n:3 | c:0.80 l:1.46 s:0.13 n:4 |
| `hrv_hfn_low equals hrv_pnn50_high => hrv_hfn_low equals hrv_pnn50_high AND hrv_hfn_low meets valence_medium AND hrv_pnn50_high meets valence_medium` | **0.680** | 0.680 | 0.141 | 0.571 | 0.571 | 0.094 | c:0.83 l:0.83 s:0.17 n:5 | c:0.57 l:0.57 s:0.14 n:4 | c:0.60 l:0.60 s:0.09 n:3 | c:0.71 l:0.71 s:0.16 n:5 |
| `hrv_hfn_low equals hrv_rmssd_low => hrv_hfn_low equals hrv_rmssd_low AND hrv_hfn_low meets arousal_medium AND hrv_rmssd_low meets arousal_medium` | **0.677** | 0.682 | 0.183 | 0.500 | 0.500 | 0.094 | c:0.83 l:0.83 s:0.33 n:10 | c:0.50 l:0.50 s:0.14 n:4 | c:0.75 l:0.75 s:0.09 n:3 | c:0.62 l:0.65 s:0.16 n:5 |
| `hrv_cvnn_high equals hrv_sdnn_high AND hrv_cvnn_high meets arousal_medium AND hrv_sdnn_high meets arousal_medium => hrv_cvnn_high equals hrv_sdnn_high AND hrv_cvnn_high meets arousal_medium AND hrv_cvnn_high meets valence_medium AND hrv_sdnn_high meets arousal_medium AND hrv_sdnn_high meets valence_medium AND arousal_medium equals valence_medium` | **0.660** | 0.660 | 0.350 | 0.500 | 0.500 | 0.094 | c:0.91 l:0.91 s:0.70 n:21 | c:0.54 l:0.54 s:0.25 n:7 | c:0.50 l:0.50 s:0.09 n:3 | c:0.69 l:0.69 s:0.35 n:11 |
| `hrv_cvnn_medium equals hrv_hfn_low => hrv_cvnn_medium equals hrv_hfn_low AND hrv_cvnn_medium meets valence_medium AND hrv_hfn_low meets valence_medium` | **0.649** | 0.649 | 0.143 | 0.500 | 0.500 | 0.094 | c:0.67 l:0.67 s:0.13 n:4 | c:0.86 l:0.86 s:0.21 n:6 | c:0.50 l:0.50 s:0.09 n:3 | c:0.57 l:0.57 s:0.13 n:4 |
| `hrv_cvsd_low equals hrv_hfn_low => hrv_cvsd_low equals hrv_hfn_low AND hrv_cvsd_low meets arousal_medium AND hrv_hfn_low meets arousal_medium` | **0.643** | 0.647 | 0.158 | 0.500 | 0.517 | 0.094 | c:0.90 l:0.90 s:0.30 n:9 | c:0.57 l:0.57 s:0.14 n:4 | c:0.60 l:0.60 s:0.09 n:3 | c:0.50 l:0.52 s:0.10 n:3 |
| `hrv_sdnn_high meets arousal_medium => hrv_sdnn_high meets arousal_medium AND hrv_sdnn_high meets valence_medium AND arousal_medium equals valence_medium` | **0.642** | 0.642 | 0.358 | 0.500 | 0.500 | 0.094 | c:0.92 l:0.92 s:0.73 n:22 | c:0.54 l:0.54 s:0.25 n:7 | c:0.50 l:0.50 s:0.09 n:3 | c:0.61 l:0.61 s:0.35 n:11 |
| `hrv_cvnn_high meets arousal_medium => hrv_cvnn_high meets arousal_medium AND hrv_cvnn_high meets valence_medium AND arousal_medium equals valence_medium` | **0.641** | 0.641 | 0.366 | 0.500 | 0.500 | 0.094 | c:0.91 l:0.91 s:0.70 n:21 | c:0.50 l:0.50 s:0.25 n:7 | c:0.50 l:0.50 s:0.09 n:3 | c:0.65 l:0.65 s:0.42 n:13 |
| `hrv_cvsd_low equals hrv_hfn_low AND hrv_cvsd_low equals hrv_rmssd_low AND hrv_hfn_low equals hrv_rmssd_low => hrv_cvsd_low equals hrv_hfn_low AND hrv_cvsd_low equals hrv_rmssd_low AND hrv_cvsd_low meets arousal_medium AND hrv_hfn_low equals hrv_rmssd_low AND hrv_hfn_low meets arousal_medium AND hrv_rmssd_low meets arousal_medium` | **0.640** | 0.644 | 0.150 | 0.500 | 0.517 | 0.094 | c:0.89 l:0.89 s:0.27 n:8 | c:0.57 l:0.57 s:0.14 n:4 | c:0.60 l:0.60 s:0.09 n:3 | c:0.50 l:0.52 s:0.10 n:3 |
| `hrv_cvsd_low equals hrv_hfn_low => hrv_cvsd_low equals hrv_hfn_low AND hrv_cvsd_low equals hrv_rmssd_low AND hrv_cvsd_low meets arousal_medium AND hrv_hfn_low equals hrv_rmssd_low AND hrv_hfn_low meets arousal_medium AND hrv_rmssd_low meets arousal_medium` | **0.618** | 2.089 | 0.150 | 0.500 | 0.912 | 0.094 | c:0.80 l:1.04 s:0.27 n:8 | c:0.57 l:1.60 s:0.14 n:4 | c:0.60 l:4.80 s:0.09 n:3 | c:0.50 l:0.91 s:0.10 n:3 |
