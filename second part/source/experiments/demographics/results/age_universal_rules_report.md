# Universal Rules by Age Experiment

This experiment aims to identify rules that are universally true across all evaluated datasets, split solely by demographic age subsets (young vs old).

| Dataset | Participants (A) | Participants (B) | Rules Filtered (A) | Rules Filtered (B) |
|-------|-------------|---------|-------|-------|
| **CASE** | 10 | 20 | 401 | 326 |
| **K-emoCon** | 24 | 4 | 310 | 485 |
| **CEAP** | 19 | 13 | 411 | 325 |
| **EmoWorker_v2** | 7 | 24 | 367 | 301 |

## Age Universal Rules Identification

- **Universal rules found in ALL datasets for young**: 6
- **Universal rules found in ALL datasets for old**: 6
- **Universal rules spanning across BOTH groups**: 0
- **Universal rules UNIQUELY for young**: 6
- **Universal rules UNIQUELY for old**: 6

### Universal Rules for young (Top metrics globally)

| Rule | Avg Conf | Avg Lift | Avg Sup | Min Conf | Min Lift | Min Sup | CASE | K-emoCon | CEAP | EmoWorker_v2 |
|---|---|---|---|---|---|---|---|---|---|---|
| `(temp_medium) => temp_medium meets valence_medium` *(unique)* | **0.828** | 0.828 | 0.583 | 0.696 | 0.696 | 0.053 | c:0.90 l:0.90 s:0.90 n:9 | c:0.70 l:0.70 s:0.67 n:16 | c:1.00 l:1.00 s:0.05 n:1 | c:0.71 l:0.71 s:0.71 n:5 |
| `(hrv_pnn50_high) => hrv_pnn50_high meets arousal_medium` *(unique)* | **0.714** | 0.742 | 0.568 | 0.591 | 0.591 | 0.158 | c:1.00 l:1.00 s:1.00 n:10 | c:0.59 l:0.59 s:0.54 n:13 | c:0.60 l:0.60 s:0.16 n:3 | c:0.67 l:0.78 s:0.57 n:4 |
| `(hrv_hfn_high) => hrv_hfn_high meets valence_medium` *(unique)* | **0.679** | 0.679 | 0.444 | 0.500 | 0.500 | 0.158 | c:1.00 l:1.00 s:1.00 n:10 | c:0.62 l:0.62 s:0.33 n:8 | c:0.60 l:0.60 s:0.16 n:3 | c:0.50 l:0.50 s:0.29 n:2 |
| `(hrv_pnn50_medium) => hrv_pnn50_medium meets eda_peaks_high` *(unique)* | **0.598** | 0.598 | 0.432 | 0.500 | 0.500 | 0.158 | c:0.50 l:0.50 s:0.50 n:5 | c:0.57 l:0.57 s:0.50 n:12 | c:0.75 l:0.75 s:0.16 n:3 | c:0.57 l:0.57 s:0.57 n:4 |
| `(hrv_cvnn_medium) => hrv_cvnn_medium meets eda_std_high` *(unique)* | **0.589** | 0.615 | 0.415 | 0.500 | 0.500 | 0.316 | c:0.50 l:0.50 s:0.50 n:5 | c:0.50 l:0.50 s:0.42 n:10 | c:0.86 l:0.96 s:0.32 n:6 | c:0.50 l:0.50 s:0.43 n:3 |
| `(hrv_hfn_low) => hrv_hfn_low meets arousal_medium` *(unique)* | **0.550** | 0.571 | 0.439 | 0.500 | 0.500 | 0.210 | c:0.70 l:0.70 s:0.70 n:7 | c:0.50 l:0.50 s:0.42 n:10 | c:0.50 l:0.50 s:0.21 n:4 | c:0.50 l:0.58 s:0.43 n:3 |

### Universal Rules for old (Top metrics globally)

| Rule | Avg Conf | Avg Lift | Avg Sup | Min Conf | Min Lift | Min Sup | CASE | K-emoCon | CEAP | EmoWorker_v2 |
|---|---|---|---|---|---|---|---|---|---|---|
| `(hrv_pnn50_low) => hrv_pnn50_low meets arousal_medium` *(unique)* | **0.747** | 0.747 | 0.603 | 0.667 | 0.667 | 0.154 | c:0.80 l:0.80 s:0.80 n:16 | c:0.75 l:0.75 s:0.75 n:3 | c:0.67 l:0.67 s:0.15 n:2 | c:0.77 l:0.77 s:0.71 n:17 |
| `(valence_high) => valence_high meets arousal_medium` *(unique)* | **0.740** | 0.740 | 0.573 | 0.667 | 0.667 | 0.250 | c:0.85 l:0.85 s:0.85 n:17 | c:0.67 l:0.67 s:0.50 n:2 | c:0.69 l:0.69 s:0.69 n:9 | c:0.75 l:0.75 s:0.25 n:6 |
| `(hrv_pnn20_low) => hrv_pnn20_low meets arousal_medium` *(unique)* | **0.667** | 0.667 | 0.559 | 0.500 | 0.500 | 0.154 | c:0.75 l:0.75 s:0.75 n:15 | c:0.75 l:0.75 s:0.75 n:3 | c:0.50 l:0.50 s:0.15 n:2 | c:0.67 l:0.67 s:0.58 n:14 |
| `(hrv_cvnn_medium) => hrv_cvnn_medium meets valence_medium` *(unique)* | **0.659** | 0.659 | 0.511 | 0.500 | 0.500 | 0.154 | c:0.85 l:0.85 s:0.85 n:17 | c:0.50 l:0.50 s:0.50 n:2 | c:0.67 l:0.67 s:0.15 n:2 | c:0.62 l:0.62 s:0.54 n:13 |
| `(hrv_cvsd_medium) => hrv_cvsd_medium meets valence_medium` *(unique)* | **0.634** | 0.634 | 0.527 | 0.500 | 0.500 | 0.308 | c:0.80 l:0.80 s:0.80 n:16 | c:0.50 l:0.50 s:0.50 n:2 | c:0.67 l:0.67 s:0.31 n:4 | c:0.57 l:0.57 s:0.50 n:12 |
| `(hrv_cvsd_medium) => hrv_cvsd_medium meets eda_medium` *(unique)* | **0.530** | 0.559 | 0.445 | 0.500 | 0.500 | 0.231 | c:0.55 l:0.55 s:0.55 n:11 | c:0.50 l:0.50 s:0.50 n:2 | c:0.50 l:0.59 s:0.23 n:3 | c:0.57 l:0.60 s:0.50 n:12 |
