# Universal Rules by Gender Experiment

This experiment aims to identify rules that are universally true across all evaluated datasets, split solely by demographic gender subsets (M vs F).

| Dataset | Participants (A) | Participants (B) | Rules Filtered (A) | Rules Filtered (B) |
|-------|-------------|---------|-------|-------|
| **CASE** | 15 | 15 | 337 | 318 |
| **K-emoCon** | 16 | 12 | 331 | 400 |
| **CEAP** | 16 | 16 | 392 | 387 |
| **EmoWorker_v2** | 24 | 7 | 325 | 316 |

## Gender Universal Rules Identification

- **Universal rules found in ALL datasets for M**: 4
- **Universal rules found in ALL datasets for F**: 6
- **Universal rules spanning across BOTH groups**: 0
- **Universal rules UNIQUELY for M**: 4
- **Universal rules UNIQUELY for F**: 6

### Universal Rules for M (Top metrics globally)

| Rule | Avg Conf | Avg Lift | Avg Sup | Min Conf | Min Lift | Min Sup | CASE | K-emoCon | CEAP | EmoWorker_v2 |
|---|---|---|---|---|---|---|---|---|---|---|
| `(hrv_rmssd_low) => hrv_rmssd_low meets valence_medium` *(unique)* | **0.787** | 0.787 | 0.519 | 0.615 | 0.615 | 0.125 | c:0.87 l:0.87 s:0.87 n:13 | c:0.62 l:0.62 s:0.50 n:8 | c:1.00 l:1.00 s:0.12 n:2 | c:0.67 l:0.67 s:0.58 n:14 |
| `(temp_low) => temp_low meets valence_medium` *(unique)* | **0.752** | 0.752 | 0.635 | 0.571 | 0.571 | 0.250 | c:1.00 l:1.00 s:1.00 n:15 | c:0.77 l:0.77 s:0.62 n:10 | c:0.57 l:0.57 s:0.25 n:4 | c:0.67 l:0.67 s:0.67 n:16 |
| `(hrv_pnn20_medium) => hrv_pnn20_medium meets valence_medium` *(unique)* | **0.642** | 0.642 | 0.446 | 0.500 | 0.500 | 0.062 | c:0.87 l:0.87 s:0.87 n:13 | c:0.70 l:0.70 s:0.44 n:7 | c:0.50 l:0.50 s:0.06 n:1 | c:0.50 l:0.50 s:0.42 n:10 |
| `(hrv_pnn50_low) => hrv_pnn50_low meets valence_medium` *(unique)* | **0.630** | 0.630 | 0.482 | 0.500 | 0.500 | 0.125 | c:0.87 l:0.87 s:0.87 n:13 | c:0.58 l:0.58 s:0.44 n:7 | c:0.50 l:0.50 s:0.12 n:2 | c:0.57 l:0.57 s:0.50 n:12 |

### Universal Rules for F (Top metrics globally)

| Rule | Avg Conf | Avg Lift | Avg Sup | Min Conf | Min Lift | Min Sup | CASE | K-emoCon | CEAP | EmoWorker_v2 |
|---|---|---|---|---|---|---|---|---|---|---|
| `(hrv_hfn_medium) => hrv_hfn_medium meets valence_medium` *(unique)* | **0.850** | 0.850 | 0.457 | 0.800 | 0.800 | 0.125 | c:0.80 l:0.80 s:0.80 n:12 | c:0.80 l:0.80 s:0.33 n:4 | c:1.00 l:1.00 s:0.12 n:2 | c:0.80 l:0.80 s:0.57 n:4 |
| `(temp_medium) => temp_medium meets valence_medium` *(unique)* | **0.722** | 0.722 | 0.596 | 0.500 | 0.500 | 0.062 | c:1.00 l:1.00 s:1.00 n:15 | c:0.82 l:0.82 s:0.75 n:9 | c:0.50 l:0.50 s:0.06 n:1 | c:0.57 l:0.57 s:0.57 n:4 |
| `(hrv_cvsd_medium) => hrv_cvsd_medium meets valence_medium` *(unique)* | **0.708** | 0.708 | 0.614 | 0.500 | 0.500 | 0.125 | c:0.87 l:0.87 s:0.87 n:13 | c:0.75 l:0.75 s:0.75 n:9 | c:0.50 l:0.50 s:0.12 n:2 | c:0.71 l:0.71 s:0.71 n:5 |
| `(hrv_hfn_high) => hrv_hfn_high meets valence_medium` *(unique)* | **0.694** | 0.694 | 0.498 | 0.500 | 0.500 | 0.125 | c:1.00 l:1.00 s:1.00 n:15 | c:0.78 l:0.78 s:0.58 n:7 | c:0.50 l:0.50 s:0.12 n:2 | c:0.50 l:0.50 s:0.29 n:2 |
| `(hrv_pnn50_high) => hrv_pnn50_high meets valence_medium` *(unique)* | **0.666** | 0.666 | 0.573 | 0.500 | 0.500 | 0.188 | c:0.87 l:0.87 s:0.87 n:13 | c:0.73 l:0.73 s:0.67 n:8 | c:0.50 l:0.50 s:0.19 n:3 | c:0.57 l:0.57 s:0.57 n:4 |
| `(hrv_pnn20_low) => hrv_pnn20_low meets arousal_medium` *(unique)* | **0.601** | 0.625 | 0.445 | 0.500 | 0.500 | 0.125 | c:0.67 l:0.67 s:0.67 n:10 | c:0.50 l:0.50 s:0.42 n:5 | c:0.67 l:0.67 s:0.12 n:2 | c:0.57 l:0.67 s:0.57 n:4 |
