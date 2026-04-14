# Universal Rules by Age Experiment

This experiment aims to identify rules that are universally true across all evaluated datasets, split solely by demographic age subsets (young vs old).

| Dataset | Participants (A) | Participants (B) | Rules Filtered (A) | Rules Filtered (B) |
|-------|-------------|---------|-------|-------|
| **CASE** | 10 | 20 | 12902 | 5070 |
| **K-emoCon** | 24 | 4 | 2746 | 119950 |
| **CEAP** | 19 | 13 | 7169 | 21209 |
| **EmoWorker_v2** | 7 | 24 | 14116 | 2310 |

## Age Universal Rules Identification

- **Universal rules found in ALL datasets for young**: 1
- **Universal rules found in ALL datasets for old**: 5
- **Universal rules spanning across BOTH groups**: 0
- **Universal rules UNIQUELY for young**: 1
- **Universal rules UNIQUELY for old**: 5

### Universal Rules for young (Top metrics globally)

| Rule | Avg Conf | Avg Lift | Avg Sup | Min Conf | Min Lift | Min Sup | CASE | K-emoCon | CEAP | EmoWorker_v2 |
|---|---|---|---|---|---|---|---|---|---|---|
| `(hrv_cvnn_medium) => hrv_cvnn_medium meets eda_std_high` *(unique)* | **0.589** | 0.615 | 0.415 | 0.500 | 0.500 | 0.316 | c:0.50 l:0.50 s:0.50 n:5 | c:0.50 l:0.50 s:0.42 n:10 | c:0.86 l:0.96 s:0.32 n:6 | c:0.50 l:0.50 s:0.43 n:3 |

### Universal Rules for old (Top metrics globally)

| Rule | Avg Conf | Avg Lift | Avg Sup | Min Conf | Min Lift | Min Sup | CASE | K-emoCon | CEAP | EmoWorker_v2 |
|---|---|---|---|---|---|---|---|---|---|---|
| `eda_low meets eda_low => eda_low meets eda_low AND eda_low meets valence_medium AND eda_low equals valence_medium` *(unique)* | **0.748** | 0.748 | 0.369 | 0.526 | 0.526 | 0.250 | c:0.67 l:0.67 s:0.50 n:10 | c:1.00 l:1.00 s:0.25 n:1 | c:0.80 l:0.80 s:0.31 n:4 | c:0.53 l:0.53 s:0.42 n:10 |
| `(hrv_cvsd_medium) => hrv_cvsd_medium meets valence_medium` *(unique)* | **0.634** | 0.634 | 0.527 | 0.500 | 0.500 | 0.308 | c:0.80 l:0.80 s:0.80 n:16 | c:0.50 l:0.50 s:0.50 n:2 | c:0.67 l:0.67 s:0.31 n:4 | c:0.57 l:0.57 s:0.50 n:12 |
| `hrv_cvsd_medium equals hrv_rmssd_medium => hrv_cvsd_medium equals hrv_rmssd_medium AND hrv_cvsd_medium meets valence_medium AND hrv_rmssd_medium meets valence_medium` *(unique)* | **0.607** | 0.607 | 0.364 | 0.500 | 0.500 | 0.231 | c:0.80 l:0.80 s:0.60 n:12 | c:0.50 l:0.50 s:0.25 n:1 | c:0.60 l:0.60 s:0.23 n:3 | c:0.53 l:0.53 s:0.38 n:9 |
| `eda_medium meets eda_medium => eda_medium meets eda_medium AND eda_medium meets valence_medium AND eda_medium equals valence_medium` *(unique)* | **0.603** | 0.603 | 0.444 | 0.500 | 0.500 | 0.308 | c:0.58 l:0.58 s:0.55 n:11 | c:0.67 l:0.67 s:0.50 n:2 | c:0.67 l:0.67 s:0.31 n:4 | c:0.50 l:0.50 s:0.42 n:10 |
| `(hrv_cvsd_medium) => hrv_cvsd_medium meets eda_medium` *(unique)* | **0.530** | 0.559 | 0.445 | 0.500 | 0.500 | 0.231 | c:0.55 l:0.55 s:0.55 n:11 | c:0.50 l:0.50 s:0.50 n:2 | c:0.50 l:0.59 s:0.23 n:3 | c:0.57 l:0.60 s:0.50 n:12 |
