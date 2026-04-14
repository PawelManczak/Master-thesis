# Universal Rules by Gender Experiment

This experiment aims to identify rules that are universally true across all evaluated datasets, split solely by demographic gender subsets (M vs F).

| Dataset | Participants (A) | Participants (B) | Rules Filtered (A) | Rules Filtered (B) |
|-------|-------------|---------|-------|-------|
| **CASE** | 15 | 15 | 10229 | 9891 |
| **K-emoCon** | 16 | 12 | 7019 | 10066 |
| **CEAP** | 16 | 16 | 10727 | 11563 |
| **EmoWorker_v2** | 24 | 7 | 2403 | 16838 |

## Gender Universal Rules Identification

- **Universal rules found in ALL datasets for M**: 1
- **Universal rules found in ALL datasets for F**: 4
- **Universal rules spanning across BOTH groups**: 0
- **Universal rules UNIQUELY for M**: 1
- **Universal rules UNIQUELY for F**: 4

### Universal Rules for M (Top metrics globally)

| Rule | Avg Conf | Avg Lift | Avg Sup | Min Conf | Min Lift | Min Sup | CASE | K-emoCon | CEAP | EmoWorker_v2 |
|---|---|---|---|---|---|---|---|---|---|---|
| `(temp_low) => temp_low meets valence_medium` *(unique)* | **0.752** | 0.752 | 0.635 | 0.571 | 0.571 | 0.250 | c:1.00 l:1.00 s:1.00 n:15 | c:0.77 l:0.77 s:0.62 n:10 | c:0.57 l:0.57 s:0.25 n:4 | c:0.67 l:0.67 s:0.67 n:16 |

### Universal Rules for F (Top metrics globally)

| Rule | Avg Conf | Avg Lift | Avg Sup | Min Conf | Min Lift | Min Sup | CASE | K-emoCon | CEAP | EmoWorker_v2 |
|---|---|---|---|---|---|---|---|---|---|---|
| `temp_high meets eda_low => temp_high meets eda_low AND temp_high meets eda_max_low AND eda_low equals eda_max_low` *(unique)* | **1.000** | 1.036 | 0.353 | 1.000 | 1.000 | 0.250 | c:1.00 l:1.00 s:0.40 n:6 | c:1.00 l:1.00 s:0.33 n:4 | c:1.00 l:1.14 s:0.25 n:4 | c:1.00 l:1.00 s:0.43 n:3 |
| `hrv_cvnn_high meets eda_low => hrv_cvnn_high meets eda_low AND hrv_cvnn_high meets eda_max_low AND eda_low equals eda_max_low` *(unique)* | **0.887** | 0.916 | 0.296 | 0.750 | 0.750 | 0.250 | c:1.00 l:1.00 s:0.40 n:6 | c:0.75 l:0.75 s:0.25 n:3 | c:0.80 l:0.91 s:0.25 n:4 | c:1.00 l:1.00 s:0.29 n:2 |
| `hrv_pnn20_high meets eda_low => hrv_pnn20_high meets eda_low AND hrv_pnn20_high meets eda_max_low AND eda_low equals eda_max_low` *(unique)* | **0.887** | 0.923 | 0.263 | 0.750 | 0.750 | 0.250 | c:0.80 l:0.80 s:0.27 n:4 | c:0.75 l:0.75 s:0.25 n:3 | c:1.00 l:1.14 s:0.25 n:4 | c:1.00 l:1.00 s:0.29 n:2 |
| `hrv_sdnn_high meets eda_low => hrv_sdnn_high meets eda_low AND hrv_sdnn_high meets eda_max_low AND eda_low equals eda_max_low` *(unique)* | **0.887** | 0.916 | 0.296 | 0.750 | 0.750 | 0.250 | c:1.00 l:1.00 s:0.40 n:6 | c:0.75 l:0.75 s:0.25 n:3 | c:0.80 l:0.91 s:0.25 n:4 | c:1.00 l:1.00 s:0.29 n:2 |
