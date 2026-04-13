# ARMADA Rules Cross-Validation

Date: 2026-04-13 15:42

## Methodology

For each combination of two training sets:
1. Find patterns common to the training pair
2. Check how many of them appear in the validation set
3. Calculate hit rate (% of validated rules)

## Parameters

- minsup: 0.3, minconf: 0.5, maxgap: 5, max_pattern_size: 3
- Filters: BVP_ONLY=True, EDA_ONLY=True, PHYSIO_CROSS=True, SINGLE_FEATURE=True

## Dataset Statistics

| Dataset | Participants | Rules (raw) | Rules (filtered) |
|-------|-------------|----------------|----------------------|
| CASE | 30 | 2668 | 1131 |
| K-emoCon | 28 | 2334 | 604 |
| CEAP | 32 | 2425 | 1333 |
| EmoWorker_v2 | 31 | 1593 | 493 |

## Cross-Validation Results

| Train (N-1)                    | Val          |   Common (Train) |   Validated |   Hit Rate (%) |
|:-------------------------------|:-------------|-----------------:|------------:|---------------:|
| K-emoCon + CEAP + EmoWorker_v2 | CASE         |                1 |           0 |              0 |
| CASE + CEAP + EmoWorker_v2     | K-emoCon     |                5 |           0 |              0 |
| CASE + K-emoCon + EmoWorker_v2 | CEAP         |               49 |           0 |              0 |
| CASE + K-emoCon + CEAP         | EmoWorker_v2 |                2 |           0 |              0 |

## Top 20 Universal Rules (by dataset count and conf)

| Rule                                                                                                                                                         |   Presence (/4) | CASE conf   | CASE lift   | CASE sup   | K-emoCon conf   | K-emoCon lift   | K-emoCon sup   | CEAP conf   | CEAP lift   | CEAP sup   | EmoWorker_v2 conf   | EmoWorker_v2 lift   | EmoWorker_v2 sup   |   Avg conf |   Avg lift |   Avg sup |
|:-------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------:|:------------|:------------|:-----------|:----------------|:----------------|:---------------|:------------|:------------|:-----------|:--------------------|:--------------------|:-------------------|-----------:|-----------:|----------:|
| hrv_cvnn_high equals hrv_pnn20_high => hrv_cvnn_high equals hrv_pnn20_high AND hrv_cvnn_high meets arousal_medium AND hrv_pnn20_high meets arousal_medium    |               3 | 0.75        | 0.75        | 0.3        | 0.688           | 0.688           | 0.393          | -           | -           | -          | 1.0                 | 1.033               | 0.452              |      0.812 |      0.824 |     0.381 |
| (arousal_medium) => arousal_medium equals valence_medium                                                                                                     |               3 | 0.633       | 0.633       | 0.633      | -               | -               | -              | 0.938       | 0.938       | 0.938      | 0.833               | 0.833               | 0.806              |      0.801 |      0.801 |     0.792 |
| (eda_medium) => eda_medium meets arousal_medium                                                                                                              |               3 | 0.9         | 0.9         | 0.9        | -               | -               | -              | 0.8         | 0.8         | 0.625      | 0.667               | 0.689               | 0.645              |      0.789 |      0.796 |     0.723 |
| hrv_pnn20_high equals hrv_rmssd_high => hrv_pnn20_high equals hrv_rmssd_high AND hrv_pnn20_high meets valence_medium AND hrv_rmssd_high meets valence_medium |               3 | 0.929       | 0.929       | 0.433      | 0.688           | 0.688           | 0.393          | -           | -           | -          | 0.722               | 0.722               | 0.419              |      0.779 |      0.779 |     0.415 |
| (valence_low) => valence_low before eda_peaks_high                                                                                                           |               3 | -           | -           | -          | 0.81            | 0.81            | 0.607          | 0.931       | 0.931       | 0.844      | 0.5                 | 0.5                 | 0.484              |      0.747 |      0.747 |     0.645 |
| (eda_std_medium) => eda_std_medium meets arousal_medium                                                                                                      |               3 | 0.767       | 0.767       | 0.767      | -               | -               | -              | 0.808       | 0.808       | 0.656      | 0.613               | 0.633               | 0.613              |      0.729 |      0.736 |     0.679 |
| (eda_medium) => eda_medium meets valence_medium                                                                                                              |               3 | 0.8         | 0.8         | 0.8        | 0.577           | 0.577           | 0.536          | 0.8         | 0.8         | 0.625      | -                   | -                   | -                  |      0.726 |      0.726 |     0.654 |
| (hrv_hfn_high) => hrv_hfn_high meets valence_medium                                                                                                          |               3 | 0.967       | 0.967       | 0.967      | 0.625           | 0.625           | 0.357          | -           | -           | -          | 0.579               | 0.579               | 0.355              |      0.724 |      0.724 |     0.56  |
| hrv_hfn_low meets arousal_medium => hrv_hfn_low meets arousal_medium AND hrv_hfn_low meets valence_medium AND arousal_medium equals valence_medium           |               3 | 0.739       | 0.739       | 0.567      | 0.727           | 0.727           | 0.286          | -           | -           | -          | 0.692               | 0.692               | 0.29               |      0.72  |      0.72  |     0.381 |
| (arousal_low) => arousal_low meets eda_scr_auc_medium                                                                                                        |               3 | 0.593       | 0.593       | 0.533      | 0.833           | 0.864           | 0.714          | -           | -           | -          | 0.714               | 0.738               | 0.645              |      0.713 |      0.732 |     0.631 |
| hrv_pnn50_high meets arousal_medium => hrv_pnn50_high meets arousal_medium AND hrv_pnn50_high meets valence_medium AND arousal_medium equals valence_medium  |               3 | 0.885       | 0.885       | 0.767      | 0.571           | 0.571           | 0.286          | -           | -           | -          | 0.667               | 0.667               | 0.387              |      0.708 |      0.708 |     0.48  |
| (arousal_medium) => arousal_medium meets eda_scr_auc_medium                                                                                                  |               3 | 0.833       | 0.833       | 0.833      | 0.679           | 0.704           | 0.679          | 0.594       | 0.826       | 0.594      | -                   | -                   | -                  |      0.702 |      0.788 |     0.702 |
| hrv_pnn20_high equals hrv_pnn50_high => hrv_pnn20_high equals hrv_pnn50_high AND hrv_pnn20_high meets valence_medium AND hrv_pnn50_high meets valence_medium |               3 | 0.917       | 0.917       | 0.367      | 0.6             | 0.6             | 0.321          | -           | -           | -          | 0.588               | 0.588               | 0.323              |      0.702 |      0.702 |     0.337 |
| hrv_cvnn_high equals hrv_sdnn_high => hrv_cvnn_high equals hrv_sdnn_high AND hrv_cvnn_high meets valence_medium AND hrv_sdnn_high meets valence_medium       |               3 | 0.821       | 0.821       | 0.767      | 0.577           | 0.577           | 0.536          | -           | -           | -          | 0.696               | 0.696               | 0.516              |      0.698 |      0.698 |     0.606 |
| hrv_pnn50_high equals hrv_sdnn_high => hrv_pnn50_high equals hrv_sdnn_high AND hrv_pnn50_high meets arousal_medium AND hrv_sdnn_high meets arousal_medium    |               3 | 0.867       | 0.867       | 0.433      | 0.545           | 0.545           | 0.429          | -           | -           | -          | 0.682               | 0.705               | 0.484              |      0.698 |      0.706 |     0.449 |
| (hrv_pnn20_high) => hrv_pnn20_high meets arousal_medium                                                                                                      |               3 | 0.8         | 0.8         | 0.8        | 0.5             | 0.5             | 0.464          | -           | -           | -          | 0.786               | 0.812               | 0.71               |      0.695 |      0.704 |     0.658 |
| hrv_cvnn_high equals hrv_rmssd_high => hrv_cvnn_high equals hrv_rmssd_high AND hrv_cvnn_high meets arousal_medium AND hrv_rmssd_high meets arousal_medium    |               3 | 0.85        | 0.85        | 0.567      | 0.545           | 0.545           | 0.429          | -           | -           | -          | 0.688               | 0.71                | 0.355              |      0.694 |      0.702 |     0.45  |
| (hrv_cvnn_high) => hrv_cvnn_high meets valence_medium                                                                                                        |               3 | 0.828       | 0.828       | 0.8        | 0.556           | 0.556           | 0.536          | -           | -           | -          | 0.69                | 0.69                | 0.645              |      0.691 |      0.691 |     0.66  |
| hrv_pnn50_high equals hrv_rmssd_high => hrv_pnn50_high equals hrv_rmssd_high AND hrv_pnn50_high meets valence_medium AND hrv_rmssd_high meets valence_medium |               3 | 0.9         | 0.9         | 0.6        | 0.522           | 0.522           | 0.429          | -           | -           | -          | 0.652               | 0.652               | 0.484              |      0.691 |      0.691 |     0.504 |
| hrv_cvnn_low equals hrv_pnn50_low => hrv_cvnn_low equals hrv_pnn50_low AND hrv_cvnn_low meets arousal_medium AND hrv_pnn50_low meets arousal_medium          |               3 | 0.8         | 0.8         | 0.4        | 0.533           | 0.533           | 0.286          | -           | -           | -          | 0.737               | 0.761               | 0.452              |      0.69  |      0.698 |     0.379 |
