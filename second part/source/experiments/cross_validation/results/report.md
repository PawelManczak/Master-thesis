# ARMADA Rules Cross-Validation

Date: 2026-03-24 22:42

## Methodology

For each combination of two training sets:
1. Find patterns common to the training pair
2. Check how many of them appear in the validation set
3. Calculate hit rate (% of validated rules)

## Parameters

- minsup: 0.3, minconf: 0.5, maxgap: 10, max_pattern_size: 2
- Filters: BVP_ONLY=True, EDA_ONLY=True, PHYSIO_CROSS=False, SINGLE_FEATURE=True

## Dataset Statistics

| Dataset | Participants | Rules (raw) | Rules (filtered) |
|-------|-------------|----------------|----------------------|
| CASE | 30 | 768 | 484 |
| K-emoCon | 28 | 746 | 401 |
| CEAP | 32 | 2066 | 1306 |
| EmoWorker_v2 | 31 | 631 | 378 |

## Cross-Validation Results

| Train (N-1)                    | Val          |   Common (Train) |   Validated |   Hit Rate (%) |
|:-------------------------------|:-------------|-----------------:|------------:|---------------:|
| K-emoCon + CEAP + EmoWorker_v2 | CASE         |                3 |           0 |              0 |
| CASE + CEAP + EmoWorker_v2     | K-emoCon     |                3 |           0 |              0 |
| CASE + K-emoCon + EmoWorker_v2 | CEAP         |               40 |           0 |              0 |
| CASE + K-emoCon + CEAP         | EmoWorker_v2 |               32 |           0 |              0 |

## Top 20 Universal Rules (by dataset count and conf)

| Rule                                                     |   Presence (/4) |   CASE conf |   CASE sup |   K-emoCon conf |   K-emoCon sup |   CEAP conf |   CEAP sup | EmoWorker_v2 conf   | EmoWorker_v2 sup   |   Avg conf |   Avg sup |
|:---------------------------------------------------------|----------------:|------------:|-----------:|----------------:|---------------:|------------:|-----------:|:--------------------|:-------------------|-----------:|----------:|
| (temp_medium) => temp_medium contains eda_peaks_high     |               3 |       0.929 |      0.433 |           0.913 |          0.75  |       1     |      0.656 | -                   | -                  |      0.947 |     0.613 |
| (temp_medium) => temp_medium contains hrv_cvnn_medium    |               3 |       0.929 |      0.433 |           0.913 |          0.75  |       1     |      0.656 | -                   | -                  |      0.947 |     0.613 |
| (temp_medium) => temp_medium contains hrv_sdnn_medium    |               3 |       0.929 |      0.433 |           0.957 |          0.786 |       0.952 |      0.625 | -                   | -                  |      0.946 |     0.615 |
| (temp_medium) => temp_medium contains eda_peaks_medium   |               3 |       1     |      0.467 |           0.87  |          0.714 |       0.952 |      0.625 | -                   | -                  |      0.941 |     0.602 |
| (temp_medium) => temp_medium contains valence_medium     |               3 |       0.929 |      0.433 |           0.913 |          0.75  |       0.952 |      0.625 | -                   | -                  |      0.931 |     0.603 |
| (temp_medium) => temp_medium contains hrv_pnn20_medium   |               3 |       0.857 |      0.4   |           0.957 |          0.786 |       0.952 |      0.625 | -                   | -                  |      0.922 |     0.604 |
| (temp_medium) => temp_medium contains hrv_pnn50_medium   |               3 |       0.857 |      0.4   |           0.957 |          0.786 |       0.952 |      0.625 | -                   | -                  |      0.922 |     0.604 |
| (temp_medium) => temp_medium contains hrv_cvsd_medium    |               3 |       0.929 |      0.433 |           0.913 |          0.75  |       0.905 |      0.594 | -                   | -                  |      0.915 |     0.592 |
| (temp_medium) => temp_medium contains hrv_rmssd_medium   |               3 |       0.929 |      0.433 |           0.913 |          0.75  |       0.905 |      0.594 | -                   | -                  |      0.915 |     0.592 |
| (temp_medium) => temp_medium contains hrv_sdnn_high      |               3 |       0.929 |      0.433 |           0.957 |          0.786 |       0.857 |      0.562 | -                   | -                  |      0.914 |     0.594 |
| (temp_medium) => temp_medium contains arousal_medium     |               3 |       0.857 |      0.4   |           0.87  |          0.714 |       1     |      0.656 | -                   | -                  |      0.909 |     0.59  |
| (temp_medium) => temp_medium contains hrv_cvnn_high      |               3 |       0.929 |      0.433 |           0.957 |          0.786 |       0.81  |      0.531 | -                   | -                  |      0.898 |     0.583 |
| (temp_medium) => temp_medium contains hrv_cvsd_high      |               3 |       0.857 |      0.4   |           0.913 |          0.75  |       0.905 |      0.594 | -                   | -                  |      0.892 |     0.581 |
| (temp_medium) => temp_medium contains hrv_pnn20_high     |               3 |       0.857 |      0.4   |           0.913 |          0.75  |       0.905 |      0.594 | -                   | -                  |      0.892 |     0.581 |
| (temp_medium) => temp_medium contains hrv_pnn20_low      |               3 |       0.857 |      0.4   |           0.913 |          0.75  |       0.905 |      0.594 | -                   | -                  |      0.892 |     0.581 |
| (temp_medium) => temp_medium contains hrv_rmssd_high     |               3 |       0.857 |      0.4   |           0.913 |          0.75  |       0.905 |      0.594 | -                   | -                  |      0.892 |     0.581 |
| (temp_medium) => temp_medium contains eda_scr_auc_medium |               3 |       1     |      0.467 |           0.913 |          0.75  |       0.714 |      0.469 | -                   | -                  |      0.876 |     0.562 |
| (temp_medium) => temp_medium contains hrv_pnn50_high     |               3 |       0.857 |      0.4   |           0.913 |          0.75  |       0.857 |      0.562 | -                   | -                  |      0.876 |     0.571 |
| (temp_medium) => temp_medium contains eda_std_medium     |               3 |       0.857 |      0.4   |           0.957 |          0.786 |       0.81  |      0.531 | -                   | -                  |      0.874 |     0.572 |
| (temp_medium) => temp_medium contains hrv_cvnn_low       |               3 |       0.857 |      0.4   |           0.87  |          0.714 |       0.81  |      0.531 | -                   | -                  |      0.845 |     0.549 |
