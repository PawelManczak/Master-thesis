# External Annotations Cross-Dataset Experiment

This experiment identifies rules that are universally present
across datasets using **external observer annotations**:
- **EMBOA**: BORIS method II (6 emotions x % agreement -> arousal/valence)
- **K-emoCon external**: Aggregated external annotations (arousal/valence 1–5)

## Experiment Parameters

- **minsup**: 0.2 (20%)
- **minconf**: 0.2 (20%)
- **maxgap**: 5s
- **max_pattern_size**: 2

## Rule Filters

- **FILTER_BVP_ONLY**: True
- **FILTER_EDA_ONLY**: True
- **FILTER_PHYSIO_CROSS**: True
- **FILTER_SINGLE_FEATURE**: True

## Dataset Processing

Evaluated on 2 datasets: EMBOA, K-emoCon-ext

| Dataset | Participants | Rules (Before Filters) | Rules (Filtered) |
|---------|-------------|------------------------|------------------|
| **EMBOA** | 16 | 383 | 127 |
| **K-emoCon-ext** | 28 | 1163 | 253 |

## Universal Rules (Present in ALL Datasets)

Total universal rules found: **11**

### Details (sorted by avg. confidence)

| Rule | Avg Conf | Avg Sup | Min Conf | Min Sup | EMBOA | K-emoCon-ext |
|---|---|---|---|---|---|---|
| `(eda_std_high) => eda_std_high starts valence_medium` | **0.871** | 0.871 | 0.812 | 0.812 | c:0.81 s:0.81 n:13 | c:0.93 s:0.93 n:26 |
| `(arousal_high) => arousal_high meets valence_medium` | **0.746** | 0.696 | 0.692 | 0.643 | c:0.80 s:0.75 n:12 | c:0.69 s:0.64 n:18 |
| `(eda_max_low) => eda_max_low meets valence_medium` | **0.701** | 0.701 | 0.688 | 0.688 | c:0.69 s:0.69 n:11 | c:0.71 s:0.71 n:20 |
| `(valence_high) => valence_high meets eda_medium` | **0.640** | 0.536 | 0.529 | 0.321 | c:0.75 s:0.75 n:12 | c:0.53 s:0.32 n:9 |
| `(valence_high) => valence_high meets eda_std_high` | **0.579** | 0.487 | 0.471 | 0.286 | c:0.69 s:0.69 n:11 | c:0.47 s:0.29 n:8 |
| `(arousal_high) => arousal_high meets eda_max_medium` | **0.564** | 0.527 | 0.462 | 0.429 | c:0.67 s:0.62 n:10 | c:0.46 s:0.43 n:12 |
| `(arousal_high) => arousal_high meets eda_medium` | **0.545** | 0.509 | 0.423 | 0.393 | c:0.67 s:0.62 n:10 | c:0.42 s:0.39 n:11 |
| `(valence_high) => valence_high meets eda_max_medium` | **0.489** | 0.420 | 0.353 | 0.214 | c:0.62 s:0.62 n:10 | c:0.35 s:0.21 n:6 |
| `(valence_high) => valence_high meets hr_low` | **0.485** | 0.393 | 0.471 | 0.286 | c:0.50 s:0.50 n:8 | c:0.47 s:0.29 n:8 |
| `(arousal_high) => arousal_high meets hr_high` | **0.382** | 0.357 | 0.231 | 0.214 | c:0.53 s:0.50 n:8 | c:0.23 s:0.21 n:6 |
| `(valence_high) => valence_high meets arousal_high` | **0.303** | 0.245 | 0.294 | 0.179 | c:0.31 s:0.31 n:5 | c:0.29 s:0.18 n:5 |

## Dataset-Specific Rules

### EMBOA
- Total filtered rules: 127
- Rules unique to EMBOA: **116**

  - `(arousal_high) => arousal_high before eda_low`
  - `(arousal_high) => arousal_high before eda_max_low`
  - `(arousal_high) => arousal_high equals eda_high`
  - `(arousal_high) => arousal_high equals eda_max_high`
  - `(arousal_high) => arousal_high equals eda_std_low`
  - `(arousal_high) => arousal_high equals eda_std_medium`
  - `(arousal_high) => arousal_high equals hr_low`
  - `(arousal_high) => arousal_high equals hr_medium`
  - `(arousal_high) => arousal_high equals valence_high`
  - `(arousal_high) => arousal_high meets eda_std_high`
  - `(arousal_low) => arousal_low before eda_low`
  - `(arousal_low) => arousal_low before eda_max_low`
  - `(arousal_low) => arousal_low before eda_max_medium`
  - `(arousal_low) => arousal_low before hr_high`
  - `(arousal_low) => arousal_low equals eda_high`
  - `(arousal_low) => arousal_low equals eda_max_high`
  - `(arousal_low) => arousal_low equals eda_medium`
  - `(arousal_low) => arousal_low equals eda_std_high`
  - `(arousal_low) => arousal_low equals hr_medium`
  - `(arousal_low) => arousal_low equals valence_low`
  - ... and 96 more

### K-emoCon-ext
- Total filtered rules: 253
- Rules unique to K-emoCon-ext: **242**

  - `(arousal_high) => arousal_high before eda_peaks_high`
  - `(arousal_high) => arousal_high before eda_peaks_low`
  - `(arousal_high) => arousal_high before eda_scr_auc_high`
  - `(arousal_high) => arousal_high equals eda_peaks_medium`
  - `(arousal_high) => arousal_high equals eda_std_high`
  - `(arousal_high) => arousal_high is-finished-by hr_low`
  - `(arousal_high) => arousal_high meets eda_high`
  - `(arousal_high) => arousal_high meets eda_scr_amp_low`
  - `(arousal_high) => arousal_high meets eda_scr_amp_medium`
  - `(arousal_high) => arousal_high meets eda_scr_auc_medium`
  - `(arousal_high) => arousal_high meets eda_std_medium`
  - `(arousal_high) => arousal_high starts eda_max_high`
  - `(arousal_high) => arousal_high starts eda_scr_auc_low`
  - `(arousal_high) => arousal_high starts eda_std_low`
  - `(arousal_high) => arousal_high starts hr_medium`
  - `(arousal_low) => arousal_low before eda_max_high`
  - `(arousal_low) => arousal_low before eda_peaks_low`
  - `(arousal_low) => arousal_low contains eda_scr_amp_medium`
  - `(arousal_low) => arousal_low contains eda_std_high`
  - `(arousal_low) => arousal_low contains eda_std_medium`
  - ... and 222 more
