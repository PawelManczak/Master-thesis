# External Annotations Cross-Dataset Experiment

This experiment identifies rules that are universally present
across datasets using **external observer annotations**:
- **EMBOA**: BORIS method II (6 emotions x % agreement -> arousal/valence)
- **K-emoCon external**: Aggregated external annotations (arousal/valence 1–5)

## Experiment Parameters

- **minsup**: 0.3 (30%)
- **minconf**: 0.5 (50%)
- **maxgap**: 5s
- **max_pattern_size**: 4

## Rule Filters

- **FILTER_BVP_ONLY**: True
- **FILTER_EDA_ONLY**: True
- **FILTER_PHYSIO_CROSS**: True
- **FILTER_SINGLE_FEATURE**: True

## Dataset Processing

Evaluated on 2 datasets: EMBOA, K-emoCon-ext

| Dataset | Participants | Rules (Before Filters) | Rules (Filtered) |
|---------|-------------|------------------------|------------------|
| **EMBOA** | 16 | 6798 | 2972 |
| **K-emoCon-ext** | 28 | 3735 | 1633 |

## Universal Rules (Present in ALL Datasets)

Total universal rules found: **7**

### Details (sorted by avg. confidence)

| Rule | Avg Conf | Avg Lift | Avg Sup | Min Conf | Min Lift | Min Sup | EMBOA | K-emoCon-ext |
|---|---|---|---|---|---|---|---|---|
| `(eda_std_high) => eda_std_high starts valence_medium` | **0.871** | 0.871 | 0.871 | 0.812 | 0.812 | 0.812 | c:0.81 l:0.81 s:0.81 n:13 | c:0.93 l:0.93 s:0.93 n:26 |
| `(arousal_high) => arousal_high meets valence_medium` | **0.746** | 0.746 | 0.696 | 0.692 | 0.692 | 0.643 | c:0.80 l:0.80 s:0.75 n:12 | c:0.69 l:0.69 s:0.64 n:18 |
| `(eda_max_low) => eda_max_low meets valence_medium` | **0.701** | 0.701 | 0.701 | 0.688 | 0.688 | 0.688 | c:0.69 l:0.69 s:0.69 n:11 | c:0.71 l:0.71 s:0.71 n:20 |
| `eda_medium meets eda_max_medium AND eda_medium meets eda_medium AND eda_max_medium equals eda_medium => eda_medium meets eda_max_medium AND eda_medium meets eda_medium AND eda_medium meets valence_medium AND eda_max_medium equals eda_medium AND eda_max_medium equals valence_medium AND eda_medium equals valence_medium` | **0.700** | 0.700 | 0.317 | 0.500 | 0.500 | 0.312 | c:0.50 l:0.50 s:0.31 n:5 | c:0.90 l:0.90 s:0.32 n:9 |
| `eda_medium meets eda_medium => eda_medium meets eda_medium AND eda_medium meets valence_medium AND eda_medium equals valence_medium` | **0.691** | 0.691 | 0.424 | 0.500 | 0.500 | 0.312 | c:0.50 l:0.50 s:0.31 n:5 | c:0.88 l:0.88 s:0.54 n:15 |
| `eda_std_medium before eda_std_medium => eda_std_medium before eda_std_medium AND eda_std_medium before valence_medium AND eda_std_medium starts valence_medium` | **0.654** | 0.654 | 0.375 | 0.571 | 0.571 | 0.250 | c:0.57 l:0.57 s:0.25 n:4 | c:0.74 l:0.74 s:0.50 n:14 |
| `(valence_high) => valence_high meets eda_medium` | **0.640** | 0.684 | 0.536 | 0.529 | 0.618 | 0.321 | c:0.75 l:0.75 s:0.75 n:12 | c:0.53 l:0.62 s:0.32 n:9 |

## Dataset-Specific Rules

### EMBOA
- Total filtered rules: 2972
- Rules unique to EMBOA: **2965**

  - `(arousal_high) => arousal_high before eda_low`
  - `(arousal_high) => arousal_high before eda_low AND arousal_high before arousal_medium AND eda_low starts arousal_medium`
  - `(arousal_high) => arousal_high before eda_low AND arousal_high before eda_max_low AND arousal_high before arousal_medium AND eda_low equals eda_max_low AND eda_low starts arousal_medium AND eda_max_low starts arousal_medium`
  - `(arousal_high) => arousal_high before eda_low AND arousal_high before eda_max_low AND arousal_high before valence_medium AND eda_low equals eda_max_low AND eda_low starts valence_medium AND eda_max_low starts valence_medium`
  - `(arousal_high) => arousal_high before eda_low AND arousal_high before eda_max_low AND eda_low equals eda_max_low`
  - `(arousal_high) => arousal_high before eda_low AND arousal_high before valence_medium AND eda_low starts valence_medium`
  - `(arousal_high) => arousal_high before eda_max_low`
  - `(arousal_high) => arousal_high before eda_max_low AND arousal_high before arousal_medium AND eda_max_low starts arousal_medium`
  - `(arousal_high) => arousal_high before eda_max_low AND arousal_high before valence_medium AND eda_max_low starts valence_medium`
  - `(arousal_high) => arousal_high equals eda_std_high AND arousal_high before eda_low AND arousal_high before eda_max_low AND eda_std_high before eda_low AND eda_std_high before eda_max_low AND eda_low equals eda_max_low`
  - `(arousal_high) => arousal_high equals eda_std_high AND arousal_high before eda_low AND eda_std_high before eda_low`
  - `(arousal_high) => arousal_high equals eda_std_high AND arousal_high before eda_max_low AND eda_std_high before eda_max_low`
  - `(arousal_high) => arousal_high equals eda_std_high AND arousal_high equals hr_medium AND arousal_high before arousal_medium AND eda_std_high equals hr_medium AND eda_std_high before arousal_medium AND hr_medium before arousal_medium`
  - `(arousal_high) => arousal_high equals eda_std_high AND arousal_high equals hr_medium AND arousal_high before valence_medium AND eda_std_high equals hr_medium AND eda_std_high before valence_medium AND hr_medium before valence_medium`
  - `(arousal_high) => arousal_high equals eda_std_high AND arousal_high equals hr_medium AND eda_std_high equals hr_medium`
  - `(arousal_high) => arousal_high equals eda_std_low`
  - `(arousal_high) => arousal_high equals eda_std_low AND arousal_high before arousal_medium AND eda_std_low before arousal_medium`
  - `(arousal_high) => arousal_high equals eda_std_low AND arousal_high before valence_medium AND eda_std_low before valence_medium`
  - `(arousal_high) => arousal_high equals hr_low`
  - `(arousal_high) => arousal_high equals hr_low AND arousal_high before arousal_medium AND hr_low before arousal_medium`
  - ... and 2945 more

### K-emoCon-ext
- Total filtered rules: 1633
- Rules unique to K-emoCon-ext: **1626**

  - `(arousal_high) => arousal_high before eda_peaks_high`
  - `(arousal_high) => arousal_high equals eda_peaks_medium`
  - `(arousal_high) => arousal_high equals eda_std_high`
  - `(arousal_high) => arousal_high meets eda_high`
  - `(arousal_high) => arousal_high meets eda_scr_amp_medium`
  - `(arousal_high) => arousal_high meets eda_scr_auc_medium`
  - `(arousal_high) => arousal_high meets eda_std_medium`
  - `(arousal_high) => arousal_high starts eda_scr_auc_low`
  - `(arousal_low) => arousal_low contains eda_scr_amp_medium`
  - `(arousal_low) => arousal_low contains eda_std_high`
  - `(arousal_low) => arousal_low contains eda_std_medium`
  - `(arousal_low) => arousal_low equals eda_max_medium`
  - `(arousal_low) => arousal_low equals eda_max_medium AND arousal_low equals valence_medium AND eda_max_medium equals valence_medium`
  - `(arousal_low) => arousal_low equals valence_medium`
  - `(arousal_low) => arousal_low is-finished-by eda_peaks_high`
  - `(arousal_low) => arousal_low is-finished-by eda_scr_auc_low`
  - `(arousal_low) => arousal_low meets eda_peaks_medium`
  - `(arousal_low) => arousal_low meets eda_scr_auc_medium`
  - `(arousal_medium) => arousal_medium before eda_peaks_high`
  - `(arousal_medium) => arousal_medium equals eda_scr_auc_medium`
  - ... and 1606 more
