# State Distribution

The main objective of this folder is to analyze the distribution of states encoded during the discretization process of physiological and emotional signals across different models. The script calculates occurrences and helps visualize how feature states map out for each dataset, providing context for subsequent pattern mining and analyses.

### Datasets & Annotations
- **Datasets:** CASE, K-emoCon, CEAP, EmoWorker_v2, EMBOA, K-emoCon (External)
- **Annotations:** Self-annotations (subjective participant ratings) and External annotations (observer ratings).

### Visualizations & Grouping
The script generates aggregated bar charts and heatmaps that logically group features for easier visual comparison:
- **Emotion & Temp:** `arousal`, `valence`, `temp`
- **EDA:** `eda`, `eda_max`, `eda_peaks`, `eda_std`, `eda_scr_amp`, `eda_scr_auc`
- **HR & HRV:** `hr`, `hrv_sdnn`, `hrv_rmssd`, `hrv_cvnn`, `hrv_cvsd`, `hrv_pnn20`, `hrv_pnn50`

Distributions are analyzed and compared across:
- **Datasets** (cross-dataset comparison)
- **Demographics:** Gender (M vs F) and Age Group (Young ≤25 vs Old >25)
