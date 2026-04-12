#!/usr/bin/env python3
"""
Experiment: State Distribution Analysis

Analyzes the frequency/proportion of each state label across all ARMADA data,
broken down by dataset, gender, and age group. Generates charts for each breakdown.
"""

import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

SCRIPT_DIR = Path(__file__).parent
EXPERIMENTS_DIR = SCRIPT_DIR.parent
PROJECT_DIR = EXPERIMENTS_DIR.parent.parent

sys.path.insert(0, str(EXPERIMENTS_DIR))
sys.path.insert(0, str(EXPERIMENTS_DIR / "demographics"))
sys.path.insert(0, str(PROJECT_DIR / "source" / "processing" / "armada"))
sys.path.insert(0, str(PROJECT_DIR / "source" / "processing" / "extracting"))

from demographic_analysis import load_demographics_from_processed

DATA_DIR = PROJECT_DIR / "data" / "armada_ready"
OUTPUT_DIR = SCRIPT_DIR / "results"

DATASETS = {
    'CASE': DATA_DIR / "armada_case.csv",
    'K-emoCon': DATA_DIR / "armada_k_emocon.csv",
    'CEAP': DATA_DIR / "armada_ceap.csv",
    'EmoWorker_v2': DATA_DIR / "armada_emoworker_v2.csv",
    'K-emo_ext': DATA_DIR / "armada_k_emocon_ext.csv",
    'EMBOA': DATA_DIR / "armada_emboa.csv"
}

FEATURE_CATEGORIES = {
    'arousal': ['arousal_high', 'arousal_medium', 'arousal_low'],
    'valence': ['valence_high', 'valence_medium', 'valence_low'],
    'eda': ['eda_high', 'eda_medium', 'eda_low'],
    'eda_max': ['eda_max_high', 'eda_max_medium', 'eda_max_low'],
    'eda_peaks': ['eda_peaks_high', 'eda_peaks_medium', 'eda_peaks_low'],
    'eda_std': ['eda_std_high', 'eda_std_medium', 'eda_std_low'],
    'eda_scr_amp': ['eda_scr_amp_high', 'eda_scr_amp_medium', 'eda_scr_amp_low'],
    'eda_scr_auc': ['eda_scr_auc_high', 'eda_scr_auc_medium', 'eda_scr_auc_low'],
    'hr': ['hr_high', 'hr_medium', 'hr_low'],
    'hrv_sdnn': ['hrv_sdnn_high', 'hrv_sdnn_medium', 'hrv_sdnn_low'],
    'hrv_rmssd': ['hrv_rmssd_high', 'hrv_rmssd_medium', 'hrv_rmssd_low'],
    'hrv_cvnn': ['hrv_cvnn_high', 'hrv_cvnn_medium', 'hrv_cvnn_low'],
    'hrv_cvsd': ['hrv_cvsd_high', 'hrv_cvsd_medium', 'hrv_cvsd_low'],
    'hrv_pnn20': ['hrv_pnn20_high', 'hrv_pnn20_medium', 'hrv_pnn20_low'],
    'hrv_pnn50': ['hrv_pnn50_high', 'hrv_pnn50_medium', 'hrv_pnn50_low'],
    'temp': ['temp_high', 'temp_medium', 'temp_low'],
}

FEATURE_GROUPS = {
    'Emotion_and_Temp': ['arousal', 'valence', 'temp'],
    'EDA': ['eda', 'eda_max', 'eda_peaks', 'eda_std', 'eda_scr_amp', 'eda_scr_auc'],
    'HR_and_HRV': ['hr', 'hrv_sdnn', 'hrv_rmssd', 'hrv_cvnn', 'hrv_cvsd', 'hrv_pnn20', 'hrv_pnn50']
}

LEVEL_COLORS = {'high': '#e74c3c', 'medium': '#f39c12', 'low': '#3498db'}
LEVEL_ORDER = ['high', 'medium', 'low']


def load_all_data():
    """Loads all ARMADA datasets into one DataFrame with a 'dataset' column."""
    frames = []
    for ds_name, path in DATASETS.items():
        if not path.exists():
            print(f"  SKIP: {path} not found")
            continue
        df = pd.read_csv(path)
        df['dataset'] = ds_name
        frames.append(df)
        print(f"  Loaded {ds_name}: {len(df)} rows, {df['client_id'].nunique()} participants")
    return pd.concat(frames, ignore_index=True)


def extract_feature_level(state: str):
    """Splits state like 'eda_peaks_high' into feature='eda_peaks', level='high'."""
    for level in LEVEL_ORDER:
        if state.endswith(f'_{level}'):
            feature = state[:-(len(level) + 1)]
            return feature, level
    return state, 'unknown'


def count_states(df: pd.DataFrame, group_col: str = None):
    """
    Counts state occurrences normalized per participant.
    Returns DataFrame: state, [group_col], count_per_participant.
    """
    if group_col:
        grouped = df.groupby([group_col, 'state', 'client_id']).size().reset_index(name='count')
        # Average per participant within each group
        result = grouped.groupby([group_col, 'state'])['count'].mean().reset_index()
        result.columns = [group_col, 'state', 'avg_count_per_participant']
    else:
        grouped = df.groupby(['state', 'client_id']).size().reset_index(name='count')
        result = grouped.groupby('state')['count'].mean().reset_index()
        result.columns = ['state', 'avg_count_per_participant']
    return result


def plot_dataset_comparison(all_data: pd.DataFrame, output_dir: Path):
    """Creates grouped bar charts comparing state distributions across datasets."""
    ds_names = sorted(all_data['dataset'].unique())

    for group_name, feats in FEATURE_GROUPS.items():
        n_feats = len(feats)
        cols = 3
        rows = (n_feats + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
        if isinstance(axes, np.ndarray):
            axes = axes.flatten()
        else:
            axes = [axes]

        for idx, feat_name in enumerate(feats):
            ax = axes[idx]
            cat_states = FEATURE_CATEGORIES.get(feat_name, [])
            if not cat_states:
                continue
            x = np.arange(len(cat_states))
            width = 0.8 / max(1, len(ds_names))

            for i, ds in enumerate(ds_names):
                ds_data = all_data[all_data['dataset'] == ds]
                counts = count_states(ds_data)
                vals = []
                for state in cat_states:
                    match = counts[counts['state'] == state]
                    vals.append(match['avg_count_per_participant'].values[0] if len(match) > 0 else 0)
                offset = (i - len(ds_names) / 2 + 0.5) * width
                ax.bar(x + offset, vals, width, label=ds, alpha=0.85)

            ax.set_title(feat_name, fontsize=12, fontweight='bold')
            if idx % cols == 0:
                ax.set_ylabel('Avg intervals/participant')
            ax.set_xticks(x)
            ax.set_xticklabels([s.split('_')[-1] for s in cat_states], fontsize=10)
            if idx == 0:
                ax.legend(fontsize=8)
            ax.grid(axis='y', alpha=0.3)

        for idx in range(n_feats, len(axes)):
            fig.delaxes(axes[idx])

        plt.suptitle(f'{group_name} — State Distribution per Dataset', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(output_dir / f'dataset_{group_name}.png', dpi=150, bbox_inches='tight')
        plt.close()

    print(f"  Saved {len(FEATURE_GROUPS)} dataset comparison charts (grouped)")


def plot_gender_comparison(all_data: pd.DataFrame, demo_df: pd.DataFrame, output_dir: Path):
    """Creates grouped bar charts comparing state distributions by gender."""
    demo_df = demo_df.copy()
    demo_df['gender_clean'] = demo_df['gender'].apply(
        lambda x: 'M' if str(x).upper() in ['M', 'MALE', '1', 'MAN'] else
                  ('F' if str(x).upper() in ['F', 'FEMALE', '2', 'WOMAN'] else None)
    )

    merged = all_data.merge(
        demo_df[['client_id', 'gender_clean']],
        on='client_id', how='left'
    ).dropna(subset=['gender_clean'])

    genders = ['M', 'F']
    gender_colors = {'M': '#2980b9', 'F': '#e74c3c'}

    for group_name, feats in FEATURE_GROUPS.items():
        n_feats = len(feats)
        cols = 3
        rows = (n_feats + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
        if isinstance(axes, np.ndarray):
            axes = axes.flatten()
        else:
            axes = [axes]

        for idx, feat_name in enumerate(feats):
            ax = axes[idx]
            cat_states = FEATURE_CATEGORIES.get(feat_name, [])
            if not cat_states:
                continue
            x = np.arange(len(cat_states))
            width = 0.35

            for i, g in enumerate(genders):
                g_data = merged[merged['gender_clean'] == g]
                counts = count_states(g_data)
                vals = []
                for state in cat_states:
                    match = counts[counts['state'] == state]
                    vals.append(match['avg_count_per_participant'].values[0] if len(match) > 0 else 0)
                offset = (i - 0.5) * width
                ax.bar(x + offset, vals, width, label=g, color=gender_colors[g], alpha=0.85)

            ax.set_title(feat_name, fontsize=12, fontweight='bold')
            if idx % cols == 0:
                ax.set_ylabel('Avg intervals/participant')
            ax.set_xticks(x)
            ax.set_xticklabels([s.split('_')[-1] for s in cat_states], fontsize=10)
            if idx == 0:
                ax.legend(fontsize=10)
            ax.grid(axis='y', alpha=0.3)

        for idx in range(n_feats, len(axes)):
            fig.delaxes(axes[idx])

        plt.suptitle(f'{group_name} — Male vs Female', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(output_dir / f'gender_{group_name}.png', dpi=150, bbox_inches='tight')
        plt.close()

    print(f"  Saved {len(FEATURE_GROUPS)} gender comparison charts (grouped)")


def plot_age_comparison(all_data: pd.DataFrame, demo_df: pd.DataFrame, output_dir: Path):
    """Creates grouped bar charts comparing state distributions by age group."""
    merged = all_data.merge(
        demo_df[['client_id', 'binary_age_group']],
        on='client_id', how='left'
    ).dropna(subset=['binary_age_group'])

    age_groups = ['young', 'old']
    age_colors = {'young': '#27ae60', 'old': '#8e44ad'}

    for group_name, feats in FEATURE_GROUPS.items():
        n_feats = len(feats)
        cols = 3
        rows = (n_feats + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
        if isinstance(axes, np.ndarray):
            axes = axes.flatten()
        else:
            axes = [axes]

        for idx, feat_name in enumerate(feats):
            ax = axes[idx]
            cat_states = FEATURE_CATEGORIES.get(feat_name, [])
            if not cat_states:
                continue
            x = np.arange(len(cat_states))
            width = 0.35

            for i, ag in enumerate(age_groups):
                ag_data = merged[merged['binary_age_group'] == ag]
                counts = count_states(ag_data)
                vals = []
                for state in cat_states:
                    match = counts[counts['state'] == state]
                    vals.append(match['avg_count_per_participant'].values[0] if len(match) > 0 else 0)
                offset = (i - 0.5) * width
                ax.bar(x + offset, vals, width, label=ag, color=age_colors[ag], alpha=0.85)

            ax.set_title(feat_name, fontsize=12, fontweight='bold')
            if idx % cols == 0:
                ax.set_ylabel('Avg intervals/participant')
            ax.set_xticks(x)
            ax.set_xticklabels([s.split('_')[-1] for s in cat_states], fontsize=10)
            if idx == 0:
                ax.legend(fontsize=10)
            ax.grid(axis='y', alpha=0.3)

        for idx in range(n_feats, len(axes)):
            fig.delaxes(axes[idx])

        plt.suptitle(f'{group_name} — Young (≤25) vs Old (>25)', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(output_dir / f'age_{group_name}.png', dpi=150, bbox_inches='tight')
        plt.close()

    print(f"  Saved {len(FEATURE_GROUPS)} age comparison charts (grouped)")


def plot_heatmap(all_data: pd.DataFrame, output_dir: Path):
    """Creates a feature x dataset normalized heatmap."""
    ds_names = sorted(all_data['dataset'].unique())
    features = list(FEATURE_CATEGORIES.keys())

    matrix = np.zeros((len(features), len(ds_names)))

    for j, ds in enumerate(ds_names):
        ds_data = all_data[all_data['dataset'] == ds]
        total = len(ds_data)
        for i, feat in enumerate(features):
            feat_states = FEATURE_CATEGORIES[feat]
            feat_count = ds_data[ds_data['state'].isin(feat_states)].shape[0]
            matrix[i, j] = feat_count / total if total > 0 else 0

    fig, ax = plt.subplots(figsize=(10, 12))
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')

    ax.set_xticks(np.arange(len(ds_names)))
    ax.set_yticks(np.arange(len(features)))
    ax.set_xticklabels(ds_names, fontsize=10)
    ax.set_yticklabels(features, fontsize=10)

    # Annotate cells
    for i in range(len(features)):
        for j in range(len(ds_names)):
            ax.text(j, i, f'{matrix[i, j]:.3f}', ha='center', va='center',
                    fontsize=8, color='black' if matrix[i, j] < 0.08 else 'white')

    ax.set_title('Feature Proportion per Dataset (normalized)', fontsize=14, fontweight='bold')
    fig.colorbar(im, ax=ax, shrink=0.6)
    plt.tight_layout()
    plt.savefig(output_dir / 'heatmap_feature_dataset.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("  Saved heatmap chart")


def generate_report(output_dir: Path):
    """Generates a Markdown report referencing all generated charts."""
    lines = []
    lines.append("# State Distribution Analysis")
    lines.append("")
    lines.append("Analysis of the frequency/proportion of each state label across all ARMADA datasets,")
    lines.append("broken down by dataset, gender (M vs F), and age group (young ≤25 vs old >25).")
    lines.append("")

    lines.append("## Feature x Dataset Heatmap")
    lines.append("")
    lines.append("![Heatmap](heatmap_feature_dataset.png)")
    lines.append("")

    lines.append("## Per-Dataset Comparison")
    lines.append("")
    for group_name in FEATURE_GROUPS.keys():
        lines.append(f"### {group_name.replace('_', ' ')}")
        lines.append(f"![{group_name} dataset](dataset_{group_name}.png)")
        lines.append("")

    lines.append("## Gender Comparison (M vs F)")
    lines.append("")
    for group_name in FEATURE_GROUPS.keys():
        lines.append(f"### {group_name.replace('_', ' ')}")
        lines.append(f"![{group_name} gender](gender_{group_name}.png)")
        lines.append("")

    lines.append("## Age Group Comparison (Young vs Old)")
    lines.append("")
    for group_name in FEATURE_GROUPS.keys():
        lines.append(f"### {group_name.replace('_', ' ')}")
        lines.append(f"![{group_name} age](age_{group_name}.png)")
        lines.append("")

    report_text = "\n".join(lines)
    with open(output_dir / "state_distribution_report.md", "w") as f:
        f.write(report_text)
    print(f"  Report saved: {output_dir / 'state_distribution_report.md'}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("EXPERIMENT: STATE DISTRIBUTION ANALYSIS")
    print("=" * 80)

    # 1. Load data
    print("\n--- Loading ARMADA data ---")
    all_data = load_all_data()
    print(f"Total: {len(all_data)} rows, {all_data['client_id'].nunique()} participants")

    # 2. Load demographics
    print("\n--- Loading demographics ---")
    demo_df = load_demographics_from_processed()
    print(f"  Demographic records: {len(demo_df)}")

    # 3. Generate charts
    print("\n--- Generating per-dataset charts ---")
    plot_dataset_comparison(all_data, OUTPUT_DIR)

    print("\n--- Generating gender charts ---")
    plot_gender_comparison(all_data, demo_df, OUTPUT_DIR)

    print("\n--- Generating age group charts ---")
    plot_age_comparison(all_data, demo_df, OUTPUT_DIR)

    print("\n--- Generating heatmap ---")
    plot_heatmap(all_data, OUTPUT_DIR)

    # 4. Generate report
    print("\n--- Generating report ---")
    generate_report(OUTPUT_DIR)

    print("\n" + "=" * 80)
    print("EXPERIMENT FINISHED")
    print(f"Results saved in: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
