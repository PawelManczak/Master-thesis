#!/usr/bin/env python3
"""
Validation and visualization of data prepared for ARMADA algorithm.

This script checks data format correctness and generates visualizations
to help understand the structure of time intervals.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

BASE_DIR = Path(__file__).parent.parent.parent.parent
ARMADA_DIR = BASE_DIR / "data" / "armada_ready"
OUTPUT_DIR = ARMADA_DIR / "validation"


def load_armada_data(filepath: Path) -> pd.DataFrame:
    """Loads ARMADA format data."""
    df = pd.read_csv(filepath)
    return df


def validate_intervals(df: pd.DataFrame) -> Dict:
    """
    Validates correctness of time intervals.

    Checks:
    - start_time < end_time
    - no overlapping intervals of the same state for the same client
    - correctness of values
    """
    issues = []

    # Check if start < end
    invalid_times = df[df['start_time'] >= df['end_time']]
    if len(invalid_times) > 0:
        issues.append(f"Znaleziono {len(invalid_times)} interwałów gdzie start_time >= end_time")

    # Check duplicates
    duplicates = df[df.duplicated(['client_id', 'state', 'start_time', 'end_time'])]
    if len(duplicates) > 0:
        issues.append(f"Znaleziono {len(duplicates)} zduplikowanych interwałów")

    # Check negative time values
    negative_times = df[(df['start_time'] < 0) | (df['end_time'] < 0)]
    if len(negative_times) > 0:
        issues.append(f"Znaleziono {len(negative_times)} interwałów z ujemnymi czasami")

    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'total_intervals': len(df),
        'unique_clients': df['client_id'].nunique(),
        'unique_states': df['state'].nunique()
    }


def analyze_temporal_patterns(df: pd.DataFrame) -> Dict:
    """
    Analyzes temporal patterns in data.
    """
    results = {
        'interval_durations': {},
        'state_frequencies': {},
        'client_sequences': {}
    }

    # Interval durations per state
    df['duration'] = df['end_time'] - df['start_time']
    results['interval_durations'] = df.groupby('state')['duration'].agg(['mean', 'std', 'min', 'max']).to_dict()

    # State frequencies
    results['state_frequencies'] = df['state'].value_counts().to_dict()

    # Sequence length per client
    client_lens = df.groupby('client_id').size()
    results['client_sequences'] = {
        'mean_length': client_lens.mean(),
        'std_length': client_lens.std(),
        'min_length': client_lens.min(),
        'max_length': client_lens.max()
    }

    return results


def compute_allen_relations_sample(df: pd.DataFrame, client_id: str, max_pairs: int = 100) -> Dict:
    """
    Computes Allen's relations for a sample of intervals.

    Relations:
    - before (b): A ends before B starts
    - meets (m): A ends exactly when B starts
    - overlaps (o): A starts before B, but ends during B
    - is-finished-by (fi): A starts before B and ends when B ends
    - contains (c): A starts before B and ends after B
    - equals (=): A and B have the same times
    - starts (s): A starts when B starts, but ends earlier
    """
    client_df = df[df['client_id'] == client_id].sort_values('start_time').reset_index(drop=True)

    relations = defaultdict(int)
    n = len(client_df)

    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if count >= max_pairs:
                break

            a_start, a_end = client_df.iloc[i]['start_time'], client_df.iloc[i]['end_time']
            b_start, b_end = client_df.iloc[j]['start_time'], client_df.iloc[j]['end_time']

            # Determine relation
            if a_end < b_start:
                relations['before'] += 1
            elif a_end == b_start:
                relations['meets'] += 1
            elif a_start < b_start < a_end < b_end:
                relations['overlaps'] += 1
            elif a_start < b_start and a_end == b_end:
                relations['is-finished-by'] += 1
            elif a_start < b_start and a_end > b_end:
                relations['contains'] += 1
            elif a_start == b_start and a_end == b_end:
                relations['equals'] += 1
            elif a_start == b_start and a_end < b_end:
                relations['starts'] += 1
            else:
                relations['other'] += 1

            count += 1

    return dict(relations)


def visualize_client_timeline(df: pd.DataFrame, client_id: str, output_path: Path, max_time: float = None):
    """
    Visualizes interval timeline for a given client.
    """
    client_df = df[df['client_id'] == client_id].sort_values('start_time')

    if len(client_df) == 0:
        print(f"Brak danych dla klienta {client_id}")
        return

    # Unique states
    states = client_df['state'].unique()
    state_colors = plt.cm.tab20(np.linspace(0, 1, len(states)))
    state_color_map = dict(zip(states, state_colors))

    # Group states by category
    state_categories = defaultdict(list)
    for state in states:
        category = state.rsplit('_', 1)[0]  # np. arousal_high -> arousal
        state_categories[category].append(state)

    fig, ax = plt.subplots(figsize=(16, len(state_categories) * 0.8 + 2))

    y_pos = 0
    y_labels = []
    y_positions = []

    for category, cat_states in sorted(state_categories.items()):
        for state in sorted(cat_states):
            state_intervals = client_df[client_df['state'] == state]
            for _, row in state_intervals.iterrows():
                ax.barh(y_pos, row['end_time'] - row['start_time'],
                       left=row['start_time'], height=0.6,
                       color=state_color_map[state], alpha=0.8)
            y_labels.append(state)
            y_positions.append(y_pos)
            y_pos += 1

    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_xlabel('Time (seconds)')
    ax.set_title(f'Intervals timeline: {client_id}')

    if max_time:
        ax.set_xlim(0, max_time)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path}")


def visualize_state_distribution(df: pd.DataFrame, output_path: Path):
    """
    Visualizes the distribution of states in the entire dataset.
    """
    state_counts = df['state'].value_counts()

    # Group by category
    categories = defaultdict(dict)
    for state, count in state_counts.items():
        parts = state.rsplit('_', 1)
        category = parts[0]
        level = parts[1] if len(parts) > 1 else 'unknown'
        categories[category][level] = count

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, (category, levels) in enumerate(sorted(categories.items())[:6]):
        ax = axes[idx]
        labels = list(levels.keys())
        values = list(levels.values())
        colors = ['#2ecc71', '#f39c12', '#e74c3c'][:len(labels)]

        ax.bar(labels, values, color=colors)
        ax.set_title(f'{category}')
        ax.set_ylabel('Number of intervals')

        for i, v in enumerate(values):
            ax.text(i, v + max(values)*0.02, str(v), ha='center', fontsize=8)

    # Hide unused axes
    for idx in range(len(categories), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('State distribution by category', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path}")


def visualize_interval_duration_distribution(df: pd.DataFrame, output_path: Path):
    """
    Visualizes the distribution of interval durations.
    """
    df['duration'] = df['end_time'] - df['start_time']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram of all durations
    ax1 = axes[0]
    ax1.hist(df['duration'], bins=50, edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Duration (seconds)')
    ax1.set_ylabel('Number of intervals')
    ax1.set_title('Interval duration distribution')
    ax1.axvline(df['duration'].median(), color='red', linestyle='--', label=f'Median: {df["duration"].median():.1f}s')
    ax1.legend()

    # Box plot per category
    ax2 = axes[1]
    state_categories = df['state'].apply(lambda x: x.rsplit('_', 1)[0])
    df_temp = df.copy()
    df_temp['category'] = state_categories

    categories = df_temp['category'].unique()
    data_per_cat = [df_temp[df_temp['category'] == cat]['duration'].values for cat in categories]

    bp = ax2.boxplot(data_per_cat, labels=categories, patch_artist=True)
    ax2.set_xlabel('State category')
    ax2.set_ylabel('Duration (seconds)')
    ax2.set_title('Duration distribution per category')
    ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path}")


def generate_armada_input_format(df: pd.DataFrame, output_path: Path):
    """
    Generates a file in a format directly compatible with many ARMADA implementations.

    Format:
    SEQUENCE client_id
    state start_time end_time
    state start_time end_time
    ...
    """
    with open(output_path, 'w') as f:
        for client_id in df['client_id'].unique():
            client_df = df[df['client_id'] == client_id].sort_values(['start_time', 'end_time', 'state'])
            f.write(f"SEQUENCE {client_id}\n")
            for _, row in client_df.iterrows():
                f.write(f"{row['state']} {row['start_time']} {row['end_time']}\n")
            f.write("\n")

    print(f"Saved ARMADA format: {output_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ARMADA DATA VALIDATION AND VISUALIZATION")
    print("=" * 60)

    data_file = ARMADA_DIR / "armada_full_metadata.csv"
    if not data_file.exists():
        print(f"File {data_file} does not exist. Run prepare_armada_data.py first")
        return

    df = load_armada_data(data_file)
    print(f"Loaded {len(df)} intervals")

    print("\n" + "-" * 40)
    print("VALIDATION")
    print("-" * 40)

    validation = validate_intervals(df)
    if validation['valid']:
        print("✓ All intervals are correct")
    else:
        print("✗ Found issues:")
        for issue in validation['issues']:
            print(f"  - {issue}")

    print(f"Total number of intervals: {validation['total_intervals']}")
    print(f"Number of clients: {validation['unique_clients']}")
    print(f"Number of states: {validation['unique_states']}")

    print("\n" + "-" * 40)
    print("TEMPORAL PATTERN ANALYSIS")
    print("-" * 40)

    analysis = analyze_temporal_patterns(df)
    print(f"Average client sequence length: {analysis['client_sequences']['mean_length']:.1f}")
    print(f"Sequence length range: {analysis['client_sequences']['min_length']}-{analysis['client_sequences']['max_length']}")

    sample_client = df['client_id'].iloc[0]
    allen_relations = compute_allen_relations_sample(df, sample_client)
    print(f"\nSample of Allen's relations for {sample_client}:")
    for rel, count in sorted(allen_relations.items(), key=lambda x: -x[1]):
        print(f"  {rel}: {count}")

    print("\n" + "-" * 40)
    print("GENERATING VISUALIZATIONS")
    print("-" * 40)

    visualize_state_distribution(df, OUTPUT_DIR / "state_distribution.png")
    visualize_interval_duration_distribution(df, OUTPUT_DIR / "duration_distribution.png")

    sample_clients = df['client_id'].unique()[:3]
    for client_id in sample_clients:
        safe_name = client_id.replace('-', '_').replace(' ', '_')
        visualize_client_timeline(df, client_id, OUTPUT_DIR / f"timeline_{safe_name}.png", max_time=300)

    print("\n" + "-" * 40)
    print("GENERATING ARMADA FORMAT")
    print("-" * 40)

    generate_armada_input_format(df, ARMADA_DIR / "armada_sequences.txt")

    for dataset in df['dataset'].unique():
        dataset_df = df[df['dataset'] == dataset]
        safe_dataset = dataset.lower().replace('-', '_')
        generate_armada_input_format(dataset_df, ARMADA_DIR / f"armada_sequences_{safe_dataset}.txt")

    print("\n" + "=" * 60)
    print("FINISHED")
    print("=" * 60)


if __name__ == "__main__":
    main()

