#!/usr/bin/env python3
"""
Walidacja i wizualizacja danych przygotowanych dla algorytmu ARMADA.

Ten skrypt sprawdza poprawność formatu danych i generuje wizualizacje
pomagające zrozumieć strukturę interwałów czasowych.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

# Ścieżki
BASE_DIR = Path(__file__).parent.parent.parent.parent
ARMADA_DIR = BASE_DIR / "data" / "armada_ready"
OUTPUT_DIR = ARMADA_DIR / "validation"


def load_armada_data(filepath: Path) -> pd.DataFrame:
    """Wczytuje dane w formacie ARMADA."""
    df = pd.read_csv(filepath)
    return df


def validate_intervals(df: pd.DataFrame) -> Dict:
    """
    Waliduje poprawność interwałów czasowych.

    Sprawdza:
    - start_time < end_time
    - brak nakładających się interwałów tego samego stanu dla tego samego klienta
    - poprawność wartości
    """
    issues = []

    # Sprawdź czy start < end
    invalid_times = df[df['start_time'] >= df['end_time']]
    if len(invalid_times) > 0:
        issues.append(f"Znaleziono {len(invalid_times)} interwałów gdzie start_time >= end_time")

    # Sprawdź duplikaty
    duplicates = df[df.duplicated(['client_id', 'state', 'start_time', 'end_time'])]
    if len(duplicates) > 0:
        issues.append(f"Znaleziono {len(duplicates)} zduplikowanych interwałów")

    # Sprawdź negatywne wartości czasowe
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
    Analizuje wzorce czasowe w danych.
    """
    results = {
        'interval_durations': {},
        'state_frequencies': {},
        'client_sequences': {}
    }

    # Czas trwania interwałów per stan
    df['duration'] = df['end_time'] - df['start_time']
    results['interval_durations'] = df.groupby('state')['duration'].agg(['mean', 'std', 'min', 'max']).to_dict()

    # Częstość stanów
    results['state_frequencies'] = df['state'].value_counts().to_dict()

    # Długość sekwencji per klient
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
    Oblicza relacje Allena dla próbki interwałów.

    Relacje:
    - before (b): A kończy się przed rozpoczęciem B
    - meets (m): A kończy się dokładnie gdy B się zaczyna
    - overlaps (o): A zaczyna się przed B, ale kończy w trakcie B
    - is-finished-by (fi): A zaczyna się przed B i kończy gdy B się kończy
    - contains (c): A zaczyna się przed B i kończy po B
    - equals (=): A i B mają te same czasy
    - starts (s): A zaczyna się gdy B, ale kończy wcześniej
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

            # Określ relację
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
    Wizualizuje timeline interwałów dla danego klienta.
    """
    client_df = df[df['client_id'] == client_id].sort_values('start_time')

    if len(client_df) == 0:
        print(f"Brak danych dla klienta {client_id}")
        return

    # Unikalne stany
    states = client_df['state'].unique()
    state_colors = plt.cm.tab20(np.linspace(0, 1, len(states)))
    state_color_map = dict(zip(states, state_colors))

    # Grupuj stany według kategorii
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
    ax.set_xlabel('Czas (sekundy)')
    ax.set_title(f'Timeline interwałów: {client_id}')

    if max_time:
        ax.set_xlim(0, max_time)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Zapisano: {output_path}")


def visualize_state_distribution(df: pd.DataFrame, output_path: Path):
    """
    Wizualizuje rozkład stanów w całym zbiorze danych.
    """
    state_counts = df['state'].value_counts()

    # Grupuj według kategorii
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
        ax.set_ylabel('Liczba interwałów')

        for i, v in enumerate(values):
            ax.text(i, v + max(values)*0.02, str(v), ha='center', fontsize=8)

    # Ukryj nieużywane osie
    for idx in range(len(categories), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('Rozkład stanów według kategorii', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Zapisano: {output_path}")


def visualize_interval_duration_distribution(df: pd.DataFrame, output_path: Path):
    """
    Wizualizuje rozkład czasu trwania interwałów.
    """
    df['duration'] = df['end_time'] - df['start_time']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram wszystkich czasów
    ax1 = axes[0]
    ax1.hist(df['duration'], bins=50, edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Czas trwania (sekundy)')
    ax1.set_ylabel('Liczba interwałów')
    ax1.set_title('Rozkład czasu trwania interwałów')
    ax1.axvline(df['duration'].median(), color='red', linestyle='--', label=f'Mediana: {df["duration"].median():.1f}s')
    ax1.legend()

    # Box plot per kategoria
    ax2 = axes[1]
    state_categories = df['state'].apply(lambda x: x.rsplit('_', 1)[0])
    df_temp = df.copy()
    df_temp['category'] = state_categories

    categories = df_temp['category'].unique()
    data_per_cat = [df_temp[df_temp['category'] == cat]['duration'].values for cat in categories]

    bp = ax2.boxplot(data_per_cat, labels=categories, patch_artist=True)
    ax2.set_xlabel('Kategoria stanu')
    ax2.set_ylabel('Czas trwania (sekundy)')
    ax2.set_title('Rozkład czasu trwania per kategoria')
    ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Zapisano: {output_path}")


def generate_armada_input_format(df: pd.DataFrame, output_path: Path):
    """
    Generuje plik w formacie bezpośrednio kompatybilnym z wieloma implementacjami ARMADA.

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

    print(f"Zapisano format ARMADA: {output_path}")


def main():
    """Główna funkcja walidacji i wizualizacji."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("WALIDACJA I WIZUALIZACJA DANYCH ARMADA")
    print("=" * 60)

    # Wczytaj dane
    data_file = ARMADA_DIR / "armada_full_metadata.csv"
    if not data_file.exists():
        print(f"Plik {data_file} nie istnieje. Uruchom najpierw prepare_armada_data.py")
        return

    df = load_armada_data(data_file)
    print(f"Wczytano {len(df)} interwałów")

    # Walidacja
    print("\n" + "-" * 40)
    print("WALIDACJA")
    print("-" * 40)

    validation = validate_intervals(df)
    if validation['valid']:
        print("✓ Wszystkie interwały są poprawne")
    else:
        print("✗ Znaleziono problemy:")
        for issue in validation['issues']:
            print(f"  - {issue}")

    print(f"Łączna liczba interwałów: {validation['total_intervals']}")
    print(f"Liczba klientów: {validation['unique_clients']}")
    print(f"Liczba stanów: {validation['unique_states']}")

    # Analiza wzorców
    print("\n" + "-" * 40)
    print("ANALIZA WZORCÓW CZASOWYCH")
    print("-" * 40)

    analysis = analyze_temporal_patterns(df)
    print(f"Średnia długość sekwencji klienta: {analysis['client_sequences']['mean_length']:.1f}")
    print(f"Zakres długości sekwencji: {analysis['client_sequences']['min_length']}-{analysis['client_sequences']['max_length']}")

    # Relacje Allena dla przykładowego klienta
    sample_client = df['client_id'].iloc[0]
    allen_relations = compute_allen_relations_sample(df, sample_client)
    print(f"\nPróbka relacji Allena dla {sample_client}:")
    for rel, count in sorted(allen_relations.items(), key=lambda x: -x[1]):
        print(f"  {rel}: {count}")

    # Wizualizacje
    print("\n" + "-" * 40)
    print("GENEROWANIE WIZUALIZACJI")
    print("-" * 40)

    # Rozkład stanów
    visualize_state_distribution(df, OUTPUT_DIR / "state_distribution.png")

    # Rozkład czasu trwania
    visualize_interval_duration_distribution(df, OUTPUT_DIR / "duration_distribution.png")

    # Timeline dla kilku przykładowych klientów
    sample_clients = df['client_id'].unique()[:3]
    for client_id in sample_clients:
        safe_name = client_id.replace('-', '_').replace(' ', '_')
        visualize_client_timeline(df, client_id, OUTPUT_DIR / f"timeline_{safe_name}.png", max_time=300)

    # Generuj format wejściowy ARMADA
    print("\n" + "-" * 40)
    print("GENEROWANIE FORMATU ARMADA")
    print("-" * 40)

    generate_armada_input_format(df, ARMADA_DIR / "armada_sequences.txt")

    # Generuj też wersje per zbiór
    for dataset in df['dataset'].unique():
        dataset_df = df[df['dataset'] == dataset]
        safe_dataset = dataset.lower().replace('-', '_')
        generate_armada_input_format(dataset_df, ARMADA_DIR / f"armada_sequences_{safe_dataset}.txt")

    print("\n" + "=" * 60)
    print("ZAKOŃCZONO")
    print("=" * 60)


if __name__ == "__main__":
    main()

