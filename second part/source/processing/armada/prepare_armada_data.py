#!/usr/bin/env python3
"""
Prepare Data for ARMADA Algorithm

Ten skrypt przekształca dane z trzech zbiorów (CASE, K-emoCon, CEAP) do formatu
wymaganego przez algorytm ARMADA do odkrywania wzorców czasowych.

Format wyjściowy ARMADA wymaga:
- client-id (identyfikator uczestnika)
- state (stan - np. arousal_high, valence_low, eda_high)
- start-time
- end-time

Normalizacja skal:
- CASE: arousal/valence 0.5-9.5 -> 0-1
- K-emoCon: arousal/valence 1-5 -> 0-1
- CEAP: arousal/valence 1-9 -> 0-1

Dyskretyzacja stanów:
- low: 0-0.33
- medium: 0.33-0.67
- high: 0.67-1.0
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Ścieżki
BASE_DIR = Path(__file__).parent.parent.parent.parent
CASE_PROCESSED = BASE_DIR / "data" / "CASE" / "processed"
KEMOCON_PROCESSED = BASE_DIR / "data" / "K-emoCon" / "processed"
CEAP_PROCESSED = BASE_DIR / "data" / "CEAP" / "processed"
OUTPUT_DIR = BASE_DIR / "data" / "armada_ready"

# Stałe dla normalizacji
SCALE_RANGES = {
    'CASE': {'arousal': (0.5, 9.5), 'valence': (0.5, 9.5)},
    'K-emoCon': {'arousal': (1, 5), 'valence': (1, 5)},
    'CEAP': {'arousal': (-1, 1), 'valence': (-1, 1)}  # CEAP używa Raw data w zakresie [-1, 1]
}

# Progi dyskretyzacji (dla skali 0-1)
# Oparte na przeglądzie metod rozpoznawania emocji z sygnałów fizjologicznych (Ahmad et al.)
# SAM (Self-Assessment Manikin) 1-9 pogrupowane jako:
#   1-3: negatywne/niskie pobudzenie → [0.00, 0.25]
#   4-6: neutralne/umiarkowane → (0.25, 0.75)
#   7-9: pozytywne/wysokie pobudzenie → [0.75, 1.00]
#
# Interpretacja dla valence:
#   low (0.00-0.25): wyraźnie nieprzyjemne (smutek, złość, wstręt)
#   medium (0.25-0.75): brak silnego nacechowania, stany obojętne lub mieszane
#   high (0.75-1.00): wyraźnie przyjemne (radość, zachwyt, zadowolenie)
#
# Interpretacja dla arousal:
#   low (0.00-0.25): bardzo niskie pobudzenie (relaks, senność, nuda)
#   medium (0.25-0.75): umiarkowane pobudzenie (codzienna czujność, koncentracja)
#   high (0.75-1.00): wysokie pobudzenie (eksytacja lub stres, zależnie od valence)
DISCRETIZE_THRESHOLDS = {
    'low': (0, 0.25),
    'medium': (0.25, 0.75),
    'high': (0.75, 1.0)
}

# Zmienne fizjologiczne do dyskretyzacji (użyjemy percentyli)
# Zaktualizowane nazwy zgodnie z nową metodologią "Global Processing, Local Aggregation"
PHYSIO_VARIABLES = ['eda_mean', 'hr_mean', 'temp_mean', 'hrv_rmssd', 'hrv_sdnn']

# Mapowanie nazw kolumn na czytelne nazwy stanów
VARIABLE_NAME_MAPPING = {
    'eda_mean': 'eda',
    'hr_mean': 'hr',
    'temp_mean': 'temp',
    'hrv_rmssd': 'hrv_rmssd',
    'hrv_sdnn': 'hrv_sdnn',
    'arousal_norm': 'arousal',
    'valence_norm': 'valence'
}

# Okno czasowe (sekundy)
WINDOW_SIZE = 5


def normalize_value(value: float, min_val: float, max_val: float) -> float:
    """Normalizuje wartość do skali 0-1."""
    if pd.isna(value):
        return np.nan
    return (value - min_val) / (max_val - min_val)


def discretize_value(value: float, thresholds: Dict[str, Tuple[float, float]]) -> Optional[str]:
    """Dyskretyzuje wartość do kategorii na podstawie progów."""
    if pd.isna(value):
        return None

    for label, (low, high) in thresholds.items():
        if low <= value < high:
            return label
        if label == 'high' and value >= high - 0.001:  # Obsłuż wartość 1.0
            return label
    return None


def compute_physio_thresholds(df: pd.DataFrame, variable: str) -> Dict[str, Tuple[float, float]]:
    """
    Oblicza progi dyskretyzacji dla zmiennej fizjologicznej na podstawie percentyli.
    Używa tercyli (33% i 67%).
    """
    values = df[variable].dropna()
    if len(values) < 3:
        return None

    p33 = values.quantile(0.33)
    p67 = values.quantile(0.67)
    min_val = values.min()
    max_val = values.max()

    return {
        'low': (min_val, p33),
        'medium': (p33, p67),
        'high': (p67, max_val + 0.001)
    }


def extract_state_intervals(
    df: pd.DataFrame,
    variable: str,
    thresholds: Dict[str, Tuple[float, float]],
    prefix: str = ""
) -> List[Dict]:
    """
    Ekstrahuje interwały stanów z szeregu czasowego.

    Zwraca listę słowników z polami:
    - state: nazwa stanu (np. "arousal_high")
    - start_time: czas rozpoczęcia interwału
    - end_time: czas zakończenia interwału
    """
    intervals = []

    if variable not in df.columns:
        return intervals

    # Sortuj po czasie
    df = df.copy().sort_values('seconds').reset_index(drop=True)

    # Dyskretyzuj wartości
    states = []
    for idx, row in df.iterrows():
        value = row[variable]
        state = discretize_value(value, thresholds)
        states.append(state)

    df['_state'] = states

    # Znajdź interwały - pomijamy wartości None
    current_state = None
    start_time = None

    for idx, row in df.iterrows():
        state = row['_state']
        time = row['seconds']

        # Pomijaj wartości None
        if state is None:
            # Zamknij poprzedni interwał jeśli istnieje
            if current_state is not None and start_time is not None:
                end_time = time
                if start_time < end_time:
                    state_name = f"{prefix}{variable}_{current_state}" if prefix else f"{variable}_{current_state}"
                    intervals.append({
                        'state': state_name,
                        'start_time': start_time,
                        'end_time': end_time
                    })
            current_state = None
            start_time = None
            continue

        if state != current_state:
            # Zamknij poprzedni interwał
            if current_state is not None and start_time is not None:
                end_time = time
                if start_time < end_time:
                    state_name = f"{prefix}{variable}_{current_state}" if prefix else f"{variable}_{current_state}"
                    intervals.append({
                        'state': state_name,
                        'start_time': start_time,
                        'end_time': end_time
                    })

            # Rozpocznij nowy interwał
            current_state = state
            start_time = time

    # Zamknij ostatni interwał
    if current_state is not None and start_time is not None:
        last_time = df['seconds'].iloc[-1] + WINDOW_SIZE
        if start_time < last_time:
            state_name = f"{prefix}{variable}_{current_state}" if prefix else f"{variable}_{current_state}"
            intervals.append({
                'state': state_name,
                'start_time': start_time,
                'end_time': last_time
            })

    return intervals


def process_participant_data(
    df: pd.DataFrame,
    dataset_name: str,
    participant_id: str
) -> pd.DataFrame:
    """
    Przetwarza dane uczestnika i konwertuje do formatu ARMADA.
    """
    if df is None or len(df) == 0:
        return None

    # Normalizuj arousal i valence
    scale = SCALE_RANGES[dataset_name]
    df = df.copy()

    if 'arousal' in df.columns:
        df['arousal_norm'] = df['arousal'].apply(
            lambda x: normalize_value(x, scale['arousal'][0], scale['arousal'][1])
        )
    if 'valence' in df.columns:
        df['valence_norm'] = df['valence'].apply(
            lambda x: normalize_value(x, scale['valence'][0], scale['valence'][1])
        )

    # Zbierz wszystkie interwały
    all_intervals = []

    # Interwały dla arousal (znormalizowanego)
    if 'arousal_norm' in df.columns:
        intervals = extract_state_intervals(df, 'arousal_norm', DISCRETIZE_THRESHOLDS, prefix='')
        # Zmień nazwę zmiennej na arousal
        for iv in intervals:
            iv['state'] = iv['state'].replace('arousal_norm', 'arousal')
        all_intervals.extend(intervals)

    # Interwały dla valence (znormalizowanego)
    if 'valence_norm' in df.columns:
        intervals = extract_state_intervals(df, 'valence_norm', DISCRETIZE_THRESHOLDS, prefix='')
        for iv in intervals:
            iv['state'] = iv['state'].replace('valence_norm', 'valence')
        all_intervals.extend(intervals)

    # Interwały dla zmiennych fizjologicznych (tercyle)
    for var in PHYSIO_VARIABLES:
        if var in df.columns:
            thresholds = compute_physio_thresholds(df, var)
            if thresholds is not None:
                intervals = extract_state_intervals(df, var, thresholds)
                # Mapuj nazwę zmiennej na czytelną nazwę stanu
                readable_name = VARIABLE_NAME_MAPPING.get(var, var)
                for iv in intervals:
                    iv['state'] = iv['state'].replace(var, readable_name)
                all_intervals.extend(intervals)

    if not all_intervals:
        return None

    # Konwertuj do DataFrame
    result_df = pd.DataFrame(all_intervals)
    result_df['client_id'] = f"{dataset_name}_{participant_id}"
    result_df['dataset'] = dataset_name
    result_df['participant_id'] = participant_id

    # Usuń interwały gdzie start_time >= end_time
    result_df = result_df[result_df['start_time'] < result_df['end_time']]

    # Usuń duplikaty
    result_df = result_df.drop_duplicates(subset=['state', 'start_time', 'end_time'])

    # Uporządkuj kolumny
    result_df = result_df[['client_id', 'dataset', 'participant_id', 'state', 'start_time', 'end_time']]

    # Sortuj po start_time, end_time, state
    result_df = result_df.sort_values(['start_time', 'end_time', 'state']).reset_index(drop=True)

    return result_df


def process_dataset(dataset_name: str, data_dir: Path) -> List[pd.DataFrame]:
    """Przetwarza wszystkie pliki z danego zbioru danych."""
    results = []

    csv_files = list(data_dir.glob("*_merged.csv"))
    print(f"\n{dataset_name}: Znaleziono {len(csv_files)} plików")

    for csv_file in sorted(csv_files):
        # Wyodrębnij ID uczestnika
        participant_id = csv_file.stem.replace("_merged", "")

        try:
            df = pd.read_csv(csv_file)
            result = process_participant_data(df, dataset_name, participant_id)

            if result is not None and len(result) > 0:
                results.append(result)
                print(f"  {participant_id}: {len(result)} interwałów")
            else:
                print(f"  {participant_id}: Brak danych")
        except Exception as e:
            print(f"  {participant_id}: Błąd - {e}")

    return results


def create_combined_dataset(all_results: List[pd.DataFrame]) -> pd.DataFrame:
    """Łączy wszystkie wyniki w jeden zbiór danych."""
    if not all_results:
        return None

    combined = pd.concat(all_results, ignore_index=True)
    return combined


def save_armada_format(df: pd.DataFrame, output_path: Path, format_type: str = 'csv'):
    """
    Zapisuje dane w formacie odpowiednim dla ARMADA.

    Format CSV:
    client_id, state, start_time, end_time
    """
    # Format podstawowy CSV
    armada_df = df[['client_id', 'state', 'start_time', 'end_time']].copy()
    armada_df = armada_df.sort_values(['client_id', 'start_time', 'end_time', 'state'])

    # Zapisz z nagłówkiem
    with open(output_path, 'w') as f:
        f.write("client_id,state,start_time,end_time\n")
        for _, row in armada_df.iterrows():
            f.write(f"{row['client_id']},{row['state']},{row['start_time']},{row['end_time']}\n")

    print(f"Zapisano: {output_path}")


def generate_summary_statistics(df: pd.DataFrame) -> Dict:
    """Generuje statystyki podsumowujące dla zbioru danych."""
    stats = {
        'total_intervals': len(df),
        'unique_clients': df['client_id'].nunique(),
        'unique_states': df['state'].nunique(),
        'states_distribution': df['state'].value_counts().to_dict(),
        'datasets_distribution': df['dataset'].value_counts().to_dict(),
        'avg_interval_duration': (df['end_time'] - df['start_time']).mean(),
    }
    return stats


def main():
    """Główna funkcja przetwarzania."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PRZYGOTOWANIE DANYCH DLA ALGORYTMU ARMADA")
    print("=" * 60)

    all_results = []

    # Przetwarzanie CASE
    if CASE_PROCESSED.exists():
        case_results = process_dataset('CASE', CASE_PROCESSED)
        all_results.extend(case_results)

    # Przetwarzanie K-emoCon
    if KEMOCON_PROCESSED.exists():
        kemocon_results = process_dataset('K-emoCon', KEMOCON_PROCESSED)
        all_results.extend(kemocon_results)

    # Przetwarzanie CEAP
    if CEAP_PROCESSED.exists():
        ceap_results = process_dataset('CEAP', CEAP_PROCESSED)
        all_results.extend(ceap_results)

    if not all_results:
        print("\nBrak danych do przetworzenia!")
        return

    # Połącz wszystkie wyniki
    print("\n" + "=" * 60)
    print("ŁĄCZENIE DANYCH")
    print("=" * 60)

    combined_df = create_combined_dataset(all_results)
    print(f"Łącznie: {len(combined_df)} interwałów z {combined_df['client_id'].nunique()} uczestników")

    # Zapisz dane
    print("\n" + "=" * 60)
    print("ZAPISYWANIE DANYCH")
    print("=" * 60)

    # 1. Pełne dane połączone
    save_armada_format(
        combined_df,
        OUTPUT_DIR / "armada_combined_all.csv"
    )

    # 2. Dane per zbiór
    for dataset in ['CASE', 'K-emoCon', 'CEAP']:
        dataset_df = combined_df[combined_df['dataset'] == dataset]
        if len(dataset_df) > 0:
            save_armada_format(
                dataset_df,
                OUTPUT_DIR / f"armada_{dataset.lower().replace('-', '_')}.csv"
            )

    # 3. Zapisz pełne metadane
    combined_df.to_csv(OUTPUT_DIR / "armada_full_metadata.csv", index=False)
    print(f"Zapisano: {OUTPUT_DIR / 'armada_full_metadata.csv'}")

    # 4. Statystyki
    print("\n" + "=" * 60)
    print("STATYSTYKI")
    print("=" * 60)

    stats = generate_summary_statistics(combined_df)
    print(f"Łączna liczba interwałów: {stats['total_intervals']}")
    print(f"Liczba uczestników: {stats['unique_clients']}")
    print(f"Liczba unikalnych stanów: {stats['unique_states']}")
    print(f"Średni czas trwania interwału: {stats['avg_interval_duration']:.2f}s")

    print("\nRozkład stanów:")
    for state, count in sorted(stats['states_distribution'].items(), key=lambda x: -x[1])[:15]:
        print(f"  {state}: {count}")

    print("\nRozkład zbiorów danych:")
    for dataset, count in stats['datasets_distribution'].items():
        print(f"  {dataset}: {count}")

    # Zapisz statystyki
    stats_df = pd.DataFrame([{
        'metric': k,
        'value': str(v) if isinstance(v, dict) else v
    } for k, v in stats.items()])
    stats_df.to_csv(OUTPUT_DIR / "armada_statistics.csv", index=False)

    print("\n" + "=" * 60)
    print("ZAKOŃCZONO")
    print("=" * 60)


if __name__ == "__main__":
    main()

