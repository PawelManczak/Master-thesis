"""
K-EmoCon Data Processing Script
Łączy dane z Empatica E4 z samoocenami (valence i arousal).
Sygnały agregowane są do okien 5-sekundowych.

Sygnały E4:
- ACC: 32 Hz (x, y, z)
- BVP: 64 Hz
- EDA: 4 Hz
- HR: 1 Hz
- IBI: nieregularne
- TEMP: 4 Hz

Samooceny: co 5 sekund
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Import wspólnych funkcji BVP
from bvp_utils import compute_metrics_from_ibi, compute_metrics_from_bvp

# Ścieżki
BASE_DIR = Path(__file__).parent.parent.parent.parent  # extracting -> processing -> source -> second part
RAW_DIR = BASE_DIR / "data" / "K-emoCon" / "raw"
E4_DIR = RAW_DIR / "e4_data"
ANNOTATIONS_DIR = RAW_DIR / "self_annotations"
OUTPUT_DIR = BASE_DIR / "data" / "K-emoCon" / "processed"


def get_participant_mapping():
    """Mapowanie między folderami E4 a plikami samoocen."""
    e4_folders = sorted([f.name for f in E4_DIR.iterdir() if f.is_dir()])
    annotation_files = sorted([f.stem.replace('.self', '') for f in ANNOTATIONS_DIR.glob("*.csv")])

    # E4 foldery są ponumerowane 1, 4, 5, 8, 9, 10, 11, ...
    # Samooceny są P1, P2, P3, ...
    # Zakładamy że folder e4 odpowiada PID z pliku E4
    return e4_folders


def load_e4_signal(participant_folder: str, signal_name: str) -> pd.DataFrame:
    """Wczytaj sygnał E4 dla danego uczestnika."""
    file_path = E4_DIR / participant_folder / f"E4_{signal_name}.csv"
    if not file_path.exists():
        return None

    df = pd.read_csv(file_path)
    return df


def load_annotations(pid: int) -> pd.DataFrame:
    """Wczytaj samooceny dla danego uczestnika."""
    file_path = ANNOTATIONS_DIR / f"P{pid}.self.csv"
    if not file_path.exists():
        return None

    df = pd.read_csv(file_path)
    # Interesują nas tylko seconds, arousal i valence
    return df[['seconds', 'arousal', 'valence']]


def compute_window_stats_df(df: pd.DataFrame, value_col: str, start_ms: float, end_ms: float) -> dict:
    """
    Oblicz statystyki dla okna czasowego.
    Zwraca średnią i wariancję wartości.
    """
    mask = (df['timestamp'] >= start_ms) & (df['timestamp'] < end_ms)
    window_data = df.loc[mask, value_col]

    if len(window_data) == 0:
        return {'mean': np.nan, 'var': np.nan}

    return {
        'mean': window_data.mean(),
        'var': window_data.var()
    }


def get_ibi_in_window(df: pd.DataFrame, start_ms: float, end_ms: float) -> np.ndarray:
    """Pobierz wartości IBI (w ms) dla okna czasowego."""
    mask = (df['timestamp'] >= start_ms) & (df['timestamp'] < end_ms)
    window_data = df.loc[mask]

    if len(window_data) == 0:
        return np.array([])

    if 'value' in window_data.columns:
        return window_data['value'].values
    return np.array([])


def compute_acc_window_stats(df: pd.DataFrame, start_ms: float, end_ms: float) -> dict:
    """
    Oblicz statystyki dla ACC (3 osie) w oknie czasowym.
    ACC ma kolumny x, y, z zamiast value.
    """
    mask = (df['timestamp'] >= start_ms) & (df['timestamp'] < end_ms)
    window_data = df.loc[mask]

    if len(window_data) == 0:
        return {
            'acc_x_mean': np.nan, 'acc_x_var': np.nan,
            'acc_y_mean': np.nan, 'acc_y_var': np.nan,
            'acc_z_mean': np.nan, 'acc_z_var': np.nan,
            'acc_magnitude_mean': np.nan, 'acc_magnitude_var': np.nan
        }

    result = {}
    for axis in ['x', 'y', 'z']:
        if axis in window_data.columns:
            result[f'acc_{axis}_mean'] = window_data[axis].mean()
            result[f'acc_{axis}_var'] = window_data[axis].var()
        else:
            result[f'acc_{axis}_mean'] = np.nan
            result[f'acc_{axis}_var'] = np.nan

    # Magnitude
    if all(ax in window_data.columns for ax in ['x', 'y', 'z']):
        magnitude = np.sqrt(window_data['x']**2 + window_data['y']**2 + window_data['z']**2)
        result['acc_magnitude_mean'] = magnitude.mean()
        result['acc_magnitude_var'] = magnitude.var()
    else:
        result['acc_magnitude_mean'] = np.nan
        result['acc_magnitude_var'] = np.nan

    return result


def process_participant(e4_folder: str, pid: int) -> pd.DataFrame:
    """
    Przetwarza dane dla jednego uczestnika.
    """
    print(f"Przetwarzanie uczestnika P{pid} (folder E4: {e4_folder})...")

    # Wczytaj samooceny
    annotations = load_annotations(pid)
    if annotations is None:
        print(f"  Brak samoocen dla P{pid}")
        return None

    # Wczytaj sygnały E4
    signals = {}
    for signal_name in ['EDA', 'HR', 'TEMP', 'BVP', 'IBI']:
        df = load_e4_signal(e4_folder, signal_name)
        if df is not None:
            signals[signal_name] = df

    # Osobno ACC (3 osie)
    acc_df = load_e4_signal(e4_folder, 'ACC')
    if acc_df is not None:
        if 'x' in acc_df.columns:
            signals['ACC'] = acc_df
        elif 'value' in acc_df.columns:
            signals['ACC'] = acc_df

    if not signals:
        print(f"  Brak sygnałów E4 dla uczestnika {pid}")
        return None

    # Znajdź początkowy timestamp (najwcześniejszy ze wszystkich sygnałów)
    start_timestamps = []
    for sig_name, sig_df in signals.items():
        if 'timestamp' in sig_df.columns and len(sig_df) > 0:
            start_timestamps.append(sig_df['timestamp'].min())

    if not start_timestamps:
        print(f"  Brak timestampów dla uczestnika {pid}")
        return None

    start_ts = min(start_timestamps)

    # Przetwórz każde okno 5-sekundowe
    results = []

    for _, row in annotations.iterrows():
        seconds = row['seconds']
        arousal = row['arousal']
        valence = row['valence']

        # Okno czasowe: [seconds-5, seconds) w sekundach, przekształcone na ms
        # Samoocena w sekundzie X odpowiada oknu [X-5, X)
        window_start_ms = start_ts + (seconds - 5) * 1000
        window_end_ms = start_ts + seconds * 1000

        record = {
            'seconds': seconds,
            'arousal': arousal,
            'valence': valence
        }

        # EDA
        if 'EDA' in signals:
            stats = compute_window_stats_df(signals['EDA'], 'value', window_start_ms, window_end_ms)
            record['eda'] = stats['mean']
            record['eda_var'] = stats['var']
        else:
            record['eda'] = np.nan
            record['eda_var'] = np.nan

        # HR
        if 'HR' in signals:
            stats = compute_window_stats_df(signals['HR'], 'value', window_start_ms, window_end_ms)
            record['hr'] = stats['mean']
            record['hr_var'] = stats['var']
        else:
            record['hr'] = np.nan
            record['hr_var'] = np.nan

        # TEMP
        if 'TEMP' in signals:
            stats = compute_window_stats_df(signals['TEMP'], 'value', window_start_ms, window_end_ms)
            record['temp'] = stats['mean']
            record['temp_var'] = stats['var']
        else:
            record['temp'] = np.nan
            record['temp_var'] = np.nan

        # BVP metrics z IBI
        if 'IBI' in signals:
            ibi_values = get_ibi_in_window(signals['IBI'], window_start_ms, window_end_ms)
            bvp_metrics = compute_metrics_from_ibi(ibi_values)
            record.update(bvp_metrics)
        else:
            record['bvp_sdnn'] = np.nan
            record['bvp_rmssd'] = np.nan
            record['bvp_pnn50'] = np.nan
            record['bvp_mean_hr'] = np.nan
            record['bvp_mean_ibi'] = np.nan
            record['bvp_lf_power'] = np.nan
            record['bvp_hf_power'] = np.nan
            record['bvp_lf_hf_ratio'] = np.nan

        # BVP
        if 'BVP' in signals:
            stats = compute_window_stats_df(signals['BVP'], 'value', window_start_ms, window_end_ms)
            record['bvp_mean'] = stats['mean']
            record['bvp_var'] = stats['var']

        # ACC
        if 'ACC' in signals:
            if 'x' in signals['ACC'].columns:
                acc_stats = compute_acc_window_stats(signals['ACC'], window_start_ms, window_end_ms)
                record.update(acc_stats)
            else:
                stats = compute_window_stats_df(signals['ACC'], 'value', window_start_ms, window_end_ms)
                record['acc_mean'] = stats['mean']
                record['acc_var'] = stats['var']

        results.append(record)

    return pd.DataFrame(results)


def main():
    """Główna funkcja przetwarzania."""
    # Utwórz folder wyjściowy
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Pobierz listę folderów E4
    e4_folders = sorted([f.name for f in E4_DIR.iterdir() if f.is_dir()], key=lambda x: int(x))

    print(f"Znaleziono {len(e4_folders)} folderów E4")
    print(f"Foldery E4: {e4_folders}")

    # Dla każdego folderu E4, wczytaj PID z pierwszego pliku
    for folder in e4_folders:
        # Wczytaj dowolny plik E4 żeby poznać pid
        sample_file = E4_DIR / folder / "E4_HR.csv"
        if not sample_file.exists():
            sample_file = E4_DIR / folder / "E4_EDA.csv"

        if not sample_file.exists():
            print(f"Pominięto folder {folder} - brak plików")
            continue

        sample_df = pd.read_csv(sample_file, nrows=1)
        pid = int(sample_df['pid'].iloc[0])

        # Przetwórz uczestnika
        result_df = process_participant(folder, pid)

        if result_df is not None and len(result_df) > 0:
            output_file = OUTPUT_DIR / f"P{pid}_merged.csv"
            result_df.to_csv(output_file, index=False)
            print(f"  Zapisano {len(result_df)} rekordów do {output_file}")
        else:
            print(f"  Brak danych do zapisania dla P{pid}")

    print("\nPrzetwarzanie zakończone!")


if __name__ == "__main__":
    main()
