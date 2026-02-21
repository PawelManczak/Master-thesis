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

METODOLOGIA PRZETWARZANIA (zgodna z publikacjami naukowymi):
===========================================================================

ZŁOTA ZASADA: "Global Processing, Local Aggregation"
- Nie liczymy HR/metryk w małych oknach 5s (za mało danych)
- Przetwarzamy cały sygnał, tworzymy ciągły przebieg, potem agregujemy

AGREGACJA SYGNAŁÓW DO OKIEN 5-SEKUNDOWYCH:
------------------------------------------
| Sygnał | Problem ze średnią              | Funkcje agregujące                    |
|--------|----------------------------------|---------------------------------------|
| EDA    | Tracisz piki stresu (SCR)       | mean, std, max, peaks_count           |
| BVP    | Średnia→0 (sygnał falowy!)      | std (amplituda), spectral_power       |
| TEMP   | Zmienia się wolno, OK           | mean, slope (trend)                   |
| ACC    | Średnia=grawitacja/pozycja      | mean (pozycja), std (ruch/drżenie)    |

Output: seconds, arousal, valence, eda_mean, eda_std, eda_max, eda_peaks,
        bvp_std, bvp_spectral_power, temp_mean, temp_slope, acc_*, hr_mean, hr_std, hrv_*
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Import wspólnych funkcji do obliczania cech
from feature_utils import (
    compute_eda_features,
    compute_bvp_features,
    compute_temp_features,
    compute_acc_features,
    compute_global_hr_from_ibi,
    compute_hr_window_features,
    compute_hrv_window_features
)

# Ścieżki
BASE_DIR = Path(__file__).parent.parent.parent.parent  # extracting -> processing -> source -> second part
RAW_DIR = BASE_DIR / "data" / "K-emoCon" / "raw"
E4_DIR = RAW_DIR / "e4_data"
ANNOTATIONS_DIR = RAW_DIR / "self_annotations"
OUTPUT_DIR = BASE_DIR / "data" / "K-emoCon" / "processed"

# Stałe częstotliwości próbkowania
FS_EDA = 4.0    # Hz
FS_HR = 1.0     # Hz
FS_TEMP = 4.0   # Hz
FS_BVP = 64.0   # Hz
FS_ACC = 32.0   # Hz


# =============================================================================
# WRAPPERY DLA FEATURE_UTILS (specyficzne dla formatu K-EmoCon)
# =============================================================================

def compute_global_hr_from_ibi_kemocon(ibi_df: pd.DataFrame, start_ts: float, max_time_ms: float) -> tuple:
    """
    Wrapper dla compute_global_hr_from_ibi dla formatu K-EmoCon.

    Args:
        ibi_df: DataFrame z kolumnami 'timestamp' i 'value' (IBI w ms)
        start_ts: początkowy timestamp (ms)
        max_time_ms: maksymalny czas (ms od start_ts)

    Returns:
        tuple (time_grid_ms, hr_timeseries) - czas w ms i HR w BPM
    """
    if ibi_df is None or len(ibi_df) < 2:
        return np.array([]), np.array([])

    timestamps = ibi_df['timestamp'].values
    ibi_values = ibi_df['value'].values  # IBI w ms

    return compute_global_hr_from_ibi(timestamps, ibi_values, start_ts + max_time_ms, ibi_unit='ms')


def compute_hrv_window_features_kemocon(ibi_df: pd.DataFrame, window_start_ms: float, window_end_ms: float) -> dict:
    """
    Wrapper dla compute_hrv_window_features dla formatu K-EmoCon.

    Args:
        ibi_df: DataFrame z kolumnami 'timestamp' i 'value'
        window_start_ms: początek okna (ms)
        window_end_ms: koniec okna (ms)

    Returns:
        dict z metrykami HRV
    """
    if ibi_df is None or len(ibi_df) < 3:
        return {
            'hrv_sdnn': np.nan, 'hrv_rmssd': np.nan, 'hrv_pnn50': np.nan,
            'hrv_lf_power': np.nan, 'hrv_hf_power': np.nan, 'hrv_lf_hf_ratio': np.nan
        }

    timestamps = ibi_df['timestamp'].values
    ibi_values = ibi_df['value'].values  # IBI w ms

    return compute_hrv_window_features(timestamps, ibi_values, window_start_ms, window_end_ms, ibi_unit='ms')


# =============================================================================
# FUNKCJE WCZYTYWANIA DANYCH
# =============================================================================

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

    Stosuje metodologię "Global Processing, Local Aggregation":
    - Najpierw przetwarza cały sygnał IBI (oblicza globalny przebieg HR)
    - Następnie agreguje do okien 5-sekundowych

    Agregacja sygnałów:
    - EDA: mean, std, max, peaks (tracisz piki stresu przy zwykłej średniej)
    - BVP: std (amplituda), spectral_power, peak_to_peak (NIE średnia!)
    - TEMP: mean, slope (zmienia się wolno)
    - ACC: mean (pozycja), std (ruch)
    - HR: mean, std z globalnie obliczonego przebiegu HR
    - HRV: sdnn, rmssd, pnn50, lf/hf z IBI
    """
    print(f"Przetwarzanie uczestnika P{pid} (folder E4: {e4_folder})...")

    # Wczytaj samooceny
    annotations = load_annotations(pid)
    if annotations is None:
        print(f"  Brak samoocen dla P{pid}")
        return None

    # Wczytaj sygnały E4
    signals = {}
    for signal_name in ['EDA', 'TEMP', 'BVP', 'IBI']:
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

    # Znajdź maksymalny czas
    max_seconds = annotations['seconds'].max()
    max_time_ms = max_seconds * 1000

    # =================================================================
    # GLOBAL PROCESSING: Oblicz ciągły przebieg HR z całego nagrania
    # =================================================================
    time_grid, hr_timeseries = np.array([]), np.array([])
    if 'IBI' in signals:
        time_grid, hr_timeseries = compute_global_hr_from_ibi_kemocon(
            signals['IBI'], start_ts, max_time_ms
        )
        print(f"  Global Processing: obliczono przebieg HR ({len(hr_timeseries)} próbek)")

    # =================================================================
    # LOCAL AGGREGATION: Okna 5-sekundowe
    # =================================================================
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

        # -----------------------------------------------------------------
        # EDA: mean, std, max, peaks (tracisz piki stresu przy zwykłej średniej!)
        # -----------------------------------------------------------------
        if 'EDA' in signals:
            eda_df = signals['EDA']
            mask = (eda_df['timestamp'] >= window_start_ms) & (eda_df['timestamp'] < window_end_ms)
            window_eda = eda_df.loc[mask, 'value'].values
            eda_features = compute_eda_features(window_eda, FS_EDA)
            record.update(eda_features)
        else:
            record.update({'eda_mean': np.nan, 'eda_std': np.nan, 'eda_max': np.nan, 'eda_peaks': 0})

        # -----------------------------------------------------------------
        # BVP: std (amplituda), spectral_power (NIE średnia - sygnał falowy!)
        # -----------------------------------------------------------------
        if 'BVP' in signals:
            bvp_df = signals['BVP']
            mask = (bvp_df['timestamp'] >= window_start_ms) & (bvp_df['timestamp'] < window_end_ms)
            window_bvp = bvp_df.loc[mask, 'value'].values
            bvp_features = compute_bvp_features(window_bvp, FS_BVP)
            record.update(bvp_features)
        else:
            record.update({'bvp_std': np.nan, 'bvp_peak_to_peak': np.nan, 'bvp_spectral_power': np.nan})

        # -----------------------------------------------------------------
        # TEMP: mean, slope (zmienia się wolno)
        # -----------------------------------------------------------------
        if 'TEMP' in signals:
            temp_df = signals['TEMP']
            mask = (temp_df['timestamp'] >= window_start_ms) & (temp_df['timestamp'] < window_end_ms)
            window_temp = temp_df.loc[mask, 'value'].values
            window_temp_ts = temp_df.loc[mask, 'timestamp'].values
            temp_features = compute_temp_features(window_temp, window_temp_ts)
            record.update(temp_features)
        else:
            record.update({'temp_mean': np.nan, 'temp_slope': np.nan})

        # -----------------------------------------------------------------
        # ACC: mean (pozycja ciała), std (intensywność ruchu)
        # -----------------------------------------------------------------
        if 'ACC' in signals and 'x' in signals['ACC'].columns:
            acc_df = signals['ACC']
            mask = (acc_df['timestamp'] >= window_start_ms) & (acc_df['timestamp'] < window_end_ms)
            window_acc = acc_df.loc[mask]
            acc_x = window_acc['x'].values if 'x' in window_acc.columns else np.array([])
            acc_y = window_acc['y'].values if 'y' in window_acc.columns else np.array([])
            acc_z = window_acc['z'].values if 'z' in window_acc.columns else np.array([])
            acc_features = compute_acc_features(acc_x, acc_y, acc_z)
            record.update(acc_features)
        else:
            record.update({
                'acc_x_mean': np.nan, 'acc_x_std': np.nan,
                'acc_y_mean': np.nan, 'acc_y_std': np.nan,
                'acc_z_mean': np.nan, 'acc_z_std': np.nan,
                'acc_magnitude_mean': np.nan, 'acc_magnitude_std': np.nan
            })

        # -----------------------------------------------------------------
        # HR: mean, std (z globalnie obliczonego przebiegu - Local Aggregation)
        # -----------------------------------------------------------------
        hr_features = compute_hr_window_features(time_grid, hr_timeseries, window_start_ms, window_end_ms)
        record.update(hr_features)

        # -----------------------------------------------------------------
        # HRV: metryki zmienności rytmu z IBI (sdnn, rmssd, pnn50, lf/hf)
        # -----------------------------------------------------------------
        hrv_features = compute_hrv_window_features_kemocon(
            signals.get('IBI'), window_start_ms, window_end_ms
        )
        record.update(hrv_features)

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
