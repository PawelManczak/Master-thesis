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
| EDA    | Tracisz piki stresu (SCR)       | SCL(tonic), SCR peaks/amp/AUC(phasic) |
| BVP    | Średnia→0 (sygnał falowy!)      | std (amplituda), spectral_power       |
| TEMP   | Zmienia się wolno, OK           | mean, slope (trend)                   |
| ACC    | Średnia=grawitacja/pozycja      | mean (pozycja), std (ruch/drżenie)    |

Output: seconds, arousal, valence, eda_mean, eda_std, eda_max, eda_peaks, eda_scr_mean_amp, eda_scr_auc,
        bvp_std, bvp_spectral_power, temp_mean, temp_slope, acc_*, hr_mean, hr_std, hrv_*
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Import wspólnych funkcji do obliczania cech
from feature_utils import (
    preprocess_eda_global,
    compute_eda_features,
    normalize_eda_features_subject,
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
    # GLOBAL PROCESSING: Dekompozycja EDA na składowe Tonic/Phasic
    # Pipeline: filtr dolnoprzepustowy 1 Hz -> dekompozycja medianowa
    # Greco et al. (2016), Benedek & Kaernbach (2010)
    # =================================================================
    eda_filtered, eda_tonic, eda_phasic = None, None, None
    eda_ts = None
    if 'EDA' in signals:
        eda_df = signals['EDA']
        full_eda = eda_df['value'].values
        eda_ts = eda_df['timestamp'].values
        eda_filtered, eda_tonic, eda_phasic = preprocess_eda_global(full_eda, FS_EDA)
        print(f"  Global Processing: dekompozycja EDA (Tonic/Phasic, {len(full_eda)} próbek)")

    # =================================================================
    # PRZYGOTOWANIE: Wyciągnij numpy arrays RAZ przed pętlą
    # Unikamy powtarzanego tworzenia masek DataFrame w każdej iteracji
    # =================================================================
    # BVP
    bvp_ts, bvp_vals = None, None
    if 'BVP' in signals:
        bvp_ts = signals['BVP']['timestamp'].values
        bvp_vals = signals['BVP']['value'].values

    # TEMP
    temp_ts, temp_vals = None, None
    if 'TEMP' in signals:
        temp_ts = signals['TEMP']['timestamp'].values
        temp_vals = signals['TEMP']['value'].values

    # ACC
    acc_ts, acc_x_arr, acc_y_arr, acc_z_arr = None, None, None, None
    if 'ACC' in signals and 'x' in signals['ACC'].columns:
        acc_df = signals['ACC']
        acc_ts = acc_df['timestamp'].values
        acc_x_arr = acc_df['x'].values
        acc_y_arr = acc_df['y'].values
        acc_z_arr = acc_df['z'].values

    # IBI (dla HRV)
    ibi_ts, ibi_vals = None, None
    if 'IBI' in signals:
        ibi_ts = signals['IBI']['timestamp'].values
        ibi_vals = signals['IBI']['value'].values

    # Adnotacje jako numpy arrays (unikamy iterrows)
    ann_seconds = annotations['seconds'].values
    ann_arousal = annotations['arousal'].values
    ann_valence = annotations['valence'].values

    # Zwolnij DataFrames z pamięci — numpy arrays wystarczą do dalszego przetwarzania
    del signals
    del annotations

    # =================================================================
    # LOCAL AGGREGATION: Okna 5-sekundowe
    # Używamy np.searchsorted zamiast masek boolean (O(log n) vs O(n))
    # =================================================================
    results = []
    n_windows = len(ann_seconds)

    for i in range(n_windows):
        seconds = ann_seconds[i]
        arousal = ann_arousal[i]
        valence = ann_valence[i]

        # Okno czasowe: [seconds-5, seconds) w sekundach, przekształcone na ms
        window_start_ms = start_ts + (seconds - 5) * 1000
        window_end_ms = start_ts + seconds * 1000

        record = {
            'seconds': seconds,
            'arousal': arousal,
            'valence': valence
        }

        # -----------------------------------------------------------------
        # EDA: SCL (tonic mean), SCR peaks/amplitude/AUC (phasic)
        # -----------------------------------------------------------------
        if eda_ts is not None and eda_filtered is not None:
            i0 = np.searchsorted(eda_ts, window_start_ms, side='left')
            i1 = np.searchsorted(eda_ts, window_end_ms, side='left')
            eda_features = compute_eda_features(
                eda_filtered[i0:i1], FS_EDA,
                tonic_values=eda_tonic[i0:i1],
                phasic_values=eda_phasic[i0:i1]
            )
            record.update(eda_features)
        else:
            record.update({
                'eda_mean': np.nan, 'eda_std': np.nan, 'eda_max': np.nan,
                'eda_peaks': 0, 'eda_scr_mean_amp': np.nan, 'eda_scr_auc': np.nan
            })

        # -----------------------------------------------------------------
        # BVP: std (amplituda), spectral_power (NIE średnia!)
        # -----------------------------------------------------------------
        if bvp_ts is not None:
            i0 = np.searchsorted(bvp_ts, window_start_ms, side='left')
            i1 = np.searchsorted(bvp_ts, window_end_ms, side='left')
            bvp_features = compute_bvp_features(bvp_vals[i0:i1], FS_BVP)
            record.update(bvp_features)
        else:
            record.update({'bvp_std': np.nan, 'bvp_peak_to_peak': np.nan, 'bvp_spectral_power': np.nan})

        # -----------------------------------------------------------------
        # TEMP: mean, slope (zmienia się wolno)
        # -----------------------------------------------------------------
        if temp_ts is not None:
            i0 = np.searchsorted(temp_ts, window_start_ms, side='left')
            i1 = np.searchsorted(temp_ts, window_end_ms, side='left')
            temp_features = compute_temp_features(temp_vals[i0:i1], temp_ts[i0:i1])
            record.update(temp_features)
        else:
            record.update({'temp_mean': np.nan, 'temp_slope': np.nan})

        # -----------------------------------------------------------------
        # ACC: mean (pozycja ciała), std (intensywność ruchu)
        # -----------------------------------------------------------------
        if acc_ts is not None:
            i0 = np.searchsorted(acc_ts, window_start_ms, side='left')
            i1 = np.searchsorted(acc_ts, window_end_ms, side='left')
            acc_features = compute_acc_features(
                acc_x_arr[i0:i1], acc_y_arr[i0:i1], acc_z_arr[i0:i1]
            )
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
        if ibi_ts is not None:
            hrv_features = compute_hrv_window_features(
                ibi_ts, ibi_vals, window_start_ms, window_end_ms, ibi_unit='ms'
            )
        else:
            hrv_features = {
                'hrv_sdnn': np.nan, 'hrv_rmssd': np.nan, 'hrv_pnn50': np.nan,
                'hrv_lf_power': np.nan, 'hrv_hf_power': np.nan, 'hrv_lf_hf_ratio': np.nan
            }
        record.update(hrv_features)

        results.append(record)

        if (i + 1) % 50 == 0:
            print(f"    Okno {i+1}/{n_windows}", flush=True)

    if not results:
        return None

    df = pd.DataFrame(results)

    # =================================================================
    # Krok 4: Normalizacja wewnątrz-osobnicza EDA (Min-Max)
    # Lykken & Venables (1971), Boucsein (2012)
    # =================================================================
    df = normalize_eda_features_subject(df)
    print(f"  Normalizacja osobnicza EDA: dodano kolumny *_norm")

    return df


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
