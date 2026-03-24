"""
K-EmoCon Data Processing Script
Merges Empatica E4 data with self-annotations (valence and arousal).
Applies 'Global Processing, Local Aggregation' methodology.
Signals are aggregated into physiological features windows.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'utils'))

# Shared feature extraction functions
from feature_utils import (
    preprocess_eda_global,
    compute_global_hr_from_ibi,
    normalize_eda_features_subject
)
from bvp_utils import _empty_hrv_result

from windowing_utils import extract_window_features
from config import WINDOW_SEC, ANNOT_STEP_SEC, ANNOTS_PER_WINDOW, FS_EDA, FS_HR, FS_TEMP, FS_BVP, FS_ACC

# Paths
BASE_DIR = Path(__file__).parent.parent.parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "K-emoCon" / "raw"
OUTPUT_DIR = BASE_DIR / "data" / "K-emoCon" / "processed"
METADATA_DIR = BASE_DIR / "data" / "K-emoCon" / "metadata"
E4_DIR = DATA_DIR / "e4_data"
ANNOTATIONS_DIR = DATA_DIR / "self_annotations"

from demographics_utils import get_age_group

def load_participant_metadata():
    """Loads participant metadata (age, gender) from CSV."""
    meta_path = METADATA_DIR / "participants.csv"
    if not meta_path.exists():
        print(f"Brak pliku metadanych: {meta_path}")
        return {}

    df = pd.read_csv(meta_path)
    metadata = {}
    for _, row in df.iterrows():
        pid = str(row['Participant_ID']).replace('P', '') # 'P1' -> '1'
        metadata[pid] = {
            'gender': row['Gender'],
            'age': row['Age'],
            'age_group': get_age_group(row['Age'])
        }
    return metadata

# Sampling frequencies imported from config.py
# (FS_EDA, FS_HR, FS_TEMP, FS_BVP, FS_ACC)


# Feature utils wrappers for K-EmoCon format

def compute_global_hr_from_ibi_kemocon(ibi_df: pd.DataFrame, start_ts: float, max_time_ms: float) -> tuple:
    """Computes global HR from IBI for K-EmoCon."""
    if ibi_df is None or len(ibi_df) < 2:
        return np.array([]), np.array([])

    timestamps = ibi_df['timestamp'].values
    ibi_values = ibi_df['value'].values  # IBI w ms

    return compute_global_hr_from_ibi(timestamps, ibi_values, start_ts + max_time_ms, ibi_unit='ms')


# Data loading functions

def get_participant_mapping():
    """Mapping between E4 folders and self-annotation files."""
    e4_folders = sorted([f.name for f in E4_DIR.iterdir() if f.is_dir()])
    annotation_files = sorted([f.stem.replace('.self', '') for f in ANNOTATIONS_DIR.glob("*.csv")])

    return e4_folders


def load_e4_signal(participant_folder: str, signal_name: str) -> pd.DataFrame:
    """Loads E4 signal for a given participant."""
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


def process_participant(e4_folder: str, pid: int, metadata: dict = None) -> pd.DataFrame:
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
    # Przygotuj słowniki z danymi przed pętlą dla nowej funkcji agregacyjnej
    # =================================================================
    eda_data_dict = {'ts': eda_ts, 'filtered': eda_filtered, 'tonic': eda_tonic, 'phasic': eda_phasic, 'fs': FS_EDA} if eda_ts is not None else None
    bvp_data_dict = {'ts': bvp_ts, 'values': bvp_vals, 'fs': FS_BVP} if bvp_ts is not None else None
    temp_data_dict = {'ts': temp_ts, 'values': temp_vals} if temp_ts is not None else None
    acc_data_dict = {'ts': acc_ts, 'x': acc_x_arr, 'y': acc_y_arr, 'z': acc_z_arr} if acc_ts is not None else None
    hr_data_dict = {'time_grid': time_grid, 'timeseries': hr_timeseries}
    hrv_data_dict = {'type': 'ibi', 'ts': ibi_ts, 'values': ibi_vals, 'unit': 'ms'} if ibi_ts is not None else None

    # LOCAL AGGREGATION: 30-second windows
    # K-emoCon annotations are every 5s -> 6 annotations per window.
    # 30s = minimum for stable HRV metrics (Task Force ESC 1996).
    # Using np.searchsorted instead of boolean masks (O(log n) vs O(n))
    ANNOT_STEP = ANNOT_STEP_SEC  # seconds between annotations

    results = []
    n_annotations = len(ann_seconds)
    n_windows = n_annotations // ANNOTS_PER_WINDOW

    for wi in range(n_windows):
        # Indeksy adnotacji dla tego okna
        idx_start = wi * ANNOTS_PER_WINDOW
        idx_end = idx_start + ANNOTS_PER_WINDOW

        # Uśrednij arousal/valence z 6 adnotacji
        window_seconds_end = ann_seconds[idx_end - 1]
        window_seconds_start = ann_seconds[idx_start] - ANNOT_STEP
        arousal = float(np.nanmean(ann_arousal[idx_start:idx_end]))
        valence = float(np.nanmean(ann_valence[idx_start:idx_end]))

        # Okno czasowe w ms
        window_start_ms = start_ts + window_seconds_start * 1000
        window_end_ms = start_ts + window_seconds_end * 1000

        record = {
            'seconds': window_seconds_end,
            'arousal': arousal,
            'valence': valence
        }



        # Ekstrakcja cech ze współdzielonego modułu
        features = extract_window_features(
            window_start_ms, window_end_ms,
            eda_data=eda_data_dict,
            bvp_data=bvp_data_dict,
            temp_data=temp_data_dict,
            acc_data=acc_data_dict,
            hr_data=hr_data_dict,
            hrv_data=hrv_data_dict
        )
        record.update(features)

        results.append(record)

        if (wi + 1) % 20 == 0:
            print(f"    Okno {wi+1}/{n_windows}", flush=True)

    if not results:
        return None

    df = pd.DataFrame(results)

    if metadata:
        df['gender'] = metadata.get('gender', 'Unknown')
        df['age'] = metadata.get('age', np.nan)
        df['age_group'] = metadata.get('age_group', 'Unknown')

    # =================================================================
    # Krok 4: Normalizacja wewnątrz-osobnicza EDA (Min-Max)
    # Lykken & Venables (1971), Boucsein (2012)
    # =================================================================
    df = normalize_eda_features_subject(df)
    print(f"  Normalizacja osobnicza EDA: dodano kolumny *_norm")

    return df


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Wczytaj metadane
    participants_metadata = load_participant_metadata()
    print(f"Wczytano participants.csv: {len(participants_metadata)} rekordów")

    # (E4_DIR i ANNOTATIONS_DIR są już zdefiniowane globalnie)

    # Pobierz foldery E4
    e4_folders = [f.name for f in E4_DIR.iterdir() if f.is_dir()]

    # Przetworz każdego uczestnika
    for pid in sorted(e4_folders, key=lambda x: int(x) if x.isdigit() else x):
        # PID w metadanych to po prostu liczba, np '1'
        participant_meta = participants_metadata.get(pid, {})

        result_df = process_participant(
            E4_DIR / pid,
            pid,
            metadata=participant_meta
        )

        if result_df is not None and len(result_df) > 0:

            output_file = OUTPUT_DIR / f"P{pid}_merged.csv"
            result_df.to_csv(output_file, index=False)
            print(f"  Zapisano {len(result_df)} rekordów do {output_file}")
        else:
            print(f"  Brak danych do zapisania dla P{pid}")

    print("\nPrzetwarzanie zakończone!")


if __name__ == "__main__":
    print("Starting K-emoCon processing...", flush=True)
    main()
