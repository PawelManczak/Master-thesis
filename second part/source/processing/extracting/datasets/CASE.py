"""
CASE Dataset Processing Script
Python translation of original MATLAB scripts.
Merges physiological data with annotations (valence and arousal).
Applies 'Global Processing, Local Aggregation' methodology.
Signals are aggregated into physiological features windows.
"""

import warnings
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'utils'))

import numpy as np
import pandas as pd

# Shared feature extraction functions
from feature_utils import (
    preprocess_eda_global,
    compute_eda_features,
    normalize_eda_features_subject,
    compute_temp_features,
    compute_global_hr_from_ecg,
    compute_hr_window_features,
    compute_hrv_from_ecg_window
)
from demographics_utils import get_age_group
from windowing_utils import extract_window_features
from config import WINDOW_SEC, CASE_FS_PHYSIO as FS_PHYSIO, CASE_FS_ANNOT as FS_ANNOT

warnings.filterwarnings('ignore')


# Paths
BASE_DIR = Path(__file__).parent.parent.parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "CASE" / "raw" / "case_dataset-master" / "data"
METADATA_DIR = BASE_DIR / "data" / "CASE" / "raw" / "case_dataset-master" / "metadata"
OUTPUT_DIR = BASE_DIR / "data" / "CASE" / "processed"
PHYSIO_DIR = DATA_DIR / "raw" / "physiological"
ANNOTATIONS_DIR = DATA_DIR / "raw" / "annotations"

# Windowing parameters (see utils/config.py)
# WINDOW_SEC and FS_PHYSIO/FS_ANNOT imported from config


def age_group_to_midpoint(age_group: str) -> float:
    """Converts an age group string (e.g. '25-29') to a midpoint float (e.g. 27)."""
    if pd.isna(age_group) or age_group == 'Unknown':
        return np.nan
    try:
        parts = age_group.split('-')
        return (int(parts[0]) + int(parts[1])) / 2
    except:
        return np.nan

def load_demographics():
    """Loads participant metadata (age, gender) from participants.xlsx."""
    meta_path = METADATA_DIR / "participants.xlsx"
    if not meta_path.exists():
        print(f"WARN: Brak pliku participants.xlsx w {meta_path}")
        return {}

    try:
        df = pd.read_excel(meta_path)
        demos = {}
        for _, row in df.iterrows():
            pid = str(row['Participant-ID'])  # e.g. 1, 2, ...

            age_group = str(row['Age-Group']).strip()  # np. '30-34'
            sex = str(row['Sex']).strip()  # 'F' lub 'M'

            # Konwertuj Sex F/M na pełne nazwy dla spójności z innymi datasetami
            gender = 'M' if sex.upper() == 'M' else ('F' if sex.upper() == 'F' else sex)

            # Oblicz przybliżony wiek jako punkt środkowy grupy wiekowej
            age_approx = age_group_to_midpoint(age_group)

            demos[pid] = {
                'gender': gender,
                'age': age_approx,
                'age_group': age_group
            }
        print(f"  Wczytano demografię dla {len(demos)} uczestników CASE")
        return demos
    except Exception as e:
        print(f"Błąd wczytywania metadanych CASE: {e}")
        return {}


def load_videos_duration() -> dict:
    """Wczytaj informacje o czasie trwania wideo."""
    file_path = METADATA_DIR / "videos_duration.txt"

    # Mapowanie nazw wideo na ID (z videos.xlsx)
    video_name_to_id = {
        'amusing-1': 1, 'amusing-2': 2, 'bluVid': 3, 'boring-1': 4, 'boring-2': 5,
        'endVid': 6, 'relaxed-1': 7, 'relaxed-2': 8, 'scary-1': 9, 'scary-2': 10,
        'startVid': 11
    }

    videos_duration = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:  # Pomiń nagłówek
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                name = parts[0].strip('"')
                duration = float(parts[1].strip())
                if name in video_name_to_id:
                    videos_duration[video_name_to_id[name]] = duration

    return videos_duration


def load_seqs_order() -> dict:
    """Wczytaj kolejność sekwencji dla każdego uczestnika."""
    file_path = METADATA_DIR / "seqs_order.txt"

    video_name_to_id = {
        'amusing-1': 1, 'amusing-2': 2, 'bluVid': 3, 'boring-1': 4, 'boring-2': 5,
        'endVid': 6, 'relaxed-1': 7, 'relaxed-2': 8, 'scary-1': 9, 'scary-2': 10,
        'startVid': 11
    }

    seqs_order = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()
        # Pierwsza linia to nagłówki (sub1, sub2, ...)
        # Kolejne linie to sekwencje wideo
        n_subs = 30
        for sub_id in range(1, n_subs + 1):
            seqs_order[sub_id] = []

        for line in lines[1:]:  # Pomiń nagłówek
            parts = line.strip().split('\t')
            for sub_id, part in enumerate(parts[:n_subs], 1):
                name = part.strip('"')
                if name in video_name_to_id:
                    seqs_order[sub_id].append(video_name_to_id[name])

    return seqs_order


def f_label_data(timevec: np.ndarray, vidseq_order: list, vids_duration: dict) -> np.ndarray:
    """Returns a vector of labels (video-ID) for the given time vector."""
    timevec_nrow = len(timevec)
    labels = np.zeros(timevec_nrow, dtype=int)

    n_vids_in_seq = len(vidseq_order)

    last_time_found = 0
    video_pos_start = 0

    for j in range(n_vids_in_seq):
        if j == 0:
            last_time_found = 0
            video_pos_start = 0
        else:
            last_time_found = video_time_end
            video_pos_start = video_pos_end + 1

        vid_searched = vidseq_order[j]
        vid_matched_dur = vids_duration.get(vid_searched, 0)

        time2match = last_time_found + vid_matched_dur

        # Znajdź najbliższy czas
        tmp_timeVec = np.abs(timevec - time2match)
        loc_time2match = np.argmin(tmp_timeVec)

        video_pos_end = loc_time2match
        video_time_end = timevec[video_pos_end]

        # Aktualizuj etykiety
        if video_pos_start < timevec_nrow:
            labels[video_pos_start:video_pos_end + 1] = vid_searched

    return labels


def transform_ecg(voltage: np.ndarray) -> np.ndarray:
    """Transformacja ECG: V -> mV"""
    return ((voltage - 2.8) / 50) * 1000


def transform_bvp(voltage: np.ndarray) -> np.ndarray:
    """Transformacja BVP: V -> %"""
    return (58.962 * voltage) - 115.09


def transform_gsr(voltage: np.ndarray) -> np.ndarray:
    """Transformacja GSR (EDA): V -> µS"""
    return (24 * voltage) - 49.2


def transform_rsp(voltage: np.ndarray) -> np.ndarray:
    """Transformacja RSP: V -> %"""
    return (58.923 * voltage) - 115.01


def transform_skt(voltage: np.ndarray) -> np.ndarray:
    """Transformacja SKT (temp): V -> °C"""
    return (21.341 * voltage) - 32.085


def transform_emg(voltage: np.ndarray) -> np.ndarray:
    """Transformacja EMG: V -> µV"""
    return ((voltage - 2.0) / 4000) * 1000000


def transform_annotation(raw_value: np.ndarray) -> np.ndarray:
    """Transformacja adnotacji: [-26225, 26225] -> [0.5, 9.5]"""
    return 0.5 + 9 * (raw_value + 26225) / 52450


# Data loading functions


def load_raw_physiological(file_path) -> pd.DataFrame:
    """Loads raw physiological data from the given path."""
    file_path = Path(file_path)
    if not file_path.exists():
        return None

    columns = ['daqtime', 'ecg', 'bvp', 'gsr', 'rsp', 'skt', 'emg_zygo', 'emg_coru', 'emg_trap']
    df = pd.read_csv(file_path, sep='\t', header=None, names=columns)
    return df


def load_raw_annotations(file_path) -> pd.DataFrame:
    """Loads raw annotation data from the given path."""
    file_path = Path(file_path)
    if not file_path.exists():
        return None

    columns = ['jstime', 'valence_raw', 'arousal_raw']
    df = pd.read_csv(file_path, sep='\t', header=None, names=columns)
    return df


def process_participant(pid, physio_file, annot_file, demographics=None):
    """Processes a single participant using 'Global Processing, Local Aggregation'."""
    print(f"Przetwarzanie uczestnika {pid}...", flush=True)

    # Wczytaj surowe dane z przekazanych ścieżek
    raw_physio = load_raw_physiological(physio_file)
    raw_annot = load_raw_annotations(annot_file)

    if raw_physio is None:
        print(f"  Brak danych fizjologicznych dla {pid}")
        return None

    if raw_annot is None:
        print(f"  Brak danych adnotacji dla {pid}")
        return None

    print(f"  Wczytano {len(raw_physio)} próbek fizjo i {len(raw_annot)} próbek adnotacji", flush=True)

    # Konwertuj czas z sekund na ms
    daqtime = raw_physio['daqtime'].values * 1000
    jstime = raw_annot['jstime'].values * 1000

    # Transformuj dane fizjologiczne (w voltach -> jednostki)
    ecg = transform_ecg(raw_physio['ecg'].values)      # mV
    gsr = transform_gsr(raw_physio['gsr'].values)      # µS (EDA)
    skt = transform_skt(raw_physio['skt'].values)      # °C (temp)

    # Transformuj adnotacje
    valence_arr = transform_annotation(raw_annot['valence_raw'].values)
    arousal_arr = transform_annotation(raw_annot['arousal_raw'].values)

    # Znajdź zakres czasowy danych
    max_time = min(daqtime.max(), jstime.max())

    # Global HR extraction (R-peaks detection)
    print(f"  Global Processing: R-peaks detection...", flush=True)
    time_grid, hr_timeseries, r_peaks = compute_global_hr_from_ecg(ecg, daqtime, FS_PHYSIO)
    print(f"  Detected {len(r_peaks)} R-peaks", flush=True)

    # Global EDA decomposition (Tonic/Phasic)
    print(f"  Global Processing: dekompozycja EDA (Tonic/Phasic)...", flush=True)
    gsr_filtered, gsr_tonic, gsr_phasic = preprocess_eda_global(gsr, FS_PHYSIO)
    print(f"  Dekompozycja EDA: {len(gsr)} próbek przetworzonych", flush=True)

    # Prepare feature dictionaries
    eda_data_dict = {
        'ts': daqtime, 'filtered': gsr_filtered, 'tonic': gsr_tonic, 'phasic': gsr_phasic, 'fs': FS_PHYSIO
    }
    temp_data_dict = {
        'ts': daqtime, 'values': skt
    }
    hr_data_dict = {
        'time_grid': time_grid, 'timeseries': hr_timeseries
    }
    hrv_data_dict = {
        'type': 'ecg', 'ts_ecg': daqtime, 'r_peaks': r_peaks, 'fs': FS_PHYSIO
    }

    # 30-second windows (Local Aggregation)
    window_size_ms = WINDOW_SEC * 1000
    results = []
    n_windows = int(max_time / window_size_ms)

    for i, window_start in enumerate(range(0, int(max_time), window_size_ms)):
        window_end = window_start + window_size_ms

        # Annotation indices
        js_mask = (jstime >= window_start) & (jstime < window_end)

        if not np.any(js_mask):
            continue

        # Arousal & Valence
        arousal = np.mean(arousal_arr[js_mask])
        valence = np.mean(valence_arr[js_mask])

        record = {
            'seconds': window_end,
            'video_id': 0, # Placeholder
            'arousal': arousal,
            'valence': valence
        }



        # Feature extraction
        features = extract_window_features(
            window_start, window_end,
            eda_data=eda_data_dict,
            temp_data=temp_data_dict,
            hr_data=hr_data_dict,
            hrv_data=hrv_data_dict
        )
        record.update(features)

        results.append(record)

        # Progress
        if (i + 1) % 100 == 0:
            print(f"    Okno {i+1}/{n_windows}", flush=True)

    if len(results) == 0:
        return None

    df = pd.DataFrame(results)

    if demographics:
        df['gender'] = demographics.get('gender', 'Unknown')
        df['age'] = demographics.get('age', np.nan)
        df['age_group'] = demographics.get('age_group', 'Unknown')

    # Subject-level EDA normalization (Min-Max)
    df = normalize_eda_features_subject(df)
    print(f"  Subject-level EDA normalization applied", flush=True)

    return df


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Wczytaj demografię
    demos = load_demographics()
    print(f"Wczytano metadane dla {len(demos)} uczestników.")

    physio_dir = DATA_DIR / "raw" / "physiological"
    annot_dir = DATA_DIR / "raw" / "annotations"

    # Przetwarzanie uczestników
    physio_files = list(physio_dir.glob("sub*.txt"))
    for pf in sorted(physio_files):
        # Format nazwy: sub10_DAQ.txt -> stem = sub10_DAQ
        pid_str = pf.stem.replace('_DAQ', '')  # sub10
        pid_num = pid_str.replace('sub', '')     # 10

        print(f"Przetwarzanie uczestnika {pid_str}...")

        # Adnotacje mają inny sufiks: sub10_joystick.txt
        af = annot_dir / f"{pid_str}_joystick.txt"

        if not af.exists():
            print(f"  Brak pliku adnotacji: {af}")
            continue

        participant_demo = demos.get(pid_num, {})

        result_df = process_participant(pid_str, pf, af, demographics=participant_demo)

        if result_df is not None and len(result_df) > 0:
            output_file = OUTPUT_DIR / f"sub{pid_num}_merged.csv"
            result_df.to_csv(output_file, index=False)
            print(f"  Zapisano {len(result_df)} rekordów do {output_file}", flush=True)
        else:
            print(f"  Brak danych do zapisania dla {pid_str}", flush=True)

    print(f"\nPrzetwarzanie zakończone!", flush=True)


if __name__ == "__main__":
    main()

