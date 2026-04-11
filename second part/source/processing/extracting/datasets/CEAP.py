#!/usr/bin/env python3
"""
CEAP-360VR Data Processing Script
Merges physiological data with annotations (valence and arousal).
Applies 'Global Processing, Local Aggregation' methodology.
Signals are aggregated into physiological features windows.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'utils'))

import numpy as np
import pandas as pd

# Shared feature extraction functions
from feature_utils import (
    preprocess_eda_global,
    compute_global_hr_from_ibi,
    normalize_eda_features_subject
)
from windowing_utils import extract_window_features
from config import CEAP_FS_EDA as FS_EDA, CEAP_FS_BVP as FS_BVP, CEAP_FS_ACC as FS_ACC, WINDOW_FAST_SEC, WINDOW_SLOW_SEC

# Paths
BASE_DIR = Path(__file__).parent.parent.parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "CEAP"
OUTPUT_DIR = DATA_DIR / "processed"

# Sampling rates and window size imported from config.py
# (CEAP_FS_EDA, CEAP_FS_BVP, CEAP_FS_ACC, WINDOW_FAST_SEC, WINDOW_SLOW_SEC)

from demographics_utils import get_age_group

def load_demographics(questionnaire_dir: Path) -> dict:
    """Loads gender and age from JSON files in '2_QuestionnaireData'."""
    if not questionnaire_dir.exists():
        print(f"WARN: Brak folderu QuestionnaireData: {questionnaire_dir}")
        return {}

    try:
        import json
        demos = {}
        json_files = sorted(questionnaire_dir.glob("P*_Questionnaire_Data.json"))

        for jf in json_files:
            with open(jf) as f:
                data = json.load(f)

            q_list = data.get('QuestionnaireData', [])
            if not q_list:
                continue

            q = q_list[0]
            pid_str = q.get('ParticipantID', '')
            pid = int(pid_str.replace('P', ''))

            gender_raw = q.get('Gender', 'Unknown')
            if str(gender_raw).lower() == 'male':
                gender = 'M'
            elif str(gender_raw).lower() == 'female':
                gender = 'F'
            else:
                gender = str(gender_raw)

            age = q.get('Age', np.nan)

            demos[pid] = {
                'gender': gender,
                'age': age,
                'age_group': get_age_group(age)
            }

        print(f"  Loaded JSON demographics for {len(demos)} CEAP participants")
        return demos
    except Exception as e:
        print(f"Error loading CEAP demographics: {e}")
        return {}

def compute_global_hr_timeseries_ceap(ibi_data: list, max_time: float) -> tuple:
    """
    Wrapper dla compute_global_hr_from_ibi dla formatu CEAP.

    Args:
        ibi_data: lista dict z 'TimeStamp' i 'IBI' (IBI w sekundach)
        max_time: maksymalny czas nagrania

    Returns:
        tuple (time_grid, hr_timeseries)
    """
    if not ibi_data or max_time <= 0:
        return np.array([]), np.array([])

    timestamps = np.array([d['TimeStamp'] for d in ibi_data])
    ibi_values = np.array([d['IBI'] for d in ibi_data])  # IBI w sekundach

    return compute_global_hr_from_ibi(timestamps, ibi_values, max_time, ibi_unit='s')





def process_video_data(video_physio: dict, video_annot: dict, demographics: dict) -> pd.DataFrame:
    """Processes a single video using 'Global Processing, Local Aggregation'."""
    video_id = video_physio['VideoID']

    # Pobierz dane - nazwy kluczy dla Raw data
    acc_data = video_physio.get('ACC_RawData', [])
    eda_data = video_physio.get('EDA_RawData', [])
    skt_data = video_physio.get('SKT_RawData', [])
    bvp_data = video_physio.get('BVP_RawData', [])
    ibi_data = video_physio.get('IBI_RawData', [])
    annot_data = video_annot.get('TimeStamp_Xvalue_Yvalue', [])

    if not annot_data:
        return None

    # Znajdź maksymalny czas
    max_time = annot_data[-1]['TimeStamp'] if annot_data else 0

    # Global HR extraction
    time_grid, hr_timeseries = compute_global_hr_timeseries_ceap(ibi_data, max_time)

    # Global EDA decomposition (Tonic/Phasic)
    eda_filtered, eda_tonic, eda_phasic = None, None, None
    eda_timestamps = None
    if eda_data:
        full_eda_values = np.array([e['EDA'] for e in eda_data])
        eda_timestamps = np.array([e['TimeStamp'] for e in eda_data])
        eda_filtered, eda_tonic, eda_phasic = preprocess_eda_global(full_eda_values, int(FS_EDA))

    # Prepare dictionaries for `extract_window_features`
    eda_data_dict = {'ts': eda_timestamps, 'filtered': eda_filtered, 'tonic': eda_tonic, 'phasic': eda_phasic, 'fs': FS_EDA} if eda_timestamps is not None else None
    
    ceap_bvp_ts = np.array([b['TimeStamp'] for b in bvp_data]) if bvp_data else None
    ceap_bvp_vals = np.array([b['BVP'] for b in bvp_data]) if bvp_data else None
    bvp_data_dict = {'ts': ceap_bvp_ts, 'values': ceap_bvp_vals, 'fs': FS_BVP} if ceap_bvp_ts is not None else None

    ceap_temp_ts = np.array([s['TimeStamp'] for s in skt_data]) if skt_data else None
    ceap_temp_vals = np.array([s['SKT'] for s in skt_data]) if skt_data else None
    temp_data_dict = {'ts': ceap_temp_ts, 'values': ceap_temp_vals} if ceap_temp_ts is not None else None

    ceap_acc_ts = np.array([a['TimeStamp'] for a in acc_data]) if acc_data else None
    ceap_acc_x = np.array([a['ACC_X'] for a in acc_data]) if acc_data else None
    ceap_acc_y = np.array([a['ACC_Y'] for a in acc_data]) if acc_data else None
    ceap_acc_z = np.array([a['ACC_Z'] for a in acc_data]) if acc_data else None
    acc_data_dict = {'ts': ceap_acc_ts, 'x': ceap_acc_x, 'y': ceap_acc_y, 'z': ceap_acc_z} if ceap_acc_ts is not None else None

    hr_data_dict = {'time_grid': time_grid, 'timeseries': hr_timeseries}

    ceap_ibi_ts = np.array([i['TimeStamp'] for i in ibi_data]) if ibi_data else None
    ceap_ibi_vals = np.array([i['IBI'] for i in ibi_data]) if ibi_data else None
    hrv_data_dict = {'type': 'ibi', 'ts': ceap_ibi_ts, 'values': ceap_ibi_vals, 'unit': 's'} if ceap_ibi_ts is not None else None

    results = []

    # --- LOOP 1: FAST WINDOWS ---
    window_end = WINDOW_FAST_SEC

    while window_end <= max_time:
        window_start = window_end - WINDOW_FAST_SEC

        record = {
            'seconds': window_end, # Timestamp in seconds
            'video_id': video_id,
            'window_type': 'fast'
        }

        # Arousal & Valence
        window_annot = [a for a in annot_data if window_start <= a['TimeStamp'] < window_end]
        if window_annot:
            arousal = np.mean([a['Y_Value'] for a in window_annot])
            valence = np.mean([a['X_Value'] for a in window_annot])
            record['arousal'] = arousal
            record['valence'] = valence
        else:
            record['arousal'] = np.nan
            record['valence'] = np.nan

        # Fast Feature extraction
        features = extract_window_features(
            window_start, window_end,
            eda_data=eda_data_dict,
            bvp_data=bvp_data_dict,
            temp_data=None,         # Exclude slow
            acc_data=acc_data_dict, # Keeping ACC on fast
            hr_data=hr_data_dict,
            hrv_data=None           # Exclude slow
        )
        record.update(features)

        results.append(record)
        window_end += WINDOW_FAST_SEC

    # --- LOOP 2: SLOW WINDOWS ---
    window_end = WINDOW_SLOW_SEC

    while window_end <= max_time:
        window_start = window_end - WINDOW_SLOW_SEC

        record = {
            'seconds': window_end, # Timestamp in seconds
            'video_id': video_id,
            'window_type': 'slow',
            'arousal': np.nan,  # Exclude from slow
            'valence': np.nan   # Exclude from slow
        }

        # Slow Feature extraction
        features = extract_window_features(
            window_start, window_end,
            eda_data=None,
            bvp_data=None,         
            temp_data=temp_data_dict,  # Keep TEMP on slow
            acc_data=None,
            hr_data=None,
            hrv_data=hrv_data_dict     # Keep HRV on slow
        )
        record.update(features)

        results.append(record)
        window_end += WINDOW_SLOW_SEC

    if not results:
        return None

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('seconds').reset_index(drop=True)
    return df_results


def process_participant(physio_file: Path, annot_file: Path, pid: int, demographics: dict = None) -> pd.DataFrame:
    """Process a single participant from JSON files."""
    # Wczytaj dane z JSON
    try:
        with open(physio_file, 'r') as f:
            physio_json = json.load(f)
        physio_data = physio_json['Physio_RawData'][0]
    except Exception as e:
        print(f"  Błąd wczytywania danych fizjo P{pid}: {e}")
        return None

    try:
        with open(annot_file, 'r') as f:
            annot_json = json.load(f)
        annot_data = annot_json['ContinuousAnnotation_RawData'][0]
    except Exception as e:
        print(f"  Błąd wczytywania adnotacji P{pid}: {e}")
        return None

    video_physio_list = physio_data.get('Video_Physio_RawData', [])
    video_annot_list = annot_data.get('Video_Annotation_RawData', [])

    all_results = []

    for video_physio in video_physio_list:
        video_id = video_physio['VideoID']
        video_annot = next((v for v in video_annot_list if v['VideoID'] == video_id), None)

        if video_annot is None:
            print(f"  Brak adnotacji dla wideo {video_id}")
            continue

        video_df = process_video_data(video_physio, video_annot, demographics)
        if video_df is not None and len(video_df) > 0:
            all_results.append(video_df)

    if not all_results:
        return None

    df = pd.concat(all_results, ignore_index=True)

    if demographics:
        df['gender'] = demographics.get('gender', 'Unknown')
        df['age'] = demographics.get('age', np.nan)
        df['age_group'] = demographics.get('age_group', 'Unknown')

    # Subject-level EDA normalization (Min-Max)
    df = normalize_eda_features_subject(df)
    print(f"  Subject-level EDA normalization applied")

    return df


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Wczytaj demografię z plików JSON w 2_QuestionnaireData
    ceap_base = DATA_DIR / "raw" / "CEAP-360VR-Dataset-master" / "CEAP-360VR"
    questionnaire_dir = ceap_base / "2_QuestionnaireData"
    demos = load_demographics(questionnaire_dir)
    print(f"Wczytano demografię dla {len(demos)} uczestników.")

    # Ścieżki do danych Raw (JSON)
    physio_dir = ceap_base / "5_PhysioData" / "Raw"
    annot_dir = ceap_base / "3_AnnotationData" / "Raw"

    participants = []
    # Szukamy plików fizjologicznych JSON
    physio_files = sorted(list(physio_dir.glob("P*_Physio_RawData.json")))

    if not physio_files:
        print(f"Brak plików fizjologicznych w {physio_dir}")
        return

    print(f"Znaleziono {len(physio_files)} plików fizjologicznych.")

    for physio_path in physio_files:
        # Nazwa pliku: P1_Physio_RawData.json
        filename = physio_path.name
        pid_str = filename.split('_')[0]  # P1
        pid = int(pid_str.replace('P', ''))

        print(f"Przetwarzanie uczestnika P{pid}...")

        # Znajdź plik adnotacji: P1_Annotation_RawData.json
        annot_path = annot_dir / f"{pid_str}_Annotation_RawData.json"

        if not annot_path.exists():
            print(f"  Brak pliku adnotacji: {annot_path}")
            continue

        participant_demo = demos.get(pid, {})

        result_df = process_participant(physio_path, annot_path, pid, demographics=participant_demo)

        if result_df is not None and len(result_df) > 0:
            output_file = OUTPUT_DIR / f"P{pid}_merged.csv"
            result_df.to_csv(output_file, index=False)
            print(f"  Zapisano {len(result_df)} rekordów do {output_file}")
            participants.append(pid)
        else:
            print(f"  Brak danych do zapisania dla P{pid}")

    print(f"\nPrzetwarzanie zakończone! Przetworzono {len(participants)} uczestników.")


if __name__ == "__main__":
    main()

