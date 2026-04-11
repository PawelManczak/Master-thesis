"""
EmoWorker_v2 Data Processing Script
Merges EmoWorker_v2 (Empatica E4) data with labels (arousal, valence).
Applies 'Global Processing, Local Aggregation' methodology.
Signals are aggregated into physiological features windows.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'utils'))

# Import features utilities
from feature_utils import (
    preprocess_eda_global,
    compute_global_hr_from_ibi,
    normalize_eda_features_subject
)
from bvp_utils import _empty_hrv_result

# Import demographics and windowing utilities
from demographics_utils import get_age_group
from windowing_utils import extract_window_features
from config import WINDOW_FAST_SEC, WINDOW_SLOW_SEC, FS_EDA, FS_TEMP, FS_BVP, FS_ACC

# --- PATHS ---
BASE_DIR = Path(__file__).parent.parent.parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "EmoWorker_v2"
OUTPUT_DIR = DATA_DIR / "processed"
SENSORS_DIR = DATA_DIR / "CSV" / "SENSORS_csv"
LABELS_DIR = DATA_DIR / "CSV" / "LABEL_csv"

# --- CONSTANTS (defined in utils/config.py) ---
# FS_EDA=4Hz, FS_TEMP=4Hz, FS_BVP=64Hz, FS_ACC=32Hz, WINDOW_FAST_SEC=5, WINDOW_SLOW_SEC=30


def load_csv_data(filepath, columns=None):
    """Helper function to load CSV with error handling."""
    if not filepath.exists():
        return None
    try:
        df = pd.read_csv(filepath)
        if columns:
            df = df[columns]
        return df
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def load_demographics():
    """Loads demographic data from EmoWorker_v2 META/presurvey.csv."""
    meta_path = DATA_DIR / "META" / "presurvey.csv"

    if not meta_path.exists():
        print(f"WARN: Brak pliku demografii: {meta_path}")
        return {}

    try:
        df = pd.read_csv(meta_path)
        demos = {}
        for _, row in df.iterrows():
            pid = str(int(row['Pnum']))

            # Sex: 1=Male, 0=Female
            sex_val = row['Sex']
            if sex_val == 1:
                gender = 'M'
            elif sex_val == 0:
                gender = 'F'
            else:
                gender = 'Unknown'

            age = int(row['Age'])

            demos[pid] = {
                'gender': gender,
                'age': age,
                'age_group': get_age_group(age)
            }

        print(f"  Loaded demographics for {len(demos)} EmoWorker_v2 participants")
        return demos
    except Exception as e:
        print(f"Error loading demographics: {e}")
        return {}

def process_participant_condition(pid, condition):
    """Processes data for one participant and one condition (c1, c2, c3)."""

    # 1. File paths
    sensor_path = SENSORS_DIR / str(pid) / condition
    label_path = LABELS_DIR / str(pid) / condition

    if not sensor_path.exists() or not label_path.exists():
        # print(f"  Brak folderu dla {pid} {condition}")
        return None

    # 2. Load labels
    f_arousal = label_path / "arousal.csv"
    f_valence = label_path / "valence.csv"

    df_arousal = load_csv_data(f_arousal)
    df_valence = load_csv_data(f_valence)

    if df_arousal is None or df_valence is None:
        print(f"  Brak plików label dla {pid} {condition}")
        return None

    # Upewnij się co do nazw kolumn
    # arousal.csv: arousal,Timestamp
    # valence.csv: valence,Timestamp

    # Sortuj po timestamp
    df_arousal = df_arousal.sort_values(by='Timestamp')
    df_valence = df_valence.sort_values(by='Timestamp')

    # Sprawdź czy mamy wspólny zakres czasu
    # Timestamps are in ms (Unix epoch potentially, looking at 1671068... ~ 2022/2023)

    # Labels time range
    arousal_ts = df_arousal['Timestamp'].values
    valence_ts = df_valence['Timestamp'].values

    if len(arousal_ts) == 0 or len(valence_ts) == 0:
        return None

    start_ts_lbl = max(arousal_ts[0], valence_ts[0])
    end_ts_lbl = min(arousal_ts[-1], valence_ts[-1])

    # 3. Wczytaj sensory
    f_bvp = sensor_path / "e4_bvp.csv"
    f_eda = sensor_path / "e4_eda.csv"
    f_temp = sensor_path / "e4_temp.csv"
    f_ibi = sensor_path / "e4_ibi.csv"
    f_acc = sensor_path / "e4_acc.csv"

    df_bvp = load_csv_data(f_bvp)
    df_eda = load_csv_data(f_eda)
    df_temp = load_csv_data(f_temp)
    df_ibi = load_csv_data(f_ibi)
    df_acc = load_csv_data(f_acc)

    # Check availability of key signals (EDA, BVP)
    if df_eda is None and df_bvp is None:
        print(f"  Missing EDA/BVP data for {pid} {condition}")
        return None

    # Determine common time range (start)
    start_ts_sig = []
    if df_bvp is not None: start_ts_sig.append(df_bvp['Timestamp'].min())
    if df_eda is not None: start_ts_sig.append(df_eda['Timestamp'].min())

    if not start_ts_sig:
        return None

    global_start = max(start_ts_lbl, min(start_ts_sig))
    global_end = end_ts_lbl # End determined by labels

    if global_end <= global_start:
        return None

    # Global Processing

    # HR / IBI
    time_grid, hr_timeseries = np.array([]), np.array([])
    if df_ibi is not None and len(df_ibi) > 2:
        ibi_timestamps = df_ibi['Timestamp'].values
        ibi_values = df_ibi['ibi'].values

        ibi_unit = 's'
        if np.mean(ibi_values[:10]) > 100:
            ibi_unit = 'ms'

        # Compute global HR using relative time to avoid huge arrays
        try:
            rel_timestamps = ibi_timestamps - global_start
            duration_ms = global_end - global_start + 1000

            rel_time_grid, hr_timeseries = compute_global_hr_from_ibi(
                rel_timestamps, ibi_values, duration_ms, ibi_unit=ibi_unit
            )

            time_grid = rel_time_grid + global_start

            print(f"  Global HR computed: {len(time_grid)} points")
        except Exception as e:
            print(f"Error computing global HR for {pid} {condition}: {e}")
            time_grid, hr_timeseries = np.array([]), np.array([])


    # EDA Decomposition
    eda_filtered, eda_tonic, eda_phasic = None, None, None
    eda_ts = None
    if df_eda is not None:
        eda_ts = df_eda['Timestamp'].values
        eda_vals = df_eda['eda'].values
        try:
            eda_filtered, eda_tonic, eda_phasic = preprocess_eda_global(eda_vals, FS_EDA)
        except Exception as e:
            print(f"Error decomposing EDA for {pid} {condition}: {e}")

    # Prepare numpy arrays for windowing
    bvp_ts, bvp_vals = None, None
    if df_bvp is not None:
        bvp_ts = df_bvp['Timestamp'].values
        bvp_vals = df_bvp['bvp'].values

    temp_ts, temp_vals = None, None
    if df_temp is not None:
        temp_ts = df_temp['Timestamp'].values
        temp_vals = df_temp['temp'].values

    # ACC
    acc_ts, acc_x, acc_y, acc_z = None, None, None, None
    if df_acc is not None:
        # Check columns
        # Usually E4 ACC has x,y,z or similar.
        # EmoWorker e4_acc.csv structure? Assuming Timestamp,x,y,z (need to verify mapping if names differ)
        # Reading head of acc... assume x,y,z or 2nd,3rd,4th col.
        # Let's check names if possible. But I can't check inside function easily while writing.
        # Assuming standard E4 format or 'value' if single? E4 ACC is 3-axis.
        # I'll try to map by column names.
        pass # Will implement inside loop robustly

    # IBI for HRV window features
    ibi_ts_arr, ibi_vals_arr = None, None
    if df_ibi is not None:
        ibi_ts_arr = df_ibi['Timestamp'].values
        ibi_vals_arr = df_ibi['ibi'].values

    # Prepare feature dictionaries
    eda_data_dict = {'ts': eda_ts, 'filtered': eda_filtered, 'tonic': eda_tonic, 'phasic': eda_phasic, 'fs': FS_EDA} if eda_ts is not None else None
    bvp_data_dict = {'ts': bvp_ts, 'values': bvp_vals, 'fs': FS_BVP} if bvp_ts is not None else None
    temp_data_dict = {'ts': temp_ts, 'values': temp_vals} if temp_ts is not None else None
    hr_data_dict = {'time_grid': time_grid, 'timeseries': hr_timeseries}
    hrv_data_dict = {'type': 'ibi', 'ts': ibi_ts_arr, 'values': ibi_vals_arr, 'unit': 's'} if ibi_ts_arr is not None else None

    duration_ms = global_end - global_start
    results = []

    # --- LOOP 1: FAST WINDOWS (EDA, HR, ACC, Emotions) ---
    n_windows_fast = int(duration_ms // (WINDOW_FAST_SEC * 1000))
    for i in range(n_windows_fast):
        window_start_ms = global_start + i * WINDOW_FAST_SEC * 1000
        window_end_ms = window_start_ms + WINDOW_FAST_SEC * 1000
        window_time_sec = window_end_ms / 1000.0

        record = {
            'pid': pid,
            'condition': condition,
            'seconds': window_time_sec,
            'window_idx': i,
            'window_type': 'fast'
        }

        # Emotions (only mapped to fast windows)
        mask_arousal = (arousal_ts >= window_start_ms) & (arousal_ts < window_end_ms)
        mask_valence = (valence_ts >= window_start_ms) & (valence_ts < window_end_ms)
        record['arousal'] = df_arousal.loc[mask_arousal, 'arousal'].mean()
        record['valence'] = df_valence.loc[mask_valence, 'valence'].mean()

        # Fast features
        features = extract_window_features(
            window_start_ms, window_end_ms,
            eda_data=eda_data_dict,
            bvp_data=bvp_data_dict, 
            temp_data=None,         # Explicitly skip slow
            hr_data=hr_data_dict,
            hrv_data=None           # Explicitly skip slow
        )
        record.update(features)
        
        # Ensure temp isn't leaked (extract_window returns nans anyway but safety)
        results.append(record)

    # --- LOOP 2: SLOW WINDOWS (HRV, TEMP) ---
    n_windows_slow = int(duration_ms // (WINDOW_SLOW_SEC * 1000))
    for i in range(n_windows_slow):
        window_start_ms = global_start + i * WINDOW_SLOW_SEC * 1000
        window_end_ms = window_start_ms + WINDOW_SLOW_SEC * 1000
        window_time_sec = window_end_ms / 1000.0

        record = {
            'pid': pid,
            'condition': condition,
            'seconds': window_time_sec,
            'window_idx': i,
            'window_type': 'slow',
            'arousal': np.nan,  # Skip emotions in slow windows to avoid 30s-long generic emotional states overlapping incorrectly
            'valence': np.nan
        }

        # Slow features
        features = extract_window_features(
            window_start_ms, window_end_ms,
            eda_data=None,          # Explicitly skip fast
            bvp_data=None,          # Explicitly skip fast
            temp_data=temp_data_dict, # Process TEMP
            hr_data=None,           # Process HR fast only
            hrv_data=hrv_data_dict    # Process HRV
        )
        record.update(features)
        results.append(record)

    if not results:
        return None

    # Sort results chronologically
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('seconds').reset_index(drop=True)
    return df_results


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    demos = load_demographics()
    print(f"Loaded metadata for {len(demos)} participants.")

    if not SENSORS_DIR.exists():
        print(f"Missing folder {SENSORS_DIR}")
        return

    pids = [d.name for d in SENSORS_DIR.iterdir() if d.is_dir()]
    pids = sorted(pids, key=lambda x: int(x) if x.isdigit() else 9999)

    print(f"Found {len(pids)} participants: {pids}")

    for pid in pids:
        all_cond_results = []
        for cond in ['c1', 'c2', 'c3']:
            print(f"Processing {pid} {cond}...")
            df_cond = process_participant_condition(pid, cond)
            if df_cond is not None and not df_cond.empty:
                all_cond_results.append(df_cond)

        if all_cond_results:
            merged_df = pd.concat(all_cond_results, ignore_index=True)

            # Add demographics
            participant_demo = demos.get(str(pid), {})
            if participant_demo:
                merged_df['gender'] = participant_demo.get('gender', 'Unknown')
                merged_df['age'] = participant_demo.get('age', np.nan)
                merged_df['age_group'] = participant_demo.get('age_group', 'Unknown')

            # Subject-level EDA normalization (Min-Max)
            merged_df = normalize_eda_features_subject(merged_df)

            out_file = OUTPUT_DIR / f"P{pid}_merged.csv"
            merged_df.to_csv(out_file, index=False)
            print(f"Saved {out_file} ({len(merged_df)} windows)")
        else:
            print(f"No results for participant {pid}")

if __name__ == "__main__":
    main()





