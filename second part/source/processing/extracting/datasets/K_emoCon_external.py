"""
K-EmoCon Data Processing Script — External Annotations

Same processing as K_emoCon.py but uses aggregated external annotations
(from external observers) instead of self-assessments.

External annotations: K-emoCon/raw/aggregated_external_annotations/P{X}.external.csv
- Columns: seconds, arousal, valence, cheerful, happy, angry, nervous, sad, ...
- Every 5 seconds (same as self-annotations)
- Arousal/valence scale: 1–5 (same as self-annotations)

The physiological processing pipeline is identical to K_emoCon.py:
- Global Processing: EDA decomposition, global HR from IBI
- Local Aggregation: dual-resolution windows (5s fast / 30s slow)
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
from config import (
    WINDOW_FAST_SEC, WINDOW_SLOW_SEC, ANNOT_STEP_SEC,
    ANNOTS_PER_FAST_WINDOW, ANNOTS_PER_SLOW_WINDOW,
    FS_EDA, FS_HR, FS_TEMP, FS_BVP, FS_ACC
)
from demographics_utils import get_age_group

# Paths
BASE_DIR = Path(__file__).parent.parent.parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "K-emoCon" / "raw"
OUTPUT_DIR = BASE_DIR / "data" / "K-emoCon" / "processed_external"
METADATA_DIR = BASE_DIR / "data" / "K-emoCon" / "metadata"
E4_DIR = DATA_DIR / "e4_data"
EXTERNAL_ANNOTATIONS_DIR = DATA_DIR / "aggregated_external_annotations"


def load_participant_metadata():
    """Loads participant metadata (age, gender) from CSV."""
    meta_path = METADATA_DIR / "participants.csv"
    if not meta_path.exists():
        print(f"No metadata file: {meta_path}")
        return {}

    df = pd.read_csv(meta_path)
    metadata = {}
    for _, row in df.iterrows():
        pid = str(row['Participant_ID']).replace('P', '')
        metadata[pid] = {
            'gender': row['Gender'],
            'age': row['Age'],
            'age_group': get_age_group(row['Age'])
        }
    return metadata


def load_e4_signal(participant_folder, signal_name):
    """Loads E4 signal for a given participant."""
    file_path = E4_DIR / str(participant_folder) / f"E4_{signal_name}.csv"
    if not file_path.exists():
        return None
    return pd.read_csv(file_path)


def load_external_annotations(pid) -> pd.DataFrame:
    """Loads aggregated external annotations for a participant."""
    file_path = EXTERNAL_ANNOTATIONS_DIR / f"P{pid}.external.csv"
    if not file_path.exists():
        return None

    df = pd.read_csv(file_path)
    # We need seconds, arousal and valence
    return df[['seconds', 'arousal', 'valence']]


def compute_global_hr_from_ibi_kemocon(ibi_df, start_ts, max_time_ms):
    """Computes global HR from IBI for K-EmoCon."""
    if ibi_df is None or len(ibi_df) < 2:
        return np.array([]), np.array([])

    timestamps = ibi_df['timestamp'].values
    ibi_values = ibi_df['value'].values

    return compute_global_hr_from_ibi(timestamps, ibi_values, start_ts + max_time_ms, ibi_unit='ms')


def process_participant(e4_folder, pid, metadata=None):
    """
    Processes data for one participant using EXTERNAL annotations.

    Identical methodology to K_emoCon.py:
    - Global Processing, Local Aggregation
    - Dual-resolution windows (5s fast / 30s slow)
    """
    print(f"Processing participant P{pid} (external annotations)...")

    # Load external annotations
    annotations = load_external_annotations(pid)
    if annotations is None:
        print(f"  No external annotations for P{pid}")
        return None

    # Load E4 signals
    signals = {}
    for signal_name in ['EDA', 'TEMP', 'BVP', 'IBI']:
        df = load_e4_signal(e4_folder, signal_name)
        if df is not None:
            signals[signal_name] = df

    acc_df = load_e4_signal(e4_folder, 'ACC')
    if acc_df is not None:
        if 'x' in acc_df.columns:
            signals['ACC'] = acc_df
        elif 'value' in acc_df.columns:
            signals['ACC'] = acc_df

    if not signals:
        print(f"  No E4 signals for participant {pid}")
        return None

    # Find start timestamp
    start_timestamps = []
    for sig_name, sig_df in signals.items():
        if 'timestamp' in sig_df.columns and len(sig_df) > 0:
            start_timestamps.append(sig_df['timestamp'].min())

    if not start_timestamps:
        print(f"  No timestamps for participant {pid}")
        return None

    start_ts = min(start_timestamps)

    max_seconds = annotations['seconds'].max()
    max_time_ms = max_seconds * 1000

    # Global Processing: HR from IBI
    time_grid, hr_timeseries = np.array([]), np.array([])
    if 'IBI' in signals:
        time_grid, hr_timeseries = compute_global_hr_from_ibi_kemocon(
            signals['IBI'], start_ts, max_time_ms
        )
        print(f"  Global Processing: HR timeseries ({len(hr_timeseries)} samples)")

    # Global Processing: EDA decomposition
    eda_filtered, eda_tonic, eda_phasic = None, None, None
    eda_ts = None
    if 'EDA' in signals:
        eda_df = signals['EDA']
        full_eda = eda_df['value'].values
        eda_ts = eda_df['timestamp'].values
        eda_filtered, eda_tonic, eda_phasic = preprocess_eda_global(full_eda, FS_EDA)
        print(f"  Global Processing: EDA decomposition ({len(full_eda)} samples)")

    # Prepare numpy arrays
    bvp_ts, bvp_vals = None, None
    if 'BVP' in signals:
        bvp_ts = signals['BVP']['timestamp'].values
        bvp_vals = signals['BVP']['value'].values

    temp_ts, temp_vals = None, None
    if 'TEMP' in signals:
        temp_ts = signals['TEMP']['timestamp'].values
        temp_vals = signals['TEMP']['value'].values

    acc_ts, acc_x_arr, acc_y_arr, acc_z_arr = None, None, None, None
    if 'ACC' in signals and 'x' in signals['ACC'].columns:
        acc_df = signals['ACC']
        acc_ts = acc_df['timestamp'].values
        acc_x_arr = acc_df['x'].values
        acc_y_arr = acc_df['y'].values
        acc_z_arr = acc_df['z'].values

    ibi_ts, ibi_vals = None, None
    if 'IBI' in signals:
        ibi_ts = signals['IBI']['timestamp'].values
        ibi_vals = signals['IBI']['value'].values

    ann_seconds = annotations['seconds'].values
    ann_arousal = annotations['arousal'].values
    ann_valence = annotations['valence'].values

    del signals
    del annotations

    # Prepare data dicts
    eda_data_dict = {'ts': eda_ts, 'filtered': eda_filtered, 'tonic': eda_tonic, 'phasic': eda_phasic, 'fs': FS_EDA} if eda_ts is not None else None
    bvp_data_dict = {'ts': bvp_ts, 'values': bvp_vals, 'fs': FS_BVP} if bvp_ts is not None else None
    temp_data_dict = {'ts': temp_ts, 'values': temp_vals} if temp_ts is not None else None
    acc_data_dict = {'ts': acc_ts, 'x': acc_x_arr, 'y': acc_y_arr, 'z': acc_z_arr} if acc_ts is not None else None
    hr_data_dict = {'time_grid': time_grid, 'timeseries': hr_timeseries}
    hrv_data_dict = {'type': 'ibi', 'ts': ibi_ts, 'values': ibi_vals, 'unit': 'ms'} if ibi_ts is not None else None

    ANNOT_STEP = ANNOT_STEP_SEC

    results = []
    n_annotations = len(ann_seconds)

    # FAST WINDOWS (5s)
    n_windows_fast = n_annotations // ANNOTS_PER_FAST_WINDOW
    for wi in range(n_windows_fast):
        idx_start = wi * ANNOTS_PER_FAST_WINDOW
        idx_end = idx_start + ANNOTS_PER_FAST_WINDOW

        window_seconds_end = ann_seconds[idx_end - 1]
        window_seconds_start = ann_seconds[idx_start] - ANNOT_STEP

        arousal = float(np.nanmean(ann_arousal[idx_start:idx_end]))
        valence = float(np.nanmean(ann_valence[idx_start:idx_end]))

        window_start_ms = start_ts + window_seconds_start * 1000
        window_end_ms = start_ts + window_seconds_end * 1000

        record = {
            'seconds': window_seconds_end,
            'arousal': arousal,
            'valence': valence,
            'window_type': 'fast'
        }

        features = extract_window_features(
            window_start_ms, window_end_ms,
            eda_data=eda_data_dict,
            bvp_data=bvp_data_dict,
            temp_data=None,
            acc_data=acc_data_dict,
            hr_data=hr_data_dict,
            hrv_data=None
        )
        record.update(features)

        record['temp_mean'] = np.nan
        record['temp_slope'] = np.nan

        results.append(record)

    # SLOW WINDOWS (30s)
    n_windows_slow = n_annotations // ANNOTS_PER_SLOW_WINDOW
    for wi in range(n_windows_slow):
        idx_start = wi * ANNOTS_PER_SLOW_WINDOW
        idx_end = idx_start + ANNOTS_PER_SLOW_WINDOW

        window_seconds_end = ann_seconds[idx_end - 1]
        window_seconds_start = ann_seconds[idx_start] - ANNOT_STEP

        window_start_ms = start_ts + window_seconds_start * 1000
        window_end_ms = start_ts + window_seconds_end * 1000

        record = {
            'seconds': window_seconds_end,
            'arousal': np.nan,
            'valence': np.nan,
            'window_type': 'slow'
        }

        features = extract_window_features(
            window_start_ms, window_end_ms,
            eda_data=None,
            bvp_data=None,
            temp_data=temp_data_dict,
            acc_data=None,
            hr_data=None,
            hrv_data=hrv_data_dict
        )
        record.update(features)
        results.append(record)

    if not results:
        return None

    df = pd.DataFrame(results)
    df = df.sort_values('seconds').reset_index(drop=True)

    if metadata:
        df['gender'] = metadata.get('gender', 'Unknown')
        df['age'] = metadata.get('age', np.nan)
        df['age_group'] = metadata.get('age_group', 'Unknown')

    # Personal EDA normalization
    df = normalize_eda_features_subject(df)
    print(f"  Personal EDA normalization: added *_norm columns")

    return df


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("K-EmoCon Processing — EXTERNAL Annotations")
    print("=" * 60)

    participants_metadata = load_participant_metadata()
    print(f"Loaded participants.csv: {len(participants_metadata)} records")

    # Get participant folders from E4 data
    e4_folders = [f.name for f in E4_DIR.iterdir() if f.is_dir()]

    for pid in sorted(e4_folders, key=lambda x: int(x) if x.isdigit() else x):
        # Check if external annotations exist
        ext_path = EXTERNAL_ANNOTATIONS_DIR / f"P{pid}.external.csv"
        if not ext_path.exists():
            print(f"  Skipping P{pid}: no external annotations")
            continue

        participant_meta = participants_metadata.get(str(pid), {})

        result_df = process_participant(
            pid,
            pid,
            metadata=participant_meta
        )

        if result_df is not None and len(result_df) > 0:
            output_file = OUTPUT_DIR / f"P{pid}_merged.csv"
            result_df.to_csv(output_file, index=False)
            print(f"  Saved {len(result_df)} records to {output_file.name}")
        else:
            print(f"  No data to save for P{pid}")

    print("\nK-emoCon external processing complete!")


if __name__ == "__main__":
    print("Starting K-emoCon external annotations processing...", flush=True)
    main()
