#!/usr/bin/env python3
"""
Prepare Data for ARMADA Algorithm

This script transforms data from three datasets (CASE, K-emoCon, CEAP) into the format
required by the ARMADA algorithm to discover temporal patterns.

Output ARMADA format requires:
- client-id (participant identifier)
- state (e.g. arousal_high, valence_low, eda_high)
- start-time
- end-time

Scale normalization:
- CASE: arousal/valence 0.5-9.5 -> 0-1
- K-emoCon: arousal/valence 1-5 -> 0-1
- CEAP: arousal/valence 1-9 -> 0-1

State discretization:
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

# Paths
BASE_DIR = Path(__file__).parent.parent.parent.parent
CASE_PROCESSED = BASE_DIR / "data" / "CASE" / "processed"
KEMOCON_PROCESSED = BASE_DIR / "data" / "K-emoCon" / "processed"
CEAP_PROCESSED = BASE_DIR / "data" / "CEAP" / "processed"
EMOWORKER_PROCESSED = BASE_DIR / "data" / "EmoWorker_v2" / "processed"
OUTPUT_DIR = BASE_DIR / "data" / "armada_ready"

# Normalization constants
SCALE_RANGES = {
    'CASE': {'arousal': (0.5, 9.5), 'valence': (0.5, 9.5)},
    'K-emoCon': {'arousal': (1, 5), 'valence': (1, 5)},
    'CEAP': {'arousal': (-1, 1), 'valence': (-1, 1)},  # CEAP używa Raw data w zakresie [-1, 1]
    'EmoWorker_v2': {'arousal': (1, 9), 'valence': (1, 9)} # Standard SAM 1-9
}

# Discretization thresholds (for 0-1 scale)
# Based on emotion recognition methods review (Ahmad et al.)
# SAM (Self-Assessment Manikin) 1-9 grouped as:
#   1-3: negative/low arousal -> [0.00, 0.25]
#   4-6: neutral/moderate -> (0.25, 0.75)
#   7-9: positive/high arousal -> [0.75, 1.00]
#
# Interpretation for valence:
#   low (0.00-0.25): clearly unpleasant (sadness, anger, disgust)
#   medium (0.25-0.75): no strong valence, mixed or indifferent states
#   high (0.75-1.00): clearly pleasant (joy, delight, contentment)
#
# Interpretation for arousal:
#   low (0.00-0.25): very low arousal (relaxation, sleepiness, boredom)
#   medium (0.25-0.75): moderate arousal (everyday alertness, concentration)
#   high (0.75-1.00): high arousal (excitement or stress, depending on valence)
DISCRETIZE_THRESHOLDS = {
    'low': (0, 0.33),
    'medium': (0.33, 0.66),
    'high': (0.66, 1.0)
}

# Discretization thresholds for EDA (personally normalized to [0,1])
# Based on EDA and emotion literature (Boucsein, Venables & Christie, Horvers et al., Greco et al.)
# EDA is a marker of sympathetic arousal - personal normalization required
#
# Interpretation:
#   low (0.00-0.33): low SCL, small number of SCR/min (1-3 SCR/min), relaxation/sleepiness
#   medium (0.33-0.66): typical wakefulness, mild stress, educational situations
#   high (0.66-1.00): increased sympathetic activity, high frequency and amplitude of SCR
EDA_THRESHOLDS = {
    'low': (0, 0.33),
    'medium': (0.33, 0.66),
    'high': (0.66, 1.0)
}

# Physiological variables to discretize (will use percentiles)
# Updated names according to the new "Global Processing, Local Aggregation" methodology
# EDA (6 features from NeuroKit2 pipeline):
#   eda_mean     - SCL, mean tonic level (Skin Conductance Level)
#   eda_std      - phasic component variability (SCR variability)
#   eda_max      - signal maximum in window
#   eda_peaks    - number of SCR peaks (most reliable arousal indicator according to Braithwaite et al.)
#   eda_scr_mean_amp - mean SCR peaks amplitude
#   eda_scr_auc  - phasic component AUC (total electrodermal activity)
# HRV (NeuroKit2 nk.hrv_time + nk.hrv_frequency):
#   Time domain:  hrv_sdnn, hrv_rmssd, hrv_pnn50, hrv_pnn20, hrv_cvnn, hrv_cvsd, …
#   Freq domain:  hrv_lf, hrv_hf, hrv_vlf, hrv_lf_hf_ratio, hrv_lfn, hrv_hfn, hrv_lnhf
PHYSIO_VARIABLES = [
    'eda_mean', 'eda_std', 'eda_max', 'eda_peaks', 'eda_scr_mean_amp', 'eda_scr_auc',
    'hr_mean', 'temp_mean',
    # HRV time domain
    'hrv_sdnn', 'hrv_rmssd', 'hrv_pnn50', 'hrv_pnn20',
    'hrv_cvnn', 'hrv_cvsd',
    # HRV frequency domain
    'hrv_lf_hf_ratio', 'hrv_lfn', 'hrv_hfn',
]

# Variables requiring personal normalization (min-max per participant)
# All physiological features are now scaled personally min-max to address baseline individual differences,
# prior to applying global terciles across the normalized dataset.
PERSONAL_NORM_VARIABLES = PHYSIO_VARIABLES.copy()

# Mapping column names to readable state names
VARIABLE_NAME_MAPPING = {
    'eda_mean': 'eda',
    'eda_std': 'eda_std',
    'eda_max': 'eda_max',
    'eda_peaks': 'eda_peaks',
    'eda_scr_mean_amp': 'eda_scr_amp',
    'eda_scr_auc': 'eda_scr_auc',
    'hr_mean': 'hr',
    'temp_mean': 'temp',
    # HRV time domain
    'hrv_sdnn': 'hrv_sdnn',
    'hrv_rmssd': 'hrv_rmssd',
    'hrv_pnn50': 'hrv_pnn50',
    'hrv_pnn20': 'hrv_pnn20',
    'hrv_cvnn': 'hrv_cvnn',
    'hrv_cvsd': 'hrv_cvsd',
    # HRV frequency domain
    'hrv_lf_hf_ratio': 'hrv_lf_hf',
    'hrv_lfn': 'hrv_lfn',
    'hrv_hfn': 'hrv_hfn',
    # Emotions
    'arousal_norm': 'arousal',
    'valence_norm': 'valence'
}

# Time window (seconds)
# 30s ensures stable HRV metrics (Task Force ESC requires min 60s for freq-domain,
# but 30s is minimum for time-domain: SDNN, RMSSD; at 5s we only have 3-5 beats)
WINDOW_SIZE = 30


def normalize_value(value: float, min_val: float, max_val: float) -> float:
    """Normalizes value to 0-1 scale."""
    if pd.isna(value):
        return np.nan
    return (value - min_val) / (max_val - min_val)


def discretize_value(value: float, thresholds: Dict[str, Tuple[float, float]]) -> Optional[str]:
    """Discretizes a value to a category based on thresholds."""
    if pd.isna(value):
        return None

    for label, (low, high) in thresholds.items():
        if low <= value < high:
            return label
        if label == 'high' and value >= high - 0.001:  # Handle 1.0 value
            return label
    return None


def compute_physio_thresholds(df: pd.DataFrame, variable: str) -> Dict[str, Tuple[float, float]]:
    """
    Computes discretization thresholds for a physiological variable based on percentiles.
    Uses terciles (33% and 67%).
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


def normalize_personal_minmax(df: pd.DataFrame, variable: str) -> pd.Series:
    """
    Personal min-max normalization for a physiological variable.

    According to psychophysiological recommendations (Birmingham EDA guide):
    EDA_norm = (EDA - EDA_min) / (EDA_max - EDA_min)

    This is required for EDA because conductance levels are highly individual.

    Args:
        df: DataFrame with participant data
        variable: name of the variable to normalize

    Returns:
        Series with values normalized to [0, 1]
    """
    values = df[variable].copy()
    valid_values = values.dropna()

    if len(valid_values) < 2:
        return pd.Series([np.nan] * len(values), index=values.index)

    min_val = valid_values.min()
    max_val = valid_values.max()

    if max_val - min_val < 1e-10:  # Avoid division by zero
        return pd.Series([0.5] * len(values), index=values.index)

    normalized = (values - min_val) / (max_val - min_val)
    return normalized


def extract_state_intervals(
    df: pd.DataFrame,
    variable: str,
    thresholds: Dict[str, Tuple[float, float]],
    prefix: str = ""
) -> List[Dict]:
    """
    Extracts state intervals from a time series.

    Returns a list of dictionaries with fields:
    - state: state name (e.g. "arousal_high")
    - start_time: interval start time
    - end_time: interval end time
    """
    intervals = []

    if variable not in df.columns:
        return intervals

    # Sort by time
    df = df.copy().sort_values('seconds').reset_index(drop=True)

    # Discretize values
    states = []
    for idx, row in df.iterrows():
        value = row[variable]
        state = discretize_value(value, thresholds)
        states.append(state)

    df['_state'] = states

    # Find intervals - skip None values
    current_state = None
    start_time = None

    for idx, row in df.iterrows():
        state = row['_state']
        time = row['seconds']

        # Skip None values
        if state is None:
            # Close previous interval if exists
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
            # Close previous interval
            if current_state is not None and start_time is not None:
                end_time = time
                if start_time < end_time:
                    state_name = f"{prefix}{variable}_{current_state}" if prefix else f"{variable}_{current_state}"
                    intervals.append({
                        'state': state_name,
                        'start_time': start_time,
                        'end_time': end_time
                    })

            # Start new interval
            current_state = state
            start_time = time

    # Close last interval
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


def compute_global_thresholds(
    data_dir: Path,
    dataset_name: str
) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """
    Computes global discretization thresholds per dataset (NOT per participant).

    Instead of min-max normalization per participant (which makes low/medium/high
    labels incomparable), we compute terciles on data from ALL participants 
    of a given dataset combined.

    All physiological features undergo personal normalization (min-max) before computing
    global terciles to account for individual baselines. Terciles are then computed 
    globally on these normalized values.

    Returns:
        dict: {variable: {'low': (min, p33), 'medium': (p33, p67), 'high': (p67, max)}}
    """
    csv_files = list(data_dir.glob("*_merged.csv"))
    if not csv_files:
        return {}

    # Load all files and combine
    all_dfs = []
    for csv_file in sorted(csv_files):
        try:
            df = pd.read_csv(csv_file)
            participant_id = csv_file.stem.replace("_merged", "")
            df['_participant_id'] = participant_id
            all_dfs.append(df)
        except Exception:
            continue

    if not all_dfs:
        return {}

    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"  Computing global thresholds from {len(all_dfs)} participants, {len(combined)} windows")

    thresholds = {}

    for var in PHYSIO_VARIABLES:
        if var not in combined.columns:
            continue

        if var in PERSONAL_NORM_VARIABLES:
            # Personal min-max normalization, then global terciles
            norm_values = []
            for pid, group in combined.groupby('_participant_id'):
                values = group[var].dropna()
                if len(values) < 2:
                    continue
                vmin, vmax = values.min(), values.max()
                if vmax - vmin < 1e-10:
                    norm_values.extend([0.5] * len(values))
                else:
                    normed = (values - vmin) / (vmax - vmin)
                    norm_values.extend(normed.tolist())

            if len(norm_values) < 10:
                continue

            norm_series = pd.Series(norm_values)
            p33 = norm_series.quantile(0.33)
            p67 = norm_series.quantile(0.67)
            thresholds[var] = {
                'low': (0.0, p33),
                'medium': (p33, p67),
                'high': (p67, 1.001)
            }
            print(f"    {var} (global after personal norm): p33={p33:.3f}, p67={p67:.3f}")
        else:
            # Other variables: global terciles on raw values
            values = combined[var].dropna()
            if len(values) < 10:
                continue

            p33 = values.quantile(0.33)
            p67 = values.quantile(0.67)
            min_val = values.min()
            max_val = values.max()
            thresholds[var] = {
                'low': (min_val, p33),
                'medium': (p33, p67),
                'high': (p67, max_val + 0.001)
            }
            print(f"    {var}: p33={p33:.3f}, p67={p67:.3f} (range {min_val:.3f}–{max_val:.3f})")

    return thresholds


def compute_cross_dataset_thresholds(
    dataset_dirs: Dict[str, Path]
) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """
    Computes GLOBAL discretization thresholds from ALL datasets combined.

    Instead of computing terciles per dataset (which gives incomparable 
    labels between datasets), we compute terciles on data from ALL participants
    across all datasets.

    For all variables: personal min-max normalization, then global terciles

    Args:
        dataset_dirs: dict {dataset_name: path_to_processed}

    Returns:
        dict: {variable: {'low': (min, p33), 'medium': (p33, p67), 'high': (p67, max)}}
    """
    print("\n" + "=" * 60)
    print("COMPUTING GLOBAL CROSS-DATASET THRESHOLDS")
    print("=" * 60)

    all_dfs = []
    for ds_name, data_dir in dataset_dirs.items():
        if not data_dir.exists():
            print(f"  SKIP: {ds_name} — folder {data_dir} not found")
            continue

        csv_files = list(data_dir.glob("*_merged.csv"))
        for csv_file in sorted(csv_files):
            try:
                df = pd.read_csv(csv_file)
                participant_id = csv_file.stem.replace("_merged", "")
                df['_participant_id'] = f"{ds_name}_{participant_id}"
                df['_dataset'] = ds_name
                all_dfs.append(df)
            except Exception:
                continue

        print(f"  {ds_name}: loaded {len(csv_files)} files")

    if not all_dfs:
        print("  No data to compute thresholds!")
        return {}

    combined = pd.concat(all_dfs, ignore_index=True)
    n_participants = combined['_participant_id'].nunique()
    n_datasets = combined['_dataset'].nunique()
    print(f"  Total: {n_participants} participants from {n_datasets} datasets, {len(combined)} windows")

    thresholds = {}

    for var in PHYSIO_VARIABLES:
        if var not in combined.columns:
            continue

        if var in PERSONAL_NORM_VARIABLES:
            # Personal min-max normalization, then global terciles
            norm_values = []
            for pid, group in combined.groupby('_participant_id'):
                values = group[var].dropna()
                if len(values) < 2:
                    continue
                vmin, vmax = values.min(), values.max()
                if vmax - vmin < 1e-10:
                    norm_values.extend([0.5] * len(values))
                else:
                    normed = (values - vmin) / (vmax - vmin)
                    norm_values.extend(normed.tolist())

            if len(norm_values) < 10:
                continue

            norm_series = pd.Series(norm_values)
            p33 = norm_series.quantile(0.33)
            p67 = norm_series.quantile(0.67)
            thresholds[var] = {
                'low': (0.0, p33),
                'medium': (p33, p67),
                'high': (p67, 1.001)
            }
            print(f"    {var} (cross-dataset after personal norm): p33={p33:.3f}, p67={p67:.3f}")
        else:
            # Other variables: global terciles on raw values
            values = combined[var].dropna()
            if len(values) < 10:
                continue

            p33 = values.quantile(0.33)
            p67 = values.quantile(0.67)
            min_val = values.min()
            max_val = values.max()
            thresholds[var] = {
                'low': (min_val, p33),
                'medium': (p33, p67),
                'high': (p67, max_val + 0.001)
            }
            print(f"    {var}: p33={p33:.3f}, p67={p67:.3f} (range {min_val:.3f}–{max_val:.3f})")

    print(f"  Computed cross-dataset thresholds for {len(thresholds)} variables")
    return thresholds


def process_participant_data(
    df: pd.DataFrame,
    dataset_name: str,
    participant_id: str,
    global_thresholds: Dict[str, Dict[str, Tuple[float, float]]] = None
) -> pd.DataFrame:
    """
    Processes participant data and converts to ARMADA format.

    Uses GLOBAL tercile thresholds (computed on whole dataset)
    instead of per-participant - this makes low/medium/high labels
    comparable between participants and datasets.

    Args:
        df: DataFrame with participant data
        dataset_name: dataset name (CASE, K-emoCon, CEAP)
        participant_id: participant identifier
        global_thresholds: thresholds computed by compute_global_thresholds()
    """
    if df is None or len(df) == 0:
        return None

    if global_thresholds is None:
        global_thresholds = {}

    # Normalize arousal and valence
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

    # Collect all intervals
    all_intervals = []

    # Intervals for arousal (normalized)
    if 'arousal_norm' in df.columns:
        intervals = extract_state_intervals(df, 'arousal_norm', DISCRETIZE_THRESHOLDS, prefix='')
        # Change variable name to arousal
        for iv in intervals:
            iv['state'] = iv['state'].replace('arousal_norm', 'arousal')
        all_intervals.extend(intervals)

    # Intervals for valence (normalized)
    if 'valence_norm' in df.columns:
        intervals = extract_state_intervals(df, 'valence_norm', DISCRETIZE_THRESHOLDS, prefix='')
        for iv in intervals:
            iv['state'] = iv['state'].replace('valence_norm', 'valence')
        all_intervals.extend(intervals)

    # Intervals for physiological variables — GLOBAL THRESHOLDS
    for var in PHYSIO_VARIABLES:
        if var not in df.columns:
            continue

        if var in global_thresholds:
            gt = global_thresholds[var]

            if var in PERSONAL_NORM_VARIABLES:
                # Personal min-max normalization, then global thresholds
                if f'{var}_norm' in df.columns:
                    col_to_use = f'{var}_norm'
                else:
                    df[f'{var}_norm'] = normalize_personal_minmax(df, var)
                    col_to_use = f'{var}_norm'

                intervals = extract_state_intervals(df, col_to_use, gt)
                readable_name = VARIABLE_NAME_MAPPING.get(var, var)
                for iv in intervals:
                    iv['state'] = iv['state'].replace(col_to_use, readable_name)
                all_intervals.extend(intervals)
            else:
                # Others: global thresholds on raw values
                intervals = extract_state_intervals(df, var, gt)
                readable_name = VARIABLE_NAME_MAPPING.get(var, var)
                for iv in intervals:
                    iv['state'] = iv['state'].replace(var, readable_name)
                all_intervals.extend(intervals)
        else:
            # Fallback: terciles per participant (if no global thresholds)
            thresholds = compute_physio_thresholds(df, var)
            if thresholds is not None:
                intervals = extract_state_intervals(df, var, thresholds)
                readable_name = VARIABLE_NAME_MAPPING.get(var, var)
                for iv in intervals:
                    iv['state'] = iv['state'].replace(var, readable_name)
                all_intervals.extend(intervals)

    if not all_intervals:
        return None

    # Convert to DataFrame
    result_df = pd.DataFrame(all_intervals)
    result_df['client_id'] = f"{dataset_name}_{participant_id}"
    result_df['dataset'] = dataset_name
    result_df['participant_id'] = participant_id

    # Remove intervals where start_time >= end_time
    result_df = result_df[result_df['start_time'] < result_df['end_time']]

    # Remove duplicates
    result_df = result_df.drop_duplicates(subset=['state', 'start_time', 'end_time'])

    # Reorder columns
    result_df = result_df[['client_id', 'dataset', 'participant_id', 'state', 'start_time', 'end_time']]

    # Sort by start_time, end_time, state
    result_df = result_df.sort_values(['start_time', 'end_time', 'state']).reset_index(drop=True)

    return result_df


def process_dataset(
    dataset_name: str,
    data_dir: Path,
    cross_dataset_thresholds: Dict[str, Dict[str, Tuple[float, float]]] = None
) -> List[pd.DataFrame]:
    """
    Processes all files from a given dataset.

    If cross_dataset_thresholds are provided, uses them (thresholds computed on
    all datasets combined). Otherwise, computes thresholds per dataset.
    """
    results = []

    csv_files = list(data_dir.glob("*_merged.csv"))
    print(f"\n{dataset_name}: Found {len(csv_files)} files")

    if cross_dataset_thresholds is not None:
        global_thresholds = cross_dataset_thresholds
        print(f"  Using CROSS-DATASET thresholds ({len(global_thresholds)} variables)")
    else:
        # Fallback: thresholds per dataset
        print(f"  Computing global thresholds per dataset...")
        global_thresholds = compute_global_thresholds(data_dir, dataset_name)
        print(f"  Computed thresholds for {len(global_thresholds)} variables")

    # Process each participant with global thresholds
    for csv_file in sorted(csv_files):
        participant_id = csv_file.stem.replace("_merged", "")

        try:
            df = pd.read_csv(csv_file)
            result = process_participant_data(df, dataset_name, participant_id, global_thresholds)

            if result is not None and len(result) > 0:
                results.append(result)
                print(f"  {participant_id}: {len(result)} intervals")
            else:
                print(f"  {participant_id}: No data")
        except Exception as e:
            print(f"  {participant_id}: Error - {e}")

    return results


def create_combined_dataset(all_results: List[pd.DataFrame]) -> pd.DataFrame:
    """Combines all results into one dataset."""
    if not all_results:
        return None

    combined = pd.concat(all_results, ignore_index=True)
    return combined


def save_armada_format(df: pd.DataFrame, output_path: Path, format_type: str = 'csv'):
    """
    Saves data in formatting required by ARMADA.

    Format CSV:
    client_id, state, start_time, end_time
    """
    # Basic CSV format
    armada_df = df[['client_id', 'state', 'start_time', 'end_time']].copy()
    armada_df = armada_df.sort_values(['client_id', 'start_time', 'end_time', 'state'])

    # Save with header
    with open(output_path, 'w') as f:
        f.write("client_id,state,start_time,end_time\n")
        for _, row in armada_df.iterrows():
            f.write(f"{row['client_id']},{row['state']},{row['start_time']},{row['end_time']}\n")

    print(f"Saved: {output_path}")


def generate_summary_statistics(df: pd.DataFrame) -> Dict:
    """Generates summary statistics for the dataset."""
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
    """Main processing function."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PREPARING DATA FOR ARMADA ALGORITHM")
    print("=" * 60)

    dataset_dirs = {}
    if CASE_PROCESSED.exists():
        dataset_dirs['CASE'] = CASE_PROCESSED
    if KEMOCON_PROCESSED.exists():
        dataset_dirs['K-emoCon'] = KEMOCON_PROCESSED
    if CEAP_PROCESSED.exists():
        dataset_dirs['CEAP'] = CEAP_PROCESSED
    if EMOWORKER_PROCESSED.exists():
        dataset_dirs['EmoWorker_v2'] = EMOWORKER_PROCESSED

    cross_thresholds = compute_cross_dataset_thresholds(dataset_dirs)

    all_results = []

    if CASE_PROCESSED.exists():
        case_results = process_dataset('CASE', CASE_PROCESSED, cross_thresholds)
        all_results.extend(case_results)

    if KEMOCON_PROCESSED.exists():
        kemocon_results = process_dataset('K-emoCon', KEMOCON_PROCESSED, cross_thresholds)
        all_results.extend(kemocon_results)

    if CEAP_PROCESSED.exists():
        ceap_results = process_dataset('CEAP', CEAP_PROCESSED, cross_thresholds)
        all_results.extend(ceap_results)

    if EMOWORKER_PROCESSED.exists():
        emoworker_results = process_dataset('EmoWorker_v2', EMOWORKER_PROCESSED, cross_thresholds)
        all_results.extend(emoworker_results)

    if not all_results:
        print("\nNo data to process!")
        return

    print("\n" + "=" * 60)
    print("COMBINING DATA")
    print("=" * 60)

    combined_df = create_combined_dataset(all_results)
    print(f"Total: {len(combined_df)} intervals from {combined_df['client_id'].nunique()} participants")

    # Save data
    print("\n" + "=" * 60)
    print("SAVING DATA")
    print("=" * 60)

    save_armada_format(
        combined_df,
        OUTPUT_DIR / "armada_combined_all.csv"
    )

    for dataset in ['CASE', 'K-emoCon', 'CEAP', 'EmoWorker_v2']:
        dataset_df = combined_df[combined_df['dataset'] == dataset]
        if len(dataset_df) > 0:
            save_armada_format(
                dataset_df,
                OUTPUT_DIR / f"armada_{dataset.lower().replace('-', '_')}.csv"
            )

    combined_df.to_csv(OUTPUT_DIR / "armada_full_metadata.csv", index=False)
    print(f"Saved: {OUTPUT_DIR / 'armada_full_metadata.csv'}")

    print("\n" + "=" * 60)
    print("STATISTICS")
    print("=" * 60)

    stats = generate_summary_statistics(combined_df)
    print(f"Total number of intervals: {stats['total_intervals']}")
    print(f"Number of participants: {stats['unique_clients']}")
    print(f"Number of unique states: {stats['unique_states']}")
    print(f"Average interval duration: {stats['avg_interval_duration']:.2f}s")

    print("\nStates distribution:")
    for state, count in sorted(stats['states_distribution'].items(), key=lambda x: -x[1])[:15]:
        print(f"  {state}: {count}")

    print("\nDatasets distribution:")
    for dataset, count in stats['datasets_distribution'].items():
        print(f"  {dataset}: {count}")

    stats_df = pd.DataFrame([{
        'metric': k,
        'value': str(v) if isinstance(v, dict) else v
    } for k, v in stats.items()])
    stats_df.to_csv(OUTPUT_DIR / "armada_statistics.csv", index=False)

    print("\n" + "=" * 60)
    print("FINISHED")
    print("=" * 60)


if __name__ == "__main__":
    main()

