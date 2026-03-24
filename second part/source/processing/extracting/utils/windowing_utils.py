#!/usr/bin/env python3
"""
Tools for aggregating physiological signals in time windows.
Abstracts common logic for slicing signals and extracting features
used by various datasets (CASE, CEAP, K-emoCon, EmoWorker).
"""
import numpy as np
from feature_utils import (
    compute_eda_features,
    compute_temp_features,
    compute_hr_window_features,
    compute_hrv_window_features,
    compute_bvp_features,
    compute_acc_features,
    compute_hrv_from_ecg_window
)
from bvp_utils import _empty_hrv_result

def _empty_bvp_result() -> dict:
    return {'bvp_std': np.nan, 'bvp_peak_to_peak': np.nan, 'bvp_spectral_power': np.nan}

def _empty_eda_result() -> dict:
    return {'eda_mean': np.nan, 'eda_std': np.nan, 'eda_max': np.nan, 'eda_peaks': 0, 'eda_scr_mean_amp': np.nan, 'eda_scr_auc': np.nan}

def _empty_temp_result() -> dict:
    return {'temp_mean': np.nan, 'temp_slope': np.nan}

def _empty_acc_result() -> dict:
    return {
        'acc_x_mean': np.nan, 'acc_x_std': np.nan,
        'acc_y_mean': np.nan, 'acc_y_std': np.nan,
        'acc_z_mean': np.nan, 'acc_z_std': np.nan,
        'acc_magnitude_mean': np.nan, 'acc_magnitude_std': np.nan
    }

def _empty_hr_result() -> dict:
    return {'hr_mean': np.nan, 'hr_std': np.nan}

def extract_window_features(
    window_start_ms: float,
    window_end_ms: float,
    eda_data: dict = None,
    bvp_data: dict = None,
    temp_data: dict = None,
    acc_data: dict = None,
    hr_data: dict = None,
    hrv_data: dict = None
) -> dict:
    """
    Extracts physiological features for a given time window (in ms).
    
    Expected keys in dictionaries:
    - eda_data: 'ts', 'filtered', 'tonic', 'phasic', 'fs'
    - bvp_data: 'ts', 'values', 'fs'
    - temp_data: 'ts', 'values'
    - acc_data: 'ts', 'x', 'y', 'z'
    - hr_data: 'time_grid', 'timeseries'
    - hrv_data: 'type' ('ibi' or 'ecg'), 'ts'/'r_peaks', 'values'/'ts_ecg', 'unit'/'fs'
    """
    record = {}

    # 1. EDA
    if eda_data and eda_data.get('ts') is not None and eda_data.get('filtered') is not None:
        ts = eda_data['ts']
        i0 = np.searchsorted(ts, window_start_ms, side='left')
        i1 = np.searchsorted(ts, window_end_ms, side='left')
        if i1 > i0:
            eda_feat = compute_eda_features(
                eda_data['filtered'][i0:i1], eda_data['fs'],
                tonic_values=eda_data['tonic'][i0:i1] if 'tonic' in eda_data else None,
                phasic_values=eda_data['phasic'][i0:i1] if 'phasic' in eda_data else None
            )
            record.update(eda_feat)
        else:
            record.update(_empty_eda_result())
    else:
        record.update(_empty_eda_result())

    # 2. BVP
    if bvp_data and bvp_data.get('ts') is not None:
        ts = bvp_data['ts']
        i0 = np.searchsorted(ts, window_start_ms, side='left')
        i1 = np.searchsorted(ts, window_end_ms, side='left')
        if i1 > i0:
            bvp_feat = compute_bvp_features(bvp_data['values'][i0:i1], bvp_data['fs'])
            record.update(bvp_feat)
        else:
            record.update(_empty_bvp_result())
    else:
        record.update(_empty_bvp_result())

    # 3. TEMP
    if temp_data and temp_data.get('ts') is not None:
        ts = temp_data['ts']
        i0 = np.searchsorted(ts, window_start_ms, side='left')
        i1 = np.searchsorted(ts, window_end_ms, side='left')
        if i1 > i0:
            temp_feat = compute_temp_features(temp_data['values'][i0:i1], ts[i0:i1])
            record.update(temp_feat)
        else:
            record.update(_empty_temp_result())
    else:
        record.update(_empty_temp_result())

    # 4. ACC
    if acc_data and acc_data.get('ts') is not None:
        ts = acc_data['ts']
        i0 = np.searchsorted(ts, window_start_ms, side='left')
        i1 = np.searchsorted(ts, window_end_ms, side='left')
        if i1 > i0:
            acc_feat = compute_acc_features(
                acc_data['x'][i0:i1],
                acc_data['y'][i0:i1],
                acc_data['z'][i0:i1]
            )
            record.update(acc_feat)
        else:
            record.update(_empty_acc_result())
    else:
        record.update(_empty_acc_result())

    # 5. HR Window
    if hr_data and hr_data.get('time_grid') is not None and len(hr_data['time_grid']) > 0:
        hr_feat = compute_hr_window_features(
            hr_data['time_grid'], hr_data['timeseries'], window_start_ms, window_end_ms
        )
        record.update(hr_feat)
    else:
        record.update(_empty_hr_result())

    # 6. HRV (from IBI or ECG)
    if hrv_data:
        hrv_type = hrv_data.get('type')
        if hrv_type == 'ibi' and hrv_data.get('ts') is not None:
            hrv_feat = compute_hrv_window_features(
                hrv_data['ts'], hrv_data['values'], window_start_ms, window_end_ms, hrv_data.get('unit', 'ms')
            )
            record.update(hrv_feat)
        elif hrv_type == 'ecg' and hrv_data.get('r_peaks') is not None:
            hrv_feat = compute_hrv_from_ecg_window(
                hrv_data['r_peaks'], hrv_data['ts_ecg'], hrv_data['fs'], window_start_ms, window_end_ms
            )
            record.update(hrv_feat)
        else:
            record.update(_empty_hrv_result())
    else:
        record.update(_empty_hrv_result())

    return record
