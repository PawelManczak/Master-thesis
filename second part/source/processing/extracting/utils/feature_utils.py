"""
Common functions for computing features from physiological signals.
Unified implementation for all datasets (K-EmoCon, CASE, CEAP).

PROCESSING METHODOLOGY (consistent with scientific literature):
===========================================================================

GOLDEN RULE: "Global Processing, Local Aggregation"
- Do not compute HR/metrics in small 5s windows (not enough data)
- Process the entire signal, create a continuous waveform, then aggregate

AGGREGATION OF SIGNALS INTO 30-SECOND WINDOWS:
------------------------------------------
| Signal | Problem with mean                 | Aggregating functions                 |
|--------|-----------------------------------|---------------------------------------|
| EDA    | Loss of stress peaks (SCR)        | SCL(tonic), SCR peaks/amp/AUC(phasic) |
| BVP    | Mean->0 (wave signal!)            | std (amplitude), spectral_power       |
| TEMP   | Changes slowly, OK                | mean, slope (trend)                   |
| ACC    | Mean=gravity/position             | mean (position), std (movement/tremor)|

30-SECOND WINDOWS (instead of 5s):
- HRV time-domain (SDNN, RMSSD) requires a minimum of 30s of data
  (Task Force ESC & NASPE, 1996)
- In a 5s window we have only 3-5 beats -> unstable metrics
- 30s gives ~30 beats -> reliable time-domain HRV

EDA PROCESSING (NeuroKit2):
1. nk.eda_clean() - low-pass filtering (Butterworth, cutoff 3 Hz / skipped for fs<7 Hz)
2. nk.eda_phasic(method='highpass') - Tonic(SCL)/Phasic(SCR) decomposition
3. nk.eda_findpeaks() - SCR peak detection with amplitudes
4. Feature extraction: SCL mean, SCR peaks/amplitude/AUC - Braithwaite et al. (2013)
5. Personal Min-Max normalization - Lykken & Venables (1971), Boucsein (2012)
6. Discretization: Low [0, 0.33), Medium [0.33, 0.67), High [0.67, 1.0]

References:
- Makowski, D. et al. (2021). NeuroKit2. Behavior Research Methods, 53, 1689-1696.
- Greco, A. et al. (2016). cvxEDA. IEEE TBME.
- Benedek, M. & Kaernbach, C. (2010). Continuous Decomposition Analysis. Psychophysiology.
- Braithwaite, J.J. et al. (2013). A Guide for Analysing EDA. Birmingham.
- Boucsein, W. (2012). Electrodermal Activity. Springer.
- Task Force of ESC and NASPE (1996). Heart rate variability: standards of measurement.
- Pham et al. (2021). Heart Rate Variability in Psychology. Frontiers in Psychology.
"""

import warnings
import numpy as np
import pandas as pd
from scipy import signal as scipy_signal
from scipy.stats import linregress
from scipy.interpolate import interp1d
from typing import Tuple

import neurokit2 as nk

from bvp_utils import compute_metrics_from_ibi, HRV_KEYS, _empty_hrv_result


# =============================================================================
# FUNCTIONS FOR EDA (Electrodermal Activity) - based on NeuroKit2
# =============================================================================
# NeuroKit2 Pipeline:
# 1. nk.eda_clean() - filter (Butterworth LP, cutoff 3 Hz; skipped for fs<7 Hz)
# 2. nk.eda_phasic(method='highpass') - Tonic(SCL)/Phasic(SCR) decomposition
# 3. nk.eda_findpeaks() - detection of SCR peaks with amplitudes (Neurokit/Kim2004/Gamboa)
# 4. Feature extraction in windows: from phasic (SCR peaks/amp/AUC) + tonic (SCL mean)
# 5. Intra-individual Min-Max normalization - Lykken & Venables (1971), Boucsein (2012)
# 6. Discretization into Low/Medium/High categories (terciles 0.33/0.67)
# =============================================================================


def preprocess_eda_global(eda_signal: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    GLOBAL PROCESSING: EDA preprocessing and decomposition using NeuroKit2.

    Pipeline:
    1. nk.eda_clean() - low-pass filtering (4th order Butterworth)
       - cutoff 3 Hz for fs >= 7 Hz (NeuroKit default)
       - skipped for fs < 7 Hz (e.g. Empatica E4 @ 4 Hz)
    2. nk.eda_phasic(method='highpass') - decomposition:
       - Tonic (SCL): Skin Conductance Level
       - Phasic (SCR): Skin Conductance Response
       - Biopac Acqknowledge method: highpass filter with 0.05 Hz cutoff

    Args:
        eda_signal: raw EDA signal (µS) from the entire recording
        fs: sampling frequency (Hz)

    Returns:
        tuple (cleaned, tonic, phasic):
            - cleaned: signal after filtering (nk.eda_clean)
            - tonic: tonic component (SCL)
            - phasic: phasic component (SCR)
    """
    eda_signal = np.array(eda_signal, dtype=float)

    if len(eda_signal) < 4:
        return eda_signal.copy(), eda_signal.copy(), np.zeros_like(eda_signal)

    sampling_rate = int(round(fs))

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Step 1: Signal cleaning (filtering)
            # NeuroKit automatically skips filtering for fs < 7 Hz
            cleaned = nk.eda_clean(eda_signal, sampling_rate=sampling_rate, method='neurokit')

            # Step 2: Tonic/Phasic decomposition
            # highpass = highpass filter with 0.05 Hz cutoff (Biopac Acqknowledge)
            # Fast and stable - NeuroKit2 default method
            decomposed = nk.eda_phasic(cleaned, sampling_rate=sampling_rate, method='highpass')

            tonic = decomposed['EDA_Tonic'].values
            phasic = decomposed['EDA_Phasic'].values

    except Exception:
        # Fallback: if NeuroKit fails
        cleaned = eda_signal.copy()
        tonic = eda_signal.copy()
        phasic = np.zeros_like(eda_signal)

    return cleaned, tonic, phasic


def compute_eda_features(
    values: np.ndarray,
    fs: float,
    tonic_values: np.ndarray = None,
    phasic_values: np.ndarray = None
) -> dict:
    """
    Compute EDA features in a time window using NeuroKit2.

    If tonic_values and phasic_values are provided (from preprocess_eda_global),
    uses nk.eda_findpeaks() for professional SCR detection.
    Otherwise fallback to simple analysis.

    Features from Phasic (SCR) component - Braithwaite et al. (2013):
    - eda_peaks: number of SCR peaks (most reliable arousal indicator)
    - eda_scr_mean_amp: mean amplitude of SCR peaks (nk.eda_findpeaks)
    - eda_scr_auc: AUC from positive part of phasic

    Features from Tonic (SCL) component:
    - eda_mean: mean SCL (Skin Conductance Level)

    General features:
    - eda_std: standard deviation of phasic (response variability)
    - eda_max: maximum of original signal in window

    Args:
        values: EDA values in window (cleaned or raw)
        fs: sampling frequency
        tonic_values: tonic component (SCL) in window (optional)
        phasic_values: phasic component (SCR) in window (optional)

    Returns:
        dict with: eda_mean, eda_std, eda_max, eda_peaks, eda_scr_mean_amp, eda_scr_auc
    """
    empty = {
        'eda_mean': np.nan, 'eda_std': np.nan, 'eda_max': np.nan,
        'eda_peaks': 0, 'eda_scr_mean_amp': np.nan, 'eda_scr_auc': np.nan
    }

    if len(values) == 0:
        return empty

    values = np.array(values, dtype=float)
    sampling_rate = int(round(fs))

    # =================================================================
    # Path with NeuroKit2 decomposition (scientifically correct)
    # =================================================================
    if tonic_values is not None and phasic_values is not None:
        tonic_values = np.array(tonic_values, dtype=float)
        phasic_values = np.array(phasic_values, dtype=float)

        # --- Tonic (SCL): mean baseline level ---
        eda_mean = float(np.mean(tonic_values))

        # --- Phasic (SCR): variability and peaks ---
        eda_std = float(np.std(phasic_values, ddof=1)) if len(phasic_values) > 1 else 0.0
        eda_max = float(np.max(values))

        # SCR peak detection using NeuroKit2
        peaks_count = 0
        scr_amplitudes = []

        if len(phasic_values) >= 4:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # nk.eda_findpeaks on phasic component
                    # amplitude_min=0.01 - minimum SCR threshold (Braithwaite et al. 2013)
                    peak_info = nk.eda_findpeaks(phasic_values, sampling_rate=sampling_rate,
                                                  method='neurokit', amplitude_min=0.01)
                    scr_peaks = peak_info.get('SCR_Peaks', [])
                    # SCR_Height = phasic amplitude at the peak location
                    scr_heights = peak_info.get('SCR_Height', [])

                    if len(scr_peaks) > 0:
                        peaks_count = len(scr_peaks)
                        valid_amps = [a for a in scr_heights if not np.isnan(a)]
                        if valid_amps:
                            scr_amplitudes = valid_amps
            except Exception:
                peaks_count = 0

        scr_mean_amp = float(np.mean(scr_amplitudes)) if scr_amplitudes else 0.0

        # AUC from positive part of phasic
        positive_phasic = np.maximum(phasic_values, 0.0)
        _trapz = getattr(np, 'trapezoid', np.trapz)  # numpy 2.0+ vs 1.x
        scr_auc = float(_trapz(positive_phasic, dx=1.0/fs)) if len(positive_phasic) > 1 else 0.0

        return {
            'eda_mean': eda_mean,
            'eda_std': eda_std,
            'eda_max': eda_max,
            'eda_peaks': peaks_count,
            'eda_scr_mean_amp': scr_mean_amp,
            'eda_scr_auc': scr_auc
        }

    # =================================================================
    # Fallback: simple analysis without decomposition
    # =================================================================
    peaks_count = 0
    if len(values) >= 4:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                peak_info = nk.eda_findpeaks(values, sampling_rate=sampling_rate,
                                              method='neurokit', amplitude_min=0.01)
                scr_peaks = peak_info.get('SCR_Peaks', [])
                peaks_count = len(scr_peaks) if len(scr_peaks) > 0 else 0
        except Exception:
            peaks_count = 0

    return {
        'eda_mean': float(np.mean(values)),
        'eda_std': float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        'eda_max': float(np.max(values)),
        'eda_peaks': peaks_count,
        'eda_scr_mean_amp': np.nan,
        'eda_scr_auc': np.nan
    }


def normalize_eda_features_subject(df: pd.DataFrame) -> pd.DataFrame:
    """
    Intra-individual (Min-Max) normalization of EDA features.

    Critical step for categorization! EDA ranges differ drastically
    between people (skin thickness, number of glands). Comparing raw
    values (e.g., "5 µS is High Arousal") is a methodological error.

    Formula: X_norm = (X - X_min) / (X_max - X_min)
    where min/max are extremes for a given subject across the entire recording.

    References:
    - Lykken & Venables (1971): Range Correction for between-subject comparisons
    - Boucsein (2012): necessity of range correction for reliable classification

    Normalized columns:
    - eda_mean (SCL) -> eda_mean_norm
    - eda_scr_mean_amp -> eda_scr_mean_amp_norm
    - eda_scr_auc -> eda_scr_auc_norm
    - eda_max -> eda_max_norm

    Args:
        df: DataFrame with data for one participant (must contain eda_* columns)

    Returns:
        DataFrame with added *_norm columns (range [0, 1])
    """
    df = df.copy()

    eda_columns = ['eda_mean', 'eda_std', 'eda_max', 'eda_peaks', 'eda_scr_mean_amp', 'eda_scr_auc']

    for col in eda_columns:
        if col not in df.columns:
            continue

        values = df[col].copy()
        valid = values.dropna()

        if len(valid) < 2:
            df[f'{col}_norm'] = np.nan
            continue

        vmin = valid.min()
        vmax = valid.max()

        if vmax - vmin < 1e-10:
            # No variability - set to 0.5
            df[f'{col}_norm'] = 0.5
        else:
            df[f'{col}_norm'] = (values - vmin) / (vmax - vmin)

    return df


# =============================================================================
# FUNCTIONS FOR BVP (Blood Volume Pulse)
# =============================================================================

def compute_bvp_features(values: np.ndarray, fs: float) -> dict:
    """
    Compute BVP features in a time window.

    NOTE: BVP is a wave signal - mean tends to 0!
    DO NOT use mean! Use std (amplitude) and spectral power.

    Args:
        values: BVP values in window
        fs: sampling frequency

    Returns:
        dict with: bvp_std (amplitude), bvp_peak_to_peak, bvp_spectral_power
    """
    if len(values) == 0:
        return {
            'bvp_std': np.nan,
            'bvp_peak_to_peak': np.nan,
            'bvp_spectral_power': np.nan
        }

    values = np.array(values)

    # Std = signal amplitude measure
    bvp_std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0

    # Peak-to-peak amplitude
    bvp_p2p = float(np.max(values) - np.min(values))

    # Spectral power (signal energy)
    spectral_power = np.nan
    if len(values) >= 8:
        try:
            # Compute PSD using Welch's method
            freqs, psd = scipy_signal.welch(values, fs=fs, nperseg=min(len(values), 64))
            # Power in cardiac band (0.5-4 Hz)
            cardiac_band = (freqs >= 0.5) & (freqs <= 4.0)
            if np.any(cardiac_band):
                _trapz = getattr(np, 'trapezoid', np.trapz)  # numpy 2.0+ vs 1.x
                spectral_power = float(_trapz(psd[cardiac_band], freqs[cardiac_band]))
        except:
            pass

    return {
        'bvp_std': bvp_std,
        'bvp_peak_to_peak': bvp_p2p,
        'bvp_spectral_power': spectral_power
    }


# =============================================================================
# FUNCTIONS FOR TEMPERATURE
# =============================================================================

def compute_temp_features(values: np.ndarray, timestamps: np.ndarray = None) -> dict:
    """
    Compute temperature features in a time window.

    Temperature changes slowly - mean is OK.
    Additionally: slope (trend) - whether it goes up or down.

    Args:
        values: temperature values in window
        timestamps: timestamps (optional)

    Returns:
        dict with: temp_mean, temp_slope
    """
    if len(values) == 0:
        return {'temp_mean': np.nan, 'temp_slope': np.nan}

    values = np.array(values)
    temp_mean = float(np.mean(values))

    # Slope (linear trend)
    temp_slope = 0.0
    if len(values) >= 2:
        try:
            if timestamps is not None and len(timestamps) == len(values):
                x = np.array(timestamps)
            else:
                x = np.arange(len(values))
            slope, _, _, _, _ = linregress(x, values)
            temp_slope = float(slope)
        except:
            temp_slope = 0.0

    return {
        'temp_mean': temp_mean,
        'temp_slope': temp_slope
    }


# =============================================================================
# FUNCTIONS FOR ACCELEROMETER
# =============================================================================

def compute_acc_features(acc_x: np.ndarray, acc_y: np.ndarray, acc_z: np.ndarray) -> dict:
    """
    Compute accelerometer features in a time window.

    Mean = body position (gravity)
    Std = movement intensity/tremor

    Args:
        acc_x, acc_y, acc_z: values for each axis

    Returns:
        dict with: mean and std for each axis + magnitude
    """
    result = {}

    for axis, values in [('x', acc_x), ('y', acc_y), ('z', acc_z)]:
        if len(values) == 0:
            result[f'acc_{axis}_mean'] = np.nan
            result[f'acc_{axis}_std'] = np.nan
        else:
            values = np.array(values)
            result[f'acc_{axis}_mean'] = float(np.mean(values))
            result[f'acc_{axis}_std'] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0

    # Magnitude
    if len(acc_x) > 0 and len(acc_y) > 0 and len(acc_z) > 0:
        magnitude = np.sqrt(np.array(acc_x)**2 + np.array(acc_y)**2 + np.array(acc_z)**2)
        result['acc_magnitude_mean'] = float(np.mean(magnitude))
        result['acc_magnitude_std'] = float(np.std(magnitude, ddof=1)) if len(magnitude) > 1 else 0.0
    else:
        result['acc_magnitude_mean'] = np.nan
        result['acc_magnitude_std'] = np.nan

    return result


# =============================================================================
# FUNCTIONS FOR HR (Heart Rate) - GLOBAL PROCESSING
# =============================================================================

def compute_global_hr_from_ibi(
    ibi_timestamps: np.ndarray,
    ibi_values: np.ndarray,
    max_time: float,
    ibi_unit: str = 'ms'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    GLOBAL PROCESSING: Compute continuous HR waveform from the entire IBI recording.

    Golden rule: "Don't compute HR in a window, average HR in a window"

    Pipeline:
    1. Extract all IBIs from the entire recording
    2. Compute instantaneous HR for each IBI
    3. Interpolate to a uniform time grid (1 Hz)

    Args:
        ibi_timestamps: IBI timestamps
        ibi_values: IBI values (in ms or s, depending on ibi_unit)
        max_time: maximum recording time (in the same unit as timestamps)
        ibi_unit: 'ms' or 's' - IBI unit

    Returns:
        tuple (time_grid, hr_timeseries) - time and HR in BPM
    """
    if len(ibi_timestamps) < 2 or max_time <= 0:
        return np.array([]), np.array([])

    ibi_timestamps = np.array(ibi_timestamps)
    ibi_values = np.array(ibi_values)

    # Convert to ms if in seconds
    if ibi_unit == 's':
        ibi_values_ms = ibi_values * 1000
    else:
        ibi_values_ms = ibi_values

    # Filter invalid IBIs (300-2000 ms = 30-200 BPM)
    valid_mask = (ibi_values_ms > 300) & (ibi_values_ms < 2000)
    timestamps = ibi_timestamps[valid_mask]
    ibi_ms = ibi_values_ms[valid_mask]

    if len(timestamps) < 2:
        return np.array([]), np.array([])

    # Compute instantaneous HR (BPM)
    hr_values = 60000 / ibi_ms

    # Create a uniform time grid (1 Hz)
    # Detect time unit (ms vs s) based on max_time
    if max_time > 10000:  # Probably ms
        step = 1000  # 1 second in ms
    else:  # Probably s
        step = 1  # 1 second

    # Time grid optimization:
    start_time = 0
    if max_time > 100000:
        # Absolute timestamps mode (large values)
        min_ts = timestamps.min()

        if min_ts > 100000 and max_time > min_ts:
            start_time = np.floor(min_ts / step) * step
        elif min_ts > 100000 and max_time <= min_ts:
            # Data starts after max_time -> no data in the window
            start_time = max_time
        # else: min_ts is small? (inconsistent data) -> leave start_time=0 if max_time is large,
        # but we assume unit consistency.

        # Safety check: if start_time is still 0 with a large max_time
        if start_time == 0:
             # Set to something close to max_time to avoid memory allocation
             start_time = max_time - step
    else:
        # Relative mode (small values)
        start_time = 0

    if start_time >= max_time:
         return np.array([]), np.array([])

    time_grid = np.arange(start_time, max_time, step)

    if len(time_grid) == 0:
        return np.array([]), np.array([])

    # Interpolate HR to the time grid
    try:
        f_interp = interp1d(timestamps, hr_values, kind='linear',
                           bounds_error=False, fill_value=np.nan)
        hr_timeseries = f_interp(time_grid)
    except:
        hr_timeseries = np.full(len(time_grid), np.nan)

    return time_grid, hr_timeseries


def compute_global_hr_from_ecg(
    ecg_data: np.ndarray,
    time_data: np.ndarray,
    fs: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    GLOBAL PROCESSING: Compute continuous HR waveform from the entire ECG signal.

    Pipeline:
    1. Detect R peaks across the entire recording
    2. Compute IBI (RR intervals)
    3. Compute instantaneous HR for each IBI
    4. Interpolate to a uniform time grid (1 Hz)

    Args:
        ecg_data: ECG signal
        time_data: time vector
        fs: ECG sampling frequency

    Returns:
        tuple (time_grid, hr_timeseries, r_peaks) - time, HR in BPM, R peak indices
    """
    # Detect R peaks
    r_peaks = detect_r_peaks_ecg(ecg_data, fs)

    if len(r_peaks) < 2:
        return np.array([]), np.array([]), r_peaks

    # Compute RR intervals (IBI) in ms
    rr_intervals = np.diff(r_peaks) / fs * 1000

    # Timestamps for RR intervals (mean between consecutive peaks)
    rr_timestamps = (time_data[r_peaks[:-1]] + time_data[r_peaks[1:]]) / 2

    # Filter invalid RRs (300-2000 ms = 30-200 BPM)
    valid_mask = (rr_intervals > 300) & (rr_intervals < 2000)
    rr_intervals = rr_intervals[valid_mask]
    rr_timestamps = rr_timestamps[valid_mask]

    if len(rr_intervals) < 2:
        return np.array([]), np.array([]), r_peaks

    # Compute instantaneous HR (BPM)
    hr_values = 60000 / rr_intervals

    # Create a uniform time grid (1 Hz = spacing in time_data unit)
    max_time = time_data.max()
    min_time = time_data.min()

    # Detect time unit
    if max_time > 10000:  # ms
        step = 1000
    else:  # s
        step = 1

    time_grid = np.arange(min_time, max_time, step)

    if len(time_grid) == 0:
        return np.array([]), np.array([]), r_peaks

    # Interpolate HR to the time grid
    try:
        f_interp = interp1d(rr_timestamps, hr_values, kind='linear',
                           bounds_error=False, fill_value=np.nan)
        hr_timeseries = f_interp(time_grid)
    except:
        hr_timeseries = np.full(len(time_grid), np.nan)

    return time_grid, hr_timeseries, r_peaks


def detect_r_peaks_ecg(ecg_data: np.ndarray, fs: float) -> np.ndarray:
    """
    Detect R peaks in ECG signal (simplified Pan-Tompkins).

    Args:
        ecg_data: ECG signal
        fs: sampling frequency (Hz)

    Returns:
        array of R peak indices
    """
    if len(ecg_data) < fs * 2:
        return np.array([])

    try:
        # ECG bandpass filter (5-15 Hz - QRS band)
        nyquist = fs / 2
        low = 5.0 / nyquist
        high = min(15.0 / nyquist, 0.99)

        b, a = scipy_signal.butter(2, [low, high], btype='band')
        filtered_ecg = scipy_signal.filtfilt(b, a, ecg_data)

        # Differentiation
        diff_ecg = np.diff(filtered_ecg)

        # Squaring
        squared = diff_ecg ** 2

        # Moving window integration (150 ms)
        window_size = int(fs * 0.15)
        if window_size < 1:
            window_size = 1
        integrated = np.convolve(squared, np.ones(window_size)/window_size, mode='same')

        # Peak detection
        min_distance = int(fs * 0.3)  # Minimum distance 300ms (max 200 BPM)
        threshold = np.mean(integrated) + 0.5 * np.std(integrated)
        peaks, _ = scipy_signal.find_peaks(integrated, distance=min_distance, height=threshold)

        return peaks

    except Exception:
        return np.array([])


def compute_hr_window_features(
    time_grid: np.ndarray,
    hr_timeseries: np.ndarray,
    window_start: float,
    window_end: float
) -> dict:
    """
    LOCAL AGGREGATION: Compute HR features in a time window from the global waveform.

    Args:
        time_grid: time grid (sorted in ascending order)
        hr_timeseries: continuous HR waveform
        window_start: window start
        window_end: window end

    Returns:
        dict with: hr_mean, hr_std
    """
    if len(hr_timeseries) == 0 or len(time_grid) == 0:
        return {'hr_mean': np.nan, 'hr_std': np.nan}

    # Find HR values in window (O(log n) instead of O(n))
    i0 = np.searchsorted(time_grid, window_start, side='left')
    i1 = np.searchsorted(time_grid, window_end, side='left')
    hr_window = hr_timeseries[i0:i1]

    # Remove NaN
    hr_valid = hr_window[~np.isnan(hr_window)]

    if len(hr_valid) == 0:
        return {'hr_mean': np.nan, 'hr_std': np.nan}

    return {
        'hr_mean': float(np.mean(hr_valid)),
        'hr_std': float(np.std(hr_valid, ddof=1)) if len(hr_valid) > 1 else 0.0
    }


# =============================================================================
# FUNCTIONS FOR HRV (Heart Rate Variability)
# =============================================================================

def compute_hrv_window_features(
    ibi_timestamps: np.ndarray,
    ibi_values: np.ndarray,
    window_start: float,
    window_end: float,
    ibi_unit: str = 'ms'
) -> dict:
    """
    Compute HRV metrics from IBI in a time window using NeuroKit2.

    Returns full set of metrics from nk.hrv_time() and nk.hrv_frequency():
    - Time domain: SDNN, RMSSD, pNN50, pNN20, MeanNN, MedianNN, CVNN, CVSD, ...
    - Frequency domain: LF, HF, VLF, LF/HF, LFn, HFn, LnHF

    Args:
        ibi_timestamps: IBI timestamps
        ibi_values: IBI values
        window_start: window start
        window_end: window end
        ibi_unit: 'ms' or 's'

    Returns:
        dict with HRV metrics (see bvp_utils.HRV_KEYS)
    """
    if len(ibi_timestamps) < 3:
        return _empty_hrv_result()

    ibi_timestamps = np.array(ibi_timestamps)
    ibi_values = np.array(ibi_values)

    # Get IBI in window (O(log n))
    i0 = np.searchsorted(ibi_timestamps, window_start, side='left')
    i1 = np.searchsorted(ibi_timestamps, window_end, side='left')
    window_ibi = ibi_values[i0:i1]

    if len(window_ibi) < 3:
        return _empty_hrv_result()

    return compute_metrics_from_ibi(window_ibi, ibi_unit=ibi_unit)


def compute_hrv_from_ecg_window(
    r_peaks: np.ndarray,
    time_data: np.ndarray,
    fs: float,
    window_start: float,
    window_end: float
) -> dict:
    """
    Compute HRV metrics from ECG R peaks in a time window using NeuroKit2.

    Returns full set of metrics from nk.hrv_time() and nk.hrv_frequency().

    Args:
        r_peaks: R peak indices
        time_data: time vector
        fs: sampling frequency
        window_start: window start
        window_end: window end

    Returns:
        dict with HRV metrics (see bvp_utils.HRV_KEYS)
    """
    if len(r_peaks) < 3:
        return _empty_hrv_result()

    # Find R peaks in window (O(log n))
    peak_times = time_data[r_peaks]
    i0 = np.searchsorted(peak_times, window_start, side='left')
    i1 = np.searchsorted(peak_times, window_end, side='left')
    window_peaks = r_peaks[i0:i1]

    if len(window_peaks) < 3:
        return _empty_hrv_result()

    # Compute RR intervals in window (ms)
    rr_intervals = np.diff(window_peaks) / fs * 1000

    # Filter invalid
    valid_rr = rr_intervals[(rr_intervals > 300) & (rr_intervals < 2000)]

    if len(valid_rr) < 3:
        return _empty_hrv_result()

    return compute_metrics_from_ibi(valid_rr, ibi_unit='ms')

