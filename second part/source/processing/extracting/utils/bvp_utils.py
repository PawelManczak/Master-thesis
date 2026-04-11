"""
Common functions for computing metrics from BVP (Blood Volume Pulse) and HRV.
Unified implementation for all datasets (K-EmoCon, CASE, CEAP).

HRV computations based on NeuroKit2 (Makowski et al. 2021):
- nk.hrv_time()       -> SDNN, RMSSD, pNN50, MeanNN, MedianNN, CVNN, CVSD, …
- nk.hrv_frequency()  -> LF, HF, VLF, LF/HF, LFn, HFn, …
- nk.intervals_to_peaks()  -> IBI to peak indices conversion

References:
- Makowski, D. et al. (2021). NeuroKit2. Behavior Research Methods, 53, 1689-1696.
- Pham, T. et al. (2021). HRV in Psychology: A Review. Sensors, 21(12), 3998.
- Shaffer, F. & Ginsberg, J.P. (2017). HRV metrics and norms. Frontiers Public Health, 5, 258.
- Task Force of ESC and NASPE (1996). HRV: standards of measurement.
"""

import warnings
import numpy as np
from scipy import signal as scipy_signal
from scipy.interpolate import interp1d
from typing import Dict, Optional

import neurokit2 as nk


def detect_peaks_bvp(bvp_data: np.ndarray, fs: float) -> np.ndarray:
    """
    Detect peaks in BVP (photoplethysmography) signal.

    Args:
        bvp_data: BVP signal
        fs: sampling frequency (Hz)

    Returns:
        array of peak indices
    """
    if len(bvp_data) < fs * 0.5:
        return np.array([])

    try:
        nyquist = fs / 2
        low = max(0.5 / nyquist, 0.01)
        high = min(8.0 / nyquist, 0.99)

        b, a = scipy_signal.butter(2, [low, high], btype='band')
        filtered_bvp = scipy_signal.filtfilt(b, a, bvp_data)

        min_distance = int(fs * 0.3)
        threshold = np.mean(filtered_bvp) + 0.3 * np.std(filtered_bvp)

        peaks, _ = scipy_signal.find_peaks(
            filtered_bvp,
            distance=min_distance,
            height=threshold,
            prominence=0.1 * np.std(filtered_bvp)
        )
        return peaks
    except Exception:
        return np.array([])


def compute_ibi_from_bvp(bvp_data: np.ndarray, fs: float) -> np.ndarray:
    """
    Compute IBI (Inter-Beat Intervals) from BVP signal.

    Args:
        bvp_data: BVP signal
        fs: sampling frequency (Hz)

    Returns:
        array of IBIs in milliseconds
    """
    peaks = detect_peaks_bvp(bvp_data, fs)

    if len(peaks) < 2:
        return np.array([])

    ibi_samples = np.diff(peaks)
    ibi_ms = (ibi_samples / fs) * 1000

    valid_ibi = ibi_ms[(ibi_ms > 300) & (ibi_ms < 2000)]
    return valid_ibi


def compute_hr_from_bvp(bvp_data: np.ndarray, fs: float) -> float:
    """
    Compute mean Heart Rate (HR) from BVP signal.

    Args:
        bvp_data: BVP signal
        fs: sampling frequency (Hz)

    Returns:
        mean heart rate in BPM or NaN
    """
    ibi_ms = compute_ibi_from_bvp(bvp_data, fs)

    if len(ibi_ms) == 0:
        return np.nan

    return 60000 / np.mean(ibi_ms)


# =============================================================================
# MAIN FUNCTION: Compute HRV metrics from IBI using NeuroKit2
# =============================================================================

# Define full set of HRV keys (time + frequency domain)
HRV_KEYS = {
    # === Time Domain (nk.hrv_time) ===
    'hrv_mean_nn': 'MeanNN',    # MeanNN: mean NN interval (ms)
    'hrv_sdnn': 'SDNN',         # SDNN: standard deviation of NN (ms) - general variability
    'hrv_rmssd': 'RMSSD',       # RMSSD: sqrt(mean(diff²)) - short-term variability (parasympathetic)
    'hrv_sdsd': 'SDSD',         # SDSD: std(diff) - similar to RMSSD
    'hrv_cvnn': 'CVNN',         # CVNN: SDNN/MeanNN - coefficient of variation
    'hrv_cvsd': 'CVSD',         # CVSD: RMSSD/MeanNN - normalized RMSSD
    'hrv_median_nn': 'MedianNN',  # MedianNN: median of NN
    'hrv_mad_nn': 'MadNN',      # MadNN: median absolute deviation NN
    'hrv_mcvnn': 'MCVNN',       # MCVNN: MadNN/MedianNN
    'hrv_iqrnn': 'IQRNN',       # IQRNN: interquartile range of NN
    'hrv_pnn50': 'pNN50',       # pNN50: % of differences > 50ms
    'hrv_pnn20': 'pNN20',       # pNN20: % of differences > 20ms
    'hrv_hti': 'HTI',           # HTI: HRV triangular index
    # === Frequency Domain (nk.hrv_frequency) ===
    'hrv_lf': 'LF Power',       # LF Power: 0.04-0.15 Hz (sympathetic + parasympathetic)
    'hrv_hf': 'HF Power',       # HF Power: 0.15-0.4 Hz (parasympathetic / vagal)
    'hrv_vlf': 'VLF Power',      # VLF Power: 0.0033-0.04 Hz
    'hrv_lf_hf_ratio': 'LF/HF Ratio', # LF/HF Ratio
    'hrv_lfn': 'LFn',           # LFn: normalized LF
    'hrv_hfn': 'HFn',           # HFn: normalized HF
    'hrv_lnhf': 'LnHF',         # LnHF: log(HF)
}


def _empty_hrv_result() -> Dict:
    """Return a dict with NaN for all HRV keys."""
    return {k: np.nan for k in HRV_KEYS}


def _safe_float(row, key: str) -> float:
    """Safely extract float from a DataFrame row."""
    try:
        if key in row.index:
            val = row[key]
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return np.nan
            return float(val)
    except Exception:
        pass
    return np.nan


def compute_metrics_from_ibi(ibi_values: np.ndarray, ibi_unit: str = 'ms') -> Dict:
    """
    Compute HRV metrics from IBI intervals using NeuroKit2.

    Pipeline:
    1. Validate and convert IBI to ms
    2. nk.intervals_to_peaks() - conversion IBI -> peak indices
    3. nk.hrv_time()       - time domain metrics (SDNN, RMSSD, pNN50, ...)
    4. nk.hrv_frequency()  - frequency domain metrics (LF, HF, LF/HF, ...)

    Args:
        ibi_values: array of IBI values
        ibi_unit: IBI unit - 'ms' (milliseconds) or 's' (seconds)

    Returns:
        dict with HRV metrics (see HRV_KEYS)
    """
    result = _empty_hrv_result()

    if len(ibi_values) < 3:
        return result

    # Convert to ms
    if ibi_unit == 's':
        ibi_ms = np.array(ibi_values, dtype=float) * 1000
    else:
        ibi_ms = np.array(ibi_values, dtype=float)

    # Filter invalid IBIs (300-2000 ms = 30-200 BPM)
    valid_ibi = ibi_ms[(ibi_ms > 300) & (ibi_ms < 2000)]

    if len(valid_ibi) < 3:
        return result

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Convert IBI -> peak indices (sampling_rate=1000)
            peaks = nk.intervals_to_peaks(valid_ibi, sampling_rate=1000)

            if len(peaks) < 3:
                return result

            # ===== TIME DOMAIN =====
            try:
                hrv_time_df = nk.hrv_time(peaks, sampling_rate=1000)
                t = hrv_time_df.iloc[0]
                result['hrv_mean_nn'] = _safe_float(t, 'HRV_MeanNN')
                result['hrv_sdnn'] = _safe_float(t, 'HRV_SDNN')
                result['hrv_rmssd'] = _safe_float(t, 'HRV_RMSSD')
                result['hrv_sdsd'] = _safe_float(t, 'HRV_SDSD')
                result['hrv_cvnn'] = _safe_float(t, 'HRV_CVNN')
                result['hrv_cvsd'] = _safe_float(t, 'HRV_CVSD')
                result['hrv_median_nn'] = _safe_float(t, 'HRV_MedianNN')
                result['hrv_mad_nn'] = _safe_float(t, 'HRV_MadNN')
                result['hrv_mcvnn'] = _safe_float(t, 'HRV_MCVNN')
                result['hrv_iqrnn'] = _safe_float(t, 'HRV_IQRNN')
                result['hrv_pnn50'] = _safe_float(t, 'HRV_pNN50')
                result['hrv_pnn20'] = _safe_float(t, 'HRV_pNN20')
                result['hrv_hti'] = _safe_float(t, 'HRV_HTI')
            except Exception:
                pass

            # ===== FREQUENCY DOMAIN =====
            if len(valid_ibi) >= 10:
                try:
                    hrv_freq_df = nk.hrv_frequency(
                        peaks, sampling_rate=1000,
                        psd_method='welch', silent=True, normalize=True
                    )
                    f = hrv_freq_df.iloc[0]
                    result['hrv_lf'] = _safe_float(f, 'HRV_LF')
                    result['hrv_hf'] = _safe_float(f, 'HRV_HF')
                    result['hrv_vlf'] = _safe_float(f, 'HRV_VLF')
                    result['hrv_lf_hf_ratio'] = _safe_float(f, 'HRV_LFHF')
                    result['hrv_lfn'] = _safe_float(f, 'HRV_LFn')
                    result['hrv_hfn'] = _safe_float(f, 'HRV_HFn')
                    result['hrv_lnhf'] = _safe_float(f, 'HRV_LnHF')
                except Exception:
                    pass

    except Exception:
        pass

    return result


def compute_metrics_from_bvp(bvp_data: np.ndarray, fs: float) -> Dict:
    """
    Compute HRV metrics from BVP signal.

    Args:
        bvp_data: BVP (Blood Volume Pulse) signal
        fs: sampling frequency (Hz)

    Returns:
        dict with HRV metrics
    """
    if len(bvp_data) < fs * 2:
        return _empty_hrv_result()

    try:
        ibi_ms = compute_ibi_from_bvp(bvp_data, fs)
        if len(ibi_ms) < 3:
            return _empty_hrv_result()
        return compute_metrics_from_ibi(ibi_ms, ibi_unit='ms')
    except Exception:
        return _empty_hrv_result()


def compute_metrics_from_ecg(ecg_data: np.ndarray, fs: float) -> Dict:
    """
    Compute HRV metrics from ECG signal.

    Args:
        ecg_data: ECG signal
        fs: sampling frequency (Hz)

    Returns:
        dict with HRV metrics
    """
    if len(ecg_data) < fs * 2:
        return _empty_hrv_result()

    try:
        rr_intervals = _detect_rr_from_ecg(ecg_data, fs)
        if rr_intervals is None or len(rr_intervals) < 3:
            return _empty_hrv_result()
        return compute_metrics_from_ibi(rr_intervals, ibi_unit='ms')
    except Exception:
        return _empty_hrv_result()


def _detect_rr_from_ecg(ecg_data: np.ndarray, fs: float) -> Optional[np.ndarray]:
    """
    Detect RR intervals from ECG signal (simplified Pan-Tompkins).

    Args:
        ecg_data: ECG signal
        fs: sampling frequency (Hz)

    Returns:
        array of RR intervals in ms or None
    """
    try:
        nyquist = fs / 2
        low = 0.5 / nyquist
        high = min(40.0 / nyquist, 0.99)

        b, a = scipy_signal.butter(2, [low, high], btype='band')
        filtered_ecg = scipy_signal.filtfilt(b, a, ecg_data)

        diff_ecg = np.diff(filtered_ecg)
        squared = diff_ecg ** 2

        window_size = int(fs * 0.15)
        if window_size < 1:
            window_size = 1
        integrated = np.convolve(squared, np.ones(window_size)/window_size, mode='same')

        min_distance = int(fs * 0.3)
        threshold = np.mean(integrated) + 0.5 * np.std(integrated)
        peaks, _ = scipy_signal.find_peaks(integrated, distance=min_distance, height=threshold)

        if len(peaks) < 3:
            return None

        rr_intervals = np.diff(peaks) / fs * 1000
        return rr_intervals
    except Exception:
        return None


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def antialiasing_filter(data: np.ndarray, original_fs: float, target_fs: float) -> np.ndarray:
    """
    Anti-aliasing (low-pass) filter before decimation.

    Args:
        data: input data
        original_fs: original sampling frequency (Hz)
        target_fs: target sampling frequency (Hz)

    Returns:
        filtered data
    """
    if len(data) < 10:
        return data

    nyquist = original_fs / 2
    cutoff = min(target_fs * 0.4, nyquist * 0.9)
    normalized_cutoff = cutoff / nyquist

    if normalized_cutoff >= 1.0 or normalized_cutoff <= 0:
        return data

    try:
        b, a = scipy_signal.butter(4, normalized_cutoff, btype='low')
        filtered = scipy_signal.filtfilt(b, a, data)
        return filtered
    except Exception:
        return data


def compute_window_stats(values: np.ndarray) -> Dict:
    """
    Compute statistics for a time window.

    Args:
        values: values in the window

    Returns:
        dict with 'mean' and 'var'
    """
    if len(values) == 0:
        return {'mean': np.nan, 'var': np.nan}

    return {
        'mean': np.mean(values),
        'var': np.var(values, ddof=1) if len(values) > 1 else 0.0
    }
