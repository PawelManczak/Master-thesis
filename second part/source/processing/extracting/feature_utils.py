"""
Wspólne funkcje do obliczania cech (features) z sygnałów fizjologicznych.
Ujednolicona implementacja dla wszystkich datasetów (K-EmoCon, CASE, CEAP).

METODOLOGIA PRZETWARZANIA (zgodna z publikacjami naukowymi):
===========================================================================

ZŁOTA ZASADA: "Global Processing, Local Aggregation"
- Nie liczymy HR/metryk w małych oknach 5s (za mało danych)
- Przetwarzamy cały sygnał, tworzymy ciągły przebieg, potem agregujemy

AGREGACJA SYGNAŁÓW DO OKIEN 5-SEKUNDOWYCH:
------------------------------------------
| Sygnał | Problem ze średnią              | Funkcje agregujące                    |
|--------|----------------------------------|---------------------------------------|
| EDA    | Tracisz piki stresu (SCR)       | mean, std, max, peaks_count           |
| BVP    | Średnia→0 (sygnał falowy!)      | std (amplituda), spectral_power       |
| TEMP   | Zmienia się wolno, OK           | mean, slope (trend)                   |
| ACC    | Średnia=grawitacja/pozycja      | mean (pozycja), std (ruch/drżenie)    |

Referencje:
- Task Force of ESC and NASPE (1996). Heart rate variability: standards of measurement
- Boucsein, W. (2012). Electrodermal Activity. Springer.
- Pham et al. (2021). Heart Rate Variability in Psychology. Frontiers in Psychology.
"""

import numpy as np
import pandas as pd
from scipy import signal as scipy_signal
from scipy.stats import linregress
from scipy.interpolate import interp1d
from typing import Dict, Tuple, Optional, Union

from bvp_utils import compute_metrics_from_ibi


# =============================================================================
# FUNKCJE DLA EDA (Electrodermal Activity)
# =============================================================================

def compute_eda_features(values: np.ndarray, fs: float) -> dict:
    """
    Oblicz cechy EDA w oknie czasowym.

    EDA (Electrodermal Activity) - tracisz piki stresu przy zwykłej średniej!

    Args:
        values: wartości EDA w oknie
        fs: częstotliwość próbkowania

    Returns:
        dict z: eda_mean, eda_std, eda_max, eda_peaks
    """
    if len(values) == 0:
        return {'eda_mean': np.nan, 'eda_std': np.nan, 'eda_max': np.nan, 'eda_peaks': 0}

    values = np.array(values)

    # Detekcja pików SCR (Skin Conductance Response)
    peaks_count = 0
    if len(values) >= 3:
        try:
            # Piki muszą być wyższe niż 0.01 µS od baseline
            threshold = np.mean(values) + 0.01
            min_distance = max(1, int(fs * 0.5))  # Minimum 0.5s między pikami
            peaks, _ = scipy_signal.find_peaks(values, height=threshold, distance=min_distance)
            peaks_count = len(peaks)
        except:
            peaks_count = 0

    return {
        'eda_mean': float(np.mean(values)),
        'eda_std': float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        'eda_max': float(np.max(values)),
        'eda_peaks': peaks_count
    }


# =============================================================================
# FUNKCJE DLA BVP (Blood Volume Pulse)
# =============================================================================

def compute_bvp_features(values: np.ndarray, fs: float) -> dict:
    """
    Oblicz cechy BVP w oknie czasowym.

    UWAGA: BVP to sygnał falowy - średnia dąży do 0!
    NIE używaj średniej! Używaj std (amplituda) i spectral power.

    Args:
        values: wartości BVP w oknie
        fs: częstotliwość próbkowania

    Returns:
        dict z: bvp_std (amplituda), bvp_peak_to_peak, bvp_spectral_power
    """
    if len(values) == 0:
        return {
            'bvp_std': np.nan,
            'bvp_peak_to_peak': np.nan,
            'bvp_spectral_power': np.nan
        }

    values = np.array(values)

    # Std = miara amplitudy sygnału
    bvp_std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0

    # Peak-to-peak amplitude
    bvp_p2p = float(np.max(values) - np.min(values))

    # Spectral power (energia sygnału)
    spectral_power = np.nan
    if len(values) >= 8:
        try:
            # Oblicz PSD metodą Welcha
            freqs, psd = scipy_signal.welch(values, fs=fs, nperseg=min(len(values), 64))
            # Moc w paśmie sercowym (0.5-4 Hz)
            cardiac_band = (freqs >= 0.5) & (freqs <= 4.0)
            if np.any(cardiac_band):
                spectral_power = float(np.trapezoid(psd[cardiac_band], freqs[cardiac_band]))
        except:
            pass

    return {
        'bvp_std': bvp_std,
        'bvp_peak_to_peak': bvp_p2p,
        'bvp_spectral_power': spectral_power
    }


# =============================================================================
# FUNKCJE DLA TEMPERATURY
# =============================================================================

def compute_temp_features(values: np.ndarray, timestamps: np.ndarray = None) -> dict:
    """
    Oblicz cechy temperatury w oknie czasowym.

    Temperatura zmienia się wolno - średnia jest OK.
    Dodatkowo: slope (trend) - czy rośnie czy spada.

    Args:
        values: wartości temperatury w oknie
        timestamps: znaczniki czasowe (opcjonalne)

    Returns:
        dict z: temp_mean, temp_slope
    """
    if len(values) == 0:
        return {'temp_mean': np.nan, 'temp_slope': np.nan}

    values = np.array(values)
    temp_mean = float(np.mean(values))

    # Slope (trend liniowy)
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
# FUNKCJE DLA AKCELEROMETRU
# =============================================================================

def compute_acc_features(acc_x: np.ndarray, acc_y: np.ndarray, acc_z: np.ndarray) -> dict:
    """
    Oblicz cechy akcelerometru w oknie czasowym.

    Mean = pozycja ciała (grawitacja)
    Std = intensywność ruchu/drżenie

    Args:
        acc_x, acc_y, acc_z: wartości dla każdej osi

    Returns:
        dict z: mean i std dla każdej osi + magnitude
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
# FUNKCJE DLA HR (Heart Rate) - GLOBAL PROCESSING
# =============================================================================

def compute_global_hr_from_ibi(
    ibi_timestamps: np.ndarray,
    ibi_values: np.ndarray,
    max_time: float,
    ibi_unit: str = 'ms'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    GLOBAL PROCESSING: Oblicz ciągły przebieg HR z całego nagrania IBI.

    Złota zasada: "Nie licz HR w oknie, uśredniaj HR w oknie"

    Pipeline:
    1. Pobierz wszystkie IBI z całego nagrania
    2. Oblicz chwilowe HR dla każdego IBI
    3. Interpoluj do równomiernej siatki czasowej (1 Hz)

    Args:
        ibi_timestamps: znaczniki czasowe IBI
        ibi_values: wartości IBI (w ms lub s, zależnie od ibi_unit)
        max_time: maksymalny czas nagrania (w tej samej jednostce co timestamps)
        ibi_unit: 'ms' lub 's' - jednostka IBI

    Returns:
        tuple (time_grid, hr_timeseries) - czas i HR w BPM
    """
    if len(ibi_timestamps) < 2 or max_time <= 0:
        return np.array([]), np.array([])

    ibi_timestamps = np.array(ibi_timestamps)
    ibi_values = np.array(ibi_values)

    # Konwersja do ms jeśli w sekundach
    if ibi_unit == 's':
        ibi_values_ms = ibi_values * 1000
    else:
        ibi_values_ms = ibi_values

    # Filtruj nieprawidłowe IBI (300-2000 ms = 30-200 BPM)
    valid_mask = (ibi_values_ms > 300) & (ibi_values_ms < 2000)
    timestamps = ibi_timestamps[valid_mask]
    ibi_ms = ibi_values_ms[valid_mask]

    if len(timestamps) < 2:
        return np.array([]), np.array([])

    # Oblicz chwilowe HR (BPM)
    hr_values = 60000 / ibi_ms

    # Stwórz równomierną siatkę czasową (1 Hz)
    # Wykryj jednostkę czasową (ms vs s) na podstawie max_time
    if max_time > 10000:  # Prawdopodobnie ms
        step = 1000  # 1 sekunda w ms
    else:  # Prawdopodobnie s
        step = 1  # 1 sekunda

    time_grid = np.arange(0, max_time, step)

    if len(time_grid) == 0:
        return np.array([]), np.array([])

    # Interpoluj HR do siatki czasowej
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
    GLOBAL PROCESSING: Oblicz ciągły przebieg HR z całego sygnału ECG.

    Pipeline:
    1. Wykryj piki R na całym nagraniu
    2. Oblicz IBI (RR intervals)
    3. Oblicz chwilowe HR dla każdego IBI
    4. Interpoluj do równomiernej siatki czasowej (1 Hz)

    Args:
        ecg_data: sygnał ECG
        time_data: wektor czasu
        fs: częstotliwość próbkowania ECG

    Returns:
        tuple (time_grid, hr_timeseries, r_peaks) - czas, HR w BPM, indeksy pików R
    """
    # Wykryj piki R
    r_peaks = detect_r_peaks_ecg(ecg_data, fs)

    if len(r_peaks) < 2:
        return np.array([]), np.array([]), r_peaks

    # Oblicz RR intervals (IBI) w ms
    rr_intervals = np.diff(r_peaks) / fs * 1000

    # Timestamps dla RR intervals (średnia między kolejnymi pikami)
    rr_timestamps = (time_data[r_peaks[:-1]] + time_data[r_peaks[1:]]) / 2

    # Filtruj nieprawidłowe RR (300-2000 ms = 30-200 BPM)
    valid_mask = (rr_intervals > 300) & (rr_intervals < 2000)
    rr_intervals = rr_intervals[valid_mask]
    rr_timestamps = rr_timestamps[valid_mask]

    if len(rr_intervals) < 2:
        return np.array([]), np.array([]), r_peaks

    # Oblicz chwilowe HR (BPM)
    hr_values = 60000 / rr_intervals

    # Stwórz równomierną siatkę czasową (1 Hz = odstęp w jednostce time_data)
    max_time = time_data.max()
    min_time = time_data.min()

    # Wykryj jednostkę czasową
    if max_time > 10000:  # ms
        step = 1000
    else:  # s
        step = 1

    time_grid = np.arange(min_time, max_time, step)

    if len(time_grid) == 0:
        return np.array([]), np.array([]), r_peaks

    # Interpoluj HR do siatki czasowej
    try:
        f_interp = interp1d(rr_timestamps, hr_values, kind='linear',
                           bounds_error=False, fill_value=np.nan)
        hr_timeseries = f_interp(time_grid)
    except:
        hr_timeseries = np.full(len(time_grid), np.nan)

    return time_grid, hr_timeseries, r_peaks


def detect_r_peaks_ecg(ecg_data: np.ndarray, fs: float) -> np.ndarray:
    """
    Wykryj piki R w sygnale ECG (uproszczony Pan-Tompkins).

    Args:
        ecg_data: sygnał ECG
        fs: częstotliwość próbkowania (Hz)

    Returns:
        tablica indeksów pików R
    """
    if len(ecg_data) < fs * 2:
        return np.array([])

    try:
        # Filtracja pasmowa ECG (5-15 Hz - pasmo QRS)
        nyquist = fs / 2
        low = 5.0 / nyquist
        high = min(15.0 / nyquist, 0.99)

        b, a = scipy_signal.butter(2, [low, high], btype='band')
        filtered_ecg = scipy_signal.filtfilt(b, a, ecg_data)

        # Różniczkowanie
        diff_ecg = np.diff(filtered_ecg)

        # Podniesienie do kwadratu
        squared = diff_ecg ** 2

        # Integracja z oknem ruchomym (150 ms)
        window_size = int(fs * 0.15)
        if window_size < 1:
            window_size = 1
        integrated = np.convolve(squared, np.ones(window_size)/window_size, mode='same')

        # Detekcja pików
        min_distance = int(fs * 0.3)  # Minimalna odległość 300ms (max 200 BPM)
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
    LOCAL AGGREGATION: Oblicz cechy HR w oknie czasowym z globalnego przebiegu.

    Args:
        time_grid: siatka czasowa
        hr_timeseries: ciągły przebieg HR
        window_start: początek okna
        window_end: koniec okna

    Returns:
        dict z: hr_mean, hr_std
    """
    if len(hr_timeseries) == 0 or len(time_grid) == 0:
        return {'hr_mean': np.nan, 'hr_std': np.nan}

    # Znajdź wartości HR w oknie
    mask = (time_grid >= window_start) & (time_grid < window_end)
    hr_window = hr_timeseries[mask]

    # Usuń NaN
    hr_valid = hr_window[~np.isnan(hr_window)]

    if len(hr_valid) == 0:
        return {'hr_mean': np.nan, 'hr_std': np.nan}

    return {
        'hr_mean': float(np.mean(hr_valid)),
        'hr_std': float(np.std(hr_valid, ddof=1)) if len(hr_valid) > 1 else 0.0
    }


# =============================================================================
# FUNKCJE DLA HRV (Heart Rate Variability)
# =============================================================================

def compute_hrv_window_features(
    ibi_timestamps: np.ndarray,
    ibi_values: np.ndarray,
    window_start: float,
    window_end: float,
    ibi_unit: str = 'ms'
) -> dict:
    """
    Oblicz metryki HRV z IBI w oknie czasowym.

    Args:
        ibi_timestamps: znaczniki czasowe IBI
        ibi_values: wartości IBI
        window_start: początek okna
        window_end: koniec okna
        ibi_unit: 'ms' lub 's'

    Returns:
        dict z metrykami HRV: hrv_sdnn, hrv_rmssd, hrv_pnn50, hrv_lf_power, hrv_hf_power, hrv_lf_hf_ratio
    """
    empty_result = {
        'hrv_sdnn': np.nan,
        'hrv_rmssd': np.nan,
        'hrv_pnn50': np.nan,
        'hrv_lf_power': np.nan,
        'hrv_hf_power': np.nan,
        'hrv_lf_hf_ratio': np.nan
    }

    if len(ibi_timestamps) < 3:
        return empty_result

    ibi_timestamps = np.array(ibi_timestamps)
    ibi_values = np.array(ibi_values)

    # Pobierz IBI w oknie
    mask = (ibi_timestamps >= window_start) & (ibi_timestamps < window_end)
    window_ibi = ibi_values[mask]

    if len(window_ibi) < 3:
        return empty_result

    # Użyj funkcji z bvp_utils
    metrics = compute_metrics_from_ibi(window_ibi, ibi_unit=ibi_unit)

    return {
        'hrv_sdnn': metrics.get('bvp_sdnn', np.nan),
        'hrv_rmssd': metrics.get('bvp_rmssd', np.nan),
        'hrv_pnn50': metrics.get('bvp_pnn50', np.nan),
        'hrv_lf_power': metrics.get('bvp_lf_power', np.nan),
        'hrv_hf_power': metrics.get('bvp_hf_power', np.nan),
        'hrv_lf_hf_ratio': metrics.get('bvp_lf_hf_ratio', np.nan)
    }


def compute_hrv_from_ecg_window(
    r_peaks: np.ndarray,
    time_data: np.ndarray,
    fs: float,
    window_start: float,
    window_end: float
) -> dict:
    """
    Oblicz metryki HRV z pików R ECG w oknie czasowym.

    Args:
        r_peaks: indeksy pików R
        time_data: wektor czasu
        fs: częstotliwość próbkowania
        window_start: początek okna
        window_end: koniec okna

    Returns:
        dict z metrykami HRV
    """
    empty_result = {
        'hrv_sdnn': np.nan,
        'hrv_rmssd': np.nan,
        'hrv_pnn50': np.nan,
        'hrv_lf_power': np.nan,
        'hrv_hf_power': np.nan,
        'hrv_lf_hf_ratio': np.nan
    }

    if len(r_peaks) < 3:
        return empty_result

    # Znajdź piki R w oknie
    peak_times = time_data[r_peaks]
    mask = (peak_times >= window_start) & (peak_times < window_end)
    window_peaks = r_peaks[mask]

    if len(window_peaks) < 3:
        return empty_result

    # Oblicz RR intervals w oknie (ms)
    rr_intervals = np.diff(window_peaks) / fs * 1000

    # Filtruj nieprawidłowe
    valid_rr = rr_intervals[(rr_intervals > 300) & (rr_intervals < 2000)]

    if len(valid_rr) < 3:
        return empty_result

    # Użyj funkcji z bvp_utils
    metrics = compute_metrics_from_ibi(valid_rr, ibi_unit='ms')

    return {
        'hrv_sdnn': metrics.get('bvp_sdnn', np.nan),
        'hrv_rmssd': metrics.get('bvp_rmssd', np.nan),
        'hrv_pnn50': metrics.get('bvp_pnn50', np.nan),
        'hrv_lf_power': metrics.get('bvp_lf_power', np.nan),
        'hrv_hf_power': metrics.get('bvp_hf_power', np.nan),
        'hrv_lf_hf_ratio': metrics.get('bvp_lf_hf_ratio', np.nan)
    }

