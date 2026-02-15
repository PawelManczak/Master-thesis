"""
Wspólne funkcje do obliczania metryk HRV (Heart Rate Variability).
Ujednolicona implementacja dla wszystkich datasetów (K-EmoCon, CASE, CEAP).
"""

import numpy as np
from scipy import signal as scipy_signal
from scipy.interpolate import interp1d
from typing import Union


def compute_hrv_from_ibi(ibi_values: np.ndarray, ibi_unit: str = 'ms') -> dict:
    """
    Oblicz metryki HRV z interwałów RR (IBI).

    Args:
        ibi_values: tablica wartości IBI (inter-beat interval)
        ibi_unit: jednostka IBI - 'ms' (milisekundy) lub 's' (sekundy)

    Returns:
        dict z metrykami HRV:
        - hrv_sdnn: odchylenie standardowe NN intervals (ms)
        - hrv_rmssd: root mean square of successive differences (ms)
        - hrv_pnn50: procent różnic NN > 50ms
        - hrv_mean_hr: średnie tętno (BPM)
        - hrv_lf_power: moc w paśmie LF (0.04-0.15 Hz)
        - hrv_hf_power: moc w paśmie HF (0.15-0.4 Hz)
        - hrv_lf_hf_ratio: stosunek LF/HF
    """
    result = {
        'hrv_sdnn': np.nan,
        'hrv_rmssd': np.nan,
        'hrv_pnn50': np.nan,
        'hrv_mean_hr': np.nan,
        'hrv_lf_power': np.nan,
        'hrv_hf_power': np.nan,
        'hrv_lf_hf_ratio': np.nan
    }

    if len(ibi_values) < 3:
        return result

    # Konwersja do ms jeśli podano sekundy
    if ibi_unit == 's':
        ibi_ms = ibi_values * 1000
    else:
        ibi_ms = ibi_values

    # Filtruj nieprawidłowe wartości IBI (300-2000 ms odpowiada 30-200 BPM)
    valid_ibi = ibi_ms[(ibi_ms > 300) & (ibi_ms < 2000)]

    if len(valid_ibi) < 3:
        return result

    try:
        # Metryki czasowe
        # SDNN - odchylenie standardowe wszystkich NN intervals
        result['hrv_sdnn'] = np.std(valid_ibi, ddof=1)  # ddof=1 dla próbki

        # RMSSD - root mean square of successive differences
        ibi_diff = np.diff(valid_ibi)
        if len(ibi_diff) > 0:
            result['hrv_rmssd'] = np.sqrt(np.mean(ibi_diff ** 2))

            # pNN50 - procent różnic > 50ms
            nn50 = np.sum(np.abs(ibi_diff) > 50)
            result['hrv_pnn50'] = (nn50 / len(ibi_diff)) * 100

        # Mean HR - średnie tętno
        mean_ibi = np.mean(valid_ibi)
        result['hrv_mean_hr'] = 60000 / mean_ibi if mean_ibi > 0 else np.nan

        # Metryki częstotliwościowe (LF/HF)
        if len(valid_ibi) >= 5:
            freq_result = _compute_frequency_domain_hrv(valid_ibi)
            result.update(freq_result)

    except Exception:
        pass

    return result


def compute_hrv_from_rr_intervals(rr_intervals: np.ndarray) -> dict:
    """
    Oblicz metryki HRV z interwałów RR w ms.
    Alias dla compute_hrv_from_ibi z jednostką ms.

    Args:
        rr_intervals: tablica interwałów RR w ms

    Returns:
        dict z metrykami HRV
    """
    return compute_hrv_from_ibi(rr_intervals, ibi_unit='ms')


def compute_hrv_from_ecg(ecg_data: np.ndarray, fs: float = 1000.0) -> dict:
    """
    Oblicz metryki HRV z sygnału ECG.

    Args:
        ecg_data: sygnał ECG (w mV lub jednostkach względnych)
        fs: częstotliwość próbkowania (Hz), domyślnie 1000 Hz

    Returns:
        dict z metrykami HRV
    """
    result = {
        'hrv_sdnn': np.nan,
        'hrv_rmssd': np.nan,
        'hrv_pnn50': np.nan,
        'hrv_mean_hr': np.nan,
        'hrv_lf_power': np.nan,
        'hrv_hf_power': np.nan,
        'hrv_lf_hf_ratio': np.nan
    }

    # Minimalna długość sygnału: 2 sekundy
    if len(ecg_data) < fs * 2:
        return result

    try:
        # Detekcja pików R z sygnału ECG
        rr_intervals = _detect_rr_from_ecg(ecg_data, fs)

        if rr_intervals is None or len(rr_intervals) < 3:
            return result

        # Oblicz metryki HRV z wykrytych interwałów RR
        return compute_hrv_from_ibi(rr_intervals, ibi_unit='ms')

    except Exception:
        return result


def _detect_rr_from_ecg(ecg_data: np.ndarray, fs: float) -> Union[np.ndarray, None]:
    """
    Wykryj interwały RR z sygnału ECG używając uproszczonego algorytmu Pan-Tompkins.

    Args:
        ecg_data: sygnał ECG
        fs: częstotliwość próbkowania (Hz)

    Returns:
        tablica interwałów RR w ms lub None
    """
    try:
        # Filtracja pasmowa ECG (0.5-40 Hz)
        nyquist = fs / 2
        low = 0.5 / nyquist
        high = min(40.0 / nyquist, 0.99)

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

        if len(peaks) < 3:
            return None

        # Oblicz interwały RR w ms
        rr_intervals = np.diff(peaks) / fs * 1000

        return rr_intervals

    except Exception:
        return None


def _compute_frequency_domain_hrv(valid_ibi: np.ndarray) -> dict:
    """
    Oblicz metryki częstotliwościowe HRV (LF, HF, LF/HF ratio).

    Args:
        valid_ibi: przefiltrowane wartości IBI w ms

    Returns:
        dict z metrykami LF/HF
    """
    result = {
        'hrv_lf_power': np.nan,
        'hrv_hf_power': np.nan,
        'hrv_lf_hf_ratio': np.nan
    }

    try:
        # Tworzenie wektora czasu (kumulatywna suma IBI)
        ibi_times = np.cumsum(valid_ibi) / 1000  # sekundy
        ibi_times = ibi_times - ibi_times[0]

        # Potrzebujemy co najmniej 1 sekundy danych
        if ibi_times[-1] <= 1:
            return result

        # Interpolacja do równomiernego próbkowania (4 Hz)
        interp_fs = 4  # Hz
        interp_times = np.arange(0, ibi_times[-1], 1/interp_fs)

        if len(interp_times) <= 4:
            return result

        # Interpolacja kubiczna
        f_interp = interp1d(ibi_times, valid_ibi, kind='cubic', fill_value='extrapolate')
        ibi_interp = f_interp(interp_times)

        # Usunięcie trendu
        ibi_detrend = scipy_signal.detrend(ibi_interp)

        # Welch PSD
        nperseg = min(len(ibi_detrend), 256)
        freqs, psd = scipy_signal.welch(ibi_detrend, fs=interp_fs, nperseg=nperseg)

        # Pasma częstotliwości
        # LF: 0.04-0.15 Hz (aktywność współczulna i przywspółczulna)
        # HF: 0.15-0.4 Hz (aktywność przywspółczulna)
        lf_mask = (freqs >= 0.04) & (freqs < 0.15)
        hf_mask = (freqs >= 0.15) & (freqs < 0.4)

        # Oblicz moc jako całkę PSD
        if np.any(lf_mask):
            lf_power = np.trapezoid(psd[lf_mask], freqs[lf_mask])
            result['hrv_lf_power'] = lf_power

        if np.any(hf_mask):
            hf_power = np.trapezoid(psd[hf_mask], freqs[hf_mask])
            result['hrv_hf_power'] = hf_power

        # LF/HF ratio
        if result['hrv_hf_power'] > 0 and not np.isnan(result['hrv_lf_power']):
            result['hrv_lf_hf_ratio'] = result['hrv_lf_power'] / result['hrv_hf_power']

    except Exception:
        pass

    return result


def antialiasing_filter(data: np.ndarray, original_fs: float, target_fs: float) -> np.ndarray:
    """
    Filtr antyaliasingowy (dolnoprzepustowy) przed decymacją.

    Args:
        data: dane wejściowe
        original_fs: oryginalna częstotliwość próbkowania (Hz)
        target_fs: docelowa częstotliwość próbkowania (Hz)

    Returns:
        przefiltrowane dane
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


def compute_window_stats(values: np.ndarray) -> dict:
    """
    Oblicz statystyki dla okna czasowego.

    Args:
        values: wartości w oknie

    Returns:
        dict z 'mean' i 'var'
    """
    if len(values) == 0:
        return {'mean': np.nan, 'var': np.nan}

    return {
        'mean': np.mean(values),
        'var': np.var(values, ddof=1) if len(values) > 1 else 0.0
    }

