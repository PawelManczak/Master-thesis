"""
Wspólne funkcje do obliczania metryk z BVP (Blood Volume Pulse).
Ujednolicona implementacja dla wszystkich datasetów (K-EmoCon, CASE, CEAP).

Z BVP obliczamy:
- HR (Heart Rate) - średnie tętno
- IBI (Inter-Beat Interval) - interwały między uderzeniami serca
- Metryki pochodne z IBI:
  - bvp_sdnn: odchylenie standardowe IBI (ms)
  - bvp_rmssd: root mean square of successive differences IBI (ms)
  - bvp_pnn50: procent różnic IBI > 50ms
  - bvp_lf_power: moc w paśmie LF (0.04-0.15 Hz)
  - bvp_hf_power: moc w paśmie HF (0.15-0.4 Hz)
  - bvp_lf_hf_ratio: stosunek LF/HF
"""

import numpy as np
from scipy import signal as scipy_signal
from scipy.interpolate import interp1d
from typing import Union, Dict, Tuple, Optional


def detect_peaks_bvp(bvp_data: np.ndarray, fs: float) -> np.ndarray:
    """
    Wykryj piki w sygnale BVP (fotopletyzmograficznym).

    Args:
        bvp_data: sygnał BVP
        fs: częstotliwość próbkowania (Hz)

    Returns:
        tablica indeksów pików
    """
    if len(bvp_data) < fs * 0.5:  # Minimum 0.5 sekundy danych
        return np.array([])

    try:
        # Filtracja pasmowa BVP (0.5-8 Hz - typowe pasmo dla sygnału PPG)
        nyquist = fs / 2
        low = max(0.5 / nyquist, 0.01)
        high = min(8.0 / nyquist, 0.99)

        b, a = scipy_signal.butter(2, [low, high], btype='band')
        filtered_bvp = scipy_signal.filtfilt(b, a, bvp_data)

        # Minimalna odległość między pikami (odpowiada max 200 BPM)
        min_distance = int(fs * 0.3)  # 300ms

        # Detekcja pików
        # Używamy progu adaptacyjnego
        threshold = np.mean(filtered_bvp) + 0.3 * np.std(filtered_bvp)

        peaks, properties = scipy_signal.find_peaks(
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
    Oblicz IBI (Inter-Beat Intervals) z sygnału BVP.

    Args:
        bvp_data: sygnał BVP
        fs: częstotliwość próbkowania (Hz)

    Returns:
        tablica IBI w milisekundach
    """
    peaks = detect_peaks_bvp(bvp_data, fs)

    if len(peaks) < 2:
        return np.array([])

    # Oblicz IBI jako różnice między kolejnymi pikami w ms
    ibi_samples = np.diff(peaks)
    ibi_ms = (ibi_samples / fs) * 1000

    # Filtruj nieprawidłowe wartości IBI (300-2000 ms = 30-200 BPM)
    valid_ibi = ibi_ms[(ibi_ms > 300) & (ibi_ms < 2000)]

    return valid_ibi


def compute_hr_from_bvp(bvp_data: np.ndarray, fs: float) -> float:
    """
    Oblicz średnie tętno (HR) z sygnału BVP.

    Args:
        bvp_data: sygnał BVP
        fs: częstotliwość próbkowania (Hz)

    Returns:
        średnie tętno w BPM lub NaN
    """
    ibi_ms = compute_ibi_from_bvp(bvp_data, fs)

    if len(ibi_ms) == 0:
        return np.nan

    mean_ibi = np.mean(ibi_ms)
    hr = 60000 / mean_ibi  # BPM

    return hr


def compute_metrics_from_ibi(ibi_values: np.ndarray, ibi_unit: str = 'ms') -> Dict:
    """
    Oblicz metryki z interwałów IBI.

    Args:
        ibi_values: tablica wartości IBI
        ibi_unit: jednostka IBI - 'ms' (milisekundy) lub 's' (sekundy)

    Returns:
        dict z metrykami:
        - bvp_sdnn: odchylenie standardowe IBI (ms)
        - bvp_rmssd: root mean square of successive differences (ms)
        - bvp_pnn50: procent różnic IBI > 50ms
        - bvp_mean_hr: średnie tętno (BPM)
        - bvp_mean_ibi: średnie IBI (ms)
        - bvp_lf_power: moc w paśmie LF (0.04-0.15 Hz)
        - bvp_hf_power: moc w paśmie HF (0.15-0.4 Hz)
        - bvp_lf_hf_ratio: stosunek LF/HF
    """
    result = {
        'bvp_sdnn': np.nan,
        'bvp_rmssd': np.nan,
        'bvp_pnn50': np.nan,
        'bvp_mean_hr': np.nan,
        'bvp_mean_ibi': np.nan,
        'bvp_lf_power': np.nan,
        'bvp_hf_power': np.nan,
        'bvp_lf_hf_ratio': np.nan
    }

    if len(ibi_values) < 3:
        return result

    # Konwersja do ms jeśli podano sekundy
    if ibi_unit == 's':
        ibi_ms = ibi_values * 1000
    else:
        ibi_ms = np.array(ibi_values, dtype=float)

    # Filtruj nieprawidłowe wartości IBI (300-2000 ms = 30-200 BPM)
    valid_ibi = ibi_ms[(ibi_ms > 300) & (ibi_ms < 2000)]

    if len(valid_ibi) < 3:
        return result

    try:
        # ===== METRYKI CZASOWE =====

        # SDNN - odchylenie standardowe wszystkich IBI
        result['bvp_sdnn'] = np.std(valid_ibi, ddof=1)

        # Mean IBI
        result['bvp_mean_ibi'] = np.mean(valid_ibi)

        # Mean HR - średnie tętno
        result['bvp_mean_hr'] = 60000 / result['bvp_mean_ibi']

        # RMSSD - root mean square of successive differences
        ibi_diff = np.diff(valid_ibi)
        if len(ibi_diff) > 0:
            result['bvp_rmssd'] = np.sqrt(np.mean(ibi_diff ** 2))

            # pNN50 - procent różnic > 50ms
            nn50 = np.sum(np.abs(ibi_diff) > 50)
            result['bvp_pnn50'] = (nn50 / len(ibi_diff)) * 100

        # ===== METRYKI CZĘSTOTLIWOŚCIOWE =====
        if len(valid_ibi) >= 5:
            freq_result = _compute_frequency_domain(valid_ibi)
            result.update(freq_result)

    except Exception:
        pass

    return result


def compute_metrics_from_bvp(bvp_data: np.ndarray, fs: float) -> Dict:
    """
    Oblicz wszystkie metryki z sygnału BVP.

    Args:
        bvp_data: sygnał BVP (Blood Volume Pulse)
        fs: częstotliwość próbkowania (Hz)

    Returns:
        dict z wszystkimi metrykami BVP
    """
    result = {
        'bvp_sdnn': np.nan,
        'bvp_rmssd': np.nan,
        'bvp_pnn50': np.nan,
        'bvp_mean_hr': np.nan,
        'bvp_mean_ibi': np.nan,
        'bvp_lf_power': np.nan,
        'bvp_hf_power': np.nan,
        'bvp_lf_hf_ratio': np.nan
    }

    # Minimalna długość sygnału: 2 sekundy
    if len(bvp_data) < fs * 2:
        return result

    try:
        # Oblicz IBI z BVP
        ibi_ms = compute_ibi_from_bvp(bvp_data, fs)

        if len(ibi_ms) < 3:
            return result

        # Oblicz metryki z IBI
        return compute_metrics_from_ibi(ibi_ms, ibi_unit='ms')

    except Exception:
        return result


def compute_metrics_from_ecg(ecg_data: np.ndarray, fs: float) -> Dict:
    """
    Oblicz metryki z sygnału ECG.
    Używa detekcji pików R do obliczenia IBI.

    Args:
        ecg_data: sygnał ECG (w mV lub jednostkach względnych)
        fs: częstotliwość próbkowania (Hz)

    Returns:
        dict z metrykami BVP (te same co z BVP)
    """
    result = {
        'bvp_sdnn': np.nan,
        'bvp_rmssd': np.nan,
        'bvp_pnn50': np.nan,
        'bvp_mean_hr': np.nan,
        'bvp_mean_ibi': np.nan,
        'bvp_lf_power': np.nan,
        'bvp_hf_power': np.nan,
        'bvp_lf_hf_ratio': np.nan
    }

    # Minimalna długość sygnału: 2 sekundy
    if len(ecg_data) < fs * 2:
        return result

    try:
        # Detekcja interwałów RR z ECG
        rr_intervals = _detect_rr_from_ecg(ecg_data, fs)

        if rr_intervals is None or len(rr_intervals) < 3:
            return result

        # Oblicz metryki z RR intervals (to są IBI)
        return compute_metrics_from_ibi(rr_intervals, ibi_unit='ms')

    except Exception:
        return result


def _detect_rr_from_ecg(ecg_data: np.ndarray, fs: float) -> Optional[np.ndarray]:
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


def _compute_frequency_domain(valid_ibi: np.ndarray) -> Dict:
    """
    Oblicz metryki częstotliwościowe (LF, HF, LF/HF ratio) z IBI.

    Args:
        valid_ibi: przefiltrowane wartości IBI w ms

    Returns:
        dict z metrykami LF/HF
    """
    result = {
        'bvp_lf_power': np.nan,
        'bvp_hf_power': np.nan,
        'bvp_lf_hf_ratio': np.nan
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
            result['bvp_lf_power'] = lf_power

        if np.any(hf_mask):
            hf_power = np.trapezoid(psd[hf_mask], freqs[hf_mask])
            result['bvp_hf_power'] = hf_power

        # LF/HF ratio
        if result['bvp_hf_power'] > 0 and not np.isnan(result['bvp_lf_power']):
            result['bvp_lf_hf_ratio'] = result['bvp_lf_power'] / result['bvp_hf_power']

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


def compute_window_stats(values: np.ndarray) -> Dict:
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

