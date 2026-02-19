#!/usr/bin/env python3
"""
CEAP-360VR Data Processing Script
Łączy dane fizjologiczne z adnotacjami (valence i arousal).
Sygnały agregowane są do okien 5-sekundowych.

Sygnały z Empatica E4 w danych Raw:
- ACC: 3-osiowy akcelerometr (x, y, z), ~4 Hz
- BVP: fotopletyzmografia, ~4 Hz
- EDA: przewodnictwo skórne w µS, ~4 Hz
- HR: tętno, ~4 Hz
- IBI: inter-beat interval w sekundach (nieregularne)
- SKT: temperatura skóry (°C), ~4 Hz

Adnotacje Raw: ciągłe valence-arousal w skali [-1, 1]

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
| HR/IBI | Tracisz zmienność rytmu (HRV)   | mean, std (SDNN), RMSSD               |
| BVP    | Średnia→0 (sygnał falowy!)      | std (amplituda), spectral_power (NIE mean!) |
| TEMP   | Zmienia się wolno, OK           | mean, slope (trend)                   |
| ACC    | Średnia=grawitacja/pozycja      | mean (pozycja), std (ruch/drżenie)    |

PIPELINE HR (Global Processing):
1. Filtr pasmowo-przepustowy 0.5-8 Hz (cały sygnał)
2. Detekcja pików na całym nagraniu
3. Obliczenie chwilowego HR (interpolacja)
4. Agregacja do okien 5s

Referencje:
- Task Force of ESC and NASPE (1996). Heart rate variability: standards of measurement
- Malik, M. (1996). Heart rate variability. Circulation, 93(5), 1043-1065.
- Pham et al. (2021). Heart Rate Variability in Psychology. Frontiers in Psychology.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal as scipy_signal
from scipy.interpolate import interp1d
from scipy.stats import linregress

# Import wspólnych funkcji BVP
from bvp_utils import compute_metrics_from_ibi

# Ścieżki
BASE_DIR = Path(__file__).parent.parent.parent.parent  # extracting -> processing -> source -> second part
CEAP_DIR = BASE_DIR / "data" / "CEAP" / "raw" / "CEAP-360VR-Dataset-master" / "CEAP-360VR"
PHYSIO_DIR = CEAP_DIR / "5_PhysioData" / "Raw"  # Używamy Raw dla oryginalnych jednostek
ANNOT_DIR = CEAP_DIR / "3_AnnotationData" / "Raw"  # Adnotacje Raw z zakresem [-1, 1]
OUTPUT_DIR = BASE_DIR / "data" / "CEAP" / "processed"

# Stałe
ORIGINAL_FS = 4.0  # Hz - dane Raw są próbkowane 4 Hz (EDA, SKT)
WINDOW_SIZE = 5.0   # sekundy


def load_physio_data(pid: int) -> dict:
    """Wczytaj dane fizjologiczne dla danego uczestnika."""
    file_path = PHYSIO_DIR / f"P{pid}_Physio_RawData.json"
    if not file_path.exists():
        return None

    with open(file_path, 'r') as f:
        data = json.load(f)

    return data['Physio_RawData'][0]


def load_annotation_data(pid: int) -> dict:
    """Wczytaj dane adnotacji dla danego uczestnika (Raw, zakres [-1, 1])."""
    file_path = ANNOT_DIR / f"P{pid}_Annotation_RawData.json"
    if not file_path.exists():
        return None

    with open(file_path, 'r') as f:
        data = json.load(f)

    return data['ContinuousAnnotation_RawData'][0]


# =============================================================================
# FUNKCJE AGREGUJĄCE DLA POSZCZEGÓLNYCH SYGNAŁÓW
# =============================================================================

def compute_eda_features(values: np.ndarray, fs: float) -> dict:
    """
    Oblicz cechy EDA w oknie czasowym.

    EDA (Electrodermal Activity) - tracisz piki stresu przy zwykłej średniej!

    Args:
        values: wartości EDA w oknie
        fs: częstotliwość próbkowania

    Returns:
        dict z: mean, std, max, peaks_count
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
            peaks, _ = scipy_signal.find_peaks(values, height=threshold, distance=int(fs * 0.5))
            peaks_count = len(peaks)
        except:
            peaks_count = 0

    return {
        'eda_mean': np.mean(values),
        'eda_std': np.std(values, ddof=1) if len(values) > 1 else 0.0,
        'eda_max': np.max(values),
        'eda_peaks': peaks_count
    }


def compute_bvp_features(values: np.ndarray, fs: float) -> dict:
    """
    Oblicz cechy BVP w oknie czasowym.

    UWAGA: BVP to sygnał falowy - średnia dąży do 0!
    NIE używaj średniej! Używaj std (amplituda) i spectral power.

    Args:
        values: wartości BVP w oknie
        fs: częstotliwość próbkowania

    Returns:
        dict z: std (amplituda), spectral_power, peak_to_peak
    """
    if len(values) == 0:
        return {
            'bvp_std': np.nan,
            'bvp_peak_to_peak': np.nan,
            'bvp_spectral_power': np.nan
        }

    values = np.array(values)

    # Std = miara amplitudy sygnału
    bvp_std = np.std(values, ddof=1) if len(values) > 1 else 0.0

    # Peak-to-peak amplitude
    bvp_p2p = np.max(values) - np.min(values)

    # Spectral power (energia sygnału)
    spectral_power = np.nan
    if len(values) >= 4:
        try:
            # Oblicz PSD metodą Welcha
            freqs, psd = scipy_signal.welch(values, fs=fs, nperseg=min(len(values), 64))
            # Moc w paśmie sercowym (0.5-4 Hz)
            cardiac_band = (freqs >= 0.5) & (freqs <= 4.0)
            if np.any(cardiac_band):
                spectral_power = np.trapezoid(psd[cardiac_band], freqs[cardiac_band])
        except:
            pass

    return {
        'bvp_std': bvp_std,
        'bvp_peak_to_peak': bvp_p2p,
        'bvp_spectral_power': spectral_power
    }


def compute_temp_features(values: np.ndarray, timestamps: np.ndarray = None) -> dict:
    """
    Oblicz cechy temperatury w oknie czasowym.

    Temperatura zmienia się wolno - średnia jest OK.
    Dodatkowo: slope (trend) - czy rośnie czy spada.

    Args:
        values: wartości temperatury w oknie
        timestamps: znaczniki czasowe (opcjonalne)

    Returns:
        dict z: mean, slope
    """
    if len(values) == 0:
        return {'temp_mean': np.nan, 'temp_slope': np.nan}

    values = np.array(values)
    temp_mean = np.mean(values)

    # Slope (trend liniowy)
    temp_slope = 0.0
    if len(values) >= 2:
        try:
            if timestamps is not None and len(timestamps) == len(values):
                x = timestamps
            else:
                x = np.arange(len(values))
            slope, _, _, _, _ = linregress(x, values)
            temp_slope = slope
        except:
            temp_slope = 0.0

    return {
        'temp_mean': temp_mean,
        'temp_slope': temp_slope
    }


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
            result[f'acc_{axis}_mean'] = np.mean(values)
            result[f'acc_{axis}_std'] = np.std(values, ddof=1) if len(values) > 1 else 0.0

    # Magnitude
    if len(acc_x) > 0 and len(acc_y) > 0 and len(acc_z) > 0:
        magnitude = np.sqrt(np.array(acc_x)**2 + np.array(acc_y)**2 + np.array(acc_z)**2)
        result['acc_magnitude_mean'] = np.mean(magnitude)
        result['acc_magnitude_std'] = np.std(magnitude, ddof=1) if len(magnitude) > 1 else 0.0
    else:
        result['acc_magnitude_mean'] = np.nan
        result['acc_magnitude_std'] = np.nan

    return result


def compute_global_hr_timeseries(ibi_data: list, max_time: float, fs: float = 1.0) -> np.ndarray:
    """
    GLOBAL PROCESSING: Oblicz ciągły przebieg HR z całego nagrania.

    Złota zasada: "Nie licz HR w oknie, uśredniaj HR w oknie"

    Pipeline:
    1. Pobierz wszystkie IBI z całego nagrania
    2. Oblicz chwilowe HR dla każdego IBI
    3. Interpoluj do równomiernej siatki czasowej
    4. Zwróć ciągły przebieg HR (1 Hz domyślnie)

    Args:
        ibi_data: lista dict z 'TimeStamp' i 'IBI' (IBI w sekundach)
        max_time: maksymalny czas nagrania
        fs: częstotliwość wyjściowa (domyślnie 1 Hz)

    Returns:
        tablica HR w BPM dla każdej sekundy (lub NaN gdzie brak danych)
    """
    if not ibi_data or max_time <= 0:
        return np.array([])

    # Pobierz timestamps i IBI
    timestamps = np.array([d['TimeStamp'] for d in ibi_data])
    ibi_values = np.array([d['IBI'] for d in ibi_data])  # IBI w sekundach

    # Filtruj nieprawidłowe IBI (0.3-2.0 s = 30-200 BPM)
    valid_mask = (ibi_values > 0.3) & (ibi_values < 2.0)
    timestamps = timestamps[valid_mask]
    ibi_values = ibi_values[valid_mask]

    if len(timestamps) < 2:
        return np.array([])

    # Oblicz chwilowe HR (BPM) dla każdego IBI
    hr_values = 60.0 / ibi_values

    # Stwórz równomierną siatkę czasową
    time_grid = np.arange(0, max_time, 1.0 / fs)

    if len(time_grid) == 0:
        return np.array([])

    # Interpoluj HR do siatki czasowej
    try:
        # Interpolacja liniowa z ekstrapolacją NaN poza zakresem
        f_interp = interp1d(timestamps, hr_values, kind='linear',
                           bounds_error=False, fill_value=np.nan)
        hr_timeseries = f_interp(time_grid)
    except:
        hr_timeseries = np.full(len(time_grid), np.nan)

    return hr_timeseries


def compute_hr_window_features(hr_timeseries: np.ndarray, window_start: float,
                                window_end: float, fs: float = 1.0) -> dict:
    """
    LOCAL AGGREGATION: Oblicz cechy HR w oknie czasowym z globalnego przebiegu.

    Args:
        hr_timeseries: ciągły przebieg HR (z compute_global_hr_timeseries)
        window_start: początek okna (sekundy)
        window_end: koniec okna (sekundy)
        fs: częstotliwość przebiegu HR

    Returns:
        dict z: hr_mean, hr_std
    """
    if len(hr_timeseries) == 0:
        return {'hr_mean': np.nan, 'hr_std': np.nan}

    # Indeksy dla okna
    start_idx = int(window_start * fs)
    end_idx = int(window_end * fs)

    # Sprawdź granice
    start_idx = max(0, start_idx)
    end_idx = min(len(hr_timeseries), end_idx)

    if start_idx >= end_idx:
        return {'hr_mean': np.nan, 'hr_std': np.nan}

    # Pobierz wartości HR w oknie
    hr_window = hr_timeseries[start_idx:end_idx]

    # Usuń NaN
    hr_valid = hr_window[~np.isnan(hr_window)]

    if len(hr_valid) == 0:
        return {'hr_mean': np.nan, 'hr_std': np.nan}

    return {
        'hr_mean': np.mean(hr_valid),
        'hr_std': np.std(hr_valid, ddof=1) if len(hr_valid) > 1 else 0.0
    }


def compute_ibi_window_features(ibi_data: list, window_start: float,
                                 window_end: float) -> dict:
    """
    Oblicz metryki HRV z IBI w oknie czasowym.

    Args:
        ibi_data: lista dict z 'TimeStamp' i 'IBI'
        window_start: początek okna
        window_end: koniec okna

    Returns:
        dict z metrykami HRV: sdnn, rmssd, pnn50, lf_power, hf_power, lf_hf_ratio
    """
    # Pobierz IBI w oknie
    window_ibi = [d['IBI'] for d in ibi_data
                  if window_start <= d['TimeStamp'] < window_end]

    if len(window_ibi) < 3:
        return {
            'hrv_sdnn': np.nan,
            'hrv_rmssd': np.nan,
            'hrv_pnn50': np.nan,
            'hrv_lf_power': np.nan,
            'hrv_hf_power': np.nan,
            'hrv_lf_hf_ratio': np.nan
        }

    # Użyj funkcji z bvp_utils (IBI w sekundach)
    metrics = compute_metrics_from_ibi(np.array(window_ibi), ibi_unit='s')

    # Przemapuj nazwy (bvp_ -> hrv_)
    return {
        'hrv_sdnn': metrics.get('bvp_sdnn', np.nan),
        'hrv_rmssd': metrics.get('bvp_rmssd', np.nan),
        'hrv_pnn50': metrics.get('bvp_pnn50', np.nan),
        'hrv_lf_power': metrics.get('bvp_lf_power', np.nan),
        'hrv_hf_power': metrics.get('bvp_hf_power', np.nan),
        'hrv_lf_hf_ratio': metrics.get('bvp_lf_hf_ratio', np.nan)
    }


def process_video_data(video_physio: dict, video_annot: dict) -> pd.DataFrame:
    """
    Przetwarza dane dla jednego wideo.

    Stosuje metodologię "Global Processing, Local Aggregation":
    - Najpierw przetwarza cały sygnał (np. detekcja pików HR)
    - Następnie agreguje do okien 5-sekundowych

    Agregacja sygnałów:
    - EDA: mean, std, max, peaks (tracisz piki stresu przy zwykłej średniej)
    - BVP: std (amplituda), spectral_power, peak_to_peak (NIE średnia - sygnał falowy!)
    - TEMP: mean, slope (zmienia się wolno)
    - ACC: mean (pozycja), std (ruch)
    - HR: mean, std z globalnie obliczonego przebiegu HR
    - HRV: sdnn, rmssd, pnn50, lf/hf z IBI
    """
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

    # =================================================================
    # GLOBAL PROCESSING: Oblicz ciągły przebieg HR dla całego wideo
    # =================================================================
    hr_timeseries = compute_global_hr_timeseries(ibi_data, max_time, fs=1.0)

    # Przetwórz okna 5-sekundowe
    results = []
    window_end = WINDOW_SIZE

    while window_end <= max_time:
        window_start = window_end - WINDOW_SIZE

        record = {
            'seconds': window_end,
            'video_id': video_id
        }

        # -----------------------------------------------------------------
        # AROUSAL i VALENCE (Raw: X=Valence, Y=Arousal, zakres [-1, 1])
        # -----------------------------------------------------------------
        window_annot = [a for a in annot_data if window_start <= a['TimeStamp'] < window_end]
        if window_annot:
            record['arousal'] = np.mean([a['Y_Value'] for a in window_annot])
            record['valence'] = np.mean([a['X_Value'] for a in window_annot])
        else:
            record['arousal'] = np.nan
            record['valence'] = np.nan

        # -----------------------------------------------------------------
        # EDA: mean, std, max, peaks (tracisz piki stresu przy zwykłej średniej!)
        # -----------------------------------------------------------------
        window_eda = np.array([e['EDA'] for e in eda_data
                               if window_start <= e['TimeStamp'] < window_end])
        eda_features = compute_eda_features(window_eda, ORIGINAL_FS)
        record.update(eda_features)

        # -----------------------------------------------------------------
        # BVP: std (amplituda), spectral_power (NIE średnia - sygnał falowy!)
        # -----------------------------------------------------------------
        window_bvp = np.array([b['BVP'] for b in bvp_data
                               if window_start <= b['TimeStamp'] < window_end])
        bvp_features = compute_bvp_features(window_bvp, ORIGINAL_FS)
        record.update(bvp_features)

        # -----------------------------------------------------------------
        # TEMP: mean, slope (zmienia się wolno)
        # -----------------------------------------------------------------
        window_skt = np.array([s['SKT'] for s in skt_data
                               if window_start <= s['TimeStamp'] < window_end])
        window_skt_times = np.array([s['TimeStamp'] for s in skt_data
                                     if window_start <= s['TimeStamp'] < window_end])
        temp_features = compute_temp_features(window_skt, window_skt_times)
        record.update(temp_features)

        # -----------------------------------------------------------------
        # ACC: mean (pozycja ciała), std (intensywność ruchu)
        # -----------------------------------------------------------------
        window_acc = [(a['ACC_X'], a['ACC_Y'], a['ACC_Z']) for a in acc_data
                      if window_start <= a['TimeStamp'] < window_end]
        if window_acc:
            acc_x = np.array([a[0] for a in window_acc])
            acc_y = np.array([a[1] for a in window_acc])
            acc_z = np.array([a[2] for a in window_acc])
        else:
            acc_x, acc_y, acc_z = np.array([]), np.array([]), np.array([])
        acc_features = compute_acc_features(acc_x, acc_y, acc_z)
        record.update(acc_features)

        # -----------------------------------------------------------------
        # HR: mean, std (z globalnie obliczonego przebiegu - Local Aggregation)
        # -----------------------------------------------------------------
        hr_features = compute_hr_window_features(hr_timeseries, window_start, window_end, fs=1.0)
        record.update(hr_features)

        # -----------------------------------------------------------------
        # HRV: metryki zmienności rytmu z IBI (sdnn, rmssd, pnn50, lf/hf)
        # -----------------------------------------------------------------
        hrv_features = compute_ibi_window_features(ibi_data, window_start, window_end)
        record.update(hrv_features)

        results.append(record)
        window_end += WINDOW_SIZE

    return pd.DataFrame(results)


def process_participant(pid: int) -> pd.DataFrame:
    """Przetwarza dane dla jednego uczestnika."""
    print(f"Przetwarzanie uczestnika P{pid}...")

    physio_data = load_physio_data(pid)
    annot_data = load_annotation_data(pid)

    if physio_data is None:
        print(f"  Brak danych fizjologicznych dla P{pid}")
        return None

    if annot_data is None:
        print(f"  Brak danych adnotacji dla P{pid}")
        return None

    video_physio_list = physio_data.get('Video_Physio_RawData', [])
    video_annot_list = annot_data.get('Video_Annotation_RawData', [])

    # Mapowanie VideoID -> dane
    annot_by_video = {v['VideoID']: v for v in video_annot_list}

    all_results = []

    for video_physio in video_physio_list:
        video_id = video_physio['VideoID']
        video_annot = annot_by_video.get(video_id)

        if video_annot is None:
            print(f"  Brak adnotacji dla wideo {video_id}")
            continue

        video_df = process_video_data(video_physio, video_annot)
        if video_df is not None and len(video_df) > 0:
            all_results.append(video_df)

    if not all_results:
        return None

    return pd.concat(all_results, ignore_index=True)


def main():
    """Główna funkcja przetwarzania."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Rozpoczynam przetwarzanie danych CEAP-360VR...")
    print(f"Folder wyjściowy: {OUTPUT_DIR}")

    total_processed = 0

    for pid in range(1, 33):
        result_df = process_participant(pid)

        if result_df is not None and len(result_df) > 0:
            output_file = OUTPUT_DIR / f"P{pid}_merged.csv"
            result_df.to_csv(output_file, index=False)
            print(f"  Zapisano {len(result_df)} rekordów do {output_file}")
            total_processed += 1
        else:
            print(f"  Brak danych do zapisania dla P{pid}")

    print(f"\nPrzetwarzanie zakończone! Przetworzono {total_processed} uczestników.")


if __name__ == "__main__":
    main()

