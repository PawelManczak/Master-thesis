"""
CASE Dataset Processing Script
Przepisanie skryptów MATLAB na Python.
Łączy dane fizjologiczne z adnotacjami (valence i arousal).
Sygnały agregowane są do okien 5-sekundowych.

Dane wejściowe (raw):
- Fizjologiczne (1000 Hz): daqtime (s), ecg, bvp, gsr, rsp, skt, emg_zygo, emg_coru, emg_trap (w voltach)
- Adnotacje (20 Hz): jstime (s), valence, arousal (zakres -26225 do +26225)

Transformacje (na podstawie s03_v2_interTransformData.m):
- ECG: (V - 2.8) / 50 * 1000 -> mV
- BVP: 58.962 * V - 115.09 -> %
- GSR (EDA): 24 * V - 49.2 -> µS
- RSP: 58.923 * V - 115.01 -> %
- SKT (temp): 21.341 * V - 32.085 -> °C
- EMG: (V - 2.0) / 4000 * 1000000 -> µV
- Valence/Arousal: 0.5 + 9 * (raw + 26225) / 52450 -> [0.5, 9.5]

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
| ECG    | Sygnał falowy!                  | HR z globalnego przebiegu             |
| TEMP   | Zmienia się wolno, OK           | mean, slope (trend)                   |

Output: seconds, arousal, valence, eda_mean, eda_std, eda_max, eda_peaks,
        hr_mean, hr_std, temp_mean, temp_slope, hrv_* metrics
"""

import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import signal as scipy_signal
from scipy.stats import linregress
from scipy.interpolate import interp1d
from bvp_utils import compute_metrics_from_ibi

warnings.filterwarnings('ignore')


BASE_DIR = Path(__file__).parent.parent.parent.parent  # extracting -> processing -> source -> second part
RAW_DIR = BASE_DIR / "data" / "CASE" / "raw" / "case_dataset-master" / "data" / "raw"
METADATA_DIR = BASE_DIR / "data" / "CASE" / "raw" / "case_dataset-master" / "metadata"
PHYSIO_DIR = RAW_DIR / "physiological"
ANNOTATIONS_DIR = RAW_DIR / "annotations"
OUTPUT_DIR = BASE_DIR / "data" / "CASE" / "processed"

# Stałe z MATLAB
VID_SEQ_DUR = 2451583.333  # Całkowity czas sekwencji w ms
FS_PHYSIO = 1000.0  # Hz (1 próbka / 1 ms)
FS_ANNOT = 20.0     # Hz (1 próbka / 50 ms)
TARGET_FS = 0.2     # 1 próbka / 5 sekund


def load_videos_duration() -> dict:
    """Wczytaj informacje o czasie trwania wideo."""
    file_path = METADATA_DIR / "videos_duration.txt"

    # Mapowanie nazw wideo na ID (z videos.xlsx)
    video_name_to_id = {
        'amusing-1': 1, 'amusing-2': 2, 'bluVid': 3, 'boring-1': 4, 'boring-2': 5,
        'endVid': 6, 'relaxed-1': 7, 'relaxed-2': 8, 'scary-1': 9, 'scary-2': 10,
        'startVid': 11
    }

    videos_duration = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:  # Pomiń nagłówek
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                name = parts[0].strip('"')
                duration = float(parts[1].strip())
                if name in video_name_to_id:
                    videos_duration[video_name_to_id[name]] = duration

    return videos_duration


def load_seqs_order() -> dict:
    """Wczytaj kolejność sekwencji dla każdego uczestnika."""
    file_path = METADATA_DIR / "seqs_order.txt"

    video_name_to_id = {
        'amusing-1': 1, 'amusing-2': 2, 'bluVid': 3, 'boring-1': 4, 'boring-2': 5,
        'endVid': 6, 'relaxed-1': 7, 'relaxed-2': 8, 'scary-1': 9, 'scary-2': 10,
        'startVid': 11
    }

    seqs_order = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()
        # Pierwsza linia to nagłówki (sub1, sub2, ...)
        # Kolejne linie to sekwencje wideo
        n_subs = 30
        for sub_id in range(1, n_subs + 1):
            seqs_order[sub_id] = []

        for line in lines[1:]:  # Pomiń nagłówek
            parts = line.strip().split('\t')
            for sub_id, part in enumerate(parts[:n_subs], 1):
                name = part.strip('"')
                if name in video_name_to_id:
                    seqs_order[sub_id].append(video_name_to_id[name])

    return seqs_order


def f_label_data(timevec: np.ndarray, vidseq_order: list, vids_duration: dict) -> np.ndarray:
    """
    Przepisanie f_labelData.m na Python.
    Zwraca wektor etykiet (video-ID) dla danych czasowych.
    """
    timevec_nrow = len(timevec)
    labels = np.zeros(timevec_nrow, dtype=int)

    n_vids_in_seq = len(vidseq_order)

    last_time_found = 0
    video_pos_start = 0

    for j in range(n_vids_in_seq):
        if j == 0:
            last_time_found = 0
            video_pos_start = 0
        else:
            last_time_found = video_time_end
            video_pos_start = video_pos_end + 1

        vid_searched = vidseq_order[j]
        vid_matched_dur = vids_duration.get(vid_searched, 0)

        time2match = last_time_found + vid_matched_dur

        # Znajdź najbliższy czas
        tmp_timeVec = np.abs(timevec - time2match)
        loc_time2match = np.argmin(tmp_timeVec)

        video_pos_end = loc_time2match
        video_time_end = timevec[video_pos_end]

        # Aktualizuj etykiety
        if video_pos_start < timevec_nrow:
            labels[video_pos_start:video_pos_end + 1] = vid_searched

    return labels


def transform_ecg(voltage: np.ndarray) -> np.ndarray:
    """Transformacja ECG: V -> mV"""
    return ((voltage - 2.8) / 50) * 1000


def transform_bvp(voltage: np.ndarray) -> np.ndarray:
    """Transformacja BVP: V -> %"""
    return (58.962 * voltage) - 115.09


def transform_gsr(voltage: np.ndarray) -> np.ndarray:
    """Transformacja GSR (EDA): V -> µS"""
    return (24 * voltage) - 49.2


def transform_rsp(voltage: np.ndarray) -> np.ndarray:
    """Transformacja RSP: V -> %"""
    return (58.923 * voltage) - 115.01


def transform_skt(voltage: np.ndarray) -> np.ndarray:
    """Transformacja SKT (temp): V -> °C"""
    return (21.341 * voltage) - 32.085


def transform_emg(voltage: np.ndarray) -> np.ndarray:
    """Transformacja EMG: V -> µV"""
    return ((voltage - 2.0) / 4000) * 1000000


def transform_annotation(raw_value: np.ndarray) -> np.ndarray:
    """Transformacja adnotacji: [-26225, 26225] -> [0.5, 9.5]"""
    return 0.5 + 9 * (raw_value + 26225) / 52450


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
    if len(values) >= 10:
        try:
            # Piki muszą być wyższe niż 0.01 µS od baseline
            threshold = np.mean(values) + 0.01
            min_distance = int(fs * 0.5)  # Minimum 0.5s między pikami
            peaks, _ = scipy_signal.find_peaks(values, height=threshold, distance=max(1, min_distance))
            peaks_count = len(peaks)
        except:
            peaks_count = 0

    return {
        'eda_mean': np.mean(values),
        'eda_std': np.std(values, ddof=1) if len(values) > 1 else 0.0,
        'eda_max': np.max(values),
        'eda_peaks': peaks_count
    }


def compute_temp_features(values: np.ndarray, fs: float = None) -> dict:
    """
    Oblicz cechy temperatury w oknie czasowym.

    Temperatura zmienia się wolno - średnia jest OK.
    Dodatkowo: slope (trend) - czy rośnie czy spada.

    Args:
        values: wartości temperatury w oknie
        fs: częstotliwość próbkowania (opcjonalne)

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
            x = np.arange(len(values))
            slope, _, _, _, _ = linregress(x, values)
            temp_slope = slope
        except:
            temp_slope = 0.0

    return {
        'temp_mean': temp_mean,
        'temp_slope': temp_slope
    }


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


def compute_global_hr_from_ecg(ecg_data: np.ndarray, daqtime: np.ndarray, fs: float) -> np.ndarray:
    """
    GLOBAL PROCESSING: Oblicz ciągły przebieg HR z całego sygnału ECG.

    Złota zasada: "Nie licz HR w oknie, uśredniaj HR w oknie"

    Pipeline:
    1. Wykryj piki R na całym nagraniu
    2. Oblicz IBI (RR intervals)
    3. Oblicz chwilowe HR dla każdego IBI
    4. Interpoluj do równomiernej siatki czasowej (1 Hz)

    Args:
        ecg_data: sygnał ECG
        daqtime: wektor czasu (ms)
        fs: częstotliwość próbkowania ECG

    Returns:
        tuple (time_grid_ms, hr_timeseries) - czas w ms i HR w BPM
    """
    # Wykryj piki R
    r_peaks = detect_r_peaks_ecg(ecg_data, fs)

    if len(r_peaks) < 2:
        return np.array([]), np.array([])

    # Oblicz RR intervals (IBI) w ms
    rr_intervals = np.diff(r_peaks) / fs * 1000

    # Timestamps dla RR intervals (średnia między kolejnymi pikami)
    rr_timestamps = (daqtime[r_peaks[:-1]] + daqtime[r_peaks[1:]]) / 2

    # Filtruj nieprawidłowe RR (300-2000 ms = 30-200 BPM)
    valid_mask = (rr_intervals > 300) & (rr_intervals < 2000)
    rr_intervals = rr_intervals[valid_mask]
    rr_timestamps = rr_timestamps[valid_mask]

    if len(rr_intervals) < 2:
        return np.array([]), np.array([])

    # Oblicz chwilowe HR (BPM)
    hr_values = 60000 / rr_intervals

    # Stwórz równomierną siatkę czasową (1 Hz = 1000 ms)
    time_grid = np.arange(daqtime.min(), daqtime.max(), 1000)

    if len(time_grid) == 0:
        return np.array([]), np.array([])

    # Interpoluj HR do siatki czasowej
    try:
        f_interp = interp1d(rr_timestamps, hr_values, kind='linear',
                           bounds_error=False, fill_value=np.nan)
        hr_timeseries = f_interp(time_grid)
    except:
        hr_timeseries = np.full(len(time_grid), np.nan)

    return time_grid, hr_timeseries


def compute_hr_window_features(time_grid: np.ndarray, hr_timeseries: np.ndarray,
                                window_start: float, window_end: float) -> dict:
    """
    LOCAL AGGREGATION: Oblicz cechy HR w oknie czasowym z globalnego przebiegu.

    Args:
        time_grid: siatka czasowa (ms)
        hr_timeseries: ciągły przebieg HR
        window_start: początek okna (ms)
        window_end: koniec okna (ms)

    Returns:
        dict z: hr_mean, hr_std
    """
    if len(hr_timeseries) == 0:
        return {'hr_mean': np.nan, 'hr_std': np.nan}

    # Znajdź wartości HR w oknie
    mask = (time_grid >= window_start) & (time_grid < window_end)
    hr_window = hr_timeseries[mask]

    # Usuń NaN
    hr_valid = hr_window[~np.isnan(hr_window)]

    if len(hr_valid) == 0:
        return {'hr_mean': np.nan, 'hr_std': np.nan}

    return {
        'hr_mean': np.mean(hr_valid),
        'hr_std': np.std(hr_valid, ddof=1) if len(hr_valid) > 1 else 0.0
    }


def compute_ibi_window_features(r_peaks: np.ndarray, daqtime: np.ndarray, fs: float,
                                 window_start: float, window_end: float) -> dict:
    """
    Oblicz metryki HRV z IBI w oknie czasowym.

    Args:
        r_peaks: indeksy pików R
        daqtime: wektor czasu (ms)
        fs: częstotliwość próbkowania
        window_start: początek okna (ms)
        window_end: koniec okna (ms)

    Returns:
        dict z metrykami HRV
    """
    if len(r_peaks) < 3:
        return {
            'hrv_sdnn': np.nan,
            'hrv_rmssd': np.nan,
            'hrv_pnn50': np.nan,
            'hrv_lf_power': np.nan,
            'hrv_hf_power': np.nan,
            'hrv_lf_hf_ratio': np.nan
        }

    # Znajdź piki R w oknie
    peak_times = daqtime[r_peaks]
    mask = (peak_times >= window_start) & (peak_times < window_end)
    window_peaks = r_peaks[mask]

    if len(window_peaks) < 3:
        return {
            'hrv_sdnn': np.nan,
            'hrv_rmssd': np.nan,
            'hrv_pnn50': np.nan,
            'hrv_lf_power': np.nan,
            'hrv_hf_power': np.nan,
            'hrv_lf_hf_ratio': np.nan
        }

    # Oblicz RR intervals w oknie (ms)
    rr_intervals = np.diff(window_peaks) / fs * 1000

    # Filtruj nieprawidłowe
    valid_rr = rr_intervals[(rr_intervals > 300) & (rr_intervals < 2000)]

    if len(valid_rr) < 3:
        return {
            'hrv_sdnn': np.nan,
            'hrv_rmssd': np.nan,
            'hrv_pnn50': np.nan,
            'hrv_lf_power': np.nan,
            'hrv_hf_power': np.nan,
            'hrv_lf_hf_ratio': np.nan
        }

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


def load_raw_physiological(subject_id: int) -> pd.DataFrame:
    """Wczytaj surowe dane fizjologiczne."""
    file_path = PHYSIO_DIR / f"sub{subject_id}_DAQ.txt"
    if not file_path.exists():
        return None

    columns = ['daqtime', 'ecg', 'bvp', 'gsr', 'rsp', 'skt', 'emg_zygo', 'emg_coru', 'emg_trap']
    df = pd.read_csv(file_path, sep='\t', header=None, names=columns)
    return df


def load_raw_annotations(subject_id: int) -> pd.DataFrame:
    """Wczytaj surowe dane adnotacji."""
    file_path = ANNOTATIONS_DIR / f"sub{subject_id}_joystick.txt"
    if not file_path.exists():
        return None

    columns = ['jstime', 'valence_raw', 'arousal_raw']
    df = pd.read_csv(file_path, sep='\t', header=None, names=columns)
    return df


def process_participant(subject_id: int, vids_duration: dict, seqs_order: dict) -> pd.DataFrame:
    """
    Przetwarza dane dla jednego uczestnika.

    Stosuje metodologię "Global Processing, Local Aggregation":
    - Najpierw przetwarza cały sygnał ECG (detekcja pików R)
    - Następnie agreguje do okien 5-sekundowych

    Agregacja sygnałów:
    - EDA: mean, std, max, peaks (tracisz piki stresu przy zwykłej średniej)
    - TEMP: mean, slope (zmienia się wolno)
    - HR: mean, std z globalnie obliczonego przebiegu HR
    - HRV: sdnn, rmssd, pnn50, lf/hf z IBI
    """
    print(f"Przetwarzanie uczestnika sub{subject_id}...", flush=True)

    # Wczytaj surowe dane
    raw_physio = load_raw_physiological(subject_id)
    raw_annot = load_raw_annotations(subject_id)

    if raw_physio is None:
        print(f"  Brak danych fizjologicznych dla sub{subject_id}")
        return None

    if raw_annot is None:
        print(f"  Brak danych adnotacji dla sub{subject_id}")
        return None

    print(f"  Wczytano {len(raw_physio)} próbek fizjo i {len(raw_annot)} próbek adnotacji", flush=True)

    # Konwertuj czas z sekund na ms
    daqtime = raw_physio['daqtime'].values * 1000
    jstime = raw_annot['jstime'].values * 1000

    # Transformuj dane fizjologiczne (w voltach -> jednostki)
    ecg = transform_ecg(raw_physio['ecg'].values)      # mV
    gsr = transform_gsr(raw_physio['gsr'].values)      # µS (EDA)
    skt = transform_skt(raw_physio['skt'].values)      # °C (temp)

    # Transformuj adnotacje
    valence = transform_annotation(raw_annot['valence_raw'].values)
    arousal = transform_annotation(raw_annot['arousal_raw'].values)

    # Znajdź zakres czasowy danych
    max_time = min(daqtime.max(), jstime.max(), VID_SEQ_DUR)

    # =================================================================
    # GLOBAL PROCESSING: Przetwórz cały sygnał ECG
    # =================================================================
    print(f"  Global Processing: detekcja pików R...", flush=True)

    # Wykryj piki R na całym nagraniu
    r_peaks = detect_r_peaks_ecg(ecg, FS_PHYSIO)
    print(f"  Wykryto {len(r_peaks)} pików R", flush=True)

    # Oblicz globalny przebieg HR
    time_grid, hr_timeseries = compute_global_hr_from_ecg(ecg, daqtime, FS_PHYSIO)

    # =================================================================
    # LOCAL AGGREGATION: Okna 5-sekundowe
    # =================================================================
    window_size_ms = 5000  # 5 sekund
    results = []
    n_windows = int(max_time / window_size_ms)

    for i, window_start in enumerate(range(0, int(max_time), window_size_ms)):
        window_end = window_start + window_size_ms

        # Indeksy dla danych DAQ
        daq_mask = (daqtime >= window_start) & (daqtime < window_end)

        # Indeksy dla danych JS
        js_mask = (jstime >= window_start) & (jstime < window_end)

        if not np.any(daq_mask) or not np.any(js_mask):
            continue

        record = {
            'seconds': (window_end / 1000)
        }

        # -----------------------------------------------------------------
        # AROUSAL i VALENCE
        # -----------------------------------------------------------------
        record['arousal'] = np.mean(arousal[js_mask])
        record['valence'] = np.mean(valence[js_mask])

        # -----------------------------------------------------------------
        # EDA: mean, std, max, peaks (tracisz piki stresu przy zwykłej średniej!)
        # -----------------------------------------------------------------
        window_gsr = gsr[daq_mask]
        eda_features = compute_eda_features(window_gsr, FS_PHYSIO)
        record.update(eda_features)

        # -----------------------------------------------------------------
        # TEMP: mean, slope (zmienia się wolno)
        # -----------------------------------------------------------------
        window_skt = skt[daq_mask]
        temp_features = compute_temp_features(window_skt, FS_PHYSIO)
        record.update(temp_features)

        # -----------------------------------------------------------------
        # HR: mean, std (z globalnie obliczonego przebiegu - Local Aggregation)
        # -----------------------------------------------------------------
        hr_features = compute_hr_window_features(time_grid, hr_timeseries, window_start, window_end)
        record.update(hr_features)

        # -----------------------------------------------------------------
        # HRV: metryki zmienności rytmu z IBI (sdnn, rmssd, pnn50, lf/hf)
        # -----------------------------------------------------------------
        hrv_features = compute_ibi_window_features(r_peaks, daqtime, FS_PHYSIO, window_start, window_end)
        record.update(hrv_features)

        results.append(record)

        # Progress
        if (i + 1) % 100 == 0:
            print(f"    Okno {i+1}/{n_windows}", flush=True)

    if len(results) == 0:
        return None

    return pd.DataFrame(results)


def main():
    """Główna funkcja przetwarzania."""
    # Utwórz folder wyjściowy
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Rozpoczynam przetwarzanie danych CASE...", flush=True)
    print(f"Dane źródłowe: {PHYSIO_DIR}", flush=True)
    print(f"Folder wyjściowy: {OUTPUT_DIR}", flush=True)

    total_processed = 0

    for subject_id in range(1, 31):
        result_df = process_participant(subject_id, {}, {})

        if result_df is not None and len(result_df) > 0:
            output_file = OUTPUT_DIR / f"sub{subject_id}_merged.csv"
            result_df.to_csv(output_file, index=False)
            print(f"  Zapisano {len(result_df)} rekordów do {output_file}", flush=True)
            total_processed += 1
        else:
            print(f"  Brak danych do zapisania dla sub{subject_id}", flush=True)

    print(f"\nPrzetwarzanie zakończone! Przetworzono {total_processed} uczestników.", flush=True)


if __name__ == "__main__":
    main()

