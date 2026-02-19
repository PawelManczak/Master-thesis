"""
CASE Dataset Processing Script
Przepisanie skryptów MATLAB na Python.
Łączy dane fizjologiczne z adnotacjami (valence i arousal).
Sygnały agregowane są do okien 5-sekundowych z filtracją antyaliasingową.

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

Output: seconds, arousal, valence, eda (gsr), hr (z ECG), temp (skt) + HRV metrics
"""

import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from bvp_utils import compute_metrics_from_ecg, antialiasing_filter

warnings.filterwarnings('ignore')


BASE_DIR = Path(__file__).parent.parent.parent.parent  # extracting -> processing -> source -> second part
RAW_DIR = BASE_DIR / "data" / "CASE" / "raw" / "case_dataset"
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
    Zoptymalizowana wersja - bez pełnej interpolacji, bezpośrednie przetwarzanie okien.
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

    # ==================== DOWNSAMPLING DO 5 SEKUND ====================

    window_size_ms = 5000  # 5 sekund
    results = []

    # Iteruj po oknach 5-sekundowych
    n_windows = int(max_time / window_size_ms)

    for i, window_start in enumerate(range(0, int(max_time), window_size_ms)):
        window_end = window_start + window_size_ms

        # Indeksy dla danych DAQ
        daq_mask = (daqtime >= window_start) & (daqtime < window_end)

        # Indeksy dla danych JS
        js_mask = (jstime >= window_start) & (jstime < window_end)

        if not np.any(daq_mask) or not np.any(js_mask):
            continue

        # Średnie adnotacje
        window_valence = np.mean(valence[js_mask])
        window_arousal = np.mean(arousal[js_mask])

        # EDA (GSR): filtracja antyaliasingowa + średnia i wariancja
        window_gsr = gsr[daq_mask]
        if len(window_gsr) >= 10:
            gsr_filtered = antialiasing_filter(window_gsr, FS_PHYSIO, TARGET_FS)
            window_eda = np.mean(gsr_filtered)
            window_eda_var = np.var(gsr_filtered)
        else:
            window_eda = np.mean(window_gsr) if len(window_gsr) > 0 else np.nan
            window_eda_var = np.var(window_gsr) if len(window_gsr) > 0 else np.nan

        # Temp (SKT): filtracja antyaliasingowa + średnia i wariancja
        window_skt = skt[daq_mask]
        if len(window_skt) >= 10:
            skt_filtered = antialiasing_filter(window_skt, FS_PHYSIO, TARGET_FS)
            window_temp = np.mean(skt_filtered)
            window_temp_var = np.var(skt_filtered)
        else:
            window_temp = np.mean(window_skt) if len(window_skt) > 0 else np.nan
            window_temp_var = np.var(window_skt) if len(window_skt) > 0 else np.nan

        # BVP metrics z ECG
        window_ecg = ecg[daq_mask]
        bvp_metrics = compute_metrics_from_ecg(window_ecg, FS_PHYSIO)

        # HR variance
        window_hr_var = np.var(bvp_metrics['bvp_sdnn']) if not np.isnan(bvp_metrics['bvp_sdnn']) else np.nan

        record = {
            'seconds': (window_end / 1000),
            'arousal': window_arousal,
            'valence': window_valence,
            'eda': window_eda,
            'eda_var': window_eda_var,
            'hr': bvp_metrics['bvp_mean_hr'],
            'hr_var': bvp_metrics['bvp_sdnn'] ** 2 if not np.isnan(bvp_metrics['bvp_sdnn']) else np.nan,
            'temp': window_temp,
            'temp_var': window_temp_var,
            'bvp_sdnn': bvp_metrics['bvp_sdnn'],
            'bvp_rmssd': bvp_metrics['bvp_rmssd'],
            'bvp_pnn50': bvp_metrics['bvp_pnn50'],
            'bvp_mean_hr': bvp_metrics['bvp_mean_hr'],
            'bvp_mean_ibi': bvp_metrics['bvp_mean_ibi'],
            'bvp_lf_power': bvp_metrics['bvp_lf_power'],
            'bvp_hf_power': bvp_metrics['bvp_hf_power'],
            'bvp_lf_hf_ratio': bvp_metrics['bvp_lf_hf_ratio']
        }

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

