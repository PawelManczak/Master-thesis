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

# Import wspólnych funkcji do obliczania cech
from feature_utils import (
    compute_eda_features,
    compute_bvp_features,
    compute_temp_features,
    compute_acc_features,
    compute_global_hr_from_ibi,
    compute_hr_window_features,
    compute_hrv_window_features
)

BASE_DIR = Path(__file__).parent.parent.parent.parent  # extracting -> processing -> source -> second part
CEAP_DIR = BASE_DIR / "data" / "CEAP" / "raw" / "CEAP-360VR-Dataset-master" / "CEAP-360VR"
PHYSIO_DIR = CEAP_DIR / "5_PhysioData" / "Raw"  # Używamy Raw dla oryginalnych jednostek
ANNOT_DIR = CEAP_DIR / "3_AnnotationData" / "Raw"  # Adnotacje Raw z zakresem [-1, 1]
OUTPUT_DIR = BASE_DIR / "data" / "CEAP" / "processed"

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


def compute_global_hr_timeseries_ceap(ibi_data: list, max_time: float) -> tuple:
    """
    Wrapper dla compute_global_hr_from_ibi dla formatu CEAP.

    Args:
        ibi_data: lista dict z 'TimeStamp' i 'IBI' (IBI w sekundach)
        max_time: maksymalny czas nagrania

    Returns:
        tuple (time_grid, hr_timeseries)
    """
    if not ibi_data or max_time <= 0:
        return np.array([]), np.array([])

    timestamps = np.array([d['TimeStamp'] for d in ibi_data])
    ibi_values = np.array([d['IBI'] for d in ibi_data])  # IBI w sekundach

    return compute_global_hr_from_ibi(timestamps, ibi_values, max_time, ibi_unit='s')


def compute_hrv_window_features_ceap(ibi_data: list, window_start: float, window_end: float) -> dict:
    """
    Wrapper dla compute_hrv_window_features dla formatu CEAP.

    Args:
        ibi_data: lista dict z 'TimeStamp' i 'IBI'
        window_start: początek okna
        window_end: koniec okna

    Returns:
        dict z metrykami HRV
    """
    if not ibi_data:
        return {
            'hrv_sdnn': np.nan, 'hrv_rmssd': np.nan, 'hrv_pnn50': np.nan,
            'hrv_lf_power': np.nan, 'hrv_hf_power': np.nan, 'hrv_lf_hf_ratio': np.nan
        }

    timestamps = np.array([d['TimeStamp'] for d in ibi_data])
    ibi_values = np.array([d['IBI'] for d in ibi_data])  # IBI w sekundach

    return compute_hrv_window_features(timestamps, ibi_values, window_start, window_end, ibi_unit='s')


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
    time_grid, hr_timeseries = compute_global_hr_timeseries_ceap(ibi_data, max_time)

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
        hr_features = compute_hr_window_features(time_grid, hr_timeseries, window_start, window_end)
        record.update(hr_features)

        # -----------------------------------------------------------------
        # HRV: metryki zmienności rytmu z IBI (sdnn, rmssd, pnn50, lf/hf)
        # -----------------------------------------------------------------
        hrv_features = compute_hrv_window_features_ceap(ibi_data, window_start, window_end)
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

