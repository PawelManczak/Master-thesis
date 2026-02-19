#!/usr/bin/env python3
"""
CEAP-360VR Data Processing Script
Łączy dane fizjologiczne z adnotacjami (valence i arousal).
Sygnały agregowane są do okien 5-sekundowych.

Sygnały z Empatica E4 (wszystkie przeskalowane do 25 Hz w danych Frame):
- ACC: 3-osiowy akcelerometr (x, y, z)
- BVP: fotopletyzmografia
- EDA: przewodnictwo skórne w µS
- HR: tętno
- IBI: inter-beat interval w ms (nieregularne)
- SKT: temperatura skóry (°C)

Adnotacje: ciągłe valence-arousal w skali 1-9, próbkowane 25 Hz
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import signal as scipy_signal

# Import wspólnych funkcji BVP
from bvp_utils import compute_metrics_from_ibi, antialiasing_filter

# Ścieżki
BASE_DIR = Path(__file__).parent.parent.parent.parent  # extracting -> processing -> source -> second part
CEAP_DIR = BASE_DIR / "data" / "CEAP" / "raw" / "CEAP-360VR-Dataset-master" / "CEAP-360VR"
PHYSIO_DIR = CEAP_DIR / "5_PhysioData" / "Raw"  # Używamy Raw dla oryginalnych jednostek
ANNOT_DIR = CEAP_DIR / "3_AnnotationData" / "Frame"  # Adnotacje z Frame (mają dobre wartości)
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
    """Wczytaj dane adnotacji dla danego uczestnika."""
    file_path = ANNOT_DIR / f"P{pid}_Annotation_FrameData.json"
    if not file_path.exists():
        return None

    with open(file_path, 'r') as f:
        data = json.load(f)

    return data['ContinuousAnnotation_FrameData'][0]



def process_video_data(video_physio: dict, video_annot: dict) -> pd.DataFrame:
    """
    Przetwarza dane dla jednego wideo.
    Agreguje dane do okien 5-sekundowych.
    """
    video_id = video_physio['VideoID']

    # Pobierz dane - nazwy kluczy dla Raw data
    acc_data = video_physio.get('ACC_RawData', [])
    eda_data = video_physio.get('EDA_RawData', [])
    hr_data = video_physio.get('HR_RawData', [])
    skt_data = video_physio.get('SKT_RawData', [])
    bvp_data = video_physio.get('BVP_RawData', [])
    ibi_data = video_physio.get('IBI_RawData', [])
    annot_data = video_annot.get('TimeStamp_Valence_Arousal', [])

    if not annot_data:
        return None

    # Znajdź maksymalny czas
    max_time = annot_data[-1]['TimeStamp'] if annot_data else 0

    # Przetwórz okna 5-sekundowe
    results = []
    window_end = WINDOW_SIZE

    while window_end <= max_time:
        window_start = window_end - WINDOW_SIZE

        record = {
            'seconds': window_end,
            'video_id': video_id
        }

        # Arousal i Valence - średnia w oknie
        window_annot = [a for a in annot_data if window_start <= a['TimeStamp'] < window_end]
        if window_annot:
            record['arousal'] = np.mean([a['Arousal'] for a in window_annot])
            record['valence'] = np.mean([a['Valence'] for a in window_annot])
        else:
            record['arousal'] = np.nan
            record['valence'] = np.nan

        # EDA
        window_eda = [e['EDA'] for e in eda_data if window_start <= e['TimeStamp'] < window_end]
        if window_eda:
            eda_filtered = antialiasing_filter(np.array(window_eda), ORIGINAL_FS, 1/WINDOW_SIZE)
            record['eda'] = np.mean(eda_filtered)
            record['eda_var'] = np.var(eda_filtered)
        else:
            record['eda'] = np.nan
            record['eda_var'] = np.nan

        # HR
        window_hr = [h['HR'] for h in hr_data if window_start <= h['TimeStamp'] < window_end]
        if window_hr:
            hr_filtered = antialiasing_filter(np.array(window_hr), ORIGINAL_FS, 1/WINDOW_SIZE)
            record['hr'] = np.mean(hr_filtered)
            record['hr_var'] = np.var(hr_filtered)
        else:
            record['hr'] = np.nan
            record['hr_var'] = np.nan

        # TEMP (SKT)
        window_skt = [s['SKT'] for s in skt_data if window_start <= s['TimeStamp'] < window_end]
        if window_skt:
            skt_filtered = antialiasing_filter(np.array(window_skt), ORIGINAL_FS, 1/WINDOW_SIZE)
            record['temp'] = np.mean(skt_filtered)
            record['temp_var'] = np.var(skt_filtered)
        else:
            record['temp'] = np.nan
            record['temp_var'] = np.nan

        # BVP metrics z IBI (IBI w CEAP jest w sekundach)
        window_ibi = [i['IBI'] for i in ibi_data if window_start <= i['TimeStamp'] < window_end]
        if window_ibi:
            bvp_metrics = compute_metrics_from_ibi(np.array(window_ibi), ibi_unit='s')
            record.update(bvp_metrics)
        else:
            record['bvp_sdnn'] = np.nan
            record['bvp_rmssd'] = np.nan
            record['bvp_pnn50'] = np.nan
            record['bvp_mean_hr'] = np.nan
            record['bvp_mean_ibi'] = np.nan
            record['bvp_lf_power'] = np.nan
            record['bvp_hf_power'] = np.nan
            record['bvp_lf_hf_ratio'] = np.nan

        # BVP
        window_bvp = [b['BVP'] for b in bvp_data if window_start <= b['TimeStamp'] < window_end]
        if window_bvp:
            bvp_filtered = antialiasing_filter(np.array(window_bvp), ORIGINAL_FS, 1/WINDOW_SIZE)
            record['bvp_mean'] = np.mean(bvp_filtered)
            record['bvp_var'] = np.var(bvp_filtered)
        else:
            record['bvp_mean'] = np.nan
            record['bvp_var'] = np.nan

        # ACC
        window_acc = [(a['ACC_X'], a['ACC_Y'], a['ACC_Z']) for a in acc_data
                      if window_start <= a['TimeStamp'] < window_end]
        if window_acc:
            acc_x = [a[0] for a in window_acc]
            acc_y = [a[1] for a in window_acc]
            acc_z = [a[2] for a in window_acc]

            record['acc_x_mean'] = np.mean(acc_x)
            record['acc_x_var'] = np.var(acc_x)
            record['acc_y_mean'] = np.mean(acc_y)
            record['acc_y_var'] = np.var(acc_y)
            record['acc_z_mean'] = np.mean(acc_z)
            record['acc_z_var'] = np.var(acc_z)

            magnitude = np.sqrt(np.array(acc_x)**2 + np.array(acc_y)**2 + np.array(acc_z)**2)
            record['acc_magnitude_mean'] = np.mean(magnitude)
            record['acc_magnitude_var'] = np.var(magnitude)
        else:
            record['acc_x_mean'] = np.nan
            record['acc_x_var'] = np.nan
            record['acc_y_mean'] = np.nan
            record['acc_y_var'] = np.nan
            record['acc_z_mean'] = np.nan
            record['acc_z_var'] = np.nan
            record['acc_magnitude_mean'] = np.nan
            record['acc_magnitude_var'] = np.nan

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
    video_annot_list = annot_data.get('Video_Annotation_FrameData', [])

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

