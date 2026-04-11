#!/usr/bin/env python3
"""
Prepare ARMADA Data from External Annotations

Converts processed data from:
- K-emoCon (external annotations) -> armada_k_emocon_external.csv
- EMBOA (BORIS method II annotations) -> armada_emboa.csv

Uses the same pipeline as prepare_armada_data.py:
- Cross-dataset thresholds (global terciles)
- Arousal/valence normalization + discretization
- Physiological state interval extraction

Key difference: EMBOA physio is simpler (only EDA, HR, TEMP — no BVP/HRV/ACC)
so some ARMADA states won't be present.
"""

import sys
from pathlib import Path

# Add the armada directory to path so we can import from prepare_armada_data
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from prepare_armada_data import (
    SCALE_RANGES,
    DISCRETIZE_THRESHOLDS,
    PHYSIO_VARIABLES,
    PERSONAL_NORM_VARIABLES,
    VARIABLE_NAME_MAPPING,
    normalize_value,
    normalize_personal_minmax,
    extract_state_intervals,
    compute_cross_dataset_thresholds,
    process_participant_data,
    save_armada_format,
    generate_summary_statistics,
    create_combined_dataset,
)

import pandas as pd
import numpy as np

# Paths
BASE_DIR = Path(__file__).parent.parent.parent.parent
KEMOCON_EXTERNAL_PROCESSED = BASE_DIR / "data" / "K-emoCon" / "processed_external"
EMBOA_PROCESSED = BASE_DIR / "data" / "EMBOA" / "processed"
OUTPUT_DIR = BASE_DIR / "data" / "armada_ready"

# Add scale ranges for EMBOA
# EMBOA arousal/valence are already normalized to [0, 1] by Russell mapping
# So we just pass them through (identity normalization)
SCALE_RANGES['EMBOA'] = {'arousal': (0, 1), 'valence': (0, 1)}

# K-emoCon external has same scale as K-emoCon self (1-5)
SCALE_RANGES['K-emoCon-ext'] = {'arousal': (1, 5), 'valence': (1, 5)}


def process_dataset_files(
    dataset_name: str,
    data_dir: Path,
    cross_dataset_thresholds: dict = None
) -> list:
    """
    Processes all merged CSV files from a dataset directory.

    Returns list of DataFrames in ARMADA format.
    """
    results = []

    csv_files = list(data_dir.glob("*_merged.csv"))
    print(f"\n{dataset_name}: Found {len(csv_files)} files")

    if cross_dataset_thresholds is not None:
        global_thresholds = cross_dataset_thresholds
        print(f"  Using CROSS-DATASET thresholds ({len(global_thresholds)} variables)")
    else:
        global_thresholds = {}

    for csv_file in sorted(csv_files):
        participant_id = csv_file.stem.replace("_merged", "")

        try:
            df = pd.read_csv(csv_file)
            result = process_participant_data(
                df, dataset_name, participant_id, global_thresholds
            )

            if result is not None and len(result) > 0:
                results.append(result)
                print(f"  {participant_id}: {len(result)} intervals")
            else:
                print(f"  {participant_id}: No data")
        except Exception as e:
            print(f"  {participant_id}: Error - {e}")

    return results


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PREPARING EXTERNAL ANNOTATIONS DATA FOR ARMADA")
    print("=" * 60)

    # Compute cross-dataset thresholds from both datasets
    dataset_dirs = {}
    if KEMOCON_EXTERNAL_PROCESSED.exists():
        dataset_dirs['K-emoCon-ext'] = KEMOCON_EXTERNAL_PROCESSED
    if EMBOA_PROCESSED.exists():
        dataset_dirs['EMBOA'] = EMBOA_PROCESSED

    if not dataset_dirs:
        print("ERROR: No processed datasets found!")
        print(f"  Expected K-emoCon external: {KEMOCON_EXTERNAL_PROCESSED}")
        print(f"  Expected EMBOA: {EMBOA_PROCESSED}")
        print("\nRun processing scripts first:")
        print("  python datasets/K_emoCon_external.py")
        print("  python datasets/EMBOA_external.py")
        sys.exit(1)

    cross_thresholds = compute_cross_dataset_thresholds(dataset_dirs)

    all_results = []

    # Process K-emoCon external
    if KEMOCON_EXTERNAL_PROCESSED.exists():
        kemocon_results = process_dataset_files(
            'K-emoCon-ext', KEMOCON_EXTERNAL_PROCESSED, cross_thresholds
        )
        all_results.extend(kemocon_results)

    # Process EMBOA
    if EMBOA_PROCESSED.exists():
        emboa_results = process_dataset_files(
            'EMBOA', EMBOA_PROCESSED, cross_thresholds
        )
        all_results.extend(emboa_results)

    if not all_results:
        print("\nNo data to process!")
        return

    print("\n" + "=" * 60)
    print("COMBINING DATA")
    print("=" * 60)

    combined_df = create_combined_dataset(all_results)
    print(f"Total: {len(combined_df)} intervals from "
          f"{combined_df['client_id'].nunique()} participants")

    # Save
    print("\n" + "=" * 60)
    print("SAVING DATA")
    print("=" * 60)

    # Save per-dataset files
    for dataset in ['K-emoCon-ext', 'EMBOA']:
        dataset_df = combined_df[combined_df['dataset'] == dataset]
        if len(dataset_df) > 0:
            fname = f"armada_{dataset.lower().replace('-', '_')}.csv"
            save_armada_format(dataset_df, OUTPUT_DIR / fname)

    # Save combined
    save_armada_format(
        combined_df,
        OUTPUT_DIR / "armada_external_combined.csv"
    )

    combined_df.to_csv(
        OUTPUT_DIR / "armada_external_full_metadata.csv", index=False
    )
    print(f"Saved: {OUTPUT_DIR / 'armada_external_full_metadata.csv'}")

    # Statistics
    print("\n" + "=" * 60)
    print("STATISTICS")
    print("=" * 60)

    stats = generate_summary_statistics(combined_df)
    print(f"Total intervals: {stats['total_intervals']}")
    print(f"Participants: {stats['unique_clients']}")
    print(f"Unique states: {stats['unique_states']}")
    print(f"Avg interval duration: {stats['avg_interval_duration']:.2f}s")

    print("\nStates distribution (top 15):")
    for state, count in sorted(
        stats['states_distribution'].items(), key=lambda x: -x[1]
    )[:15]:
        print(f"  {state}: {count}")

    print("\nDatasets distribution:")
    for dataset, count in stats['datasets_distribution'].items():
        print(f"  {dataset}: {count}")

    print("\n" + "=" * 60)
    print("EXTERNAL ANNOTATIONS ARMADA PREPARATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
