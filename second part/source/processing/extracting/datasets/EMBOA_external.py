"""
EMBOA Data Processing Script (External Annotations — BORIS method II)

Processes EMBOA sub-datasets (GUT, ITU-YU, MAAP):
- Physiological input: _input.csv (EDA, TEMP, HR, FR) at 1 Hz
- External annotations: _BORIS_method_II.csv (Happy, Sad, Scared,
  Disgusted, Surprised, Angry — percentage agreement) at 1 Hz

Aggregation to 5-second windows (compatible with K-emoCon @ 5s):
- Physio: mean EDA, mean TEMP, mean HR per 5s window
- Annotations: mean % per emotion -> mapped to arousal/valence
  using Russell's circumplex model

Russell's circumplex mapping (Posner, Russell, Peterson 2005):
  Happy:     valence +1,   arousal +0.5
  Sad:       valence -1,   arousal -1
  Scared:    valence -1,   arousal +1
  Disgusted: valence -1,   arousal -0.5
  Surprised: valence +0.2, arousal +1
  Angry:     valence -1,   arousal +1
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
import sys

# Paths
BASE_DIR = Path(__file__).parent.parent.parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "EMBOA"
OUTPUT_DIR = BASE_DIR / "data" / "EMBOA" / "processed"

# EMBOA sub-datasets
SUBDATASETS = ["MAAP", "GUT", "ITU-YU"]

# Aggregation window (seconds) — matches K-emoCon annotation interval
WINDOW_SEC = 5

# ===========================================================================
# Russell's circumplex model mapping for BORIS method II emotions
# Values are weights applied to percentage agreement (0–100%)
# ===========================================================================
# Format: emotion -> (valence_weight, arousal_weight)
RUSSELL_MAPPING = {
    'Happy':     (+1.0, +0.5),
    'Sad':       (-1.0, -1.0),
    'Scared':    (-1.0, +1.0),
    'Disgusted': (-1.0, -0.5),
    'Surprised': (+0.2, +1.0),
    'Angry':     (-1.0, +1.0),
}

EMOTIONS = list(RUSSELL_MAPPING.keys())


def compute_arousal_valence(row: pd.Series) -> dict:
    """
    Computes arousal and valence from BORIS method II emotion % values.

    Uses Russell's circumplex model weights.
    Raw values are in range [-100, +100], normalized to [0, 1].
    """
    valence_raw = 0.0
    arousal_raw = 0.0
    total_weight = 0.0

    for emo, (v_w, a_w) in RUSSELL_MAPPING.items():
        pct = row.get(emo, 0.0)
        if pd.isna(pct):
            pct = 0.0
        # pct is 0-100 scale (percentage agreement among annotators)
        valence_raw += v_w * pct
        arousal_raw += a_w * pct
        total_weight += abs(pct)

    # Normalize: theoretical range is [-100, +100] -> map to [0, 1]
    # Using theoretical max: if all annotators agree on one max-weight emotion
    # Max = 100 * max(|weight|) = 100. So range ~ [-100, 100]
    valence_norm = (valence_raw + 100) / 200  # [-100,+100] -> [0,1]
    arousal_norm = (arousal_raw + 100) / 200

    # Clamp to [0, 1]
    valence_norm = np.clip(valence_norm, 0.0, 1.0)
    arousal_norm = np.clip(arousal_norm, 0.0, 1.0)

    return {'arousal': arousal_norm, 'valence': valence_norm}


def discover_files(subdataset_dir: Path, subdataset: str) -> dict:
    """
    Discovers all (subject, clip) pairs that have both _input.csv
    and _BORIS_method_II.csv files.

    Returns: {(subject, clip): {'input': Path, 'boris': Path}}
    """
    pattern = re.compile(
        rf'^{re.escape(subdataset)}_S(\d+)_C(\d+)_(input|BORIS_method_II)\.csv$'
    )

    files = {}
    for f in sorted(subdataset_dir.iterdir()):
        m = pattern.match(f.name)
        if not m:
            continue
        subject, clip, ftype = m.group(1), m.group(2), m.group(3)
        key = (subject, clip)
        if key not in files:
            files[key] = {}
        if ftype == 'input':
            files[key]['input'] = f
        else:
            files[key]['boris'] = f

    # Keep only complete pairs
    complete = {k: v for k, v in files.items() if 'input' in v and 'boris' in v}
    return complete


def process_emboa_file(subdataset: str, subject: str, clip: str,
                       input_path: Path = None, boris_path: Path = None) -> pd.DataFrame:
    """
    Processes a single EMBOA recording (one subject x one clip).

    1. Reads _input.csv (EDA, TEMP, HR, FR) at 1 Hz
    2. Reads _BORIS_method_II.csv (6 emotions x %) at 1 Hz
    3. Merges on row index (both are time-aligned 1 Hz)
    4. Aggregates to 5s windows
    5. Maps emotions -> arousal/valence

    Returns: DataFrame with columns [seconds, arousal, valence,
             eda_mean, temp_mean, hr_mean, window_type, ...]
    """
    if input_path is None:
        sdir = DATA_DIR / subdataset
        input_path = sdir / f"{subdataset}_S{subject}_C{clip}_input.csv"
        boris_path = sdir / f"{subdataset}_S{subject}_C{clip}_BORIS_method_II.csv"

    # Load physio data
    physio_df = pd.read_csv(input_path)
    # Columns: EDA, TEMP, HR, FR

    # Load BORIS method II annotations
    boris_df = pd.read_csv(boris_path)
    # Columns: Happy, Sad, Scared, Disgusted, Surprised, Angry

    # Both files have the same number of rows (1 per second)
    # The first row of BORIS is a header indicator/number — skip row 0 if non-numeric
    # Actually, examining the data: BORIS method II first row IS a header.
    # Let's check row counts align
    n_physio = len(physio_df)
    n_boris = len(boris_df)

    if n_physio != n_boris:
        print(f"  WARNING: {subdataset}_S{subject}_C{clip}: "
              f"physio rows={n_physio}, boris rows={n_boris}, using min")
    n = min(n_physio, n_boris)

    # Assign time index (seconds, 1-indexed like BORIS)
    physio_df = physio_df.iloc[:n].copy().reset_index(drop=True)
    boris_df = boris_df.iloc[:n].copy().reset_index(drop=True)
    physio_df['seconds'] = np.arange(1, n + 1)
    boris_df['seconds'] = np.arange(1, n + 1)

    # Merge on seconds
    merged = pd.merge(physio_df, boris_df, on='seconds')

    # Aggregate to 5s windows
    merged['window_id'] = (merged['seconds'] - 1) // WINDOW_SEC

    results = []
    for wid, group in merged.groupby('window_id'):
        window_end_sec = (wid + 1) * WINDOW_SEC

        record = {
            'seconds': window_end_sec,
            'window_type': 'fast',
        }

        # Physio features — simple aggregation (data is already at 1Hz)
        record['eda_mean'] = group['EDA'].mean()
        record['eda_std'] = group['EDA'].std() if len(group) > 1 else 0.0
        record['eda_max'] = group['EDA'].max()
        record['temp_mean'] = group['TEMP'].mean()
        record['hr_mean'] = group['HR'].mean()
        record['hr_std'] = group['HR'].std() if len(group) > 1 else 0.0

        # Emotion annotations — mean % per emotion in window
        emo_means = {}
        for emo in EMOTIONS:
            if emo in group.columns:
                emo_means[emo] = group[emo].mean()
            else:
                emo_means[emo] = 0.0

        # Map to arousal/valence
        av = compute_arousal_valence(pd.Series(emo_means))
        record['arousal'] = av['arousal']
        record['valence'] = av['valence']

        # Store raw emotion percentages for analysis
        for emo in EMOTIONS:
            record[f'emo_{emo.lower()}'] = emo_means[emo]

        results.append(record)

    if not results:
        return None

    df = pd.DataFrame(results)
    return df


def process_subdataset(subdataset: str) -> list:
    """
    Processes all files in an EMBOA sub-dataset.

    Returns list of (client_id, DataFrame) tuples.
    """
    sdir = DATA_DIR / subdataset
    if not sdir.exists():
        print(f"  Sub-dataset directory not found: {sdir}")
        return []

    file_pairs = discover_files(sdir, subdataset)
    print(f"  {subdataset}: found {len(file_pairs)} (subject, clip) pairs")

    results = []
    for (subject, clip), paths in sorted(file_pairs.items()):
        client_id = f"EMBOA_{subdataset}_S{subject}_C{clip}"

        df = process_emboa_file(
            subdataset, subject, clip,
            input_path=paths['input'],
            boris_path=paths['boris']
        )

        if df is not None and len(df) > 0:
            df['client_id'] = client_id
            results.append((client_id, df))
            print(f"    S{subject}_C{clip}: {len(df)} windows")
        else:
            print(f"    S{subject}_C{clip}: no data")

    return results


def main():
    """Processes all EMBOA sub-datasets and saves merged CSV files."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PROCESSING EMBOA DATA (External BORIS Annotations)")
    print("=" * 60)

    all_results = []

    for subdataset in SUBDATASETS:
        print(f"\n--- {subdataset} ---")
        results = process_subdataset(subdataset)
        all_results.extend(results)

    print(f"\nTotal recordings processed: {len(all_results)}")

    # Group by subject across clips -> one merged file per subject per subdataset
    # Client IDs look like: EMBOA_MAAP_S01_C02
    subject_data = {}
    for client_id, df in all_results:
        # Extract subdataset and subject
        parts = client_id.split('_')  # EMBOA, MAAP, S01, C02
        sub = parts[1]
        subj = parts[2]
        key = f"{sub}_{subj}"

        if key not in subject_data:
            subject_data[key] = []
        subject_data[key].append(df)

    # Save per-subject merged files
    for key, dfs in sorted(subject_data.items()):
        merged = pd.concat(dfs, ignore_index=True)
        merged = merged.sort_values('seconds').reset_index(drop=True)

        output_file = OUTPUT_DIR / f"{key}_merged.csv"
        merged.to_csv(output_file, index=False)
        print(f"  Saved {key}: {len(merged)} rows -> {output_file.name}")

    print(f"\nEMBOA processing complete. Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
