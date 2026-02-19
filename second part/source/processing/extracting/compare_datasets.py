"""
Skrypt do porównania i ujednolicenia danych K-EmoCon, CASE i CEAP.
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent.parent  # extracting -> processing -> source -> second part
KEMOCON_DIR = BASE_DIR / "data" / "K-emoCon" / "processed"
CASE_DIR = BASE_DIR / "data" / "CASE" / "processed"
CEAP_DIR = BASE_DIR / "data" / "CEAP" / "processed"

# Wspólne kolumny do porównania
COMMON_COLUMNS = [
    'seconds', 'arousal', 'valence',
    'eda', 'eda_var', 'hr', 'hr_var', 'temp', 'temp_var',
    'bvp_sdnn', 'bvp_rmssd', 'bvp_pnn50', 'bvp_mean_hr', 'bvp_mean_ibi',
    'bvp_lf_power', 'bvp_hf_power', 'bvp_lf_hf_ratio'
]


def normalize_kemocon_arousal_valence(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizuj arousal i valence z K-EmoCon (1-5) do skali CASE (0.5-9.5).
    Formuła: new_value = 0.5 + (old_value - 1) * (9 / 4)
    """
    df = df.copy()
    df['arousal'] = 0.5 + (df['arousal'] - 1) * (9 / 4)
    df['valence'] = 0.5 + (df['valence'] - 1) * (9 / 4)
    return df


def normalize_case_arousal_valence(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizuj arousal i valence z CASE (0.5-9.5) do skali K-EmoCon (1-5).
    Formuła: new_value = 1 + (old_value - 0.5) * (4 / 9)
    """
    df = df.copy()
    df['arousal'] = 1 + (df['arousal'] - 0.5) * (4 / 9)
    df['valence'] = 1 + (df['valence'] - 0.5) * (4 / 9)
    return df


def normalize_to_01(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """
    Normalizuj arousal i valence do skali 0-1.
    K-EmoCon: (x - 1) / 4  (skala 1-5)
    CASE: (x - 0.5) / 9   (skala 0.5-9.5)
    CEAP: (x - 1) / 8     (skala 1-9)
    """
    df = df.copy()
    if dataset == 'K-EmoCon':
        df['arousal'] = (df['arousal'] - 1) / 4
        df['valence'] = (df['valence'] - 1) / 4
    elif dataset == 'CASE':
        df['arousal'] = (df['arousal'] - 0.5) / 9
        df['valence'] = (df['valence'] - 0.5) / 9
    elif dataset == 'CEAP':
        df['arousal'] = (df['arousal'] - 1) / 8
        df['valence'] = (df['valence'] - 1) / 8
    return df


def compare_datasets():
    """Porównaj strukturę i statystyki wszystkich datasetów."""

    # Wczytaj pliki
    kemocon_files = sorted(KEMOCON_DIR.glob("*.csv")) if KEMOCON_DIR.exists() else []
    case_files = sorted(CASE_DIR.glob("*.csv")) if CASE_DIR.exists() else []
    ceap_files = sorted(CEAP_DIR.glob("*.csv")) if CEAP_DIR.exists() else []

    print("=" * 70)
    print("PORÓWNANIE DATASETÓW K-EmoCon, CASE i CEAP")
    print("=" * 70)

    print(f"\nK-EmoCon: {len(kemocon_files)} uczestników")
    print(f"CASE: {len(case_files)} uczestników")
    print(f"CEAP: {len(ceap_files)} uczestników")

    # Analiza kolumn
    samples = {}

    if kemocon_files:
        samples['K-EmoCon'] = pd.read_csv(kemocon_files[0])
        print(f"\nK-EmoCon kolumny ({len(samples['K-EmoCon'].columns)}):")
        print(list(samples['K-EmoCon'].columns))

    if case_files:
        samples['CASE'] = pd.read_csv(case_files[0])
        print(f"\nCASE kolumny ({len(samples['CASE'].columns)}):")
        print(list(samples['CASE'].columns))

    if ceap_files:
        samples['CEAP'] = pd.read_csv(ceap_files[0])
        print(f"\nCEAP kolumny ({len(samples['CEAP'].columns)}):")
        print(list(samples['CEAP'].columns))

    # Wspólne kolumny
    if len(samples) >= 2:
        all_cols = [set(df.columns) for df in samples.values()]
        common = set.intersection(*all_cols)
        print(f"\nWspólne kolumny ({len(common)}):")
        print(sorted(common))

        for name, df in samples.items():
            only_here = set(df.columns) - common
            if only_here:
                print(f"\nTylko w {name} ({len(only_here)}):")
                print(sorted(only_here))

    # Statystyki arousal/valence
    print("\n" + "=" * 70)
    print("STATYSTYKI AROUSAL/VALENCE")
    print("=" * 70)

    all_data = {}

    if kemocon_files:
        all_data['K-EmoCon'] = pd.concat([pd.read_csv(f) for f in kemocon_files], ignore_index=True)
        df = all_data['K-EmoCon']
        print(f"\nK-EmoCon (oryginalna skala 1-5):")
        print(f"  Arousal: min={df['arousal'].min():.2f}, max={df['arousal'].max():.2f}, mean={df['arousal'].mean():.2f}")
        print(f"  Valence: min={df['valence'].min():.2f}, max={df['valence'].max():.2f}, mean={df['valence'].mean():.2f}")

    if case_files:
        all_data['CASE'] = pd.concat([pd.read_csv(f) for f in case_files], ignore_index=True)
        df = all_data['CASE']
        print(f"\nCASE (oryginalna skala 0.5-9.5):")
        print(f"  Arousal: min={df['arousal'].min():.2f}, max={df['arousal'].max():.2f}, mean={df['arousal'].mean():.2f}")
        print(f"  Valence: min={df['valence'].min():.2f}, max={df['valence'].max():.2f}, mean={df['valence'].mean():.2f}")

    if ceap_files:
        all_data['CEAP'] = pd.concat([pd.read_csv(f) for f in ceap_files], ignore_index=True)
        df = all_data['CEAP']
        print(f"\nCEAP (oryginalna skala 1-9):")
        print(f"  Arousal: min={df['arousal'].min():.2f}, max={df['arousal'].max():.2f}, mean={df['arousal'].mean():.2f}")
        print(f"  Valence: min={df['valence'].min():.2f}, max={df['valence'].max():.2f}, mean={df['valence'].mean():.2f}")

    # Statystyki sygnałów fizjologicznych
    print("\n" + "=" * 70)
    print("STATYSTYKI SYGNAŁÓW FIZJOLOGICZNYCH")
    print("=" * 70)

    for signal in ['eda', 'hr', 'temp']:
        print(f"\n{signal.upper()}:")
        for name, df in all_data.items():
            if signal in df.columns:
                valid = df[signal].dropna()
                if len(valid) > 0:
                    print(f"  {name:12s}: min={valid.min():8.2f}, max={valid.max():8.2f}, mean={valid.mean():8.2f}, missing={df[signal].isna().sum()}")
                else:
                    print(f"  {name:12s}: brak danych")

    # Podsumowanie
    print("\n" + "=" * 70)
    print("PODSUMOWANIE PORÓWNYWALNOŚCI")
    print("=" * 70)

    print("""
Wszystkie trzy datasety mają wspólne kolumny:
- seconds, arousal, valence
- eda, eda_var, hr, hr_var, temp, temp_var
- bvp_sdnn, bvp_rmssd, bvp_pnn50, bvp_mean_hr, bvp_mean_ibi
- bvp_lf_power, bvp_hf_power, bvp_lf_hf_ratio

RÓŻNICE W SKALACH AROUSAL/VALENCE:
- K-EmoCon: 1-5
- CASE: 0.5-9.5
- CEAP: 1-9

NORMALIZACJA DO SKALI 0-1:
- K-EmoCon: (x - 1) / 4
- CASE: (x - 0.5) / 9
- CEAP: (x - 1) / 8

DODATKOWE KOLUMNY:
- K-EmoCon i CEAP mają dane ACC (akcelerometr) i BVP
- CASE nie ma tych danych
- CEAP ma dodatkową kolumnę video_id
    """)


if __name__ == "__main__":
    compare_datasets()

