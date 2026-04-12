#!/usr/bin/env python3
"""
Generate a heatmap comparing the intersection of three datasets with remaining datasets.
Uses RQ1 parameters (minsup=0.5, minconf=0.5, maxgap=5s).

Matrix: 20 rows (all unique triplets from 6 datasets) x 6 columns (individual datasets).
Cell value = len(A & B & C & D) where (A,B,C) is the row triplet and D is the column dataset.
Grey cells mark when D is already part of the triplet.
"""

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import itertools

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = SCRIPT_DIR.parent
PROJECT_DIR = EXPERIMENTS_DIR.parent.parent

sys.path.insert(0, str(PROJECT_DIR / "source" / "processing" / "armada"))
sys.path.insert(0, str(EXPERIMENTS_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from experiment_utils import (
    run_armada_on_df,
    extract_rule_signatures,
    filter_rules,
)

# Parameters (same as RQ1)
MINSUP = 0.5
MINCONF = 0.5
MAXGAP = 5
MAX_PATTERN_SIZE = 2

FILTER_BVP_ONLY = True
FILTER_EDA_ONLY = True
FILTER_PHYSIO_CROSS = True
FILTER_SINGLE_FEATURE = True

DATA_DIR = PROJECT_DIR / "data" / "armada_ready"

DATASETS = {
    "CASE": DATA_DIR / "armada_case.csv",
    "K-emoCon": DATA_DIR / "armada_k_emocon.csv",
    "CEAP": DATA_DIR / "armada_ceap.csv",
    "EmoWorker": DATA_DIR / "armada_emoworker_v2.csv",
    "K-emo (ext)": DATA_DIR / "armada_k_emocon_ext.csv",
    "EMBOA": DATA_DIR / "armada_emboa.csv",
}

OUTPUT_DIR = Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _clean(name):
    """Remove newlines from dataset name for display."""
    return name.replace('\n', ' ')


def main():
    # Mine rules for each dataset
    results = {}
    for name, path in DATASETS.items():
        print(f"Processing {_clean(name)}...")
        df = pd.read_csv(path)
        armada, patterns, rules = run_armada_on_df(df, MINSUP, MINCONF, MAXGAP, MAX_PATTERN_SIZE)

        raw_sigs = extract_rule_signatures(rules)
        filtered = filter_rules(raw_sigs, FILTER_BVP_ONLY, FILTER_EDA_ONLY,
                                FILTER_PHYSIO_CROSS, FILTER_SINGLE_FEATURE)
        results[name] = filtered
        print(f"  {_clean(name)}: {len(filtered)} filtered rules")

    names = list(results.keys())

    # Generate all triplets (C(6,3) = 20)
    triplets = list(itertools.combinations(names, 3))
    n_rows = len(triplets)
    n_cols = len(names)

    print(f"\nTotal triplets: {n_rows}")

    # Build triplet intersection matrix
    matrix = np.zeros((n_rows, n_cols), dtype=int)
    for i, (ds1, ds2, ds3) in enumerate(triplets):
        shared_triplet = results[ds1] & results[ds2] & results[ds3]
        for j, target_ds in enumerate(names):
            matrix[i][j] = len(shared_triplet & results[target_ds])

    # Create heatmap
    fig, ax = plt.subplots(figsize=(8, 12))

    cmap = plt.cm.YlOrRd
    off_diag = matrix.copy().astype(float)

    # Mask when the column dataset is part of the row triplet
    for i, tri in enumerate(triplets):
        for j, target_ds in enumerate(names):
            if target_ds in tri:
                off_diag[i][j] = np.nan

    # Determine vmax from non-self cells
    valid_max = np.nanmax(off_diag) if not np.all(np.isnan(off_diag)) else 1
    if valid_max == 0:
        valid_max = 1

    # Plot off-diagonal with color map
    im = ax.imshow(off_diag, cmap=cmap, aspect='auto', vmin=0, vmax=valid_max)

    # Plot triplet-member cells with grey
    for i, tri in enumerate(triplets):
        for j, target_ds in enumerate(names):
            if target_ds in tri:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                           fill=True, color='#e0e0e0', zorder=2))

    # Add text annotations
    for i in range(n_rows):
        for j in range(n_cols):
            val = matrix[i][j]
            target_ds = names[j]
            tri = triplets[i]
            if target_ds in tri:
                text_color = '#333333'
                fontweight = 'bold'
                fontsize = 10
            else:
                norm_val = val / valid_max
                text_color = 'white' if norm_val > 0.6 else 'black'
                fontweight = 'normal'
                fontsize = 10
            ax.text(j, i, str(val), ha='center', va='center',
                    color=text_color, fontweight=fontweight, fontsize=fontsize,
                    zorder=3)

    # Labels
    row_labels = [_clean(t[0]) + ' + ' + _clean(t[1]) + ' + ' + _clean(t[2]) for t in triplets]
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=8, va='center')

    col_labels = [_clean(n) for n in names]
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, fontsize=10, ha='right', rotation=45)

    ax.set_xlabel('Fourth Dataset')
    ax.set_ylabel('Intersected Dataset Triplets')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label('Shared rules (Quadruple Intersection)', fontsize=10)

    # Title
    ax.set_title('Generalization of Triplet Shared Rules\nto Fourth Datasets',
                 fontsize=12, fontweight='bold', pad=15)

    # Border
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('#cccccc')

    plt.tight_layout()

    out_png = OUTPUT_DIR / "heatmap_triplets_generalization.png"
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    print(f"\nSaved PNG to {out_png}")

    # Print triplet base intersections
    print("\nTriplet Base Intersections (rules shared by all 3):")
    for i, (ds1, ds2, ds3) in enumerate(triplets):
        shared = results[ds1] & results[ds2] & results[ds3]
        label = _clean(ds1) + ' + ' + _clean(ds2) + ' + ' + _clean(ds3)
        print(f"  {label}: {len(shared)} rules")

    # Top quadruplet generalizations
    print("\nTop 5 Generalizing Quadruplets:")
    quads = []
    for i, (ds1, ds2, ds3) in enumerate(triplets):
        for j, target_ds in enumerate(names):
            if target_ds not in (ds1, ds2, ds3):
                quads.append({
                    'triplet': _clean(ds1) + ' + ' + _clean(ds2) + ' + ' + _clean(ds3),
                    'target': _clean(target_ds),
                    'rules': matrix[i][j]
                })

    quad_df = pd.DataFrame(quads)
    quad_df = quad_df.sort_values(by='rules', ascending=False).head(5)
    print(quad_df.to_string(index=False))


if __name__ == "__main__":
    main()
