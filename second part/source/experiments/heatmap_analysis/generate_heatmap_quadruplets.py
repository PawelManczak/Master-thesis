#!/usr/bin/env python3
"""
Generate a heatmap comparing the intersection of four datasets with remaining datasets.
Uses RQ1 parameters (minsup=0.3, minconf=0.5, maxgap=5s).

Matrix: 15 rows (all unique quadruplets from 6 datasets) x 6 columns (individual datasets).
Cell value = len(A & B & C & D & E) where (A,B,C,D) is the row quadruplet and E is the target dataset.
Grey cells mark when E is already part of the quadruplet.
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
MINSUP = 0.3
MINCONF = 0.5
MAXGAP = 5
MAX_PATTERN_SIZE = 4

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

    # Generate all quadruplets (C(6,4) = 15)
    quads = list(itertools.combinations(names, 4))
    n_rows = len(quads)
    n_cols = len(names)

    print(f"\nTotal quadruplets: {n_rows}")

    # Build quadruplet intersection matrix
    matrix = np.zeros((n_rows, n_cols), dtype=int)
    for i, (ds1, ds2, ds3, ds4) in enumerate(quads):
        shared_quad = results[ds1] & results[ds2] & results[ds3] & results[ds4]
        for j, target_ds in enumerate(names):
            matrix[i][j] = len(shared_quad & results[target_ds])

    # Create heatmap
    fig, ax = plt.subplots(figsize=(8, 8))

    cmap = plt.cm.YlOrRd
    off_diag = matrix.copy().astype(float)

    # Mask when the column dataset is part of the row quadruplet
    for i, quad in enumerate(quads):
        for j, target_ds in enumerate(names):
            if target_ds in quad:
                off_diag[i][j] = np.nan

    # Determine vmax from non-self cells
    valid_max = np.nanmax(off_diag) if not np.all(np.isnan(off_diag)) else 1
    if valid_max == 0:
        valid_max = 1

    # Plot off-diagonal with color map
    im = ax.imshow(off_diag, cmap=cmap, aspect='auto', vmin=0, vmax=valid_max)

    # Plot quadruplet-member cells with grey
    for i, quad in enumerate(quads):
        for j, target_ds in enumerate(names):
            if target_ds in quad:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                           fill=True, color='#e0e0e0', zorder=2))

    # Add text annotations
    for i in range(n_rows):
        for j in range(n_cols):
            val = matrix[i][j]
            target_ds = names[j]
            quad = quads[i]
            if target_ds in quad:
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
    row_labels = [_clean(q[0]) + ' + ' + _clean(q[1]) + ' + ' + _clean(q[2]) + ' + ' + _clean(q[3]) for q in quads]
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=8, va='center')

    col_labels = [_clean(n) for n in names]
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, fontsize=10, ha='right', rotation=45)

    ax.set_xlabel('Fifth Dataset')
    ax.set_ylabel('Intersected Dataset Quadruplets')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label('Shared rules (Quintuple Intersection)', fontsize=10)

    # Title
    ax.set_title('Generalization of Quadruplet Shared Rules\nto Fifth Datasets',
                 fontsize=12, fontweight='bold', pad=15)

    # Border
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('#cccccc')

    plt.tight_layout()

    out_png = OUTPUT_DIR / "heatmap_quadruplets_generalization.png"
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    print(f"\nSaved PNG to {out_png}")

    # Print quadruplet base intersections
    print("\nQuadruplet Base Intersections (rules shared by all 4):")
    for i, (ds1, ds2, ds3, ds4) in enumerate(quads):
        shared = results[ds1] & results[ds2] & results[ds3] & results[ds4]
        label = _clean(ds1) + ' + ' + _clean(ds2) + ' + ' + _clean(ds3) + ' + ' + _clean(ds4)
        print(f"  {label}: {len(shared)} rules")

    # Top quintuplet generalizations
    print("\nTop 5 Generalizing Quintuplets (Quadruplet + 1):")
    quints = []
    for i, (ds1, ds2, ds3, ds4) in enumerate(quads):
        for j, target_ds in enumerate(names):
            if target_ds not in (ds1, ds2, ds3, ds4):
                quints.append({
                    'quadruplet': _clean(ds1) + ' + ' + _clean(ds2) + ' + ' + _clean(ds3) + ' + ' + _clean(ds4),
                    'target': _clean(target_ds),
                    'rules': matrix[i][j]
                })

    quint_df = pd.DataFrame(quints)
    quint_df = quint_df.sort_values(by='rules', ascending=False).head(5)
    print(quint_df.to_string(index=False))


if __name__ == "__main__":
    main()
