#!/usr/bin/env python3
"""
Generate a heatmap comparing the intersection of five datasets with the remaining sixth.
Uses RQ1 parameters (minsup=0.3, minconf=0.5, maxgap=5s).

Matrix: 6 rows (all unique quintuplets from 6 datasets) x 6 columns (individual datasets).
Cell value = len(A & B & C & D & E & F) where (A,B,C,D,E) is the row quintuplet and F is the column.
Grey cells mark when F is already part of the quintuplet.
"""

import sys
from pathlib import Path
import itertools

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
    return name.replace('\n', ' ')


def main(sparse=False):
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

    all_names = list(results.keys())

    # Generate all quintuplets (C(6,5) = 6)
    all_quints = list(itertools.combinations(all_names, 5))

    print(f"\nTotal quintuplets: {len(all_quints)}")

    # Build quintuplet full intersection matrix
    matrix_full = np.zeros((len(all_quints), len(all_names)), dtype=int)
    for i, quin in enumerate(all_quints):
        shared = results[quin[0]]
        for ds in quin[1:]:
            shared = shared & results[ds]
        for j, target_ds in enumerate(all_names):
            matrix_full[i, j] = len(shared & results[target_ds])

    if sparse:
        # Sprawdzamy czyste wartości w macierzy. Jeśli cały rząd lub cała kolumna ma same zera - wyrzucamy
        row_mask = np.any(matrix_full > 0, axis=1)
        col_mask = np.any(matrix_full > 0, axis=0)

        print(f"  sparse: keeping {row_mask.sum()} / {len(all_quints)} rows, "
              f"{col_mask.sum()} / {len(all_names)} cols")

        if not row_mask.any() or not col_mask.any():
            print("Sparse matrix is completely empty, skipping.")
            return

        # Explicit integer indices — avoids numpy bool array edge cases
        row_idx = np.where(row_mask)[0].tolist()
        col_idx = np.where(col_mask)[0].tolist()

        quints = [all_quints[i] for i in row_idx]
        names = [all_names[j] for j in col_idx]
        matrix = matrix_full[np.ix_(row_idx, col_idx)]
    else:
        quints = all_quints
        names = all_names
        matrix = matrix_full

    n_rows, n_cols = matrix.shape

    # Dynamic figure size based on remaining rows and columns
    fig_height = max(4, n_rows * 0.6 + 2.0)
    fig_width = max(6, n_cols * 1.3 + 3.0)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    cmap = plt.cm.YlOrRd
    off_diag = matrix.astype(float)

    # Mask when the column dataset is part of the row quintuplet
    for i, quin in enumerate(quints):
        for j, target_ds in enumerate(names):
            if target_ds in quin:
                off_diag[i, j] = np.nan

    valid_max = float(np.nanmax(off_diag)) if not np.all(np.isnan(off_diag)) else 1.0
    if valid_max == 0:
        valid_max = 1.0

    im = ax.imshow(off_diag, cmap=cmap, aspect='auto', vmin=0, vmax=valid_max)

    # Grey out quintuplet-member cells
    for i, quin in enumerate(quints):
        for j, target_ds in enumerate(names):
            if target_ds in quin:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                           fill=True, color='#e0e0e0', zorder=2))

    for i in range(n_rows):
        for j in range(n_cols):
            val = matrix[i, j]
            target_ds = names[j]
            quin = quints[i]
            
            if target_ds in quin:
                text_color = '#333333'
                fontweight = 'bold'
                fontsize = 11
            else:
                norm_val = val / valid_max
                text_color = 'white' if norm_val > 0.6 else 'black'
                fontweight = 'normal'
                fontsize = 11
                
            ax.text(j, i, str(val), ha='center', va='center',
                    color=text_color, fontweight=fontweight, fontsize=fontsize,
                    zorder=3)

    # Labels — show the ONE excluded dataset for readability based on all_names
    row_labels = []
    for quin in quints:
        excluded = [n for n in all_names if n not in quin]
        row_labels.append('All except ' + _clean(excluded[0]))
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=10, va='center')

    col_labels = [_clean(n) for n in names]
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, fontsize=10, ha='right', rotation=45)

    ax.set_xlabel('Sixth Dataset (Leave-One-Out Target)')
    ax.set_ylabel('Quintuplet (5 Intersected Datasets)')

    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label('Shared rules (Full 6-way Intersection)', fontsize=10)

    ax.set_title('Leave-One-Out: Quintuplet Rules Generalization',
                 fontsize=12, fontweight='bold', pad=15)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('#cccccc')

    plt.tight_layout()

    out_suffix = "_sparse" if sparse else ""
    out_png = OUTPUT_DIR / f"heatmap_quintuplets_generalization{out_suffix}.png"
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved PNG to {out_png}")

    print("\nQuintuplet Base Intersections (rules shared by all 5):")
    for i, quin in enumerate(quints):
        shared = results[quin[0]]
        for ds in quin[1:]:
            shared = shared & results[ds]
        excluded = [n for n in all_names if n not in quin]
        label = 'All except ' + _clean(excluded[0])
        print(f"  {label}: {len(shared)} rules")

    # Leave-one-out results
    print("\nLeave-One-Out Full 6-way Intersection:")
    for i, quin in enumerate(quints):
        excluded = [n for n in all_names if n not in quin]
        target = excluded[0]
        shared = results[quin[0]]
        for ds in quin[1:]:
            shared = shared & results[ds]
        full_six = len(shared & results[target])
        label = 'All except ' + _clean(target)
        print(f"  {label} => add {_clean(target)}: {full_six} rules survive as universal 6-way")


if __name__ == "__main__":
    main()
    main(sparse=True)