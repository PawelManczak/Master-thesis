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
    results = {}
    for name, path in DATASETS.items():
        print(f"Processing {_clean(name)}...")
        df = pd.read_csv(path)
        armada, patterns, rules = run_armada_on_df(df, MINSUP, MINCONF, MAXGAP, MAX_PATTERN_SIZE)
        raw_sigs = extract_rule_signatures(rules)
        filtered = filter_rules(
            raw_sigs,
            FILTER_BVP_ONLY,
            FILTER_EDA_ONLY,
            FILTER_PHYSIO_CROSS,
            FILTER_SINGLE_FEATURE,
        )
        results[name] = filtered
        print(f"  {_clean(name)}: {len(filtered)} filtered rules")

    all_names = list(results.keys())
    all_quads = list(itertools.combinations(all_names, 4))

    print(f"\nTotal quadruplets: {len(all_quads)}")

    # Build full 15x6 matrix on all_names / all_quads
    matrix_full = np.zeros((len(all_quads), len(all_names)), dtype=int)
    for i, quad in enumerate(all_quads):
        shared_quad = results[quad[0]] & results[quad[1]] & results[quad[2]] & results[quad[3]]
        for j, target_ds in enumerate(all_names):
            matrix_full[i, j] = len(shared_quad & results[target_ds])

    if sparse:
        # Sprawdzamy czyste wartości w macierzy. Jeśli cały rząd lub cała kolumna ma same zera - wyrzucamy
        row_mask = np.any(matrix_full > 0, axis=1)
        col_mask = np.any(matrix_full > 0, axis=0)

        print(f"  sparse: keeping {row_mask.sum()} / {len(all_quads)} rows, "
              f"{col_mask.sum()} / {len(all_names)} cols")

        if not row_mask.any() or not col_mask.any():
            print("Sparse matrix is completely empty, skipping.")
            return

        # Explicit integer indices — avoids numpy bool array edge cases
        row_idx = np.where(row_mask)[0].tolist()
        col_idx = np.where(col_mask)[0].tolist()

        quads = [all_quads[i] for i in row_idx]
        names = [all_names[j] for j in col_idx]
        matrix = matrix_full[np.ix_(row_idx, col_idx)]
    else:
        quads = all_quads
        names = all_names
        matrix = matrix_full

    n_rows, n_cols = matrix.shape

    # Build off_diag: NaN for grey cells (target in quad)
    off_diag = matrix.astype(float)
    for i, quad in enumerate(quads):
        for j, target_ds in enumerate(names):
            if target_ds in quad:
                off_diag[i, j] = np.nan

    valid_max = float(np.nanmax(off_diag)) if not np.all(np.isnan(off_diag)) else 1.0
    if valid_max == 0:
        valid_max = 1.0

    # Dynamic figure size
    fig_height = max(3, n_rows * 0.6 + 2.0)
    fig_width = max(6, n_cols * 1.3 + 3.0)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    cmap = plt.cm.YlOrRd
    im = ax.imshow(off_diag, cmap=cmap, aspect='auto', vmin=0, vmax=valid_max)

    # Grey rectangles for self-member cells
    for i, quad in enumerate(quads):
        for j, target_ds in enumerate(names):
            if target_ds in quad:
                ax.add_patch(plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    fill=True, color='#e0e0e0', zorder=2,
                ))

    # Text annotations
    for i in range(n_rows):
        for j in range(n_cols):
            target_ds = names[j]
            quad = quads[i]
            cell_val = matrix[i, j]
            cell_off = off_diag[i, j]

            if np.isnan(cell_off) and target_ds not in quad:
                continue

            if target_ds in quad:
                ax.text(j, i, str(cell_val),
                        ha='center', va='center',
                        color='#333333', fontweight='bold', fontsize=9, zorder=3)
            else:
                norm_val = cell_val / valid_max
                text_color = 'white' if norm_val > 0.6 else 'black'
                ax.text(j, i, str(cell_val),
                        ha='center', va='center',
                        color=text_color, fontweight='normal', fontsize=9, zorder=3)

    row_labels = [' + '.join(_clean(d) for d in q) for q in quads]
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=8, va='center')

    col_labels = [_clean(n) for n in names]
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, fontsize=10, ha='right', rotation=45)

    ax.set_xlabel('Fifth Dataset')
    ax.set_ylabel('Intersected Dataset Quadruplets')

    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label('Shared rules (Quintuple Intersection)', fontsize=10)

    ax.set_title(
        'Generalization of Quadruplet Shared Rules\nto Fifth Datasets',
        fontsize=12, fontweight='bold', pad=15,
    )

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('#cccccc')

    plt.tight_layout()

    out_suffix = "_sparse" if sparse else ""
    out_png = OUTPUT_DIR / f"heatmap_quadruplets_generalization{out_suffix}.png"
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved PNG to {out_png}")

    print("\nQuadruplet Base Intersections (rules shared by all 4):")
    for quad in quads:
        shared = results[quad[0]] & results[quad[1]] & results[quad[2]] & results[quad[3]]
        label = ' + '.join(_clean(d) for d in quad)
        print(f"  {label}: {len(shared)} rules")

    print("\nTop 5 Generalizing Quintuplets (Quadruplet + 1):")
    quints = []
    for i, quad in enumerate(quads):
        for j, target_ds in enumerate(names):
            if target_ds not in quad:
                quints.append({
                    'quadruplet': ' + '.join(_clean(d) for d in quad),
                    'target': _clean(target_ds),
                    'rules': matrix[i, j],
                })
    quint_df = pd.DataFrame(quints)
    if not quint_df.empty:
        print(quint_df.sort_values('rules', ascending=False).head(5).to_string(index=False))
    else:
        print("No quintuplets to display.")


if __name__ == "__main__":
    main()
    main(sparse=True)