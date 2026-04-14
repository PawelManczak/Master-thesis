#!/usr/bin/env python3
"""
Generate ALL four heatmaps (pairwise, pairs→3rd, triplets→4th, quintuplets→6th)
for the DISCRETE emotion model (Russell's 9-label circumplex).

Uses the same RQ1 parameters and filters as the dimensional heatmaps.
Reads pre-built discrete ARMADA CSVs from emotion_labels/results/.
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

DISCRETE_DIR = EXPERIMENTS_DIR / "emotion_labels" / "results"

DATASETS = {
    "CASE": DISCRETE_DIR / "armada_discrete_CASE.csv",
    "K-emoCon": DISCRETE_DIR / "armada_discrete_K-emoCon.csv",
    "CEAP": DISCRETE_DIR / "armada_discrete_CEAP.csv",
    "EmoWorker": DISCRETE_DIR / "armada_discrete_EmoWorker_v2.csv",
    "K-emo (ext)": DISCRETE_DIR / "armada_discrete_K-emoCon_ext.csv",
    "EMBOA": DISCRETE_DIR / "armada_discrete_EMBOA.csv",
}

OUTPUT_DIR = SCRIPT_DIR / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _clean(name):
    return name.replace('\n', ' ')


def mine_all_rules():
    """Mine rules for each discrete dataset."""
    results = {}
    for name, path in DATASETS.items():
        if not path.exists():
            print(f"  SKIP {_clean(name)}: missing {path}")
            continue
        print(f"Processing {_clean(name)}...")
        df = pd.read_csv(path)
        armada, patterns, rules = run_armada_on_df(df, MINSUP, MINCONF, MAXGAP, MAX_PATTERN_SIZE)
        raw_sigs = extract_rule_signatures(rules)
        filtered = filter_rules(raw_sigs, FILTER_BVP_ONLY, FILTER_EDA_ONLY,
                                FILTER_PHYSIO_CROSS, FILTER_SINGLE_FEATURE)
        results[name] = filtered
        print(f"  {_clean(name)}: {len(filtered)} filtered rules")
    return results


def make_heatmap_pairwise(results, all_names, sparse=False):
    """Standard pairwise shared-rules heatmap."""
    matrix_full = np.zeros((len(all_names), len(all_names)), dtype=int)
    for i, name1 in enumerate(all_names):
        for j, name2 in enumerate(all_names):
            matrix_full[i, j] = len(results[name1] & results[name2])

    if sparse:
        row_mask = np.any(matrix_full > 0, axis=1)
        col_mask = np.any(matrix_full > 0, axis=0)

        if not row_mask.any() or not col_mask.any():
            print("  [Pairwise] Sparse matrix is completely empty, skipping.")
            return

        row_idx = np.where(row_mask)[0].tolist()
        col_idx = np.where(col_mask)[0].tolist()

        row_names = [all_names[i] for i in row_idx]
        col_names = [all_names[j] for j in col_idx]
        matrix = matrix_full[np.ix_(row_idx, col_idx)]
    else:
        row_names = all_names
        col_names = all_names
        matrix = matrix_full

    n_rows, n_cols = matrix.shape

    fig_height = max(4, n_rows * 0.8 + 2.0)
    fig_width = max(5, n_cols * 1.0 + 2.5)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    cmap = plt.cm.YlOrRd
    off_diag = matrix.astype(float)
    
    for i in range(n_rows):
        for j in range(n_cols):
            if row_names[i] == col_names[j]:
                off_diag[i, j] = np.nan

    valid_max = float(np.nanmax(off_diag)) if not np.all(np.isnan(off_diag)) else 1.0
    if valid_max == 0:
        valid_max = 1.0

    im = ax.imshow(off_diag, cmap=cmap, aspect='auto', vmin=0, vmax=valid_max)

    for i in range(n_rows):
        for j in range(n_cols):
            if row_names[i] == col_names[j]:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                           fill=True, color='#e0e0e0', zorder=2))

    for i in range(n_rows):
        for j in range(n_cols):
            val = matrix[i, j]
            if row_names[i] == col_names[j]:
                tc, fw, fs = '#333333', 'bold', 11
            else:
                nv = val / valid_max
                tc = 'white' if nv > 0.6 else 'black'
                fw, fs = 'normal', 10
            ax.text(j, i, str(val), ha='center', va='center',
                    color=tc, fontweight=fw, fontsize=fs, zorder=3)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([_clean(n) for n in col_names], fontsize=10, ha='center')
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([_clean(n) for n in row_names], fontsize=10, va='center')
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Shared rules', fontsize=10)
    ax.set_title('Pairwise Shared Rules (Discrete Model)', fontsize=13, fontweight='bold', pad=12)
    
    for s in ax.spines.values(): 
        s.set_visible(True)
        s.set_color('#cccccc')
        
    plt.tight_layout()
    
    out_suffix = "_sparse" if sparse else ""
    out_dir = OUTPUT_DIR / "discrete" / ("sparse" if sparse else "normal")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"heatmap_discrete_pairwise{out_suffix}.png"
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


def make_heatmap_combo(results, all_names, combo_size, filename, title, ylabel, xlabel, clabel, sparse=False):
    """Generic heatmap for combinations of combo_size vs individual datasets."""
    all_combos = list(itertools.combinations(all_names, combo_size))
    
    matrix_full = np.zeros((len(all_combos), len(all_names)), dtype=int)
    for i, combo in enumerate(all_combos):
        shared = results[combo[0]]
        for ds in combo[1:]:
            shared = shared & results[ds]
        for j, target in enumerate(all_names):
            matrix_full[i, j] = len(shared & results[target])

    if sparse:
        row_mask = np.any(matrix_full > 0, axis=1)
        col_mask = np.any(matrix_full > 0, axis=0)

        if not row_mask.any() or not col_mask.any():
            print(f"  [{title}] Sparse matrix is completely empty, skipping.")
            return

        row_idx = np.where(row_mask)[0].tolist()
        col_idx = np.where(col_mask)[0].tolist()

        combos = [all_combos[i] for i in row_idx]
        names = [all_names[j] for j in col_idx]
        matrix = matrix_full[np.ix_(row_idx, col_idx)]
    else:
        combos = all_combos
        names = all_names
        matrix = matrix_full

    n_rows, n_cols = matrix.shape

    fig_height = max(4, n_rows * 0.5 + 2.0)
    fig_width = max(6, n_cols * 1.3 + 3.0)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    cmap = plt.cm.YlOrRd
    off_diag = matrix.astype(float)

    for i, combo in enumerate(combos):
        for j, target in enumerate(names):
            if target in combo:
                off_diag[i, j] = np.nan

    valid_max = float(np.nanmax(off_diag)) if not np.all(np.isnan(off_diag)) else 1.0
    if valid_max == 0:
        valid_max = 1.0

    im = ax.imshow(off_diag, cmap=cmap, aspect='auto', vmin=0, vmax=valid_max)

    for i, combo in enumerate(combos):
        for j, target in enumerate(names):
            if target in combo:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                           fill=True, color='#e0e0e0', zorder=2))

    for i in range(n_rows):
        for j in range(n_cols):
            val = matrix[i, j]
            target = names[j]
            combo = combos[i]
            if target in combo:
                tc, fw, fs = '#333333', 'bold', 10
            else:
                nv = val / valid_max
                tc = 'white' if nv > 0.6 else 'black'
                fw, fs = 'normal', 10
            ax.text(j, i, str(val), ha='center', va='center',
                    color=tc, fontweight=fw, fontsize=fs, zorder=3)

    # Row labels
    if combo_size >= len(all_names) - 1:
        row_labels = []
        for combo in combos:
            excl = [n for n in all_names if n not in combo]
            row_labels.append('All except ' + _clean(excl[0]))
    else:
        row_labels = [' + '.join(_clean(c) for c in combo) for combo in combos]

    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=8 if n_rows > 10 else 10, va='center')
    
    col_labels = [_clean(n) for n in names]
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, fontsize=10, ha='right', rotation=45)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(clabel, fontsize=10)
    
    ax.set_title(title, fontsize=12, fontweight='bold', pad=15)
    
    for s in ax.spines.values(): 
        s.set_visible(True)
        s.set_color('#cccccc')
        
    plt.tight_layout()
    
    if sparse:
        filename = filename.replace(".png", "_sparse.png")
    out_dir = OUTPUT_DIR / "discrete" / ("sparse" if sparse else "normal")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / filename
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


def main():
    print("\n" + "=" * 80)
    print("DISCRETE MODEL: ALL HEATMAPS")
    print("=" * 80)

    results = mine_all_rules()
    names = list(results.keys())

    if len(names) < 2:
        print("Not enough datasets with rules!")
        return

    for sparse in [False, True]:
        suffix_msg = " (SPARSE)" if sparse else " (STANDARD)"
        print(f"\n--- GENERATING VERSIONS{suffix_msg} ---")

        # 1. Pairwise
        make_heatmap_pairwise(results, names, sparse=sparse)

        # 2. Pairs → 3rd
        make_heatmap_combo(results, names, 2,
                           "heatmap_discrete_pairs_generalization.png",
                           "Pairs → Third Dataset (Discrete Model)",
                           "Dataset Pairs", "Third Dataset",
                           "Shared rules (Triple Intersection)",
                           sparse=sparse)

        # 3. Triplets → 4th
        make_heatmap_combo(results, names, 3,
                           "heatmap_discrete_triplets_generalization.png",
                           "Triplets → Fourth Dataset (Discrete Model)",
                           "Dataset Triplets", "Fourth Dataset",
                           "Shared rules (Quadruple Intersection)",
                           sparse=sparse)

        # 4. Quintuplets → 6th
        make_heatmap_combo(results, names, 5,
                           "heatmap_discrete_quintuplets_generalization.png",
                           "Leave-One-Out Quintuplets (Discrete Model)",
                           "Quintuplet (5 Datasets)", "Sixth Dataset",
                           "Shared rules (Full 6-way)",
                           sparse=sparse)

    print("\nDone! All discrete heatmaps generated (Standard & Sparse).")


if __name__ == "__main__":
    main()