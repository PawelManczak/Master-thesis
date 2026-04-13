#!/usr/bin/env python3
"""
Generate ALL four heatmaps (pairwise, pairs→3rd, triplets→4th, quintuplets→6th)
for the DISCRETE emotion model (Russell's 9-label circumplex).

Uses the same RQ1 parameters and filters as the dimensional heatmaps.
Reads pre-built discrete ARMADA CSVs from emotion_labels/results/.
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

DISCRETE_DIR = EXPERIMENTS_DIR / "emotion_labels" / "results"

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


def make_heatmap_pairwise(results, names):
    """Standard pairwise shared-rules heatmap."""
    n = len(names)
    matrix = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i][j] = len(results[names[i]])
            else:
                matrix[i][j] = len(results[names[i]] & results[names[j]])

    fig, ax = plt.subplots(figsize=(7.5, 6))
    cmap = plt.cm.YlOrRd
    mask_diag = np.eye(n, dtype=bool)
    off_diag = matrix.copy().astype(float)
    off_diag[mask_diag] = np.nan

    im = ax.imshow(off_diag, cmap=cmap, aspect='auto',
                   vmin=0, vmax=max(np.nanmax(off_diag), 1))

    for i in range(n):
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1,
                                   fill=True, color='#e0e0e0', zorder=2))

    for i in range(n):
        for j in range(n):
            val = matrix[i][j]
            if i == j:
                tc, fw, fs = '#333333', 'bold', 11
            else:
                nv = val / max(np.nanmax(off_diag), 1)
                tc = 'white' if nv > 0.6 else 'black'
                fw, fs = 'normal', 10
            ax.text(j, i, str(val), ha='center', va='center',
                    color=tc, fontweight=fw, fontsize=fs, zorder=3)

    display = [_clean(n) for n in names]
    ax.set_xticks(range(n));
    ax.set_xticklabels(display, fontsize=10)
    ax.set_yticks(range(n));
    ax.set_yticklabels(display, fontsize=10)
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Shared rules', fontsize=10)
    ax.set_title('Pairwise Shared Rules (Discrete Model)', fontsize=13, fontweight='bold', pad=12)
    for s in ax.spines.values(): s.set_visible(True); s.set_color('#cccccc')
    plt.tight_layout()
    out = OUTPUT_DIR / "heatmap_discrete_pairwise.png"
    plt.savefig(out, dpi=300, bbox_inches='tight');
    plt.close()
    print(f"Saved: {out}")


def make_heatmap_combo(results, names, combo_size, filename, title, ylabel, xlabel, clabel):
    """Generic heatmap for combinations of combo_size vs individual datasets."""
    combos = list(itertools.combinations(names, combo_size))
    n_rows = len(combos)
    n_cols = len(names)

    matrix = np.zeros((n_rows, n_cols), dtype=int)
    for i, combo in enumerate(combos):
        shared = results[combo[0]]
        for ds in combo[1:]:
            shared = shared & results[ds]
        for j, target in enumerate(names):
            matrix[i][j] = len(shared & results[target])

    fig_h = max(5, n_rows * 0.5 + 2)
    fig, ax = plt.subplots(figsize=(8, fig_h))
    cmap = plt.cm.YlOrRd
    off_diag = matrix.copy().astype(float)

    for i, combo in enumerate(combos):
        for j, target in enumerate(names):
            if target in combo:
                off_diag[i][j] = np.nan

    valid_max = np.nanmax(off_diag) if not np.all(np.isnan(off_diag)) else 1
    if valid_max == 0 or np.isnan(valid_max):
        valid_max = 1

    im = ax.imshow(off_diag, cmap=cmap, aspect='auto', vmin=0, vmax=valid_max)

    for i, combo in enumerate(combos):
        for j, target in enumerate(names):
            if target in combo:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                           fill=True, color='#e0e0e0', zorder=2))

    for i in range(n_rows):
        for j in range(n_cols):
            val = matrix[i][j]
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
    if combo_size >= 5:
        row_labels = []
        for combo in combos:
            excl = [n for n in names if n not in combo]
            row_labels.append('All except ' + _clean(excl[0]))
    else:
        row_labels = [' + '.join(_clean(c) for c in combo) for combo in combos]

    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=8 if n_rows > 10 else 10, va='center')
    col_labels = [_clean(n) for n in names]
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, fontsize=10, ha='right', rotation=45)
    ax.set_xlabel(xlabel);
    ax.set_ylabel(ylabel)
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(clabel, fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=15)
    for s in ax.spines.values(): s.set_visible(True); s.set_color('#cccccc')
    plt.tight_layout()
    out = OUTPUT_DIR / filename
    plt.savefig(out, dpi=300, bbox_inches='tight');
    plt.close()
    print(f"Saved: {out}")


def main():
    print("=" * 80)
    print("DISCRETE MODEL: ALL HEATMAPS (pairwise, pairs, triplets, quintuplets)")
    print("=" * 80)

    results = mine_all_rules()
    names = list(results.keys())

    if len(names) < 2:
        print("Not enough datasets with rules!")
        return

    # 1. Pairwise
    make_heatmap_pairwise(results, names)

    # 2. Pairs → 3rd
    make_heatmap_combo(results, names, 2,
                       "heatmap_discrete_pairs_generalization.png",
                       "Pairs → Third Dataset (Discrete Model)",
                       "Dataset Pairs", "Third Dataset",
                       "Shared rules (Triple Intersection)")

    # 3. Triplets → 4th
    make_heatmap_combo(results, names, 3,
                       "heatmap_discrete_triplets_generalization.png",
                       "Triplets → Fourth Dataset (Discrete Model)",
                       "Dataset Triplets", "Fourth Dataset",
                       "Shared rules (Quadruple Intersection)")

    # 4. Quintuplets → 6th
    make_heatmap_combo(results, names, 5,
                       "heatmap_discrete_quintuplets_generalization.png",
                       "Leave-One-Out Quintuplets (Discrete Model)",
                       "Quintuplet (5 Datasets)", "Sixth Dataset",
                       "Shared rules (Full 6-way)")

    print("\nDone! All discrete heatmaps generated.")


if __name__ == "__main__":
    main()
