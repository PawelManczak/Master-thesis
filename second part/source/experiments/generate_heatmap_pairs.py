#!/usr/bin/env python3
"""
Generate a heatmap comparing the intersection of two datasets with a third distinct dataset.
Uses RQ1 parameters (minsup=0.5, minconf=0.5, maxgap=5s).
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import itertools

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent.parent

sys.path.insert(0, str(PROJECT_DIR / "source" / "processing" / "armada"))
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
    "CASE":       DATA_DIR / "armada_case.csv",
    "K-emoCon":   DATA_DIR / "armada_k_emocon.csv",
    "CEAP":       DATA_DIR / "armada_ceap.csv",
    "EmoWorker":  DATA_DIR / "armada_emoworker_v2.csv",
    "K-emo\n(ext)": DATA_DIR / "armada_k_emocon_ext.csv",
    "EMBOA":      DATA_DIR / "armada_emboa.csv",
}

OUTPUT_DIR = Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    # Mine rules for each dataset
    results = {}
    for name, path in DATASETS.items():
        print(f"Processing {name}...")
        df = pd.read_csv(path)
        armada, patterns, rules = run_armada_on_df(df, MINSUP, MINCONF, MAXGAP, MAX_PATTERN_SIZE)
        
        raw_sigs = extract_rule_signatures(rules)
        filtered = filter_rules(raw_sigs, FILTER_BVP_ONLY, FILTER_EDA_ONLY,
                                FILTER_PHYSIO_CROSS, FILTER_SINGLE_FEATURE)
        results[name] = filtered
        print(f"  {name}: {len(filtered)} filtered rules")

    names = list(results.keys())
    
    # Generate all pairs
    pairs = list(itertools.combinations(names, 2))
    n_rows = len(pairs)
    n_cols = len(names)

    # Build pairs intersection matrix
    matrix = np.zeros((n_rows, n_cols), dtype=int)
    for i, (ds1, ds2) in enumerate(pairs):
        shared_pair = results[ds1] & results[ds2]
        for j, target_ds in enumerate(names):
            # Intersection of pair with target dataset
            matrix[i][j] = len(shared_pair & results[target_ds])

    # Create heatmap
    fig, ax = plt.subplots(figsize=(8, 10))

    cmap = plt.cm.YlOrRd
    off_diag = matrix.copy().astype(float)
    
    # Mask exactly when the column dataset is PART of the row pair
    for i, (ds1, ds2) in enumerate(pairs):
        for j, target_ds in enumerate(names):
            if target_ds in (ds1, ds2):
                off_diag[i][j] = np.nan

    # Plot off-diagonal with color map
    im = ax.imshow(off_diag, cmap=cmap, aspect='auto',
                   vmin=0, vmax=np.nanmax(off_diag) if np.nanmax(off_diag) > 0 else 1)

    # Plot pair components with grey color
    for i, (ds1, ds2) in enumerate(pairs):
        for j, target_ds in enumerate(names):
            if target_ds in (ds1, ds2):
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                            fill=True, color='#e0e0e0', zorder=2))

    # Add text annotations
    for i in range(n_rows):
        for j in range(n_cols):
            val = matrix[i][j]
            target_ds = names[j]
            ds1, ds2 = pairs[i]
            if target_ds in (ds1, ds2):
                text_color = '#333333'
                fontweight = 'bold'
                fontsize = 11
            else:
                norm_val = val / max(np.nanmax(off_diag) if not np.isnan(np.nanmax(off_diag)) else 1, 1)
                text_color = 'white' if norm_val > 0.6 else 'black'
                fontweight = 'normal'
                fontsize = 10
            ax.text(j, i, str(val), ha='center', va='center',
                    color=text_color, fontweight=fontweight, fontsize=fontsize,
                    zorder=3)

    # Labels
    row_labels = [p[0].replace('\n', ' ') + ' + ' + p[1].replace('\n', ' ') for p in pairs]
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=10, va='center')
    
    col_labels = [n.replace('\n', ' ') for n in names]
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, fontsize=10, ha='right', rotation=45)

    ax.set_xlabel('Third Dataset')
    ax.set_ylabel('Intersected Dataset Pairs')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label('Shared rules (Triple Intersection)', fontsize=10)

    # Title
    ax.set_title('Generalization of Pairwise Shared Rules to Third Datasets', fontsize=12, fontweight='bold', pad=15)

    # Border
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('#cccccc')

    plt.tight_layout()

    out_png = OUTPUT_DIR / "heatmap_pairs_generalization.png"
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    print(f"Saved PNG to {out_png}")
    
    # Print the top triplets for the report
    print("\nTop 5 Generalizing Triplets:")
    triplets = []
    for i, (ds1, ds2) in enumerate(pairs):
        for j, target_ds in enumerate(names):
            if target_ds not in (ds1, ds2):
                triplets.append({
                    'pair': ds1.replace('\n', ' ') + ' + ' + ds2.replace('\n', ' '),
                    'target': target_ds.replace('\n', ' '),
                    'rules': matrix[i][j]
                })
    
    triplet_df = pd.DataFrame(triplets)
    triplet_df = triplet_df.sort_values(by='rules', ascending=False).head(5)
    print(triplet_df.to_string(index=False))

if __name__ == "__main__":
    main()
