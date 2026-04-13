#!/usr/bin/env python3
"""
Generate a pairwise shared-rules heatmap for all 6 datasets.
Uses RQ1 parameters (minsup=0.5, minconf=0.5, maxgap=5s).
"""

import sys
from pathlib import Path

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
    "K-emoCon\n(ext)": DATA_DIR / "armada_k_emocon_ext.csv",
    "EMBOA": DATA_DIR / "armada_emboa.csv",
}

OUTPUT_DIR = Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
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
                   vmin=0, vmax=np.nanmax(off_diag))

    for i in range(n):
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1,
                                   fill=True, color='#e0e0e0', zorder=2))

    for i in range(n):
        for j in range(n):
            val = matrix[i][j]
            if i == j:
                text_color = '#333333'
                fontweight = 'bold'
                fontsize = 11
            else:
                norm_val = val / max(np.nanmax(off_diag), 1)
                text_color = 'white' if norm_val > 0.6 else 'black'
                fontweight = 'normal'
                fontsize = 10
            ax.text(j, i, str(val), ha='center', va='center',
                    color=text_color, fontweight=fontweight, fontsize=fontsize,
                    zorder=3)

    display_names = [n.replace('\n', '\n') for n in names]
    ax.set_xticks(range(n))
    ax.set_xticklabels(display_names, fontsize=10, ha='center')
    ax.set_yticks(range(n))
    ax.set_yticklabels(display_names, fontsize=10, va='center')

    ax.set_xlabel('')
    ax.set_ylabel('')

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Shared rules', fontsize=10)

    ax.set_title('Pairwise Shared Rules Between Datasets', fontsize=13, fontweight='bold', pad=12)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('#cccccc')

    plt.tight_layout()

    out_path = OUTPUT_DIR / "heatmap_shared_rules.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {out_path}")

    out_png = OUTPUT_DIR / "heatmap_shared_rules.png"
    fig.savefig(out_png, dpi=200, bbox_inches='tight')
    print(f"Saved: {out_png}")

    plt.close()


if __name__ == "__main__":
    main()
