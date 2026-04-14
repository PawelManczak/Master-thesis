#!/usr/bin/env python3
"""
Experiment: Shuffling Baseline (Random Permutation)

Aim: To validate whether the temporal association rules discovered by ARMADA
capture genuine physiological-emotional mechanisms, or whether they could 
arise purely by chance due to the marginal frequencies of the events.

Methodology:
1. Load each dataset in its dimensional representation.
2. For each participant (`client_id`), randomly shuffle the `state` labels
   among their existing time intervals (`start_time`, `end_time`).
   This destroys all real temporal and logical associations between concurrent 
   and sequential events, while strictly preserving:
     a) The exact number of events per participant.
     b) The exact marginal frequencies of every feature state per participant.
     c) The exact temporal interval structure of the dataset.
3. Run ARMADA identically on both the original and the shuffled datasets.
4. Compare the number of significant, filtered cross-modal rules. 
   A sharp drop in rules for the shuffled data confirms that the original 
   rules are structurally meaningful and not random artifacts.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

SCRIPT_DIR = Path(__file__).parent
EXPERIMENTS_DIR = SCRIPT_DIR.parent
PROJECT_DIR = EXPERIMENTS_DIR.parent.parent

sys.path.insert(0, str(PROJECT_DIR / "source" / "processing" / "armada"))
sys.path.insert(0, str(EXPERIMENTS_DIR))

from experiment_utils import (
    run_armada_on_df,
    extract_rule_signatures,
    filter_rules
)

# Standard RQ1 Parameters
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
    "CASE":           DATA_DIR / "armada_case.csv",
    "K-emoCon":       DATA_DIR / "armada_k_emocon.csv",
    "CEAP-360VR":     DATA_DIR / "armada_ceap.csv",
    "EmoWorker_v2":   DATA_DIR / "armada_emoworker_v2.csv",
    "K-emoCon (ext)": DATA_DIR / "armada_k_emocon_ext.csv",
    "EMBOA":          DATA_DIR / "armada_emboa.csv",
}

DISCRETE_DATASETS = {
    "CASE (Discrete)":           EXPERIMENTS_DIR / "emotion_labels" / "results" / "armada_discrete_CASE.csv",
    "K-emoCon (Discrete)":       EXPERIMENTS_DIR / "emotion_labels" / "results" / "armada_discrete_K-emoCon.csv",
    "CEAP-360VR (Discrete)":     EXPERIMENTS_DIR / "emotion_labels" / "results" / "armada_discrete_CEAP.csv",
    "EmoWorker_v2 (Discrete)":   EXPERIMENTS_DIR / "emotion_labels" / "results" / "armada_discrete_EmoWorker_v2.csv",
    "K-emoCon (ext) (Discrete)": EXPERIMENTS_DIR / "emotion_labels" / "results" / "armada_discrete_K-emoCon_ext.csv",
    "EMBOA (Discrete)":          EXPERIMENTS_DIR / "emotion_labels" / "results" / "armada_discrete_EMBOA.csv",
}

RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Set random seed for reproducibility of the shuffle
np.random.seed(42)

def generate_report(results: list):
    """Generate Markdown report summarizing the shuffling baseline results."""
    report_path = RESULTS_DIR / "shuffling_baseline_report.md"
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Experiment: Shuffling Baseline (Random Permutation)\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Overview\n")
        f.write("To validate that the discovered association rules capture genuine physiological-emotional mechanisms rather than artifacts of event frequencies, this experiment compares the rule yield of the original data against a randomized baseline.\n\n")
        f.write("**Shuffling strategy:** For each participant, the physiological and emotional labels (`state` column) were randomly reassigned across their existing time intervals. This completely destroys real temporal relationships but preserves the exact frequency of every event for every participant.\n\n")
        
        f.write("### ARMADA Parameters\n")
        f.write(f"- **minsup**: {MINSUP} ({MINSUP * 100:.0f}%)\n")
        f.write(f"- **minconf**: {MINCONF}\n")
        f.write(f"- **maxgap**: {MAXGAP}s\n")
        f.write(f"- **max_pattern_size**: {MAX_PATTERN_SIZE}\n\n")
        
        f.write("## Results\n\n")
        f.write("| Dataset | Original Rules | Shuffled Rules | Reduction (%) |\n")
        f.write("|---------|----------------|----------------|---------------|\n")
        
        for res in results:
            ds_name = res['dataset']
            orig = res['original_rules']
            shuf = res['shuffled_rules']
            
            if orig > 0:
                reduction = (1.0 - (shuf / orig)) * 100
                red_str = f"{reduction:.1f}%"
            else:
                red_str = "N/A"
                
            f.write(f"| {ds_name} | {orig} | {shuf} | {red_str} |\n")
            
        f.write("\n## Conclusion\n")
        f.write("A drastic reduction in the number of rules discovered in the shuffled datasets indicates that the ARMADA algorithm is capturing genuine, structurally significant temporal relationships rather than random noise.\n")
        
def process_dataset(name: str, path: Path, minsup: float = MINSUP) -> dict:
    print(f"\nProcessing {name} (minsup={minsup})...")
    if not path.exists():
        print(f"  [!] Missing file: {path}")
        return {"dataset": name, "original_rules": 0, "shuffled_rules": 0, "orig_set": set(), "shuf_set": set()}
        
    # 1. Load data
    df = pd.read_csv(path)
    
    # 2. Run ARMADA on original
    print(f"  Running ARMADA on original data...")
    _, _, orig_rules = run_armada_on_df(df.copy(), minsup, MINCONF, MAXGAP, MAX_PATTERN_SIZE)
    orig_sigs = extract_rule_signatures(orig_rules)
    orig_filtered = filter_rules(orig_sigs, FILTER_BVP_ONLY, FILTER_EDA_ONLY, FILTER_PHYSIO_CROSS, FILTER_SINGLE_FEATURE)
    
    # 3. Create shuffled baseline
    print(f"  Shuffling events per participant...")
    df_shuffled = df.copy()
    # Randomly permute the 'state' values within each 'client_id' group
    df_shuffled['state'] = df_shuffled.groupby('client_id')['state'].transform(np.random.permutation)
    
    # 4. Run ARMADA on shuffled
    print(f"  Running ARMADA on shuffled data...")
    _, _, shuf_rules = run_armada_on_df(df_shuffled, minsup, MINCONF, MAXGAP, MAX_PATTERN_SIZE)
    shuf_sigs = extract_rule_signatures(shuf_rules)
    shuf_filtered = filter_rules(shuf_sigs, FILTER_BVP_ONLY, FILTER_EDA_ONLY, FILTER_PHYSIO_CROSS, FILTER_SINGLE_FEATURE)
    
    print(f"  -> Original rules: {len(orig_filtered)}")
    print(f"  -> Shuffled rules: {len(shuf_filtered)}")
    
    return {
        "dataset": name,
        "original_rules": len(orig_filtered),
        "shuffled_rules": len(shuf_filtered),
        "orig_set": orig_filtered,
        "shuf_set": shuf_filtered
    }

def main():
    print("=" * 60)
    print("EXPERIMENT 1: SHUFFLING BASELINE (ALL DATASETS, 30% SUP)")
    print("=" * 60)
    
    results = []
    
    for name, path in DATASETS.items():
        res = process_dataset(name, path)
        results.append(res)
        
    for name, path in DISCRETE_DATASETS.items():
        res = process_dataset(name, path)
        results.append(res)
        
    print("\nGenerating report for Experiment 1...")
    generate_report(results)
    
    print("\nDone! Full report saved to results/shuffling_baseline_report.md")

if __name__ == "__main__":
    main()
