#!/usr/bin/env python3
"""
Experiment 2: Universal Rules Shuffling Baseline (Self-Annotated Only)

Aim: To validate whether the "Universal Rules" discovered across multiple datasets 
(at lower support thresholds like 10%) are genuine or structural artifacts.

Methodology:
1. Focuses on the 4 self-annotated datasets at minsup = 10%.
2. For each participant, randomly shuffle the `state` labels among their 
   existing time intervals to destroy real temporal sequences.
3. Find the set of universal intersecting rules in the original datasets.
4. Find the set of universal intersecting rules in the shuffled datasets.
5. Check if any real universal patterns overlap with the randomized ones.
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

MINSUP = 0.1
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
}

RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)
        
def process_dataset(name: str, path: Path, minsup: float) -> dict:
    print(f"\nProcessing {name} (minsup={minsup})...")
    if not path.exists():
        print(f"  [!] Missing file: {path}")
        return {"dataset": name, "orig_set": set(), "shuf_set": set(), "orig_map": {}, "shuf_map": {}}
        
    # 1. Load data
    df = pd.read_csv(path)
    
    # 2. Run ARMADA on original
    print(f"  Running ARMADA on original data...")
    _, _, orig_rules = run_armada_on_df(df.copy(), minsup, MINCONF, MAXGAP, MAX_PATTERN_SIZE)
    orig_rule_map = {}
    for r in orig_rules:
        sig = f"{r.antecedent.get_relation_description()} => {r.consequent.get_relation_description()}"
        orig_rule_map[sig] = r

    orig_sigs = set(orig_rule_map.keys())
    orig_filtered = filter_rules(orig_sigs, FILTER_BVP_ONLY, FILTER_EDA_ONLY, FILTER_PHYSIO_CROSS, FILTER_SINGLE_FEATURE)
    
    # 3. Create shuffled baseline
    print(f"  Shuffling events per participant...")
    df_shuffled = df.copy()
    df_shuffled['state'] = df_shuffled.groupby('client_id')['state'].transform(np.random.permutation)
    
    # 4. Run ARMADA on shuffled
    print(f"  Running ARMADA on shuffled data...")
    _, _, shuf_rules = run_armada_on_df(df_shuffled, minsup, MINCONF, MAXGAP, MAX_PATTERN_SIZE)
    shuf_rule_map = {}
    for r in shuf_rules:
        sig = f"{r.antecedent.get_relation_description()} => {r.consequent.get_relation_description()}"
        shuf_rule_map[sig] = r

    shuf_sigs = set(shuf_rule_map.keys())
    shuf_filtered = filter_rules(shuf_sigs, FILTER_BVP_ONLY, FILTER_EDA_ONLY, FILTER_PHYSIO_CROSS, FILTER_SINGLE_FEATURE)
    
    print(f"  -> Original rules: {len(orig_filtered)}")
    print(f"  -> Shuffled rules: {len(shuf_filtered)}")
    
    return {
        "dataset": name,
        "orig_set": orig_filtered,
        "shuf_set": shuf_filtered,
        "orig_map": orig_rule_map,
        "shuf_map": shuf_rule_map
    }

def main():
    print("=" * 60)
    print("EXPERIMENT 2: UNIVERSAL RULES SHUFFLING (SELF-ANNOTATED, 10% SUP)")
    print("=" * 60)
    
    target_datasets = ["CASE", "K-emoCon", "CEAP-360VR", "EmoWorker_v2"]
    
    orig_sets = {}
    shuf_sets = {}
    
    orig_maps = {}
    shuf_maps = {}
    
    for name in target_datasets:
        path = DATASETS[name]
        res = process_dataset(name, path, minsup=MINSUP)
        orig_sets[name] = res["orig_set"]
        shuf_sets[name] = res["shuf_set"]
        orig_maps[name] = res["orig_map"]
        shuf_maps[name] = res["shuf_map"]
        
    # Find universal intersection across all 4 datasets
    universal_orig = set.intersection(*orig_sets.values()) if orig_sets else set()
    universal_shuf = set.intersection(*shuf_sets.values()) if shuf_sets else set()
    
    # Check if ANY of the real universal rules appear in the shuffled universal rules
    shared_in_both = universal_orig & universal_shuf
    
    report_path = RESULTS_DIR / "shuffling_universal_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Experiment 2: Universal Rules Shuffling (Self-Annotated)\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Overview\n")
        f.write(f"Focuses on the 4 self-annotated datasets at a lowered threshold (**minsup**: {MINSUP}). Computes the intersection of rules across all 4 datasets to find 'universal' rules, and compares this against the universal rules found in randomly shuffled data to ensure the universality is genuine.\n\n")
        
        f.write("## Universal Intersections (Across all 4 datasets)\n\n")
        f.write(f"- **Universal Original Rules discovered:** {len(universal_orig)}\n")
        f.write(f"- **Universal Shuffled Rules discovered:** {len(universal_shuf)}\n")
        f.write(f"- **Original Universal Rules present in Shuffled Universal:** {len(shared_in_both)}\n\n")
        
        if universal_orig:
            f.write("### Original Universal Rules:\n")
            # Calculate averages and sort by Avg Conf
            orig_stats = []
            for r in universal_orig:
                avg_conf = np.mean([orig_maps[d][r].confidence for d in target_datasets if r in orig_maps[d]])
                avg_sup = np.mean([orig_maps[d][r].support for d in target_datasets if r in orig_maps[d]])
                avg_lift = np.mean([orig_maps[d][r].lift for d in target_datasets if r in orig_maps[d]])
                orig_stats.append((r, avg_conf, avg_sup, avg_lift))
            
            orig_stats.sort(key=lambda x: -x[1]) # Sort descending by avg_conf
            
            f.write("| Rule Signature | Avg Conf | Avg Sup | Avg Lift |\n")
            f.write("| --- | --- | --- | --- |\n")
            for stat in orig_stats:
                f.write(f"| `{stat[0]}` | {stat[1]:.4f} | {stat[2]:.4f} | {stat[3]:.4f} |\n")
                
        if universal_shuf:
            f.write("\n### Shuffled Universal Rules:\n")
            shuf_stats = []
            for r in universal_shuf:
                avg_conf = np.mean([shuf_maps[d][r].confidence for d in target_datasets if r in shuf_maps[d]])
                avg_sup = np.mean([shuf_maps[d][r].support for d in target_datasets if r in shuf_maps[d]])
                avg_lift = np.mean([shuf_maps[d][r].lift for d in target_datasets if r in shuf_maps[d]])
                shuf_stats.append((r, avg_conf, avg_sup, avg_lift))
            
            shuf_stats.sort(key=lambda x: -x[1]) # Sort descending by avg_conf
            
            f.write("| Rule Signature | Avg Conf | Avg Sup | Avg Lift |\n")
            f.write("| --- | --- | --- | --- |\n")
            for stat in shuf_stats:
                f.write(f"| `{stat[0]}` | {stat[1]:.4f} | {stat[2]:.4f} | {stat[3]:.4f} |\n")
                
        if shared_in_both:
            f.write("\n### Rules crossing over (found in both):\n")
            for r in sorted(list(shared_in_both)):
                f.write(f"- `{r}`\n")
                
    print("\nDone! Full report saved to results/shuffling_universal_report.md")

if __name__ == "__main__":
    main()
