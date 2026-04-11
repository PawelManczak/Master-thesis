#!/usr/bin/env python3
"""
Experiment: RQ 2.3 (Multimodality Advantage)

Analyzes how the performance (Confidence, Support) and quantity of discovered rules
change when increasing the dimensionality (pattern size) from 2 to 3 and 4.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).parent
EXPERIMENTS_DIR = SCRIPT_DIR.parent
PROJECT_DIR = SCRIPT_DIR.parent.parent.parent

sys.path.insert(0, str(PROJECT_DIR / "source" / "processing" / "armada"))
sys.path.insert(0, str(EXPERIMENTS_DIR))

from experiment_utils import (
    run_armada_on_df,
    extract_rule_signatures,
    filter_rules,
)

# Parameters
MINSUP = 0.3  # Lowered to 0.3 to ensure we find complex rules of size 4
MINCONF = 0.5
MAXGAP = 5

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
}

OUTPUT_DIR = SCRIPT_DIR / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_rule_size(rule):
    """
    Calculate total number of items in a temporal rule.
    Assuming Rule has .antecedent and .consequent (both are Pattern objects).
    A Pattern has a string representation or an internal list.
    """
    # Quick hack to count size: count the number of states mentioned
    sig = f"{rule.antecedent.get_relation_description()} => {rule.consequent.get_relation_description()}"
    # Count states. Example: "hr_high precedes eda_low => arousal_medium"
    # We can just count the number of states by seeing how many base items are in the signature.
    # Actually, pattern.size or pattern.get_length() might exist. Let's just string split.
    # Antecedent might be "A" (size 1), "A precedes B" (size 2), "A precedes B precedes C" (size 3)
    a_str = rule.antecedent.get_relation_description()
    c_str = rule.consequent.get_relation_description()
    
    # number of elements = number of strings not equal to 'precedes', 'meets', etc.
    # Easier: count operators and add 1
    # Operators in armada strings: " precedes ", " meets "
    # But wait, what if two events are concurrent? (e.g. hr_high == eda_low)
    # Let's count occurrences of known states
    states = a_str.replace(" precedes ", ",").replace(" meets ", ",").replace(" contains ", ",").replace("==", ",")
    a_items = len(states.split(","))
    
    states_c = c_str.replace(" precedes ", ",").replace(" meets ", ",").replace(" contains ", ",").replace("==", ",")
    c_items = len(states_c.split(","))
    
    return a_items + c_items


def main():
    print("=" * 80)
    print("RQ 2.3: MULTIMODALITY ADVANTAGE")
    print(f"Parameters: minsup={MINSUP}, minconf={MINCONF}, maxgap={MAXGAP}s")
    print("=" * 80)

    # We will run with max_pattern_size = 4 and then categorize the resulting rules
    all_results = []

    for name, path in DATASETS.items():
        if not path.exists():
            print(f"Skipping {name}, not found.")
            continue
            
        print(f"\nProcessing {name}...")
        df = pd.read_csv(path)
        armada, patterns, rules = run_armada_on_df(df, MINSUP, MINCONF, MAXGAP, max_pattern_size=4)
        
        # Apply strict filters to only keep valid Physio -> Emotion rules
        # First filter just signatures
        raw_sigs = extract_rule_signatures(rules)
        filtered_sigs = filter_rules(raw_sigs, FILTER_BVP_ONLY, FILTER_EDA_ONLY,
                                     FILTER_PHYSIO_CROSS, filter_single_feature=FILTER_SINGLE_FEATURE)
        
        # Now keep only actual rule objects that passed the filter
        filtered_rules = [r for r in rules if 
                          f"{r.antecedent.get_relation_description()} => {r.consequent.get_relation_description()}" in filtered_sigs]
        
        print(f"  Total raw rules (size <= 4): {len(rules)}")
        print(f"  Valid Physio->Emotion rules: {len(filtered_rules)}")
        
        # Group by size
        stats = {2: [], 3: [], 4: []}
        
        for r in filtered_rules:
            size = get_rule_size(r)
            if size in stats:
                stats[size].append(r)
            elif size > 4:
                # ARMADA with max_size=4 shouldn't produce >4, but just in case
                pass
                
        for size, r_list in stats.items():
            if not r_list:
                all_results.append({
                    'dataset': name, 'size': size,
                    'count': 0, 'avg_conf': 0, 'max_conf': 0, 'avg_sup': 0
                })
                continue
                
            confs = [r.confidence for r in r_list]
            sups = [r.support for r in r_list]
            
            all_results.append({
                'dataset': name,
                'size': size,
                'count': len(r_list),
                'avg_conf': np.mean(confs),
                'max_conf': np.max(confs),
                'avg_sup': np.mean(sups)
            })
            print(f"    Size {size}: count={len(r_list)}, avg_conf={np.mean(confs):.3f}, max_conf={np.max(confs):.3f}")

    # Generate aggregate stats and plots
    df_res = pd.DataFrame(all_results)
    
    # 1. Bar Chart: Count vs Size
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    colors = ['#4A90E2', '#50E3C2', '#F5A623']
    
    # Group by size across all datasets (sum counts)
    total_counts = df_res.groupby('size')['count'].sum()
    axes[0].bar(total_counts.index.astype(str), total_counts.values, color=colors)
    axes[0].set_title('Total Rules vs Modality Size', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Pattern Size (N-items)')
    axes[0].set_ylabel('Total Number of Rules')
    for i, v in enumerate(total_counts.values):
        axes[0].text(i, v + (max(total_counts.values)*0.02), str(v), ha='center')

    # Group by size (average of dataset averages)
    avg_conf = df_res[df_res['count'] > 0].groupby('size')['avg_conf'].mean()
    axes[1].bar(avg_conf.index.astype(str), avg_conf.values, color=colors)
    axes[1].set_title('Average Confidence vs Modality Size', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Pattern Size (N-items)')
    axes[1].set_ylabel('Average Confidence')
    axes[1].set_ylim(0.4, 1.0)
    for i, v in enumerate(avg_conf.values):
        axes[1].text(i, v + 0.02, f"{v:.3f}", ha='center')

    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plot_path = OUTPUT_DIR / "multimodality_advantage.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved plot to {plot_path}")
    
    # Print LaTeX table snippet
    print("\n--- LaTeX Table Snippet ---")
    print(r"\begin{table}[!ht]")
    print(r"\centering")
    print(r"\small")
    print(r"\caption{RQ 2.3: Multimodality advantage (minsup=0.3). Volume and confidence of multi-item temporal rules across self-annotated datasets.}")
    print(r"\begin{tabular}{|l|c|c|c|c|c|c|}")
    print(r"\hline")
    print(r"\multirow{2}{*}{\textbf{Dataset}} & \multicolumn{2}{c|}{\textbf{Size 2 (1$\rightarrow$1)}} & \multicolumn{2}{c|}{\textbf{Size 3 (2$\rightarrow$1)}} & \multicolumn{2}{c|}{\textbf{Size 4 (3$\rightarrow$1)}} \\")
    print(r"\cline{2-7}")
    print(r" & \textbf{N} & \textbf{Avg Cnf} & \textbf{N} & \textbf{Avg Cnf} & \textbf{N} & \textbf{Avg Cnf} \\")
    print(r"\hline")
    
    for name in DATASETS.keys():
        row = f"{name}"
        for size in [2, 3, 4]:
            stat = df_res[(df_res['dataset'] == name) & (df_res['size'] == size)]
            if len(stat) > 0 and stat['count'].iloc[0] > 0:
                c = stat['count'].iloc[0]
                avg = stat['avg_conf'].iloc[0]
                row += f" & {c} & {avg:.3f}"
            else:
                row += " & 0 & -"
        print(row + r" \\ \hline")
        
    print(r"\end{tabular}")
    print(r"\label{tab:multimodality}")
    print(r"\end{table}")


if __name__ == "__main__":
    main()
