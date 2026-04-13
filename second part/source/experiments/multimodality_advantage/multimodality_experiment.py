#!/usr/bin/env python3
"""
Experiment: RQ 2.3 (Multimodality Advantage)

Analyzes how the performance (Confidence, Support) and quantity of discovered rules
change when increasing the dimensionality (pattern size) from 2 to 3 and 4.
Now extended to test self-annotations (dimensional & discrete) and external annotations.
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

MINSUP = 0.3  # Set to 0.3 to ensure we find complex rules of size 4
MINCONF = 0.5
MAXGAP = 5

FILTER_BVP_ONLY = True
FILTER_EDA_ONLY = True
FILTER_PHYSIO_CROSS = True
FILTER_SINGLE_FEATURE = True

DATA_DIR = PROJECT_DIR / "data" / "armada_ready"
DISCRETE_DIR = EXPERIMENTS_DIR / "emotion_labels" / "results"

GROUPS = {
    "Self Dimensional": {
        "CASE":       DATA_DIR / "armada_case.csv",
        "K-emoCon":   DATA_DIR / "armada_k_emocon.csv",
        "CEAP":       DATA_DIR / "armada_ceap.csv",
        "EmoWorker":  DATA_DIR / "armada_emoworker_v2.csv",
    },
    "Self Discrete": {
        "CASE":       DISCRETE_DIR / "armada_discrete_CASE.csv",
        "K-emoCon":   DISCRETE_DIR / "armada_discrete_K-emoCon.csv",
        "CEAP":       DISCRETE_DIR / "armada_discrete_CEAP.csv",
        "EmoWorker":  DISCRETE_DIR / "armada_discrete_EmoWorker_v2.csv",
    },
    "External Dimensional": {
        "K-emoCon ext": DATA_DIR / "armada_k_emocon_ext.csv",
        "EMBOA":        DATA_DIR / "armada_emboa.csv",
    },
    "External Discrete": {
        "K-emoCon ext": DISCRETE_DIR / "armada_discrete_K-emoCon_ext.csv",
        "EMBOA":        DISCRETE_DIR / "armada_discrete_EMBOA.csv",
    }
}

OUTPUT_DIR = SCRIPT_DIR / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_rule_size(rule):
    a_str = rule.antecedent.get_relation_description()
    c_str = rule.consequent.get_relation_description()
    
    states = a_str.replace(" precedes ", ",").replace(" meets ", ",").replace(" contains ", ",").replace("==", ",")
    a_items = len(states.split(","))
    
    states_c = c_str.replace(" precedes ", ",").replace(" meets ", ",").replace(" contains ", ",").replace("==", ",")
    c_items = len(states_c.split(","))
    
    return a_items + c_items


def process_dataset(df, name):
    print(f"\nProcessing {name}...")
    armada, patterns, rules = run_armada_on_df(df, MINSUP, MINCONF, MAXGAP, max_pattern_size=4)
    
    raw_sigs = extract_rule_signatures(rules)
    filtered_sigs = filter_rules(raw_sigs, FILTER_BVP_ONLY, FILTER_EDA_ONLY,
                                 FILTER_PHYSIO_CROSS, filter_single_feature=FILTER_SINGLE_FEATURE)
    
    filtered_rules = [r for r in rules if 
                      f"{r.antecedent.get_relation_description()} => {r.consequent.get_relation_description()}" in filtered_sigs]
    
    print(f"  Total raw rules (size <= 4): {len(rules)}")
    print(f"  Valid Physio->Emotion rules: {len(filtered_rules)}")
    
    stats = {2: [], 3: [], 4: []}
    
    for r in filtered_rules:
        size = get_rule_size(r)
        if size in stats:
            stats[size].append(r)
            
    stats_out = []
    for size, r_list in stats.items():
        if not r_list:
            stats_out.append({
                'size': size, 'count': 0, 'avg_conf': 0, 'max_conf': 0, 'avg_sup': 0, 'avg_lift': 0, 'max_lift': 0
            })
            continue
            
        confs = [r.confidence for r in r_list]
        sups = [r.support for r in r_list]
        lifts = [r.lift for r in r_list]
        
        stats_out.append({
            'size': size,
            'count': len(r_list),
            'avg_conf': np.mean(confs),
            'max_conf': np.max(confs),
            'avg_sup': np.mean(sups),
            'avg_lift': np.mean(lifts),
            'max_lift': np.max(lifts)
        })
        print(f"    Size {size}: count={len(r_list)}, avg_conf={np.mean(confs):.3f}, avg_lift={np.mean(lifts):.3f}, max_conf={np.max(confs):.3f}")
        
    return stats_out


def main():
    print("=" * 80)
    print("RQ 2.3: MULTIMODALITY ADVANTAGE (CROSS-ANNOTATION SUPPORT)")
    print(f"Parameters: minsup={MINSUP}, minconf={MINCONF}, maxgap={MAXGAP}s")
    print("=" * 80)

    all_results = []
    
    all_self_dfs = []
    all_ext_dfs = []
    all_combined_df = []

    for group_name, datasets in GROUPS.items():
        print(f"\n==================== {group_name.upper()} ====================")
        for name, path in datasets.items():
            if not path.exists():
                print(f"Skipping {name} in {group_name}, file not found: {path.name}")
                continue
                
            df = pd.read_csv(path)
            
            # Prepare df for the ALL COMBINED pass
            df_combined = df.copy()
            # Prefix client_id so independent groups don't merge identical client_ids organically
            df_combined['client_id'] = f"{group_name}_{name}_" + df_combined['client_id'].astype(str)
            all_combined_df.append(df_combined)
            
            if "Self" in group_name:
                all_self_dfs.append(df_combined)
            elif "External" in group_name:
                all_ext_dfs.append(df_combined)
            
            stats_out = process_dataset(df, name)
            for s in stats_out:
                s['group'] = group_name
                s['dataset'] = name
                all_results.append(s)

    # ----------------------------------------------------
    # COMBINED PASSES
    # ----------------------------------------------------
    if all_self_dfs:
        print(f"\n==================== ALL SELF (BASELINE) ====================")
        df_all_self = pd.concat(all_self_dfs, ignore_index=True)
        stats_out = process_dataset(df_all_self, "ALL_SELF_MERGED")
        for s in stats_out:
            s['group'] = "Combined baseline"
            s['dataset'] = "ALL_SELF"
            all_results.append(s)

    if all_ext_dfs:
        print(f"\n==================== ALL EXTERNAL (BASELINE) ====================")
        df_all_ext = pd.concat(all_ext_dfs, ignore_index=True)
        stats_out = process_dataset(df_all_ext, "ALL_EXT_MERGED")
        for s in stats_out:
            s['group'] = "Combined baseline"
            s['dataset'] = "ALL_EXTERNAL"
            all_results.append(s)

    if all_combined_df:
        print(f"\n==================== ALL COMBINED (NOISY BASELINE) ====================")
        df_all = pd.concat(all_combined_df, ignore_index=True)
        stats_out = process_dataset(df_all, "ALL_DATASETS_MERGED")
        for s in stats_out:
            s['group'] = "Combined baseline"
            s['dataset'] = "ALL_COMBINED"
            all_results.append(s)

    # Generate Report
    df_res = pd.DataFrame(all_results)
    
    # Optional: Generate bar charts (excluding the Combined baseline to avoid double-counting in sum)
    df_plot = df_res[df_res['group'] != "Combined baseline"]
    if not df_plot.empty:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        colors = ['#4A90E2', '#50E3C2', '#F5A623']
        
        fig.suptitle(f'Multimodality Advantage: Impact of Pattern Size on Physiological Rules\n(Aggregated Across All Datasets & Annotation Topologies | minsup={MINSUP})', fontsize=13, fontweight='bold', y=1.05)
        
        total_counts = df_plot.groupby('size')['count'].sum()
        axes[0].bar(total_counts.index.astype(str), total_counts.values, color=colors)
        axes[0].set_title('Volume of Discovered Rules vs Modality Count', fontsize=11, fontweight='bold', pad=10)
        axes[0].set_xlabel('Pattern Size (N-items)')
        axes[0].set_ylabel('Total Number of Rules')
        for i, v in enumerate(total_counts.values):
            axes[0].text(i, v + (max(total_counts.values)*0.02), str(v), ha='center')

        avg_conf = df_plot[df_plot['count'] > 0].groupby('size')['avg_conf'].mean()
        axes[1].bar(avg_conf.index.astype(str), avg_conf.values, color=colors)
        axes[1].set_title('Average Rule Confidence vs Modality Count', fontsize=11, fontweight='bold', pad=10)
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

    report_lines = [
        "# Multimodality Advantage Report\n",
        f"**Parameters:** minsup={MINSUP}, minconf={MINCONF}, maxgap={MAXGAP}s\n",
        "| Group | Dataset | Size 2 (N) | Size 2 (Avg Conf) | Size 2 (Avg Lift) | Size 3 (N) | Size 3 (Avg Conf) | Size 3 (Avg Lift) | Size 4 (N) | Size 4 (Avg Conf) | Size 4 (Avg Lift) |",
        "|---|---|---|---|---|---|---|---|---|---|---|"
    ]

    # Maintain orderly grouping in the table
    groups_in_order = list(GROUPS.keys()) + ["Combined baseline"]
    
    for group_name in groups_in_order:
        group_df = df_res[df_res['group'] == group_name]
        if group_df.empty:
            continue
            
        datasets_in_group = group_df['dataset'].unique()
        for name in datasets_in_group:
            row = [group_name, name]
            for size in [2, 3, 4]:
                stat = group_df[(group_df['dataset'] == name) & (group_df['size'] == size)]
                if not stat.empty and stat['count'].iloc[0] > 0:
                    row.extend([str(stat['count'].iloc[0]), f"{stat['avg_conf'].iloc[0]:.3f}", f"{stat['avg_lift'].iloc[0]:.3f}"])
                else:
                    row.extend(["0", "-", "-"])
            report_lines.append("| " + " | ".join(row) + " |")

    report_path = OUTPUT_DIR / "multimodality_report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"Saved multimodality report to {report_path}")


if __name__ == "__main__":
    main()
