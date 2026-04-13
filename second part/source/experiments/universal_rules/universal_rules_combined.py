#!/usr/bin/env python3
"""
Universal Rules - Combined Variant

Instead of extracting rules per dataset and finding the strict intersection,
this script combines all datasets into a single global dataset. ARMADA constraints
are evaluated globally on this mixed data pool.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

SCRIPT_DIR = Path(__file__).parent
SCRIPT_DIR = Path(__file__).parent
EXPERIMENTS_DIR = SCRIPT_DIR.parent
PROJECT_DIR = EXPERIMENTS_DIR.parent.parent

sys.path.insert(0, str(PROJECT_DIR / "source" / "processing" / "armada"))
sys.path.insert(0, str(EXPERIMENTS_DIR))

from experiment_utils import (
    run_armada_on_df,
    extract_rule_signatures,
    filter_rules,
)

MINSUP = 0.1
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
}

OUTPUT_DIR = SCRIPT_DIR / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_markdown_report(rules, output_dir):
    lines = [
        "# Aggregated Universal Rules (Combined Global Minimum Support)",
        "",
        "This experiment evaluates universal rules by completely mixing the datasets ",
        "into a single, massive dataframe before applying ARMADA.",
        "",
        "## Parameters",
        "| Parameter | Value |",
        "|-----------|-------|",
        f"| minsup | {MINSUP} (Global) |",
        f"| minconf | {MINCONF} |",
        f"| maxgap | {MAXGAP}s |",
        f"| max_pattern_size | {MAX_PATTERN_SIZE} |",
        "",
        "## Discovered Rules",
        "",
        "| Antecedent | Consequent | Confidence | Lift | Support |",
        "|------------|------------|------------|------|---------|"
    ]

    for rule in rules:
        ante = rule.antecedent.get_relation_description()
        cons = rule.consequent.get_relation_description()
        
        # Escape markdown pipe if any format issues occur, though unlikely in our states
        ante = ante.replace('|', '\\|')
        cons = cons.replace('|', '\\|')
        
        lines.append(f"| `{ante}` | `{cons}` | {rule.confidence:.4f} | {rule.lift:.4f} | {rule.support:.4f} |")

    report_path = output_dir / "universal_rules_combined_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main():
    print("=" * 80)
    print("UNIVERSAL RULES: COMBINED AGGREGATION")
    print(f"Parameters: minsup={MINSUP}, minconf={MINCONF}")
    print("=" * 80)

    all_dfs = []
    
    for ds_name, data_file in DATASETS.items():
        if not data_file.exists():
            print(f"ERROR: {data_file} not found!")
            sys.exit(1)
            
        print(f"Loading {ds_name}...")
        df = pd.read_csv(data_file)
        # Prevent identical client_ids from merging
        df['client_id'] = f"{ds_name}_" + df['client_id'].astype(str)
        all_dfs.append(df)

    global_df = pd.concat(all_dfs, ignore_index=True)
    total_clients = global_df['client_id'].nunique()
    
    print(f"\nCreated global aggregated dataset with {total_clients} distinct participant sessions.")
    print("Mining global patterns...")
    
    armada, patterns, rules = run_armada_on_df(global_df, MINSUP, MINCONF, MAXGAP, MAX_PATTERN_SIZE)
    
    print(f"\nGenerated {len(rules)} raw rules.")
    
    raw_sigs = extract_rule_signatures(rules)
    filtered_sigs = filter_rules(
        raw_sigs,
        FILTER_BVP_ONLY,
        FILTER_EDA_ONLY,
        FILTER_PHYSIO_CROSS,
        FILTER_SINGLE_FEATURE
    )
    
    # Filter the actual rule objects based on valid signatures
    valid_rules = []
    for r in rules:
        sig = f"{r.antecedent.get_relation_description()} => {r.consequent.get_relation_description()}"
        if sig in filtered_sigs:
            valid_rules.append(r)
            
    print(f"Rules surviving physiological filters: {len(valid_rules)}")
    
    # Sort rules by confidence descending
    valid_rules.sort(key=lambda r: r.confidence, reverse=True)
    
    out_path = generate_markdown_report(valid_rules, OUTPUT_DIR)
    
    print(f"\nFinished. Results saved to: {out_path}")


if __name__ == "__main__":
    main()
