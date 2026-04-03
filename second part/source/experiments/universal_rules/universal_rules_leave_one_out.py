#!/usr/bin/env python3
"""
Experiment: Leave-One-Out Universal Rules Detection

1. Runs ARMADA on each dataset separately (CASE, K-emoCon, CEAP, EmoWorker_v2).
2. Applies predefined rule filters to reject non-emotional or univariate rules.
3. Finds rules that are present in exactly 3 out of 4 datasets.
4. Highlights the dataset that is missing the rule and outputs metrics for the 3 datasets where it is present.
"""

import sys
from pathlib import Path
import pandas as pd
from typing import Dict, List, Set, Tuple

SCRIPT_DIR = Path(__file__).parent
EXPERIMENTS_DIR = SCRIPT_DIR.parent
PROJECT_DIR = EXPERIMENTS_DIR.parent.parent

sys.path.insert(0, str(PROJECT_DIR / "source" / "processing" / "armada"))
# We add experiments dir as we need experiment_utils which is there
sys.path.insert(0, str(EXPERIMENTS_DIR))

# ============================================================================
# EXPERIMENT PARAMETERS
# ============================================================================
MINSUP = 0.3       # 10% minimum support
MINCONF = 0.5      # 10% minimum confidence
MAXGAP = 20        # 20s max gap
MAX_PATTERN_SIZE = 2  # max depth 3

# ============================================================================
# RULE FILTERS
# ============================================================================
# True -> reject rules where ALL states are BVP/HRV/HR-related
FILTER_BVP_ONLY = True

# True -> reject rules where ALL states are EDA-related
FILTER_EDA_ONLY = True

# True -> reject rules where ALL states are peripheral signals without Arousal/Valence
FILTER_PHYSIO_CROSS = True

# True -> reject rules where all states are of the same feature type
FILTER_SINGLE_FEATURE = True

# Import experiment_utils from parent directory
try:
    from experiment_utils import (
        run_armada_on_df,
        extract_rule_signatures,
        filter_rules
    )
except ImportError as e:
    print(f"Error importing experiment_utils: {e}. Make sure sys.path is correct.")
    sys.exit(1)

# Import ARMADA
try:
    from armada_algorithm import ARMADA
except ImportError as e:
    print(f"Error importing armada_algorithm: {e}.")
    sys.exit(1)


def run_armada_on_dataset(
    data_file: Path,
    minsup: float = MINSUP,
    minconf: float = MINCONF,
    maxgap: float = MAXGAP,
    max_pattern_size: int = MAX_PATTERN_SIZE
) -> Tuple[ARMADA, List, List]:
    """Runs ARMADA on a single dataset."""
    print(f"  Loading CSV: {data_file}")
    df = pd.read_csv(data_file)
    return run_armada_on_df(df, minsup, minconf, maxgap, max_pattern_size)


def generate_markdown_report(
    df_loo: pd.DataFrame,
    datasets_count: int,
    output_dir: Path
) -> None:
    """Generates a minimalist Markdown report for leave-one-out rules."""
    lines = []
    lines.append("# Leave-One-Out Universal Rules Report")
    lines.append("")
    lines.append("This report lists the temporal patterns (rules) that are **almost universal** – they occur with high confidence in exactly 3 out of 4 datasets, but are completely absent in the 4th dataset.")
    lines.append("")
    lines.append("## Summary of Missing Rules per Dataset")
    lines.append("")
    lines.append(f"We analyzed {datasets_count} datasets. A total of **{len(df_loo)}** rules were identified as present in exactly 3 datasets.")
    lines.append("")
    
    missing_counts = df_loo['missing_dataset'].value_counts()
    for ds, count in missing_counts.items():
        lines.append(f"- **{ds}** is the outlier for **{count}** rules.")
        
    lines.append("")
    lines.append("## Top Rules Missing per Dataset")
    lines.append("")
    lines.append("The following tables show the top rules missing in each dataset, sorted by their average confidence in the 3 datasets where they *do* appear.")
    lines.append("")
    
    for ds in sorted(df_loo['missing_dataset'].unique()):
        ds_rules = df_loo[df_loo['missing_dataset'] == ds]
        
        lines.append(f"### Missing in: {ds}")
        lines.append("")
        lines.append("| Rule | Avg Conf | Avg Sup | Formed By |")
        lines.append("|---|---|---|---|")
        
        # Take up to top 20 for the report so it's minimalist but informative
        for _, row in ds_rules.head(20).iterrows():
            lines.append(f"| `{row['rule']}` | {row['avg_confidence']:.4f} | {row['avg_support']:.4f} | {row['present_in']} |")
            
        lines.append("")
        
    report_text = "\n".join(lines)
    with open(output_dir / "leave_one_out_report.md", "w") as f:
        f.write(report_text)
    
    print(f"  Zapisano raport Markdown do: {output_dir / 'leave_one_out_report.md'}")


def main():
    """Main experiment function."""

    # Paths
    DATA_DIR = PROJECT_DIR / "data" / "armada_ready"
    OUTPUT_DIR = SCRIPT_DIR / "results"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("EXPERIMENT: LEAVE-ONE-OUT UNIVERSAL RULES DETECTION")
    print("=" * 80)
    print(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Results directory: {OUTPUT_DIR}")
    print()

    # Datasets mapping
    datasets = {
        'CASE': DATA_DIR / "armada_case.csv",
        'K-emoCon': DATA_DIR / "armada_k_emocon.csv",
        'CEAP': DATA_DIR / "armada_ceap.csv",
        'EmoWorker_v2': DATA_DIR / "armada_emoworker_v2.csv"
    }

    # Verify input datasets
    for ds_name, data_file in datasets.items():
        if not data_file.exists():
            print(f"ERROR: Dataset file {data_file} does not exist!")
            print("Have you regenerated the armada CSVs after multi-resolution update?")
            sys.exit(1)

    all_results = {}
    rules_signatures = {}

    for ds_name, data_file in datasets.items():
        print(f"\n{'='*60}")
        print(f"Processing Dataset: {ds_name}")
        print(f"{'='*60}")

        armada, patterns, rules = run_armada_on_dataset(data_file)
        all_results[ds_name] = (armada, patterns, rules)
        
        # Extract purely signatures to filter
        raw_signatures = extract_rule_signatures(rules)
        filtered_signatures = filter_rules(
            raw_signatures, 
            FILTER_BVP_ONLY, 
            FILTER_EDA_ONLY, 
            FILTER_PHYSIO_CROSS, 
            FILTER_SINGLE_FEATURE
        )
        
        rules_signatures[ds_name] = set(filtered_signatures)
        
        print(f"  Total Rules: {len(rules)}")
        print(f"  Rules after filtering: {len(filtered_signatures)}")

    # Finding the intersection of rules across all datasets
    print("\n" + "=" * 80)
    print("CALCULATING LEAVE-ONE-OUT INTERSECTION")
    print("=" * 80)

    dataset_names = set(datasets.keys())
    all_unique_rules = set()
    for rules in rules_signatures.values():
        all_unique_rules.update(rules)
        
    leave_one_out_rules = []
    
    # Precompute signature maps to avoid O(N^2) complexity
    sig_to_rule_map = {}
    for ds_name, (armada, patterns, rules_obj) in all_results.items():
        sig_map = {}
        for r in rules_obj:
            sig = f"{r.antecedent.get_relation_description()} => {r.consequent.get_relation_description()}"
            sig_map[sig] = r
        sig_to_rule_map[ds_name] = sig_map
    
    for rule in all_unique_rules:
        present_in = {ds for ds, rules in rules_signatures.items() if rule in rules}
        
        if len(present_in) == 3:
            missing_ds = (dataset_names - present_in).pop()
            
            # Extract metrics from the 3 datasets where it is present
            confidences = []
            supports = []
            
            for ds in present_in:
                r_match = sig_to_rule_map[ds].get(rule)
                if r_match:
                    confidences.append(r_match.confidence)
                    supports.append(r_match.support)
                else:
                    print(f"WARNING: Rule {rule} not found in {ds} objects despite signature match!")
                        
            avg_conf = sum(confidences) / len(confidences) if confidences else 0
            avg_sup = sum(supports) / len(supports) if supports else 0
            
            leave_one_out_rules.append({
                'rule': rule,
                'missing_dataset': missing_ds,
                'present_in': ", ".join(present_in),
                'avg_confidence': round(avg_conf, 4),
                'avg_support': round(avg_sup, 4)
            })

    df_loo = pd.DataFrame(leave_one_out_rules)
    if len(df_loo) > 0:
        df_loo = df_loo.sort_values(['missing_dataset', 'avg_confidence'], ascending=[True, False])
        output_csv = OUTPUT_DIR / "leave_one_out_rules.csv"
        df_loo.to_csv(output_csv, index=False)
        print(f"\nFound {len(df_loo)} rules present in exactly 3 datasets.")
        print(f"Saved details to: {output_csv}")
        
        generate_markdown_report(df_loo, len(datasets), OUTPUT_DIR)
        
        # Summary per missing dataset
        print("\nSummary of missing rules per dataset:")
        missing_counts = df_loo['missing_dataset'].value_counts()
        for ds, count in missing_counts.items():
            print(f"  {ds} is the outlier for {count} rules")
            
        # Top 3 rules per missing dataset
        print("\nTop almost-universal rules (sorted by avg. confidence in their 3 datasets):")
        for ds in sorted(dataset_names):
            ds_rules = df_loo[df_loo['missing_dataset'] == ds]
            if len(ds_rules) > 0:
                print(f"\n  Missing in {ds}:")
                for _, row in ds_rules.head(3).iterrows():
                    print(f"    - {row['rule']} (avg_conf: {row['avg_confidence']})")
    else:
        print("\nNo rules present in exactly 3 datasets found.")

    print("\n" + "=" * 80)
    print("EXPERIMENT FINISHED")
    print("=" * 80)

if __name__ == "__main__":
    main()
