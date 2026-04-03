#!/usr/bin/env python3
"""
Experiment: Find Universal Rules Across All Datasets By Age

1. Loads 4 datasets and corresponding demographic data.
2. Splits each dataset into Male (M) and Female (F) subsets.
3. Runs ARMADA on Male and Female subsets separately across all datasets.
4. Finds purely "Universal Young Rules" (present in ALL evaluated datasets for Young).
5. Finds purely "Universal Old Rules" (present in ALL evaluated datasets for Old).
6. Generates metrics tracking how often these rules appear for each group globally.
"""

import sys
from pathlib import Path
import pandas as pd
import json
from collections import defaultdict
from typing import Dict, List, Set, Tuple

SCRIPT_DIR = Path(__file__).parent
EXPERIMENTS_DIR = SCRIPT_DIR.parent
PROJECT_DIR = EXPERIMENTS_DIR.parent.parent

sys.path.insert(0, str(PROJECT_DIR / "source" / "processing" / "armada"))
sys.path.insert(0, str(EXPERIMENTS_DIR))

# ============================================================================
# EXPERIMENT PARAMETERS
# ============================================================================
MINSUP = 0.1       # 10% minimum support
MINCONF = 0.1      # 10% minimum confidence
MAXGAP = 20        # 30s max gap
MAX_PATTERN_SIZE = 2  # max depth 2

# ============================================================================
# RULE FILTERS
# ============================================================================
FILTER_BVP_ONLY = True
FILTER_EDA_ONLY = True
FILTER_PHYSIO_CROSS = False
FILTER_SINGLE_FEATURE = True

# Imports
try:
    from experiment_utils import (
        run_armada_on_df,
        extract_rule_signatures,
        filter_rules
    )
    from demographic_analysis import (
        load_demographics_from_processed,
        split_data_by_group
    )
except ImportError as e:
    print(f"Error importing utils: {e}. Check sys.path.")
    sys.exit(1)

try:
    from armada_algorithm import ARMADA
except ImportError as e:
    print(f"Error importing armada_algorithm: {e}.")
    sys.exit(1)


def save_age_universal_rules_details(
    universal_rules: Set[str],
    all_results: Dict[str, Tuple],
    output_dir: Path,
    age_tag: str
) -> pd.DataFrame:
    """Saves details of universal rules with per-dataset metrics for specific gender."""
    details = []

    for rule_sig in sorted(universal_rules):
        entry = {"rule": rule_sig}
        confidences = []
        supports = []

        for ds_name, (armada, patterns, rules) in all_results.items():
            for r in rules:
                sig = f"{r.antecedent.get_relation_description()} => {r.consequent.get_relation_description()}"
                if sig == rule_sig:
                    entry[f"{ds_name}_confidence"] = round(r.confidence, 4)
                    entry[f"{ds_name}_support"] = round(r.support, 4)
                    entry[f"{ds_name}_count"] = r.consequent.support_count
                    confidences.append(r.confidence)
                    supports.append(r.support)
                    break

        if confidences:
            entry["avg_confidence"] = round(sum(confidences) / len(confidences), 4)
            entry["min_confidence"] = round(min(confidences), 4)
            entry["avg_support"] = round(sum(supports) / len(supports), 4)
            entry["min_support"] = round(min(supports), 4)

        details.append(entry)

    df = pd.DataFrame(details)

    if 'avg_confidence' in df.columns:
        df = df.sort_values('avg_confidence', ascending=False)
    
    df.to_csv(output_dir / f"universal_{age_tag}_rules_details.csv", index=False)
    return df


def generate_markdown_report(
    datasets_count: int,
    stats_data: Dict[str, Dict],
    universal_rules_young: Set[str],
    universal_rules_old: Set[str],
    df_young: pd.DataFrame,
    df_old: pd.DataFrame,
    output_dir: Path,
    dataset_names: List[str]
) -> None:
    """Generates the Markdown report analyzing rules by age."""

    lines = []
    lines.append("# Universal Rules by Age Experiment")
    lines.append("")
    lines.append("This experiment aims to identify rules that are universally true across all evaluated datasets, split solely by demographic age subsets (Young <= 25 vs Old > 25).")
    lines.append("")
    lines.append("## Experiment Parameters")
    lines.append("")
    lines.append(f"- **minsup**: {MINSUP} ({MINSUP*100:.0f}% participants)")
    lines.append(f"- **minconf**: {MINCONF} ({MINCONF*100:.0f}% confidence)")
    lines.append(f"- **maxgap**: {MAXGAP} seconds")
    lines.append(f"- **max_pattern_size**: {MAX_PATTERN_SIZE}")
    lines.append("")
    
    # Dataset statistics
    lines.append("## Dataset Processing")
    lines.append("")
    
    lines.append("| Dataset | Participants (Young) | Participants (Old) | Rules Filtered (Young) | Rules Filtered (Old) |")
    lines.append("|-------|-------------|---------|-------|-------|")

    for ds_name in dataset_names:
        stats = stats_data.get(ds_name, {})
        lines.append(
            f"| **{ds_name}** "
            f"| {stats.get('young_clients', 0)} "
            f"| {stats.get('old_clients', 0)} "
            f"| {stats.get('young_filtered', 0)} "
            f"| {stats.get('old_filtered', 0)} |"
        )
    lines.append("")

    # Universal comparison summary
    common_both = universal_rules_young & universal_rules_old
    unique_young = universal_rules_young - universal_rules_old
    unique_old = universal_rules_old - universal_rules_young

    lines.append("## Age Universal Rules Identification")
    lines.append("")
    lines.append(f"- **Universal rules found in ALL datasets for Young**: {len(universal_rules_young)}")
    lines.append(f"- **Universal rules found in ALL datasets for Old**: {len(universal_rules_old)}")
    lines.append(f"- **Universal rules spanning across BOTH Age Groups**: {len(common_both)}")
    lines.append(f"- **Universal rules UNIQUELY for Young**: {len(unique_young)}")
    lines.append(f"- **Universal rules UNIQUELY for Old**: {len(unique_old)}")
    lines.append("")

    # MALE TABLE
    if len(df_young) > 0:
        lines.append(f"### Universal Rules for Young (Top metrics globally)")
        lines.append("")
        ds_headers = " | ".join(dataset_names)
        header = f"| Rule | Avg Conf | Avg Sup | Min Conf | Min Sup | {ds_headers} |"
        separator = "|---|---|---|---|---|" + "|".join(["---" for _ in dataset_names]) + "|"
        lines.append(header)
        lines.append(separator)

        for _, row in df_young.iterrows():
            avg_conf = f"{row.get('avg_confidence', 0):.3f}"
            avg_sup = f"{row.get('avg_support', 0):.3f}"
            min_conf = f"{row.get('min_confidence', 0):.3f}"
            min_sup = f"{row.get('min_support', 0):.3f}"
            
            # Marker if common
            rule_str = f"`{row['rule']}`"
            if row['rule'] in common_both:
                rule_str += " *(common)*"
            elif row['rule'] in unique_young:
                rule_str += " *(unique)*"

            line_parts = [f"| {rule_str} | **{avg_conf}** | {avg_sup} | {min_conf} | {min_sup} |"]
            
            for ds_name in dataset_names:
                ds_conf = row.get(f'{ds_name}_confidence', 'N/A')
                ds_sup = row.get(f'{ds_name}_support', 'N/A')
                ds_count = row.get(f'{ds_name}_count', 'N/A')
                
                if isinstance(ds_conf, float): ds_conf = f"{ds_conf:.2f}"
                if isinstance(ds_sup, float): ds_sup = f"{ds_sup:.2f}"
                if isinstance(ds_count, (float, int)): ds_count = f"{int(ds_count)}"
                
                line_parts.append(f" c:{ds_conf} s:{ds_sup} n:{ds_count} |")
                
            lines.append("".join(line_parts))
        lines.append("")

    # FEMALE TABLE
    if len(df_old) > 0:
        lines.append(f"### Universal Rules for Old (Top metrics globally)")
        lines.append("")
        ds_headers = " | ".join(dataset_names)
        header = f"| Rule | Avg Conf | Avg Sup | Min Conf | Min Sup | {ds_headers} |"
        separator = "|---|---|---|---|---|" + "|".join(["---" for _ in dataset_names]) + "|"
        lines.append(header)
        lines.append(separator)

        for _, row in df_old.iterrows():
            avg_conf = f"{row.get('avg_confidence', 0):.3f}"
            avg_sup = f"{row.get('avg_support', 0):.3f}"
            min_conf = f"{row.get('min_confidence', 0):.3f}"
            min_sup = f"{row.get('min_support', 0):.3f}"
            
            # Marker if common
            rule_str = f"`{row['rule']}`"
            if row['rule'] in common_both:
                rule_str += " *(common)*"
            elif row['rule'] in unique_old:
                rule_str += " *(unique)*"

            line_parts = [f"| {rule_str} | **{avg_conf}** | {avg_sup} | {min_conf} | {min_sup} |"]
            
            for ds_name in dataset_names:
                ds_conf = row.get(f'{ds_name}_confidence', 'N/A')
                ds_sup = row.get(f'{ds_name}_support', 'N/A')
                ds_count = row.get(f'{ds_name}_count', 'N/A')
                
                if isinstance(ds_conf, float): ds_conf = f"{ds_conf:.2f}"
                if isinstance(ds_sup, float): ds_sup = f"{ds_sup:.2f}"
                if isinstance(ds_count, (float, int)): ds_count = f"{int(ds_count)}"
                
                line_parts.append(f" c:{ds_conf} s:{ds_sup} n:{ds_count} |")
                
            lines.append("".join(line_parts))
        lines.append("")

    # Save report
    report_text = "\n".join(lines)
    with open(output_dir / "age_universal_rules_report.md", "w") as f:
        f.write(report_text)

    print(f"Report saved to: {output_dir / 'age_universal_rules_report.md'}")


def main():
    """Main experiment function."""

    DATA_DIR = PROJECT_DIR / "data" / "armada_ready"
    OUTPUT_DIR = SCRIPT_DIR / "results"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("EXPERIMENT: AGE UNIVERSAL RULES DETECTION")
    print("=" * 80)
    
    # 1. Load overarching demographic context
    print("Loading global demographics mapping...")
    demo_df = load_demographics_from_processed()
    # Align mapping tags
    demo_df['gender_clean'] = demo_df['gender'].apply(
        lambda x: 'M' if str(x).upper() in ['M', 'MALE', '1', 'MAN'] else
                  ('F' if str(x).upper() in ['F', 'FEMALE', '2', 'WOMAN'] else None)
    )

    datasets = {
        'CASE': DATA_DIR / "armada_case.csv",
        'K-emoCon': DATA_DIR / "armada_k_emocon.csv",
        'CEAP': DATA_DIR / "armada_ceap.csv",
        'EmoWorker_v2': DATA_DIR / "armada_emoworker_v2.csv"
    }

    # Verify existing inputs
    for ds_name, data_file in datasets.items():
        if not data_file.exists():
            print(f"ERROR: Dataset file {data_file} does not exist!")
            sys.exit(1)

    all_results_young = {}
    all_results_old = {}
    
    rules_signatures_young = {}
    rules_signatures_old = {}
    
    # Metric tracking
    stats_data = {}

    for ds_name, data_file in datasets.items():
        print(f"\n{'='*60}")
        print(f"Processing Dataset: {ds_name}")
        print(f"{'='*60}")
        
        # Load ARMADA CSV dataset specifically
        df = pd.read_csv(data_file)
            
        # Isolate demographic lookup for the current dataset ONLY
        ds_demo = demo_df[demo_df['dataset'] == ds_name].copy()
        print(f"DEBUG: {ds_name} df client_ids example: {df['client_id'].unique()[:3]}")
        print(f"DEBUG: {ds_name} ds_demo client_ids example: {ds_demo['client_id'].unique()[:3]}")
        
        # Split internally into nested partitions: young and old
        groups = split_data_by_group(df, ds_demo, 'binary_age_group')
        
        # MALE GROUP PROCESSING
        if 'young' in groups and len(groups['young']['client_id'].unique()) >= 1:
            print(f"\n  [YOUNG] Running ARMADA for {ds_name}...")
            armada_young, patterns_young, rules_young = run_armada_on_df(groups['young'], MINSUP, MINCONF, MAXGAP, MAX_PATTERN_SIZE)
            all_results_young[ds_name] = (armada_young, patterns_young, rules_young)
            
            raw_sigs_young = extract_rule_signatures(rules_young)
            filtered_sigs_young = filter_rules(raw_sigs_young, FILTER_BVP_ONLY, FILTER_EDA_ONLY, FILTER_PHYSIO_CROSS, FILTER_SINGLE_FEATURE)
            rules_signatures_young[ds_name] = filtered_sigs_young
            
            print(f"    Raw Rules: {len(rules_young)}, Filtered: {len(filtered_sigs_young)}, Clients: {armada_young.num_clients}")
            young_clients_var = armada_young.num_clients
            young_filtered_var = len(filtered_sigs_young)
        else:
            print(f"  [YOUNG] Skipping {ds_name} - No Young Participants found.")
            rules_signatures_young[ds_name] = set()
            young_clients_var = 0
            young_filtered_var = 0
            
        # FEMALE GROUP PROCESSING
        if 'old' in groups and len(groups['old']['client_id'].unique()) >= 1:
            print(f"\n  [OLD] Running ARMADA for {ds_name}...")
            armada_old, patterns_old, rules_old = run_armada_on_df(groups['old'], MINSUP, MINCONF, MAXGAP, MAX_PATTERN_SIZE)
            all_results_old[ds_name] = (armada_old, patterns_old, rules_old)
            
            raw_sigs_old = extract_rule_signatures(rules_old)
            filtered_sigs_old = filter_rules(raw_sigs_old, FILTER_BVP_ONLY, FILTER_EDA_ONLY, FILTER_PHYSIO_CROSS, FILTER_SINGLE_FEATURE)
            rules_signatures_old[ds_name] = filtered_sigs_old
            
            print(f"    Raw Rules: {len(rules_old)}, Filtered: {len(filtered_sigs_old)}, Clients: {armada_old.num_clients}")
            old_clients_var = armada_old.num_clients
            old_filtered_var = len(filtered_sigs_old)
        else:
            print(f"  [OLD] Skipping {ds_name} - No Old Participants found.")
            rules_signatures_old[ds_name] = set()
            old_clients_var = 0
            old_filtered_var = 0
            
        stats_data[ds_name] = {
            'young_clients': young_clients_var, 'young_filtered': young_filtered_var,
            'old_clients': old_clients_var, 'old_filtered': old_filtered_var
        }


    print("\n" + "=" * 80)
    print("CALCULATING AGE-WISE UNIVERSAL INTERSECTIONS")
    print("=" * 80)

    dataset_names = list(datasets.keys())
    
    # Men global intersection
    universal_young = set(rules_signatures_young[dataset_names[0]])
    for ds_name in dataset_names[1:]:
        universal_young &= rules_signatures_young[ds_name]
        
    # Female global intersection
    universal_old = set(rules_signatures_old[dataset_names[0]])
    for ds_name in dataset_names[1:]:
        universal_old &= rules_signatures_old[ds_name]
        
    print(f"\nUniversal Rules across all datasets for YOUNG: {len(universal_young)}")
    print(f"Universal Rules across all datasets for OLD: {len(universal_old)}")
    print(f"Rules common to BOTH age groups globally: {len(universal_young & universal_old)}")
    
    # Save CSV outputs
    df_young = pd.DataFrame()
    df_old = pd.DataFrame()
    
    if len(universal_young) > 0:
        df_young = save_age_universal_rules_details(universal_young, all_results_young, OUTPUT_DIR, "young")
    if len(universal_old) > 0:
        df_old = save_age_universal_rules_details(universal_old, all_results_old, OUTPUT_DIR, "old")
        
    # Write aggregated metrics globally via MD report
    generate_markdown_report(
        datasets_count=len(dataset_names),
        stats_data=stats_data,
        universal_rules_young=universal_young,
        universal_rules_old=universal_old,
        df_young=df_young,
        df_old=df_old,
        output_dir=OUTPUT_DIR,
        dataset_names=dataset_names
    )
        
    print("\n" + "=" * 80)
    print("EXPERIMENT FINISHED")
    print(f"Results saved in: {OUTPUT_DIR}")
    print("=" * 80)

if __name__ == "__main__":
    main()
