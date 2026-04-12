#!/usr/bin/env python3
"""
Experiment: Find Universal Rules Across All Datasets By Gender

1. Loads 4 datasets and corresponding demographic data.
2. Splits each dataset into Male (M) and Female (F) subsets.
3. Runs ARMADA on Male and Female subsets separately across all datasets.
4. Finds purely "Universal Male Rules" (present in ALL evaluated datasets for Men).
5. Finds purely "Universal Female Rules" (present in ALL evaluated datasets for Women).
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

MINSUP = 0.1       # 10% minimum support
MINCONF = 0.1      # 10% minimum confidence
MAXGAP = 20
MAX_PATTERN_SIZE = 2

FILTER_BVP_ONLY = True
FILTER_EDA_ONLY = True
FILTER_PHYSIO_CROSS = False
FILTER_SINGLE_FEATURE = True

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


def save_gender_universal_rules_details(
    universal_rules: Set[str],
    all_results: Dict[str, Tuple],
    output_dir: Path,
    gender_tag: str
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
    
    df.to_csv(output_dir / f"universal_{gender_tag}_rules_details.csv", index=False)
    return df


def generate_markdown_report(
    datasets_count: int,
    stats_data: Dict[str, Dict],
    universal_rules_m: Set[str],
    universal_rules_f: Set[str],
    df_m: pd.DataFrame,
    df_f: pd.DataFrame,
    output_dir: Path,
    dataset_names: List[str]
) -> None:
    """Generates the Markdown report analyzing rules by gender."""

    lines = []
    lines.append("# Universal Rules by Gender Experiment")
    lines.append("")
    lines.append("This experiment aims to identify rules that are universally true across all evaluated datasets, split solely by demographic gender subsets (Male vs Female).")
    lines.append("")
    lines.append("## Experiment Parameters")
    lines.append("")
    lines.append(f"- **minsup**: {MINSUP} ({MINSUP*100:.0f}% participants)")
    lines.append(f"- **minconf**: {MINCONF} ({MINCONF*100:.0f}% confidence)")
    lines.append(f"- **maxgap**: {MAXGAP} seconds")
    lines.append(f"- **max_pattern_size**: {MAX_PATTERN_SIZE}")
    lines.append("")

    lines.append("## Dataset Processing")
    lines.append("")
    
    lines.append("| Dataset | Participants (M) | Participants (F) | Rules Filtered (M) | Rules Filtered (F) |")
    lines.append("|-------|-------------|---------|-------|-------|")

    for ds_name in dataset_names:
        stats = stats_data.get(ds_name, {})
        lines.append(
            f"| **{ds_name}** "
            f"| {stats.get('M_clients', 0)} "
            f"| {stats.get('F_clients', 0)} "
            f"| {stats.get('M_filtered', 0)} "
            f"| {stats.get('F_filtered', 0)} |"
        )
    lines.append("")

    common_both = universal_rules_m & universal_rules_f
    unique_m = universal_rules_m - universal_rules_f
    unique_f = universal_rules_f - universal_rules_m

    lines.append("## Gender Universal Rules Identification")
    lines.append("")
    lines.append(f"- **Universal rules found in ALL datasets for Males (M)**: {len(universal_rules_m)}")
    lines.append(f"- **Universal rules found in ALL datasets for Females (F)**: {len(universal_rules_f)}")
    lines.append(f"- **Universal rules spanning across BOTH Genders**: {len(common_both)}")
    lines.append(f"- **Universal rules UNIQUELY for Males (M)**: {len(unique_m)}")
    lines.append(f"- **Universal rules UNIQUELY for Females (F)**: {len(unique_f)}")
    lines.append("")

    if len(df_m) > 0:
        lines.append(f"### Universal Rules for Males (Top metrics globally)")
        lines.append("")
        ds_headers = " | ".join(dataset_names)
        header = f"| Rule | Avg Conf | Avg Sup | Min Conf | Min Sup | {ds_headers} |"
        separator = "|---|---|---|---|---|" + "|".join(["---" for _ in dataset_names]) + "|"
        lines.append(header)
        lines.append(separator)

        for _, row in df_m.iterrows():
            avg_conf = f"{row.get('avg_confidence', 0):.3f}"
            avg_sup = f"{row.get('avg_support', 0):.3f}"
            min_conf = f"{row.get('min_confidence', 0):.3f}"
            min_sup = f"{row.get('min_support', 0):.3f}"
            
            # Marker if common
            rule_str = f"`{row['rule']}`"
            if row['rule'] in common_both:
                rule_str += " *(common)*"
            elif row['rule'] in unique_m:
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

    if len(df_f) > 0:
        lines.append(f"### Universal Rules for Females (Top metrics globally)")
        lines.append("")
        ds_headers = " | ".join(dataset_names)
        header = f"| Rule | Avg Conf | Avg Sup | Min Conf | Min Sup | {ds_headers} |"
        separator = "|---|---|---|---|---|" + "|".join(["---" for _ in dataset_names]) + "|"
        lines.append(header)
        lines.append(separator)

        for _, row in df_f.iterrows():
            avg_conf = f"{row.get('avg_confidence', 0):.3f}"
            avg_sup = f"{row.get('avg_support', 0):.3f}"
            min_conf = f"{row.get('min_confidence', 0):.3f}"
            min_sup = f"{row.get('min_support', 0):.3f}"
            
            # Marker if common
            rule_str = f"`{row['rule']}`"
            if row['rule'] in common_both:
                rule_str += " *(common)*"
            elif row['rule'] in unique_f:
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

    report_text = "\n".join(lines)
    with open(output_dir / "gender_universal_rules_report.md", "w") as f:
        f.write(report_text)

    print(f"Report saved to: {output_dir / 'gender_universal_rules_report.md'}")


def main():
    """Main experiment function."""

    DATA_DIR = PROJECT_DIR / "data" / "armada_ready"
    OUTPUT_DIR = SCRIPT_DIR / "results"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("EXPERIMENT: GENDER UNIVERSAL RULES DETECTION")
    print("=" * 80)

    print("Loading global demographics mapping...")
    demo_df = load_demographics_from_processed()
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

    for ds_name, data_file in datasets.items():
        if not data_file.exists():
            print(f"ERROR: Dataset file {data_file} does not exist!")
            sys.exit(1)

    all_results_m = {}
    all_results_f = {}
    
    rules_signatures_m = {}
    rules_signatures_f = {}

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
        
        # Split internally into nested partitions: M and F 
        groups = split_data_by_group(df, ds_demo, 'gender_clean')
        
        # MALE GROUP PROCESSING
        if 'M' in groups and len(groups['M']['client_id'].unique()) >= 1:
            print(f"\n  [MALE] Running ARMADA for {ds_name}...")
            armada_m, patterns_m, rules_m = run_armada_on_df(groups['M'], MINSUP, MINCONF, MAXGAP, MAX_PATTERN_SIZE)
            all_results_m[ds_name] = (armada_m, patterns_m, rules_m)
            
            raw_sigs_m = extract_rule_signatures(rules_m)
            filtered_sigs_m = filter_rules(raw_sigs_m, FILTER_BVP_ONLY, FILTER_EDA_ONLY, FILTER_PHYSIO_CROSS, FILTER_SINGLE_FEATURE)
            rules_signatures_m[ds_name] = filtered_sigs_m
            
            print(f"    Raw Rules: {len(rules_m)}, Filtered: {len(filtered_sigs_m)}, Clients: {armada_m.num_clients}")
            m_clients = armada_m.num_clients
            m_filtered = len(filtered_sigs_m)
        else:
            print(f"  [MALE] Skipping {ds_name} - No Male Participants found.")
            rules_signatures_m[ds_name] = set()
            m_clients = 0
            m_filtered = 0

        if 'F' in groups and len(groups['F']['client_id'].unique()) >= 1:
            print(f"\n  [FEMALE] Running ARMADA for {ds_name}...")
            armada_f, patterns_f, rules_f = run_armada_on_df(groups['F'], MINSUP, MINCONF, MAXGAP, MAX_PATTERN_SIZE)
            all_results_f[ds_name] = (armada_f, patterns_f, rules_f)
            
            raw_sigs_f = extract_rule_signatures(rules_f)
            filtered_sigs_f = filter_rules(raw_sigs_f, FILTER_BVP_ONLY, FILTER_EDA_ONLY, FILTER_PHYSIO_CROSS, FILTER_SINGLE_FEATURE)
            rules_signatures_f[ds_name] = filtered_sigs_f
            
            print(f"    Raw Rules: {len(rules_f)}, Filtered: {len(filtered_sigs_f)}, Clients: {armada_f.num_clients}")
            f_clients = armada_f.num_clients
            f_filtered = len(filtered_sigs_f)
        else:
            print(f"  [FEMALE] Skipping {ds_name} - No Female Participants found.")
            rules_signatures_f[ds_name] = set()
            f_clients = 0
            f_filtered = 0
            
        stats_data[ds_name] = {
            'M_clients': m_clients, 'M_filtered': m_filtered,
            'F_clients': f_clients, 'F_filtered': f_filtered
        }


    print("\n" + "=" * 80)
    print("CALCULATING GENDER-WISE UNIVERSAL INTERSECTIONS")
    print("=" * 80)

    dataset_names = list(datasets.keys())
    
    # Men global intersection
    universal_m = set(rules_signatures_m[dataset_names[0]])
    for ds_name in dataset_names[1:]:
        universal_m &= rules_signatures_m[ds_name]
        
    # Female global intersection
    universal_f = set(rules_signatures_f[dataset_names[0]])
    for ds_name in dataset_names[1:]:
        universal_f &= rules_signatures_f[ds_name]
        
    print(f"\nUniversal Rules across all datasets for MEN (M): {len(universal_m)}")
    print(f"Universal Rules across all datasets for WOMEN (F): {len(universal_f)}")
    print(f"Rules common to BOTH genders globally: {len(universal_m & universal_f)}")

    df_m = pd.DataFrame()
    df_f = pd.DataFrame()
    
    if len(universal_m) > 0:
        df_m = save_gender_universal_rules_details(universal_m, all_results_m, OUTPUT_DIR, "male")
    if len(universal_f) > 0:
        df_f = save_gender_universal_rules_details(universal_f, all_results_f, OUTPUT_DIR, "female")

    generate_markdown_report(
        datasets_count=len(dataset_names),
        stats_data=stats_data,
        universal_rules_m=universal_m,
        universal_rules_f=universal_f,
        df_m=df_m,
        df_f=df_f,
        output_dir=OUTPUT_DIR,
        dataset_names=dataset_names
    )
        
    print("\n" + "=" * 80)
    print("EXPERIMENT FINISHED")
    print(f"Results saved in: {OUTPUT_DIR}")
    print("=" * 80)

if __name__ == "__main__":
    main()
