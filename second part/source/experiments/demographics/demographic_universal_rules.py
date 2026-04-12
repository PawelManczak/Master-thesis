#!/usr/bin/env python3
"""
Experiment: Demographic Universal Rules Detection (Gender & Age)

1. Loads 4 datasets and corresponding demographic data.
2. For Gender (M/F) and Age (Young/Old), splits datasets.
3. Runs ARMADA independently on all demographic subsets.
4. Extrapolates strictly "Universal" rules per subgroup (e.g. valid across all datasets solely for young).
5. Generates markdown reports and CSV metrics.
"""

import sys
from pathlib import Path
import pandas as pd
from typing import Dict, List, Set, Tuple

SCRIPT_DIR = Path(__file__).parent
EXPERIMENTS_DIR = SCRIPT_DIR.parent
PROJECT_DIR = EXPERIMENTS_DIR.parent.parent

sys.path.insert(0, str(PROJECT_DIR / "source" / "processing" / "armada"))
sys.path.insert(0, str(EXPERIMENTS_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

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


def save_universal_rules_details(
    universal_rules: Set[str],
    all_results: Dict[str, Tuple],
    output_dir: Path,
    tag: str
) -> pd.DataFrame:
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
    
    df.to_csv(output_dir / f"universal_{tag}_rules_details.csv", index=False)
    return df


def generate_markdown_report(
    datasets_count: int,
    stats_data: Dict[str, Dict],
    universal_rules_a: Set[str],
    universal_rules_b: Set[str],
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    output_dir: Path,
    dataset_names: List[str],
    group_name: str,
    tag_a: str,
    tag_b: str
) -> None:
    lines = []
    lines.append(f"# Universal Rules by {group_name.capitalize()} Experiment")
    lines.append("")
    lines.append(f"This experiment aims to identify rules that are universally true across all evaluated datasets, split solely by demographic {group_name} subsets ({tag_a} vs {tag_b}).")
    lines.append("")
    
    lines.append("| Dataset | Participants (A) | Participants (B) | Rules Filtered (A) | Rules Filtered (B) |")
    lines.append("|-------|-------------|---------|-------|-------|")
    for ds_name in dataset_names:
        stats = stats_data.get(ds_name, {})
        lines.append(
            f"| **{ds_name}** | {stats.get('A_clients', 0)} | {stats.get('B_clients', 0)} "
            f"| {stats.get('A_filtered', 0)} | {stats.get('B_filtered', 0)} |"
        )
    lines.append("")

    common_both = universal_rules_a & universal_rules_b
    unique_a = universal_rules_a - universal_rules_b
    unique_b = universal_rules_b - universal_rules_a

    lines.append(f"## {group_name.capitalize()} Universal Rules Identification")
    lines.append("")
    lines.append(f"- **Universal rules found in ALL datasets for {tag_a}**: {len(universal_rules_a)}")
    lines.append(f"- **Universal rules found in ALL datasets for {tag_b}**: {len(universal_rules_b)}")
    lines.append(f"- **Universal rules spanning across BOTH groups**: {len(common_both)}")
    lines.append(f"- **Universal rules UNIQUELY for {tag_a}**: {len(unique_a)}")
    lines.append(f"- **Universal rules UNIQUELY for {tag_b}**: {len(unique_b)}")
    lines.append("")

    for current_df, tag, unique_set in [(df_a, tag_a, unique_a), (df_b, tag_b, unique_b)]:
        if len(current_df) > 0:
            lines.append(f"### Universal Rules for {tag} (Top metrics globally)")
            lines.append("")
            ds_headers = " | ".join(dataset_names)
            header = f"| Rule | Avg Conf | Avg Sup | Min Conf | Min Sup | {ds_headers} |"
            separator = "|---|---|---|---|---|" + "|".join(["---" for _ in dataset_names]) + "|"
            lines.append(header)
            lines.append(separator)

            for _, row in current_df.iterrows():
                avg_conf = f"{row.get('avg_confidence', 0):.3f}"
                avg_sup = f"{row.get('avg_support', 0):.3f}"
                min_conf = f"{row.get('min_confidence', 0):.3f}"
                min_sup = f"{row.get('min_support', 0):.3f}"
                
                rule_str = f"`{row['rule']}`"
                if row['rule'] in common_both:
                    rule_str += " *(common)*"
                elif row['rule'] in unique_set:
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
    with open(output_dir / f"{group_name}_universal_rules_report.md", "w") as f:
        f.write(report_text)


def run_demographic_experiment(demo_df, datasets, split_column, tag_a, tag_b, group_name, output_dir):
    print(f"\n{'=' * 80}")
    print(f"EXPERIMENT: {group_name.upper()} UNIVERSAL RULES DETECTION")
    print(f"{'=' * 80}")
    
    dataset_names = list(datasets.keys())
    all_results_a = {}
    all_results_b = {}
    rules_signatures_a = {}
    rules_signatures_b = {}
    stats_data = {}

    for ds_name, data_file in datasets.items():
        df = pd.read_csv(data_file)
        ds_demo = demo_df[demo_df['dataset'] == ds_name].copy()
        groups = split_data_by_group(df, ds_demo, split_column)
        
        def process_subgroup(tag_name):
            if tag_name in groups and len(groups[tag_name]['client_id'].unique()) >= 1:
                armada, patterns, rules = run_armada_on_df(groups[tag_name], MINSUP, MINCONF, MAXGAP, MAX_PATTERN_SIZE)
                raw_sigs = extract_rule_signatures(rules)
                filtered_sigs = filter_rules(raw_sigs, FILTER_BVP_ONLY, FILTER_EDA_ONLY, FILTER_PHYSIO_CROSS, FILTER_SINGLE_FEATURE)
                return armada, patterns, rules, filtered_sigs
            return None, None, None, set()

        # Group A
        armada_a, p_a, r_a, sigs_a = process_subgroup(tag_a)
        if armada_a:
            all_results_a[ds_name] = (armada_a, p_a, r_a)
            rules_signatures_a[ds_name] = sigs_a
            stats_a = (armada_a.num_clients, len(sigs_a))
        else:
            rules_signatures_a[ds_name] = set()
            stats_a = (0, 0)
            
        # Group B
        armada_b, p_b, r_b, sigs_b = process_subgroup(tag_b)
        if armada_b:
            all_results_b[ds_name] = (armada_b, p_b, r_b)
            rules_signatures_b[ds_name] = sigs_b
            stats_b = (armada_b.num_clients, len(sigs_b))
        else:
            rules_signatures_b[ds_name] = set()
            stats_b = (0, 0)
            
        stats_data[ds_name] = {
            'A_clients': stats_a[0], 'A_filtered': stats_a[1],
            'B_clients': stats_b[0], 'B_filtered': stats_b[1]
        }

    # Intersections
    universal_a = set(rules_signatures_a[dataset_names[0]]) if dataset_names else set()
    for ds_name in dataset_names[1:]:
        universal_a &= rules_signatures_a[ds_name]
        
    universal_b = set(rules_signatures_b[dataset_names[0]]) if dataset_names else set()
    for ds_name in dataset_names[1:]:
        universal_b &= rules_signatures_b[ds_name]

    df_a = pd.DataFrame()
    df_b = pd.DataFrame()
    if len(universal_a) > 0:
        df_a = save_universal_rules_details(universal_a, all_results_a, output_dir, tag_a.lower())
    if len(universal_b) > 0:
        df_b = save_universal_rules_details(universal_b, all_results_b, output_dir, tag_b.lower())

    generate_markdown_report(
        len(dataset_names), stats_data, universal_a, universal_b, df_a, df_b, output_dir, dataset_names, group_name, tag_a, tag_b
    )


def main():
    DATA_DIR = PROJECT_DIR / "data" / "armada_ready"
    OUTPUT_DIR = SCRIPT_DIR / "results"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

    print("\nRunning Gender Universal Rules Extraction...")
    run_demographic_experiment(demo_df, datasets, 'gender_clean', 'M', 'F', 'gender', OUTPUT_DIR)
    
    print("\nRunning Age Universal Rules Extraction...")
    run_demographic_experiment(demo_df, datasets, 'binary_age_group', 'young', 'old', 'age', OUTPUT_DIR)


if __name__ == "__main__":
    main()
