#!/usr/bin/env python3
"""
Experiment: Find Universal Rules Across All Datasets

1. Runs ARMADA on each dataset separately (CASE, K-emoCon, CEAP, EmoWorker_v2).
2. Applies predefined rule filters to reject non-emotional or univariate rules.
3. Finds rules that are strictly "universal" (present in ALL evaluated datasets).
4. Generates a report with rules support and confidence cross-dataset metrics.
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
# We add experiments dir as we need experiment_utils which is there
sys.path.insert(0, str(EXPERIMENTS_DIR))

# ============================================================================
# EXPERIMENT PARAMETERS
# ============================================================================
MINSUP = 0.1       # 30% minimum support
MINCONF = 0.5      # 50% minimum confidence
MAXGAP = 5       # 10s max gap
MAX_PATTERN_SIZE = 2  # max depth 3

# ============================================================================
# RULE FILTERS
# ============================================================================
# True -> reject rules where ALL states are BVP/HRV/HR-related
FILTER_BVP_ONLY = True

# True -> reject rules where ALL states are EDA-related
FILTER_EDA_ONLY = True

# True -> reject rules where ALL states are peripheral signals (EDA + BVP/HRV/HR) without arousal/valence/temp
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
    if str(data_file).endswith('.csv'):
        print(f"  Loading CSV: {data_file}")
        df = pd.read_csv(data_file)
        return run_armada_on_df(df, minsup, minconf, maxgap, max_pattern_size)
    else:
        # Old TXT format not actively supported but preserved logic
        armada = ARMADA(
            minsup=minsup,
            minconf=minconf,
            maxgap=maxgap,
            max_pattern_size=max_pattern_size
        )
        patterns, rules = armada.run(filepath=data_file)
        return armada, patterns, rules


def save_universal_rules_details(
    universal_rules: Set[str],
    all_results: Dict[str, Tuple],
    output_dir: Path
) -> pd.DataFrame:
    """Saves details of universal rules with per-dataset metrics."""
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
    
    df.to_csv(output_dir / "universal_rules_details.csv", index=False)
    return df


def generate_markdown_report(
    datasets_count: int,
    all_results: Dict[str, Tuple],
    universal_rules: Set[str],
    universal_rules_df: pd.DataFrame,
    output_dir: Path
) -> None:
    """Generates Markdown report."""

    lines = []
    lines.append("# Universal Rules Experiment")
    lines.append("")
    lines.append("This experiment aims to identify rules that are universally true across all evaluated datasets.")
    lines.append("")
    lines.append("## Experiment Parameters")
    lines.append("")
    lines.append(f"- **minsup**: {MINSUP} ({MINSUP*100:.0f}% participants)")
    lines.append(f"- **minconf**: {MINCONF} ({MINCONF*100:.0f}% confidence)")
    lines.append(f"- **maxgap**: {MAXGAP} seconds")
    lines.append(f"- **max_pattern_size**: {MAX_PATTERN_SIZE}")
    lines.append("")
    lines.append("## Rule Filters")
    lines.append("")
    lines.append(f"- **FILTER_BVP_ONLY**: {FILTER_BVP_ONLY}")
    lines.append(f"- **FILTER_EDA_ONLY**: {FILTER_EDA_ONLY}")
    lines.append(f"- **FILTER_PHYSIO_CROSS**: {FILTER_PHYSIO_CROSS}")
    lines.append(f"- **FILTER_SINGLE_FEATURE**: {FILTER_SINGLE_FEATURE}")
    lines.append("")
    
    # Dataset statistics
    lines.append("## Dataset Processing")
    lines.append("")
    lines.append(f"Evaluated on {datasets_count} datasets: {', '.join(all_results.keys())}")
    lines.append("")
    lines.append("| Dataset | Participants | Rules (Before Filters) | Rules (Filtered) |")
    lines.append("|-------|-------------|---------|-------|")

    for ds_name, (armada, patterns, rules) in all_results.items():
        # calculate filtered count manually for report completeness (we only needed sigs during logic)
        raw_sigs = extract_rule_signatures(rules)
        filtered_sigs = filter_rules(raw_sigs, FILTER_BVP_ONLY, FILTER_EDA_ONLY, FILTER_PHYSIO_CROSS, FILTER_SINGLE_FEATURE)
        lines.append(f"| **{ds_name}** | {armada.num_clients} | {len(rules)} | {len(filtered_sigs)} |")

    lines.append("")

    # Universal comparison
    lines.append("## Universal Rules Identification")
    lines.append("")
    lines.append(f"TOTAL Universal rules found across all {datasets_count} datasets: **{len(universal_rules)}**")
    lines.append("")

    if len(universal_rules_df) > 0:
        lines.append("### Universal Rules Details (sorted by avg_confidence)")
        lines.append("")

        ds_names = list(all_results.keys())
        ds_headers = " | ".join(ds_names)
        header = f"| Rule | Avg Conf | Avg Sup | Min Conf | Min Sup | {ds_headers} |"
        separator = "|---|---|---|---|---|" + "|".join(["---" for _ in ds_names]) + "|"
        lines.append(header)
        lines.append(separator)

        for _, row in universal_rules_df.iterrows():
            avg_conf = row.get('avg_confidence', 'N/A')
            avg_sup = row.get('avg_support', 'N/A')
            min_conf = row.get('min_confidence', 'N/A')
            min_sup = row.get('min_support', 'N/A')
            
            if isinstance(avg_conf, float):
                avg_conf = f"{avg_conf:.3f}"
            if isinstance(avg_sup, float):
                avg_sup = f"{avg_sup:.3f}"
            if isinstance(min_conf, float):
                min_conf = f"{min_conf:.3f}"
            if isinstance(min_sup, float):
                min_sup = f"{min_sup:.3f}"

            line = f"| `{row['rule']}` | **{avg_conf}** | {avg_sup} | {min_conf} | {min_sup} |"
            
            for ds_name in ds_names:
                ds_conf = row.get(f'{ds_name}_confidence', 'N/A')
                ds_sup = row.get(f'{ds_name}_support', 'N/A')
                ds_count = row.get(f'{ds_name}_count', 'N/A')
                if isinstance(ds_conf, float):
                    ds_conf = f"{ds_conf:.2f}"
                if isinstance(ds_sup, float):
                    ds_sup = f"{ds_sup:.2f}"
                if isinstance(ds_count, (float, int)):
                    ds_count = f"{int(ds_count)}"
                line += f" c:{ds_conf} s:{ds_sup} n:{ds_count} |"
                
            lines.append(line)

    lines.append("")

    # Save report
    report_text = "\n".join(lines)
    with open(output_dir / "universal_rules_report.md", "w") as f:
        f.write(report_text)

    print(f"Zapisano raport: {output_dir / 'universal_rules_report.md'}")


def main():
    """Main experiment function."""

    # Paths
    DATA_DIR = PROJECT_DIR / "data" / "armada_ready"
    OUTPUT_DIR = SCRIPT_DIR / "results"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("EXPERIMENT: UNIVERSAL RULES DETECTION")
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
        
        rules_signatures[ds_name] = filtered_signatures
        
        print(f"  Total Rules: {len(rules)}")
        print(f"  Rules after filtering: {len(filtered_signatures)}")

    # Finding the intersection of rules across all datasets
    print("\n" + "=" * 80)
    print("CALCULATING UNIVERSAL RULES INTERSECTION")
    print("=" * 80)

    dataset_names = list(datasets.keys())
    universal_rules = set(rules_signatures[dataset_names[0]])
    
    for ds_name in dataset_names[1:]:
        universal_rules &= rules_signatures[ds_name]
        
    print(f"\nRules universal across ALL {len(dataset_names)} datasets: {len(universal_rules)}")
    
    # Dump metrics output into CSV
    universal_rules_df = save_universal_rules_details(universal_rules, all_results, OUTPUT_DIR)
    
    # Markdown documentation 
    generate_markdown_report(
        datasets_count=len(dataset_names),
        all_results=all_results,
        universal_rules=universal_rules,
        universal_rules_df=universal_rules_df,
        output_dir=OUTPUT_DIR
    )

    if len(universal_rules_df) > 0:
        print("\nTOP UNIVERSAL RULES (sorted by avg. confidence):")
        for _, row in universal_rules_df.head(10).iterrows():
            print(f"  {row['rule']}")
            print(f"    avg_conf={row.get('avg_confidence', 'N/A')}, avg_sup={row.get('avg_support', 'N/A')}")
        
    print("\n" + "=" * 80)
    print("EXPERIMENT FINISHED")
    print(f"Results saved in: {OUTPUT_DIR}")
    print("=" * 80)

if __name__ == "__main__":
    main()
