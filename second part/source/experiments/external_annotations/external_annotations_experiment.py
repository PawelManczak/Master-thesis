#!/usr/bin/env python3
"""
Experiment: Cross-Dataset Analysis of External Annotations

Compares ARMADA rules found in:
- EMBOA (BORIS method II external annotations, 1 Hz -> 5s windows)
- K-emoCon (aggregated external annotations, 5s intervals)

Both datasets use **external observer** annotations, making them
methodologically comparable for cross-dataset pattern analysis.

Pipeline:
1. Runs ARMADA on each dataset separately.
2. Applies rule filters (BVP-only, EDA-only, single-feature).
3. Finds universal rules (intersection across both datasets).
4. Generates a detailed Markdown report.
"""

import sys
from pathlib import Path
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Set, Tuple

SCRIPT_DIR = Path(__file__).parent
EXPERIMENTS_DIR = SCRIPT_DIR.parent
PROJECT_DIR = EXPERIMENTS_DIR.parent.parent

sys.path.insert(0, str(PROJECT_DIR / "source" / "processing" / "armada"))
sys.path.insert(0, str(EXPERIMENTS_DIR))


MINSUP = 0.2       # 10% minimum support
MINCONF = 0.2       # 10% minimum confidence
MAXGAP = 5
MAX_PATTERN_SIZE = 2

FILTER_BVP_ONLY = True
FILTER_EDA_ONLY = True
FILTER_PHYSIO_CROSS = True
FILTER_SINGLE_FEATURE = True

try:
    from experiment_utils import (
        run_armada_on_df,
        extract_rule_signatures,
        filter_rules
    )
except ImportError as e:
    print(f"Error importing experiment_utils: {e}")
    sys.exit(1)

try:
    from armada_algorithm import ARMADA
except ImportError as e:
    print(f"Error importing armada_algorithm: {e}")
    sys.exit(1)


def run_armada_on_dataset(
    data_file: Path,
    minsup: float = MINSUP,
    minconf: float = MINCONF,
    maxgap: float = MAXGAP,
    max_pattern_size: int = MAX_PATTERN_SIZE
) -> Tuple[ARMADA, List, List]:
    """Runs ARMADA on a single dataset file."""
    print(f"  Loading CSV: {data_file}")
    df = pd.read_csv(data_file)
    return run_armada_on_df(df, minsup, minconf, maxgap, max_pattern_size)


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

    df.to_csv(output_dir / "external_universal_rules_details.csv", index=False)
    return df


def generate_report(
    datasets_count: int,
    all_results: Dict[str, Tuple],
    universal_rules: Set[str],
    universal_rules_df: pd.DataFrame,
    output_dir: Path
) -> None:
    """Generates Markdown report."""

    lines = []
    lines.append("# External Annotations Cross-Dataset Experiment")
    lines.append("")
    lines.append("This experiment identifies rules that are universally present")
    lines.append("across datasets using **external observer annotations**:")
    lines.append("- **EMBOA**: BORIS method II (6 emotions x % agreement -> arousal/valence)")
    lines.append("- **K-emoCon external**: Aggregated external annotations (arousal/valence 1–5)")
    lines.append("")

    lines.append("## Experiment Parameters")
    lines.append("")
    lines.append(f"- **minsup**: {MINSUP} ({MINSUP*100:.0f}%)")
    lines.append(f"- **minconf**: {MINCONF} ({MINCONF*100:.0f}%)")
    lines.append(f"- **maxgap**: {MAXGAP}s")
    lines.append(f"- **max_pattern_size**: {MAX_PATTERN_SIZE}")
    lines.append("")

    lines.append("## Rule Filters")
    lines.append("")
    lines.append(f"- **FILTER_BVP_ONLY**: {FILTER_BVP_ONLY}")
    lines.append(f"- **FILTER_EDA_ONLY**: {FILTER_EDA_ONLY}")
    lines.append(f"- **FILTER_PHYSIO_CROSS**: {FILTER_PHYSIO_CROSS}")
    lines.append(f"- **FILTER_SINGLE_FEATURE**: {FILTER_SINGLE_FEATURE}")
    lines.append("")

    lines.append("## Dataset Processing")
    lines.append("")
    lines.append(f"Evaluated on {datasets_count} datasets: {', '.join(all_results.keys())}")
    lines.append("")
    lines.append("| Dataset | Participants | Rules (Before Filters) | Rules (Filtered) |")
    lines.append("|---------|-------------|------------------------|------------------|")

    for ds_name, (armada, patterns, rules) in all_results.items():
        raw_sigs = extract_rule_signatures(rules)
        filtered_sigs = filter_rules(
            raw_sigs, FILTER_BVP_ONLY, FILTER_EDA_ONLY,
            FILTER_PHYSIO_CROSS, FILTER_SINGLE_FEATURE
        )
        lines.append(
            f"| **{ds_name}** | {armada.num_clients} | "
            f"{len(rules)} | {len(filtered_sigs)} |"
        )

    lines.append("")

    lines.append("## Universal Rules (Present in ALL Datasets)")
    lines.append("")
    lines.append(f"Total universal rules found: **{len(universal_rules)}**")
    lines.append("")

    if len(universal_rules_df) > 0:
        lines.append("### Details (sorted by avg. confidence)")
        lines.append("")

        ds_names = list(all_results.keys())
        ds_headers = " | ".join(ds_names)
        header = f"| Rule | Avg Conf | Avg Sup | Min Conf | Min Sup | {ds_headers} |"
        separator = ("|---|---|---|---|---|"
                     + "|".join(["---" for _ in ds_names]) + "|")
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

    # Per-dataset unique rules
    all_filtered_sigs = {}
    for ds_name, (armada, patterns, rules) in all_results.items():
        raw_sigs = extract_rule_signatures(rules)
        filtered = filter_rules(
            raw_sigs, FILTER_BVP_ONLY, FILTER_EDA_ONLY,
            FILTER_PHYSIO_CROSS, FILTER_SINGLE_FEATURE
        )
        all_filtered_sigs[ds_name] = filtered

    lines.append("## Dataset-Specific Rules")
    lines.append("")
    for ds_name, sigs in all_filtered_sigs.items():
        unique_to_ds = sigs - universal_rules
        other_ds = set()
        for other_name, other_sigs in all_filtered_sigs.items():
            if other_name != ds_name:
                other_ds |= other_sigs
        truly_unique = unique_to_ds - other_ds

        lines.append(f"### {ds_name}")
        lines.append(f"- Total filtered rules: {len(sigs)}")
        lines.append(f"- Rules unique to {ds_name}: **{len(truly_unique)}**")
        lines.append("")

        if truly_unique:
            for sig in sorted(truly_unique)[:20]:
                lines.append(f"  - `{sig}`")
            if len(truly_unique) > 20:
                lines.append(f"  - ... and {len(truly_unique) - 20} more")
            lines.append("")

    # Save report
    report_text = "\n".join(lines)
    with open(output_dir / "external_annotations_report.md", "w") as f:
        f.write(report_text)

    print(f"Report saved: {output_dir / 'external_annotations_report.md'}")


def main():
    DATA_DIR = PROJECT_DIR / "data" / "armada_ready"
    OUTPUT_DIR = SCRIPT_DIR / "results"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("EXPERIMENT: EXTERNAL ANNOTATIONS CROSS-DATASET ANALYSIS")
    print("=" * 80)
    print(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Results: {OUTPUT_DIR}")
    print()

    datasets = {
        'EMBOA': DATA_DIR / "armada_emboa.csv",
        'K-emoCon-ext': DATA_DIR / "armada_k_emocon_ext.csv",
    }

    missing = []
    for ds_name, data_file in datasets.items():
        if not data_file.exists():
            missing.append(f"  {ds_name}: {data_file}")

    if missing:
        print("ERROR: Missing dataset files:")
        for m in missing:
            print(m)
        print("\nRun the preparation scripts first:")
        print("  1. python source/processing/extracting/datasets/EMBOA_external.py")
        print("  2. python source/processing/extracting/datasets/K_emoCon_external.py")
        print("  3. python source/processing/armada/prepare_external_annotations_armada.py")
        sys.exit(1)

    all_results = {}
    rules_signatures = {}

    for ds_name, data_file in datasets.items():
        print(f"\n{'='*60}")
        print(f"Processing Dataset: {ds_name}")
        print(f"{'='*60}")

        armada, patterns, rules = run_armada_on_dataset(data_file)
        all_results[ds_name] = (armada, patterns, rules)

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

    print("\n" + "=" * 80)
    print("CALCULATING UNIVERSAL RULES INTERSECTION")
    print("=" * 80)

    dataset_names = list(datasets.keys())
    universal_rules = set(rules_signatures[dataset_names[0]])

    for ds_name in dataset_names[1:]:
        universal_rules &= rules_signatures[ds_name]

    print(f"\nRules universal across ALL {len(dataset_names)} datasets: "
          f"{len(universal_rules)}")

    universal_rules_df = save_universal_rules_details(
        universal_rules, all_results, OUTPUT_DIR
    )

    generate_report(
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
            print(f"    avg_conf={row.get('avg_confidence', 'N/A')}, "
                  f"avg_sup={row.get('avg_support', 'N/A')}")

    print("\n" + "=" * 80)
    print("EXPERIMENT FINISHED")
    print(f"Results saved in: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
