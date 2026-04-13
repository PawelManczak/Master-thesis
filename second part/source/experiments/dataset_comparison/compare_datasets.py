#!/usr/bin/env python3
"""
Compare ARMADA patterns between datasets.

1. Runs ARMADA on each dataset separately (CASE, K-emoCon, CEAP, EmoWorker_v2).
2. Compares discovered patterns and rules.
3. Finds common patterns across all datasets.
4. Generates a comparison report.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = Path(__file__).parent
EXPERIMENTS_DIR = SCRIPT_DIR.parent
PROJECT_DIR = EXPERIMENTS_DIR.parent.parent
sys.path.insert(0, str(PROJECT_DIR / "source" / "processing" / "armada"))
sys.path.insert(0, str(EXPERIMENTS_DIR))

from armada_algorithm import ARMADA

MINSUP = 0.3
MINCONF = 0.5
MAXGAP = 5
MAX_PATTERN_SIZE = 3

# True -> reject rules where ALL states are BVP/HRV/HR-related
FILTER_BVP_ONLY = True

# True -> reject rules where ALL states are EDA-related
FILTER_EDA_ONLY = True

# True -> reject rules where ALL states are peripheral signals (EDA + BVP/HRV/HR) without arousal/valence/temp
FILTER_PHYSIO_CROSS = True

# True -> reject rules where all states are of the same feature type
FILTER_SINGLE_FEATURE = True

from experiment_utils import (
    run_armada_on_df,
    extract_rule_signatures,
    extract_pattern_signatures,
    filter_rules
)

def run_armada_on_dataset(
    data_file: Path,
    minsup: float = MINSUP,
    minconf: float = MINCONF,
    maxgap: float = MAXGAP,
    max_pattern_size: int = MAX_PATTERN_SIZE
) -> Tuple[ARMADA, List, List]:
    """Runs ARMADA on a single dataset."""
    # CSV support
    if str(data_file).endswith('.csv'):
        print(f"  Loading CSV: {data_file}")
        df = pd.read_csv(data_file)
        return run_armada_on_df(df, minsup, minconf, maxgap, max_pattern_size)
    else:
        # Old TXT format
        armada = ARMADA(
            minsup=minsup,
            minconf=minconf,
            maxgap=maxgap,
            max_pattern_size=max_pattern_size
        )
        patterns, rules = armada.run(filepath=data_file)
        return armada, patterns, rules


def compare_pattern_sets(patterns_dict: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
    """Compares pattern sets between datasets."""
    datasets = list(patterns_dict.keys())
    result = {}

    # Common for all
    if len(datasets) >= 2:
        common_all = patterns_dict[datasets[0]].copy()
        for ds in datasets[1:]:
            common_all &= patterns_dict[ds]
        result['common_all'] = common_all

    # Common for at least 3
    if len(datasets) >= 3:
        all_items = set().union(*patterns_dict.values())
        common_3_plus = set()
        for item in all_items:
            count = sum(1 for ds in datasets if item in patterns_dict[ds])
            if count >= 3:
                common_3_plus.add(item)
        result['common_3_plus'] = common_3_plus

    # Common for pairs
    for i, ds1 in enumerate(datasets):
        for ds2 in datasets[i+1:]:
            key = f"common_{ds1}_{ds2}"
            common = patterns_dict[ds1] & patterns_dict[ds2]
            result[key] = common

    # Unique per dataset
    for ds in datasets:
        others = set()
        for other_ds in datasets:
            if other_ds != ds:
                others |= patterns_dict[other_ds]
        result[f"unique_{ds}"] = patterns_dict[ds] - others

    return result


def create_comparison_visualizations(
    patterns_dict: Dict[str, Set[str]],
    comparison: Dict[str, Set[str]],
    output_dir: Path
) -> None:
    """Creates comparison visualizations."""
    datasets = list(patterns_dict.keys())

    # Chart 1: Number of patterns per dataset
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1a. Bar chart
    ax1 = axes[0]
    ds_names = list(patterns_dict.keys())
    counts = [len(patterns_dict[ds]) for ds in ds_names]
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']
    if len(ds_names) > len(colors):
        colors = colors * (len(ds_names) // len(colors) + 1)

    bars = ax1.bar(ds_names, counts, color=colors[:len(ds_names)])
    ax1.set_ylabel('Pattern Count')
    ax1.set_title('Patterns per Dataset')
    for bar, count in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                str(count), ha='center', va='bottom', fontweight='bold')

    # 1b. Common vs Unique
    ax2 = axes[1]

    if len(datasets) >= 3:
        # Calculate categories
        unique_counts = {ds: len(patterns_dict[ds] - set().union(*[patterns_dict[o] for o in datasets if o != ds])) for ds in datasets}

        # Common for all
        common_all = set(patterns_dict[datasets[0]])
        for ds in datasets[1:]:
            common_all &= patterns_dict[ds]
        count_common_all = len(common_all)

        labels = [f"Unique\n{ds}" for ds in datasets] + ["Common\nALL"]
        values = [unique_counts[ds] for ds in datasets] + [count_common_all]

        colors2 = colors[:len(datasets)] + ['#f1c40f'] # Gold for common

        bars2 = ax2.bar(range(len(labels)), values, color=colors2)
        ax2.set_xticks(range(len(labels)))
        ax2.set_xticklabels(labels, fontsize=8, rotation=45, ha='right')
        ax2.set_ylabel('Pattern Count')
        ax2.set_title('Unique vs Common')

        for bar, val in zip(bars2, values):
            if val > 0:
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        str(val), ha='center', va='bottom', fontsize=9)
    else:
        ax2.text(0.5, 0.5, "Requires min 3 datasets", ha='center')

    # 1c. Percentage of common patterns
    ax3 = axes[2]
    common_all = len(comparison.get('common_all', set()))
    total_unique = len(set().union(*patterns_dict.values()))

    sizes = [common_all, total_unique - common_all]
    labels_pie = [f'Common\n({common_all})', f'Others\n({total_unique - common_all})']
    colors_pie = ['#f1c40f', '#95a5a6']
    ax3.pie(sizes, labels=labels_pie, colors=colors_pie, autopct='%1.1f%%', startangle=90)
    ax3.set_title('Udział wspólnych wzorców')

    plt.tight_layout()
    plt.savefig(output_dir / 'patterns_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()


    print(f"Saved charts to {output_dir}")


def save_common_patterns_details(
    common_patterns: Set[str],
    all_results: Dict[str, Tuple],
    output_dir: Path
) -> pd.DataFrame:
    """Saves details of common patterns with per-dataset metrics."""
    details = []

    for pattern_sig in sorted(common_patterns):
        entry = {"pattern": pattern_sig}
        supports = []

        for ds_name, (armada, patterns, rules) in all_results.items():
            for p in patterns:
                if p.get_relation_description() == pattern_sig:
                    entry[f"{ds_name}_support"] = round(p.support, 4)
                    entry[f"{ds_name}_count"] = p.support_count
                    supports.append(p.support)
                    break

        if supports:
            entry["avg_support"] = round(sum(supports) / len(supports), 4)
            entry["min_support"] = round(min(supports), 4)

        details.append(entry)

    df = pd.DataFrame(details)

    if 'avg_support' in df.columns:
        df = df.sort_values('avg_support', ascending=False)

    df.to_csv(output_dir / "common_patterns_details.csv", index=False)

    return df


def save_common_rules_details(
    common_rules: Set[str],
    all_results: Dict[str, Tuple],
    output_dir: Path
) -> pd.DataFrame:
    details = []

    for rule_sig in sorted(common_rules):
        entry = {"rule": rule_sig}
        confidences = []
        lifts = []
        supports = []

        for ds_name, (armada, patterns, rules) in all_results.items():
            for r in rules:
                sig = f"{r.antecedent.get_relation_description()} => {r.consequent.get_relation_description()}"
                if sig == rule_sig:
                    entry[f"{ds_name}_confidence"] = round(r.confidence, 4)
                    entry[f"{ds_name}_lift"] = round(r.lift, 4)
                    entry[f"{ds_name}_support"] = round(r.support, 4)
                    confidences.append(r.confidence)
                    lifts.append(r.lift)
                    supports.append(r.support)
                    break

        if confidences:
            entry["avg_confidence"] = round(sum(confidences) / len(confidences), 4)
            entry["avg_lift"] = round(sum(lifts) / len(lifts), 4)
            entry["min_confidence"] = round(min(confidences), 4)
            entry["avg_support"] = round(sum(supports) / len(supports), 4)

        details.append(entry)

    df = pd.DataFrame(details)

    if 'avg_confidence' in df.columns:
        df = df.sort_values('avg_confidence', ascending=False)

    df.to_csv(output_dir / "common_rules_details.csv", index=False)

    return df


def generate_markdown_report(
    patterns_comparison: Dict[str, Set[str]],
    rules_comparison: Dict[str, Set[str]],
    all_results: Dict[str, Tuple],
    common_patterns_df: pd.DataFrame,
    common_rules_df: pd.DataFrame,
    output_dir: Path
) -> None:

    lines = []
    lines.append("# ARMADA Patterns Comparison Between Datasets")
    lines.append("")
    lines.append("## Experiment Parameters")
    lines.append("")
    lines.append(f"- **minsup**: {MINSUP} ({MINSUP*100:.0f}% participants)")
    lines.append(f"- **minconf**: {MINCONF} ({MINCONF*100:.0f}% confidence)")
    lines.append(f"- **maxgap**: {MAXGAP} sekund")
    lines.append(f"- **max_pattern_size**: {MAX_PATTERN_SIZE}")
    lines.append("")
    lines.append("## Rule Filters")
    lines.append("")
    lines.append(f"- **FILTER_BVP_ONLY**: {FILTER_BVP_ONLY} - {'rejects rules with only BVP/HRV/HR states' if FILTER_BVP_ONLY else 'disabled'}")
    lines.append(f"- **FILTER_EDA_ONLY**: {FILTER_EDA_ONLY} - {'rejects rules with only EDA states' if FILTER_EDA_ONLY else 'disabled'}")
    lines.append(f"- **FILTER_PHYSIO_CROSS**: {FILTER_PHYSIO_CROSS} - {'rejects purely physiological rules (EDA+BVP mix without emotion/temp)' if FILTER_PHYSIO_CROSS else 'disabled'}")
    lines.append(f"- **FILTER_SINGLE_FEATURE**: {FILTER_SINGLE_FEATURE} - {'rejects single-feature rules (e.g. only arousal)' if FILTER_SINGLE_FEATURE else 'disabled'}")
    lines.append("")

    lines.append("## Dataset Statistics")
    lines.append("")
    lines.append("| Dataset | Participants | Patterns | Rules | Unique Patterns |")
    lines.append("|-------|-------------|---------|-------|-----------------|")

    for ds_name, (armada, patterns, rules) in all_results.items():
        unique = len(patterns_comparison.get(f'unique_{ds_name}', set()))
        unique_pct = unique / len(patterns) * 100 if patterns else 0
        lines.append(f"| **{ds_name}** | {armada.num_clients} | {len(patterns)} | {len(rules)} | {unique} ({unique_pct:.1f}%) |")

    lines.append("")

    lines.append("## Datasets Comparison")
    lines.append("")
    common_all = len(patterns_comparison.get('common_all', set()))
    common_rules_all = len(rules_comparison.get('common_all', set()))
    lines.append(f"- **Patterns common to ALL datasets**: {common_all}")
    lines.append(f"- **Rules common to ALL datasets**: {common_rules_all}")
    lines.append("")

    lines.append("## Common Patterns (All datasets)")
    lines.append("")

    if len(common_patterns_df) > 0:
        lines.append("### Top 20 Common Patterns (by avg_support)")
        lines.append("")

        header = "| Pattern | Avg Support | " + " | ".join(all_results.keys()) + " |"
        separator = "|---------|-------------|-" + "-|-".join(["-"*len(k) for k in all_results.keys()]) + "-|"
        lines.append(header)
        lines.append(separator)

        for _, row in common_patterns_df.head(20).iterrows():
            avg_sup = row.get('avg_support', 'N/A')
            if isinstance(avg_sup, float):
                avg_sup = f"{avg_sup:.2f}"

            line_parts = [f"| `{row['pattern']}` | {avg_sup} "]
            for ds_name in all_results.keys():
                sup = row.get(f'{ds_name}_support', 'N/A')
                if isinstance(sup, float):
                    sup = f"{sup:.2f}"
                line_parts.append(f"| {sup} ")
            line_parts.append("|")
            lines.append("".join(line_parts))

    lines.append("")

    lines.append("## Common Rules (All datasets)")
    lines.append("")

    if len(common_rules_df) > 0:
        lines.append("### All Common Rules (by avg_confidence)")
        lines.append("")

        for _, row in common_rules_df.iterrows():
            rule = row['rule']
            avg_conf = row.get('avg_confidence', 'N/A')
            avg_lift = row.get('avg_lift', 'N/A')
            avg_sup = row.get('avg_support', 'N/A')

            if isinstance(avg_conf, float):
                avg_conf = f"{avg_conf:.2f}"
            if isinstance(avg_lift, float):
                avg_lift = f"{avg_lift:.2f}"
            if isinstance(avg_sup, float):
                avg_sup = f"{avg_sup:.2f}"

            lines.append(f"- `{rule}` (conf={avg_conf}, lift={avg_lift}, sup={avg_sup})")

    lines.append("")

    lines.append("## Conclusions")
    lines.append("")
    lines.append("### Repeating Patterns Across Datasets")
    lines.append("")

    if common_all > 0:
        lines.append(f"Yes - {common_all} patterns are common to all datasets, making them universal.")
    else:
        lines.append("**NO** - No patterns found common to all datasets.")

    lines.append("")

    report_text = "\n".join(lines)
    with open(output_dir / "comparison_report.md", "w") as f:
        f.write(report_text)

    print(f"Zapisano raport: {output_dir / 'comparison_report.md'}")


def run_combined_analysis(
    datasets_files: Dict[str, Path],
    minsup: float = MINSUP,
    minconf: float = MINCONF,
    maxgap: float = MAXGAP,
    max_pattern_size: int = MAX_PATTERN_SIZE,
    output_dir: Path = None
) -> Tuple[List, List]:
    """
    Runs analysis on combined datasets (Meta-analysis).
    Finds patterns globally frequent, even if locally rare.
    """
    print(f"\n{'='*60}")
    print(f"Przetwarzanie: COMBINED (Wszystkie zbiory razem)")
    print(f"{'='*60}")

    combined_df_list = []

    for ds_name, file_path in datasets_files.items():
        if str(file_path).endswith('.csv'):
            df = pd.read_csv(file_path)

            # Standardize column names to lowercase (as expected by ARMADA load_from_dataframe)
            if 'ClientID' in df.columns:
                df = df.rename(columns={'ClientID': 'client_id'})
            if 'State' in df.columns:
                df = df.rename(columns={'State': 'state'})
            if 'Start' in df.columns:
                df = df.rename(columns={'Start': 'start_time'})
            if 'End' in df.columns:
                df = df.rename(columns={'End': 'end_time'})

            # Add prefix to client_id to avoid collisions (e.g. P1 in CASE and CEAP)
            if 'client_id' in df.columns:
                df['client_id'] = f"{ds_name}_" + df['client_id'].astype(str)
            else:
                print(f"WARN: Missing client_id in {ds_name}. Columns: {df.columns.tolist()}")

            combined_df_list.append(df)
            print(f"  Dodano {len(df)} wierszy z {ds_name}")

    if not combined_df_list:
        print("No data to combine.")
        return [], []

    full_df = pd.concat(combined_df_list, ignore_index=True)
    print(f"Łącznie: {len(full_df)} wierszy, {full_df['client_id'].nunique()} uczestników")

    # Run ARMADA on combined data
    armada = ARMADA(
        minsup=minsup,
        minconf=minconf,
        maxgap=maxgap,
        max_pattern_size=max_pattern_size
    )

    patterns, rules = armada.run(df=full_df)

    print(f"  Wzorców (Combined): {len(patterns)}")
    print(f"  Reguł (Combined): {len(rules)}")

    if output_dir:
        comb_out = output_dir / "combined_all"
        comb_out.mkdir(exist_ok=True)
        armada.save_results(comb_out)

    return patterns, rules

def main():
    """Main experiment function."""

    DATA_DIR = PROJECT_DIR / "data" / "armada_ready"
    OUTPUT_DIR = SCRIPT_DIR / "results"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("EKSPERYMENT: PORÓWNANIE WZORCÓW ARMADA MIĘDZY ZBIORAMI DANYCH")
    print("=" * 80)
    print(f"Data: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Parametry: minsup={MINSUP}, minconf={MINCONF}, maxgap={MAXGAP}, max_size={MAX_PATTERN_SIZE}")
    print(f"Filtry reguł: FILTER_BVP_ONLY={FILTER_BVP_ONLY}, FILTER_EDA_ONLY={FILTER_EDA_ONLY}, FILTER_PHYSIO_CROSS={FILTER_PHYSIO_CROSS}, FILTER_SINGLE_FEATURE={FILTER_SINGLE_FEATURE}")
    print(f"Wyniki: {OUTPUT_DIR}")
    print()

    datasets = {
        'CASE': DATA_DIR / "armada_case.csv",
        'K-emoCon': DATA_DIR / "armada_k_emocon.csv",
        'CEAP': DATA_DIR / "armada_ceap.csv",
        'EmoWorker_v2': DATA_DIR / "armada_emoworker_v2.csv"
    }

    for ds_name, data_file in datasets.items():
        if not data_file.exists():
            print(f"ERROR: File {data_file} does not exist!")
            print("Run: python prepare_armada_data.py && python validate_armada_data.py")
            return

    all_results = {}
    patterns_signatures = {}
    rules_signatures = {}

    for ds_name, data_file in datasets.items():
        print(f"\n{'='*60}")
        print(f"Przetwarzanie: {ds_name}")
        print(f"{'='*60}")

        armada, patterns, rules = run_armada_on_dataset(data_file)

        all_results[ds_name] = (armada, patterns, rules)
        patterns_signatures[ds_name] = extract_pattern_signatures(patterns)
        rules_signatures[ds_name] = extract_rule_signatures(rules)

        print(f"  Wzorców: {len(patterns)}")
        print(f"  Reguł: {len(rules)}")

        # Save per-dataset results
        ds_output = OUTPUT_DIR / ds_name.lower().replace('-', '_')
        ds_output.mkdir(exist_ok=True)
        armada.save_results(ds_output)

    print("\n" + "=" * 80)
    print("PORÓWNANIE WZORCÓW")
    print("=" * 80)

    patterns_comparison = compare_pattern_sets(patterns_signatures)

    filtered_rules_signatures = {}
    for ds_name, rules_set in rules_signatures.items():
        original_count = len(rules_set)
        filtered = filter_rules(rules_set, FILTER_BVP_ONLY, FILTER_EDA_ONLY, FILTER_PHYSIO_CROSS, FILTER_SINGLE_FEATURE)
        filtered_rules_signatures[ds_name] = filtered
        removed = original_count - len(filtered)
        if removed > 0:
            print(f"  {ds_name}: filtered {removed} rules (out of {original_count})")

    rules_comparison = compare_pattern_sets(filtered_rules_signatures)

    common_patterns = patterns_comparison.get('common_all', set())
    common_rules = rules_comparison.get('common_all', set())

    common_rules_3plus = rules_comparison.get('common_3_plus', set())

    print(f"\nWzorce wspólne dla WSZYSTKICH zbiorów: {len(common_patterns)}")
    print(f"Reguły wspólne dla WSZYSTKICH zbiorów: {len(common_rules)}")
    print(f"Reguły wspólne dla >=3 zbiorów: {len(common_rules_3plus)}")

    for ds in patterns_signatures:
        unique = len(patterns_comparison.get(f'unique_{ds}', set()))
        print(f"Wzorce unikalne dla {ds}: {unique}")

    common_patterns_df = save_common_patterns_details(common_patterns, all_results, OUTPUT_DIR)

    # Save common 3+ if common all is empty
    if len(common_rules) == 0 and len(common_rules_3plus) > 0:
        print("Saving 3+ rules because common(4) is empty.")
        common_rules_df = save_common_rules_details(common_rules_3plus, all_results, OUTPUT_DIR)
    else:
        common_rules_df = save_common_rules_details(common_rules, all_results, OUTPUT_DIR)

    create_comparison_visualizations(patterns_signatures, patterns_comparison, OUTPUT_DIR)

    generate_markdown_report(
        patterns_comparison, rules_comparison, all_results,
        common_patterns_df, common_rules_df, OUTPUT_DIR
    )

    # Save JSON summary
    summary = {
        "experiment_date": pd.Timestamp.now().isoformat(),
        "parameters": {
            "minsup": MINSUP,
            "minconf": MINCONF,
            "maxgap": MAXGAP,
            "max_pattern_size": MAX_PATTERN_SIZE
        },
        "filters": {
            "filter_bvp_only": FILTER_BVP_ONLY,
            "filter_eda_only": FILTER_EDA_ONLY,
            "filter_physio_cross": FILTER_PHYSIO_CROSS,
            "filter_single_feature": FILTER_SINGLE_FEATURE
        },
        "datasets": {},
        "comparison": {
            "common_all_patterns": len(common_patterns),
            "common_all_rules": len(common_rules)
        }
    }

    for ds_name, (armada, patterns, rules) in all_results.items():
        summary["datasets"][ds_name] = {
            "num_clients": armada.num_clients,
            "total_patterns": len(patterns),
            "total_rules": len(rules),
            "unique_patterns": len(patterns_comparison.get(f'unique_{ds_name}', set()))
        }


    with open(OUTPUT_DIR / "experiment_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("COMMON PATTERNS (all 3 datasets)")
    print("=" * 80)

    for _, row in common_patterns_df.head(15).iterrows():
        print(f"  {row['pattern']} (avg_sup={row.get('avg_support', 'N/A')})")

    if len(common_patterns_df) > 15:
        print(f"  ... and {len(common_patterns_df) - 15} more")

    print("\n" + "=" * 80)
    print("COMMON RULES (all 3 datasets)")
    print("=" * 80)

    for _, row in common_rules_df.head(10).iterrows():
        print(f"  {row['rule']}")
        print(f"    avg_conf={row.get('avg_confidence', 'N/A')}, avg_lift={row.get('avg_lift', 'N/A')}, avg_sup={row.get('avg_support', 'N/A')}")

    if len(common_rules_df) > 10:
        print(f"  ... and {len(common_rules_df) - 10} more")

    combined_patterns, combined_rules = run_combined_analysis(
        datasets,
        minsup=MINSUP,
        minconf=MINCONF,
        maxgap=MAXGAP,
        max_pattern_size=MAX_PATTERN_SIZE,
        output_dir=OUTPUT_DIR
    )

    combined_rules_sigs = extract_rule_signatures(combined_rules)
    filtered_combined = filter_rules(combined_rules_sigs, FILTER_BVP_ONLY, FILTER_EDA_ONLY, FILTER_PHYSIO_CROSS, FILTER_SINGLE_FEATURE)

    print(f"  Rules (Combined) after filtering: {len(filtered_combined)}")

    # Save combined rules details
    # Using save_common_rules_details format but for single column
    comb_details = []
    for sig in filtered_combined:
        # Find rule object
        rule_obj = next((r for r in combined_rules if f"{r.antecedent.get_relation_description()} => {r.consequent.get_relation_description()}" == sig), None)
        if rule_obj:
            comb_details.append({
                "rule": sig,
                "confidence": rule_obj.confidence,
                "lift": rule_obj.lift,
                "support": rule_obj.support,
            })

    comb_df = pd.DataFrame(comb_details)
    if not comb_df.empty:
        comb_df = comb_df.sort_values("confidence", ascending=False)
        comb_df.to_csv(OUTPUT_DIR / "combined_rules_details.csv", index=False)

        print("\n" + "=" * 80)
        print("RULES FROM COMBINED ANALYSIS (TOP 15)")
        print("=" * 80)
        for _, row in comb_df.head(15).iterrows():
            print(f"  {row['rule']} (conf={row['confidence']:.2f}, lift={row['lift']:.2f}, sup={row['support']:.2f})")

    print("\n" + "=" * 80)
    print("EXPERIMENT FINISHED")
    print(f"Results saved in: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()

