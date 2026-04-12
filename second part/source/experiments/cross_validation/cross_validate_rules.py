#!/usr/bin/env python3
"""
Experiment: Cross-validation of ARMADA rules between datasets.

For each combination of two training sets and one validation set:
1. Find rules common to two training sets.
2. Check how many of them appear in the validation set.
3. Compare confidence and support.

Combinations:
  - Train: CASE + K-emoCon    -> Val: CEAP
  - Train: CASE + CEAP        -> Val: K-emoCon
  - Train: K-emoCon + CEAP    -> Val: CASE
"""

import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

SCRIPT_DIR = Path(__file__).parent
EXPERIMENTS_DIR = SCRIPT_DIR.parent
PROJECT_DIR = EXPERIMENTS_DIR.parent.parent
sys.path.insert(0, str(PROJECT_DIR / "source" / "processing" / "armada"))
sys.path.insert(0, str(EXPERIMENTS_DIR))

from armada_algorithm import ARMADA, TemporalRule
from compare_datasets import (
    run_armada_on_dataset,
    extract_rule_signatures,
    filter_rules,
    MINSUP, MINCONF, MAXGAP, MAX_PATTERN_SIZE,
    FILTER_BVP_ONLY, FILTER_EDA_ONLY, FILTER_PHYSIO_CROSS, FILTER_SINGLE_FEATURE,
)

DATA_DIR = PROJECT_DIR / "data" / "armada_ready"
OUTPUT_DIR = SCRIPT_DIR / "results"

DATASETS = {
    'CASE': DATA_DIR / "armada_case.csv",
    'K-emoCon': DATA_DIR / "armada_k_emocon.csv",
    'CEAP': DATA_DIR / "armada_ceap.csv",
    'EmoWorker_v2': DATA_DIR / "armada_emoworker_v2.csv",
}


def generate_combinations(datasets: List[str]) -> List[Tuple[List[str], str]]:
    """Generates (Train Sets) -> Test Set (Leave-One-Out) combinations."""
    combinations = []
    for test_set in datasets:
        train_sets = [d for d in datasets if d != test_set]
        combinations.append((train_sets, test_set))
    return combinations


def get_rule_details(rules: List[TemporalRule]) -> Dict[str, Dict]:
    """
    Maps rule signature to its details (confidence, support).

    Returns:
        dict: {signature: {'confidence': float, 'support': float}}
    """
    details = {}
    for r in rules:
        sig = f"{r.antecedent.get_relation_description()} => {r.consequent.get_relation_description()}"
        details[sig] = {
            'confidence': r.confidence,
            'support': r.support,
        }
    return details


def run_cross_validation():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("EXPERIMENT: ARMADA RULES CROSS-VALIDATION")
    print("=" * 90)
    print(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Parameters: minsup={MINSUP}, minconf={MINCONF}, maxgap={MAXGAP}, max_size={MAX_PATTERN_SIZE}")
    print(f"Filters: BVP_ONLY={FILTER_BVP_ONLY}, EDA_ONLY={FILTER_EDA_ONLY}, "
          f"PHYSIO_CROSS={FILTER_PHYSIO_CROSS}, SINGLE_FEAT={FILTER_SINGLE_FEATURE}")
    print()

    # Check files
    for ds_name, path in DATASETS.items():
        if not path.exists():
            print(f"ERROR: Missing file {path}")
            return

    # ================================================================
    # 1. Run ARMADA on each dataset
    # ================================================================
    all_results = {}  # {name: (armada, patterns, rules)}
    all_rule_sigs = {}  # {name: set(signatures)}
    all_rule_details = {}  # {name: {sig: {conf, sup}}}

    for ds_name, data_file in DATASETS.items():
        print(f"Processing: {ds_name}...", end=" ", flush=True)
        armada, patterns, rules = run_armada_on_dataset(data_file)
        all_results[ds_name] = (armada, patterns, rules)

        # Filter rules
        raw_sigs = extract_rule_signatures(rules)
        filtered_sigs = filter_rules(raw_sigs, FILTER_BVP_ONLY, FILTER_EDA_ONLY,
                                     FILTER_PHYSIO_CROSS, FILTER_SINGLE_FEATURE)
        all_rule_sigs[ds_name] = filtered_sigs
        all_rule_details[ds_name] = get_rule_details(rules)

        n_clients = armada.num_clients
        print(f"{n_clients} participants, {len(patterns)} patterns, "
              f"{len(rules)} rules ({len(filtered_sigs)} after filtering)")

    # ================================================================
    # 2. Cross-validation: (N-1) train -> 1 val
    # ================================================================
    ds_names = list(DATASETS.keys())
    combinations = generate_combinations(ds_names)

    cv_results = []

    print()
    print("=" * 90)
    print("CROSS-VALIDATION")
    print("=" * 90)

    for train_sets, val_set in combinations:
        train_names_str = " + ".join(train_sets)
        print(f"\n--- Train: {train_names_str}  ->  Val: {val_set} ---")

        # Rules common for ALL training sets
        common_train = None
        for t in train_sets:
            if common_train is None:
                common_train = set(all_rule_sigs[t])
            else:
                common_train &= all_rule_sigs[t]

        if common_train is None: common_train = set()

        print(f"  Common rules ({train_names_str}): {len(common_train)}")

        # How many of them are in the validation set
        validated = common_train & all_rule_sigs[val_set]
        print(f"  Validated in {val_set}: {len(validated)}")

        hit_rate = len(validated) / len(common_train) * 100 if common_train else 0
        print(f"  Hit rate: {hit_rate:.1f}%")

        # Collect validated rules details
        validated_details = []
        for sig in sorted(validated):
            detail = {'rule': sig}

            # Values for training sets
            train_confs = []
            train_sups = []
            for t in train_sets:
                d = all_rule_details[t].get(sig, {})
                c = d.get('confidence', 0)
                s = d.get('support', 0)
                detail[f'{t}_conf'] = round(c, 3)
                detail[f'{t}_sup'] = round(s, 3)
                train_confs.append(c)
                train_sups.append(s)

            # Values for validation set
            dv = all_rule_details[val_set].get(sig, {})
            val_conf = dv.get('confidence', 0)
            val_sup = dv.get('support', 0)
            detail[f'{val_set}_conf'] = round(val_conf, 3)
            detail[f'{val_set}_sup'] = round(val_sup, 3)

            # Averages
            train_avg_conf = sum(train_confs) / len(train_confs) if train_confs else 0
            detail['train_avg_conf'] = round(train_avg_conf, 3)
            detail['val_conf'] = round(val_conf, 3)
            detail['conf_diff'] = round(val_conf - train_avg_conf, 3)

            validated_details.append(detail)

        cv_results.append({
            'train_sets': train_sets,
            'val_set': val_set,
            'common_train': len(common_train),
            'validated': len(validated),
            'hit_rate': hit_rate,
            'details': validated_details,
        })

    # ================================================================
    # 3. Rules validated in ALL combinations
    # ================================================================
    # Check rules that are universal (present in EVERY dataset).
    all_universal = None
    for ds in ds_names:
        if all_universal is None:
            all_universal = set(all_rule_sigs[ds])
        else:
            all_universal &= all_rule_sigs[ds]

    print()
    print("=" * 90)
    print(f"RULES PRESENT IN EVERY DATASET ({len(all_universal)})")
    print("=" * 90)

    # ================================================================
    # 4. Generate summary table
    # ================================================================
    summary_rows = []
    for cv in cv_results:
        train_str = " + ".join(cv['train_sets'])
        summary_rows.append({
            'Train (N-1)': train_str,
            'Val': cv['val_set'],
            'Common (Train)': cv['common_train'],
            'Validated': cv['validated'],
            'Hit Rate (%)': round(cv['hit_rate'], 1),
        })

    summary_df = pd.DataFrame(summary_rows)
    print()
    print(summary_df.to_string(index=False))

    # ================================================================
    # 5. Detailed table of validated rules
    # ================================================================
    # Count how many datasets contain each rule
    rule_occurence = defaultdict(int)
    for ds in ds_names:
        for sig in all_rule_sigs[ds]:
            rule_occurence[sig] += 1

    # List of all unique rules
    all_unique_rules = set()
    for ds in ds_names:
        all_unique_rules.update(all_rule_sigs[ds])

    # Collect overall conf/sup
    total_datasets = len(ds_names)
    universal_rules = []

    for sig in sorted(all_unique_rules):
        count = rule_occurence[sig]
        # Filter to show only those in min 2 datasets
        if count < 2:
            continue

        all_confs = []
        all_sups = []
        per_ds = {}
        for ds in ds_names:
            dd = all_rule_details[ds].get(sig, {})
            c = dd.get('confidence', None)
            s = dd.get('support', None)
            if c is not None:
                all_confs.append(c)
                per_ds[f'{ds} conf'] = round(c, 3)
            else:
                per_ds[f'{ds} conf'] = '-'
            if s is not None:
                all_sups.append(s)
                per_ds[f'{ds} sup'] = round(s, 3)
            else:
                per_ds[f'{ds} sup'] = '-'

        universal_rules.append({
            'Rule': sig,
            f'Presence (/{total_datasets})': count,
            **per_ds,
            'Avg conf': round(sum(all_confs) / len(all_confs), 3) if all_confs else '-',
            'Avg sup': round(sum(all_sups) / len(all_sups), 3) if all_sups else '-',
        })

    universal_df = pd.DataFrame(universal_rules)
    if len(universal_df) > 0:
        col_count = f'Presence (/{total_datasets})'
        universal_df = universal_df.sort_values(
            by=[col_count, 'Avg conf'],
            ascending=[False, False]
        )

    # ================================================================
    # 6. Save results
    # ================================================================
    summary_df.to_csv(OUTPUT_DIR / "cross_validation_summary.csv", index=False)
    universal_df.to_csv(OUTPUT_DIR / "cross_validated_rules.csv", index=False)

    for cv in cv_results:
        if cv['details']:
            df = pd.DataFrame(cv['details']).sort_values('train_avg_conf', ascending=False)
            fname = f"validated_{'_'.join(cv['train_sets'])}_to_{cv['val_set']}.csv"
            fname = fname.replace(' ', '').replace('+', '_').replace('-', '').lower()
            df.to_csv(OUTPUT_DIR / fname, index=False)

    # ================================================================
    # 7. Generate Markdown report
    # ================================================================
    lines = []
    lines.append("# ARMADA Rules Cross-Validation")
    lines.append("")
    lines.append(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")
    lines.append("## Methodology")
    lines.append("")
    lines.append("For each combination of two training sets:")
    lines.append("1. Find patterns common to the training pair")
    lines.append("2. Check how many of them appear in the validation set")
    lines.append("3. Calculate hit rate (% of validated rules)")
    lines.append("")
    lines.append("## Parameters")
    lines.append("")
    lines.append(f"- minsup: {MINSUP}, minconf: {MINCONF}, maxgap: {MAXGAP}, max_pattern_size: {MAX_PATTERN_SIZE}")
    lines.append(f"- Filters: BVP_ONLY={FILTER_BVP_ONLY}, EDA_ONLY={FILTER_EDA_ONLY}, "
                 f"PHYSIO_CROSS={FILTER_PHYSIO_CROSS}, SINGLE_FEATURE={FILTER_SINGLE_FEATURE}")
    lines.append("")

    lines.append("## Dataset Statistics")
    lines.append("")
    lines.append("| Dataset | Participants | Rules (raw) | Rules (filtered) |")
    lines.append("|-------|-------------|----------------|----------------------|")
    for ds in ds_names:
        armada, patterns, rules = all_results[ds]
        lines.append(f"| {ds} | {armada.num_clients} | {len(rules)} | {len(all_rule_sigs[ds])} |")
    lines.append("")

    lines.append("## Cross-Validation Results")
    lines.append("")
    lines.append(summary_df.to_markdown(index=False))
    lines.append("")

    lines.append("## Top 20 Universal Rules (by dataset count and conf)")
    lines.append("")
    if len(universal_df) > 0:
        lines.append(universal_df.head(20).to_markdown(index=False))
    else:
        lines.append("No common rules found.")
    lines.append("")

    with open(OUTPUT_DIR / "report.md", 'w') as f:
        f.write("\n".join(lines))

    print(f"Report saved to: {OUTPUT_DIR / 'report.md'}")


if __name__ == "__main__":
    run_cross_validation()
