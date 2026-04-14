#!/usr/bin/env python3
"""
Demographic experiments: ARMADA patterns analysis considering age and gender.

Experiments:
1. Compare ARMADA patterns between genders (M vs F)
2. Compare patterns between age groups
3. Interaction gender x age
4. Breakdown of Combined rules support into demographic groups

Requirements:
- Processed data in data/armada_ready/ (from prepare_armada_data.py)
- Demographic data in processed CSVs (gender, age, age_group)
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).parent
EXPERIMENTS_DIR = SCRIPT_DIR.parent
PROJECT_DIR = EXPERIMENTS_DIR.parent.parent  # second part/

sys.path.insert(0, str(PROJECT_DIR / "source" / "processing" / "armada"))
sys.path.insert(0, str(PROJECT_DIR / "source" / "processing" / "extracting"))
sys.path.insert(0, str(EXPERIMENTS_DIR))

from armada_algorithm import ARMADA, TemporalRule, TemporalPattern
from experiment_utils import run_armada_on_df, extract_rule_signatures, jaccard_similarity

MINSUP = 0.3
MINCONF = 0.5
MAXGAP = 5
MAX_PATTERN_SIZE = 4

DATA_DIR = PROJECT_DIR / "data" / "armada_ready"
PROCESSED_DIRS = {
    'CASE': PROJECT_DIR / "data" / "CASE" / "processed",
    'K-emoCon': PROJECT_DIR / "data" / "K-emoCon" / "processed",
    'CEAP': PROJECT_DIR / "data" / "CEAP" / "processed",
    'EmoWorker_v2': PROJECT_DIR / "data" / "EmoWorker_v2" / "processed",
}

OUTPUT_DIR = SCRIPT_DIR / "results"

AGE_GROUP_YOUNG = 'young'  # <= 25
AGE_GROUP_OLD = 'old'      # > 25


def load_demographics_from_processed() -> pd.DataFrame:
    """
    Loads demographic data from all processed CSVs.

    Looks for 'gender' and 'age' columns in *_merged.csv files.
    Returns DataFrame with columns: dataset, participant_id, gender, age, age_group.
    """
    records = []

    for ds_name, proc_dir in PROCESSED_DIRS.items():
        if not proc_dir.exists():
            print(f"  SKIP: {ds_name} — missing folder {proc_dir}")
            continue

        csv_files = list(proc_dir.glob("*_merged.csv"))

        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file, nrows=1)  # Read only the first line
                participant_id = csv_file.stem.replace("_merged", "")

                gender = None
                age = None
                age_group = None

                if 'gender' in df.columns:
                    gender = df['gender'].iloc[0]
                    if pd.notna(gender):
                        g_str = str(gender).strip().upper()
                        if g_str in ['M', 'MALE', '1', 'MAN']:
                            gender = 'M'
                        elif g_str in ['F', 'FEMALE', '2', 'WOMAN']:
                            gender = 'F'
                        else:
                            gender = None
                if 'age' in df.columns:
                    age = df['age'].iloc[0]
                if 'age_group' in df.columns:
                    age_group = df['age_group'].iloc[0]

                # Calculate binary age group (young/old)
                binary_age_group = None
                if age is not None and not pd.isna(age):
                    binary_age_group = AGE_GROUP_YOUNG if age <= 25 else AGE_GROUP_OLD

                records.append({
                    'dataset': ds_name,
                    'participant_id': participant_id,
                    'client_id': f"{ds_name}_{participant_id}",
                    'gender': gender,
                    'age': age,
                    'age_group': age_group,
                    'binary_age_group': binary_age_group
                })
            except Exception as e:
                print(f"  Error loading {csv_file}: {e}")

    demo_df = pd.DataFrame(records)
    return demo_df


def load_armada_data(data_file: Path) -> pd.DataFrame:
    """Loads ARMADA data from CSV."""
    return pd.read_csv(data_file)


def split_data_by_group(
    armada_df: pd.DataFrame,
    demo_df: pd.DataFrame,
    group_column: str
) -> Dict[str, pd.DataFrame]:
    """
    Splits ARMADA data into groups based on a demographic column.

    Args:
        armada_df: ARMADA data (client_id, state, start_time, end_time)
        demo_df: demographic data (client_id, gender, age, ...)
        group_column: column to group by (e.g. 'gender', 'binary_age_group')

    Returns:
        dict: {group_name: armada_df_for_group}
    """
    # Merge dynamically on client_id
    merged = armada_df.merge(demo_df[['client_id', group_column]], on='client_id', how='left')

    # Drop rows without demographic data
    merged = merged.dropna(subset=[group_column])

    groups = {}
    for group_name, group_data in merged.groupby(group_column):
        groups[str(group_name)] = group_data[['client_id', 'state', 'start_time', 'end_time']].copy()

    return groups



def compare_groups(
    group_results: Dict[str, Tuple],
    output_dir: Path,
    comparison_name: str
) -> pd.DataFrame:
    """
    Compares patterns and rules between groups.

    Returns:
        DataFrame with comparison results
    """
    group_names = list(group_results.keys())

    # Extract rule signatures
    group_rules = {}
    for name, (armada, patterns, rules) in group_results.items():
        sigs = extract_rule_signatures(rules)
        group_rules[name] = sigs
        print(f"  {name}: {len(patterns)} patterns, {len(rules)} rules, {armada.num_clients} participants")

    # Jaccard matrix
    jaccard_matrix = pd.DataFrame(
        index=group_names, columns=group_names, dtype=float
    )
    for i, g1 in enumerate(group_names):
        for j, g2 in enumerate(group_names):
            jaccard_matrix.loc[g1, g2] = jaccard_similarity(group_rules[g1], group_rules[g2])

    print(f"\n  Jaccard Matrix ({comparison_name}):")
    print(jaccard_matrix.to_string())

    # Common and unique rules
    all_rules = set()
    for sigs in group_rules.values():
        all_rules |= sigs

    common_all = set.intersection(*group_rules.values()) if group_rules else set()

    print(f"\n  Rules common for ALL groups: {len(common_all)}")
    print(f"  Total rules (union): {len(all_rules)}")

    for name, sigs in group_rules.items():
        others = set()
        for other_name, other_sigs in group_rules.items():
            if other_name != name:
                others |= other_sigs
        unique = sigs - others
        print(f"  Rules unique to {name}: {len(unique)}")

    results = []
    for rule_sig in sorted(all_rules):
        entry = {'rule': rule_sig}
        for name in group_names:
            entry[f'{name}_present'] = rule_sig in group_rules[name]
            # Find confidence and support
            if rule_sig in group_rules[name]:
                armada, patterns, rules = group_results[name]
                for r in rules:
                    sig = f"{r.antecedent.get_relation_description()} => {r.consequent.get_relation_description()}"
                    if sig == rule_sig:
                        entry[f'{name}_confidence'] = round(r.confidence, 4)
                        entry[f'{name}_lift'] = round(r.lift, 4)
                        entry[f'{name}_support'] = round(r.support, 4)
                        break
        results.append(entry)

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / f"rules_comparison_{comparison_name}.csv", index=False)

    jaccard_matrix.to_csv(output_dir / f"jaccard_matrix_{comparison_name}.csv")

    return results_df


def experiment_1_gender(armada_combined_df: pd.DataFrame, demo_df: pd.DataFrame, output_dir: Path):
    """
    Experiment 1: Compare ARMADA patterns between genders (M vs F).

    Hypothesis: women may exhibit higher EDA reactivity (Boucsein, 2012).
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 1: M vs F COMPARISON")
    print("=" * 80)

    # Normalize gender values
    demo_df = demo_df.copy()
    demo_df['gender_clean'] = demo_df['gender'].apply(
        lambda x: 'M' if str(x).upper() in ['M', 'MALE', '1', 'MAN'] else
                  ('F' if str(x).upper() in ['F', 'FEMALE', '2', 'WOMAN'] else None)
    )

    groups = split_data_by_group(armada_combined_df, demo_df.rename(
        columns={'gender_clean': 'gender_group'}
    ).assign(gender_group=demo_df['gender_clean']), 'gender_group')

    if len(groups) < 2:
        print("  SKIP: not enough groups (M and F needed)")
        print(f"  Found groups: {list(groups.keys())}")
        return None

    # Run ARMADA on each group
    group_results = {}
    for group_name, group_df in groups.items():
        print(f"\n  Running ARMADA for group {group_name}...")
        armada, patterns, rules = run_armada_on_df(group_df, MINSUP, MINCONF, MAXGAP, MAX_PATTERN_SIZE)
        group_results[group_name] = (armada, patterns, rules)

    results_df = compare_groups(group_results, output_dir, "gender")
    return results_df


def experiment_2_age(armada_combined_df: pd.DataFrame, demo_df: pd.DataFrame, output_dir: Path):
    """
    Experiment 2: Compare patterns between age groups (young <= 25, old > 25).

    Hypothesis: HRV decreases with age (Pham et al., 2021).
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: YOUNG vs OLD COMPARISON")
    print("=" * 80)

    groups = split_data_by_group(armada_combined_df, demo_df, 'binary_age_group')

    if len(groups) < 2:
        print("  SKIP: not enough age groups")
        print(f"  Found groups: {list(groups.keys())}")
        return None

    group_results = {}
    for group_name, group_df in groups.items():
        print(f"\n  Running ARMADA for group {group_name}...")
        armada, patterns, rules = run_armada_on_df(group_df, MINSUP, MINCONF, MAXGAP, MAX_PATTERN_SIZE)
        group_results[group_name] = (armada, patterns, rules)

    results_df = compare_groups(group_results, output_dir, "age")
    return results_df


def experiment_3_interaction(armada_combined_df: pd.DataFrame, demo_df: pd.DataFrame, output_dir: Path):
    """
    Experiment 3: Interaction gender x age (4 groups: Young-M, Young-F, Old-M, Old-F).
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 3: GENDER x AGE INTERACTION")
    print("=" * 80)

    demo_df = demo_df.copy()
    demo_df['gender_clean'] = demo_df['gender'].apply(
        lambda x: 'M' if str(x).upper() in ['M', 'MALE', '1', 'MAN'] else
                  ('F' if str(x).upper() in ['F', 'FEMALE', '2', 'WOMAN'] else None)
    )

    # Create interaction group
    demo_df['interaction_group'] = None
    mask = demo_df['gender_clean'].notna() & demo_df['binary_age_group'].notna()
    demo_df.loc[mask, 'interaction_group'] = (
        demo_df.loc[mask, 'binary_age_group'].astype(str) + '_' +
        demo_df.loc[mask, 'gender_clean'].astype(str)
    )

    groups = split_data_by_group(armada_combined_df, demo_df, 'interaction_group')

    if len(groups) < 2:
        print("  SKIP: not enough interaction groups")
        print(f"  Found groups: {list(groups.keys())}")
        return None

    group_results = {}
    for group_name, group_df in groups.items():
        n_clients = group_df['client_id'].nunique()
        if n_clients < 3:
            print(f"  SKIP group {group_name}: not enough participants ({n_clients})")
            continue
        print(f"\n  Running ARMADA for group {group_name} ({n_clients} participants)...")
        armada, patterns, rules = run_armada_on_df(group_df, MINSUP, MINCONF, MAXGAP, MAX_PATTERN_SIZE)
        group_results[group_name] = (armada, patterns, rules)

    if len(group_results) < 2:
        print("  SKIP: not enough groups with enough participants")
        return None

    results_df = compare_groups(group_results, output_dir, "interaction")
    return results_df


def experiment_4_combined_breakdown(armada_combined_df: pd.DataFrame, demo_df: pd.DataFrame, output_dir: Path):
    """
    Experiment 4: Demographic analysis of Combined rules.

    For each rule discovered in the entire Combined dataset, calculates
    support breakdown into demographic groups (gender and age).
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 4: DEMOGRAPHIC BREAKDOWN OF COMBINED RULES")
    print("=" * 80)

    # Run ARMADA on whole Combined
    print("  Running ARMADA on Combined...")
    armada, patterns, rules = run_armada_on_df(armada_combined_df, MINSUP, MINCONF, MAXGAP, MAX_PATTERN_SIZE)
    print(f"  Combined: {len(patterns)} patterns, {len(rules)} rules, {armada.num_clients} participants")

    # Prepare demographic data
    demo_df = demo_df.copy()
    demo_df['gender_clean'] = demo_df['gender'].apply(
        lambda x: 'M' if str(x).upper() in ['M', 'MALE', '1', 'MAN'] else
                  ('F' if str(x).upper() in ['F', 'FEMALE', '2', 'WOMAN'] else None)
    )

    # Map client_id -> demographic group
    client_gender = dict(zip(demo_df['client_id'], demo_df['gender_clean']))
    client_age = dict(zip(demo_df['client_id'], demo_df['binary_age_group']))

    # Calculate support breakdown per rule
    breakdown_records = []
    all_client_ids = set(armada_combined_df['client_id'].unique())

    for rule in sorted(rules, key=lambda r: -r.confidence):
        sig = f"{rule.antecedent.get_relation_description()} => {rule.consequent.get_relation_description()}"

        record = {
            'rule': sig,
            'confidence': round(rule.confidence, 4),
            'lift': round(rule.lift, 4),
            'support': round(rule.support, 4),
        }

        # Here we cannot directly calculate per-group support from the rule object,
        # but we can calculate per-group pattern presence.
        # Simplification: we report overall support and demographics overview.

        # Count participants with demographic data
        n_m = sum(1 for cid in all_client_ids if client_gender.get(cid) == 'M')
        n_f = sum(1 for cid in all_client_ids if client_gender.get(cid) == 'F')
        n_young = sum(1 for cid in all_client_ids if client_age.get(cid) == AGE_GROUP_YOUNG)
        n_old = sum(1 for cid in all_client_ids if client_age.get(cid) == AGE_GROUP_OLD)

        record['n_male'] = n_m
        record['n_female'] = n_f
        record['n_young'] = n_young
        record['n_old'] = n_old

        breakdown_records.append(record)

    # Top 50 rules
    breakdown_df = pd.DataFrame(breakdown_records[:50])
    breakdown_df.to_csv(output_dir / "combined_rules_demographic_breakdown.csv", index=False)

    print(f"\n  Saved demographic breakdown of top 50 rules")
    print(f"  Participants with data: M={n_m}, F={n_f}, young={n_young}, old={n_old}")

    return breakdown_df


def create_summary_visualizations(demo_df: pd.DataFrame, output_dir: Path):
    """Creates summary visualizations for demographic data."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Gender distribution per dataset
    ax1 = axes[0]
    gender_counts = demo_df.groupby(['dataset', 'gender']).size().unstack(fill_value=0)
    if not gender_counts.empty:
        gender_counts.plot(kind='bar', ax=ax1, colormap='Set2')
        ax1.set_title('Gender distribution per dataset')
        ax1.set_ylabel('Number of participants')
        ax1.tick_params(axis='x', rotation=45)

    # 2. Age distribution
    ax2 = axes[1]
    valid_ages = demo_df['age'].dropna()
    if len(valid_ages) > 0:
        ax2.hist(valid_ages, bins=15, color='#3498db', edgecolor='white', alpha=0.8)
        ax2.set_title('Age distribution (all participants)')
        ax2.set_xlabel('Age')
        ax2.set_ylabel('Count')
        ax2.axvline(x=25, color='red', linestyle='--', label='Young/old split (25)')
        ax2.legend()

    # 3. Age groups per dataset
    ax3 = axes[2]
    age_counts = demo_df.groupby(['dataset', 'binary_age_group']).size().unstack(fill_value=0)
    if not age_counts.empty:
        age_counts.plot(kind='bar', ax=ax3, colormap='Set1')
        ax3.set_title('Age groups per dataset')
        ax3.set_ylabel('Number of participants')
        ax3.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(output_dir / 'demographics_overview.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'demographics_overview.png'}")


def generate_report(
    demo_df: pd.DataFrame,
    exp1_df: Optional[pd.DataFrame],
    exp2_df: Optional[pd.DataFrame],
    exp3_df: Optional[pd.DataFrame],
    exp4_df: Optional[pd.DataFrame],
    output_dir: Path
):
    lines = []
    lines.append("# Demographic analysis of ARMADA patterns")
    lines.append("")
    lines.append("## Parameters")
    lines.append(f"- **minsup**: {MINSUP}")
    lines.append(f"- **minconf**: {MINCONF}")
    lines.append(f"- **maxgap**: {MAXGAP}")
    lines.append(f"- **max_pattern_size**: {MAX_PATTERN_SIZE}")
    lines.append("")

    lines.append("## Demographic data")
    lines.append("")
    lines.append(f"- Total participants: {len(demo_df)}")
    n_with_gender = demo_df['gender'].notna().sum()
    n_with_age = demo_df['age'].notna().sum()
    lines.append(f"- With gender data: {n_with_gender}")
    lines.append(f"- With age data: {n_with_age}")
    lines.append("")

    lines.append("| Dataset | Participants | Female | Male | Age Range | Young | Old |")
    lines.append("|---------|--------------|--------|------|-----------|-------|-----|")
    for ds in sorted(demo_df['dataset'].unique()):
        ds_data = demo_df[demo_df['dataset'] == ds]
        
        n_female = (ds_data['gender'] == 'F').sum()
        n_male = (ds_data['gender'] == 'M').sum()
        
        n_young = (ds_data['binary_age_group'] == 'young').sum() if 'binary_age_group' in ds_data.columns else 0
        n_old = (ds_data['binary_age_group'] == 'old').sum() if 'binary_age_group' in ds_data.columns else 0
        
        ages = ds_data['age'].dropna()
        if len(ages) > 0:
            age_range = f"{ages.min():.0f}-{ages.max():.0f}"
        else:
            age_range = "N/A"

        lines.append(f"| {ds} | {len(ds_data)} | {n_female} | {n_male} | {age_range} | {n_young} | {n_old} |")
    lines.append("")

    # Experiments
    for i, (name, df) in enumerate([
        ("Experiment 1: M vs F", exp1_df),
        ("Experiment 2: Young vs Old", exp2_df),
        ("Experiment 3: Gender x Age Interaction", exp3_df),
        ("Experiment 4: Demographic Breakdown of Combined", exp4_df)
    ], 1):
        lines.append(f"## {name}")
        lines.append("")
        if df is not None and len(df) > 0:
            lines.append(f"Number of rules in analysis: {len(df)}")
            lines.append(f"Details: `rules_comparison_*.csv`")
        else:
            lines.append("**SKIPPED** — not enough demographic data.")
        lines.append("")

    report_text = "\n".join(lines)
    with open(output_dir / "demographic_report.md", "w") as f:
        f.write(report_text)

    print(f"\nSaved report: {output_dir / 'demographic_report.md'}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("DEMOGRAPHIC EXPERIMENTS: ARMADA + AGE/GENDER")
    print("=" * 80)

    # 1. Load demographic data
    print("\n--- Loading demographic data ---")
    demo_df = load_demographics_from_processed()
    print(f"\nTotal participants: {len(demo_df)}")
    print(f"With gender data: {demo_df['gender'].notna().sum()}")
    print(f"With age data: {demo_df['age'].notna().sum()}")

    if demo_df['gender'].notna().sum() == 0 and demo_df['age'].notna().sum() == 0:
        print("\nERROR: No demographic data found!")
        print("Check if *_merged.csv files contain 'gender' and 'age' columns.")
        print("You might need to re-run data extraction scripts (CASE.py, K_emoCon.py, etc.)")
        return

    # Summary per dataset
    print("\n--- Demographic summary per dataset ---")
    
    print("\nBreakdown of Men and Women by Dataset:")
    gender_table = demo_df.groupby(['dataset', 'gender']).size().unstack(fill_value=0)
    if 'F' not in gender_table.columns:
        gender_table['F'] = 0
    if 'M' not in gender_table.columns:
        gender_table['M'] = 0
    gender_table = gender_table[['F', 'M']]
    gender_table = gender_table.rename(columns={'F': 'Female', 'M': 'Male'})
    print(gender_table.to_string())
    print()

    for ds in demo_df['dataset'].unique():
        ds_data = demo_df[demo_df['dataset'] == ds]
        n_gender = ds_data['gender'].notna().sum()
        n_age = ds_data['age'].notna().sum()
        genders = ds_data['gender'].dropna().unique()
        ages = ds_data['age'].dropna()
        age_str = f"age: {ages.min():.0f}-{ages.max():.0f}" if len(ages) > 0 else "none"
        print(f"  {ds}: {len(ds_data)} participants, gender={n_gender} ({genders}), {age_str}")

    # 2. Load ARMADA data (Combined)
    combined_file = DATA_DIR / "armada_combined_all.csv"
    if not combined_file.exists():
        print(f"\nERROR: Missing file {combined_file}")
        print("Run first: python prepare_armada_data.py")
        return

    armada_df = load_armada_data(combined_file)
    print(f"\nARMADA data: {len(armada_df)} intervals, {armada_df['client_id'].nunique()} participants")

    # 3. Demographics visualization
    create_summary_visualizations(demo_df, OUTPUT_DIR)

    # 4. Experiments
    exp1_df = experiment_1_gender(armada_df, demo_df, OUTPUT_DIR)
    exp2_df = experiment_2_age(armada_df, demo_df, OUTPUT_DIR)
    exp3_df = experiment_3_interaction(armada_df, demo_df, OUTPUT_DIR)
    exp4_df = experiment_4_combined_breakdown(armada_df, demo_df, OUTPUT_DIR)

    # 5. Report
    generate_report(demo_df, exp1_df, exp2_df, exp3_df, exp4_df, OUTPUT_DIR)

    print("\n" + "=" * 80)
    print("EXPERIMENTS FINISHED")
    print(f"Results: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
