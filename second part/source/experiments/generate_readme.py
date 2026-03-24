#!/usr/bin/env python3
"""
README generator for ARMADA experiment results.

This script:
1. Loads results from experiment_summary.json
2. Loads common patterns and rules from CSV files
3. Generates a comprehensive README.md with results interpretation
"""

import json
from pathlib import Path
from datetime import datetime
import pandas as pd

# Paths
SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results"
OUTPUT_FILE = RESULTS_DIR / "README.md"


def load_experiment_data():
    """Loads experiment data."""
    # Load JSON summary
    summary_file = RESULTS_DIR / "experiment_summary.json"
    if not summary_file.exists():
        raise FileNotFoundError(f"Missing file {summary_file}. Run compare_datasets.py first")

    with open(summary_file) as f:
        summary = json.load(f)

    # Load common patterns
    patterns_file = RESULTS_DIR / "common_patterns_details.csv"
    patterns_df = pd.read_csv(patterns_file) if patterns_file.exists() else pd.DataFrame()

    # Load common rules
    rules_file = RESULTS_DIR / "common_rules_details.csv"
    rules_df = pd.read_csv(rules_file) if rules_file.exists() else pd.DataFrame()

    return summary, patterns_df, rules_df


def interpret_pattern(pattern: str) -> str:
    """Interprets a pattern in natural language."""

    # Mapping states to descriptions
    state_descriptions = {
             'arousal_low': 'low emotional arousal (relaxation, sleepiness)',
        'arousal_medium': 'moderate arousal (normal alertness)',
        'arousal_high': 'high arousal (excitement or stress)',

        # Valence
        'valence_low': 'negative valence (unpleasant emotions)',
        'valence_medium': 'neutral valence',
        'valence_high': 'positive valence (pleasant emotions)',

        # EDA (SCL - Skin Conductance Level, tonic component)
        'eda_low': 'low skin conductance (low stress, relaxation)',
        'eda_medium': 'medium skin conductance (normal wakefulness)',
        'eda_high': 'high skin conductance (high stress/arousal)',

        # EDA SCR amplitude (Skin Conductance Response - phasic component)
        'eda_scr_amp_low': 'low SCR amplitude (weak emotional responses)',
        'eda_scr_amp_medium': 'medium SCR amplitude',
        'eda_scr_amp_high': 'high SCR amplitude (strong emotional responses)',

        # EDA SCR AUC (area under curve - total phasic activity)
        'eda_scr_auc_low': 'low total electrodermal activity (few responses)',
        'eda_scr_auc_medium': 'medium total electrodermal activity',
        'eda_scr_auc_high': 'high total electrodermal activity (intense responses)',

        # EDA std (phasic component variability - SCR variability)
        'eda_std_low': 'low SCR variability (stable signal)',
        'eda_std_medium': 'medium SCR variability',
        'eda_std_high': 'high SCR variability (unstable signal)',

        # EDA max (maximum signal in window)
        'eda_max_low': 'low maximum skin conductance',
        'eda_max_medium': 'medium maximum skin conductance',
        'eda_max_high': 'high maximum skin conductance (strong response)',

        # EDA peaks (number of SCR peaks - response frequency)
        'eda_peaks_low': 'low number of SCRs (1-3 SCR/min, relaxation)',
        'eda_peaks_medium': 'medium number of SCRs',
        'eda_peaks_high': 'high number of SCRs (high sympathetic activity)',

        # HR
        'hr_low': 'low heart rate',
        'hr_medium': 'normal heart rate',
        'hr_high': 'elevated heart rate',

        # TEMP
        'temp_low': 'low skin temperature',
        'temp_medium': 'normal skin temperature',
        'temp_high': 'elevated skin temperature',

        # HRV
        'hrv_sdnn_low': 'low heart rate variability (SDNN)',
        'hrv_sdnn_medium': 'medium heart rate variability (SDNN)',
        'hrv_sdnn_high': 'high heart rate variability (SDNN)',
        'hrv_rmssd_low': 'low heart rate variability (RMSSD)',
        'hrv_rmssd_medium': 'medium heart rate variability (RMSSD)',
        'hrv_rmssd_high': 'high heart rate variability (RMSSD)',
    }

    # Relation mapping
    relation_descriptions = {
        'equals': 'co-occurs with',
        'before': 'precedes',
        'meets': 'immediately precedes',
        'overlaps': 'overlaps with',
        'contains': 'contains',
        'starts': 'starts with',
        'is-finished-by': 'ends with'
    }

    # Parse pattern
    interpretation = pattern

    for state, desc in state_descriptions.items():
        if state in interpretation:
            interpretation = interpretation.replace(state, f"**{state}** ({desc})")

    for rel, desc in relation_descriptions.items():
        if rel in interpretation:
            interpretation = interpretation.replace(rel, f"*{desc}*")

    return interpretation


def interpret_rule(rule: str) -> str:
    """Interprets a rule in natural language."""

    # Split into antecedent and consequent
    if '=>' not in rule:
        return rule

    parts = rule.split('=>')
    antecedent = parts[0].strip()
    consequent = parts[1].strip()

    # Interpret both parts
    ant_interp = interpret_pattern(antecedent)
    cons_interp = interpret_pattern(consequent)

    return f"If {ant_interp}, then {cons_interp}"


def generate_methodology_section(summary: dict) -> str:
    """Generuje sekcję metodologii."""

    params = summary.get('parameters', {})
    filters = summary.get('filters', {})

    lines = []
    lines.append("## Methodology")
    lines.append("")
    lines.append("### ARMADA Algorithm")
    lines.append("")
    lines.append("ARMADA (Association Rule Mining for Anomaly Detection in Affective data) is an algorithm")
    lines.append("for discovering temporal patterns in affective data. It uses Allen's temporal relations")
    lines.append("(equals, before, meets, overlaps, contains, starts, is-finished-by) to describe")
    lines.append("dependencies between emotional and physiological states.")
    lines.append("")
    lines.append("### Experiment parameters")
    lines.append("")
    lines.append("| Parameter | Value | Description |")
    lines.append("|-----------|-------|-------------|")
    lines.append(f"| minsup | {params.get('minsup', 'N/A')} | Minimum percentage of participants with the pattern |")
    lines.append(f"| minconf | {params.get('minconf', 'N/A')} | Minimum rule confidence |")
    lines.append(f"| maxgap | {params.get('maxgap', 'N/A')}s | Maximum gap between states |")
    lines.append(f"| max_pattern_size | {params.get('max_pattern_size', 'N/A')} | Maximum number of states in a pattern |")
    lines.append("")
    lines.append("### Rule filters")
    lines.append("")

    if filters.get('filter_bvp_only', False):
        lines.append("- ✅ **Filtered BVP-only rules** - rules containing only HRV metrics (bvp_*) were removed")
    else:
        lines.append("- ❌ BVP-only filter disabled")

    if filters.get('filter_single_feature', False):
        lines.append("- ✅ **Filtered single-feature rules** - rules describing only one feature (e.g. arousal) were removed")
    else:
        lines.append("- ❌ Single-feature filter disabled")

    lines.append("")
    lines.append("### Variable discretization")
    lines.append("")
    lines.append("#### Arousal and Valence (SAM scale)")
    lines.append("")
    lines.append("According to the literature (Ahmad et al.), SAM values 1-9 were grouped into three levels:")
    lines.append("")
    lines.append("| Level | Range (0-1) | SAM (1-9) | Interpretation |")
    lines.append("|-------|-------------|-----------|----------------|")
    lines.append("| low | [0.00, 0.25] | 1-3 | Negative/low |")
    lines.append("| medium | (0.25, 0.75) | 4-6 | Neutral/moderate |")
    lines.append("| high | [0.75, 1.00] | 7-9 | Positive/high |")
    lines.append("")
    lines.append("#### EDA (skin conductance)")
    lines.append("")
    lines.append("5-step pipeline (Greco et al. 2016, Benedek & Kaernbach 2010, Braithwaite et al. 2013):")
    lines.append("")
    lines.append("1. **Low-pass filtering** 1 Hz (4th order Butterworth) — SPR recommendation")
    lines.append("2. **Tonic/Phasic decomposition** — 4s median filter (CDA)")
    lines.append("   - Tonic (SCL): slow-varying baseline level")
    lines.append("   - Phasic (SCR): fast-varying skin responses")
    lines.append("3. **Feature extraction in 5s windows**: SCL mean, SCR peaks/amplitude/AUC")
    lines.append("4. **Intra-individual normalization** Min-Max: `EDA_norm = (EDA - EDA_min) / (EDA_max - EDA_min)`")
    lines.append("   - Lykken & Venables (1971), Boucsein (2012)")
    lines.append("5. **Discretization thresholds**: low [0, 0.33], medium (0.33, 0.66], high (0.66, 1.00]")
    lines.append("")
    lines.append("#### Other physiological variables")
    lines.append("")
    lines.append("HR, TEMP, HRV: terciles cross-dataset (33% and 67% percentiles)")
    lines.append("")

    return "\n".join(lines)


def generate_datasets_section(summary: dict) -> str:
    """Generates the datasets section."""

    datasets = summary.get('datasets', {})

    lines = []
    lines.append("## Datasets")
    lines.append("")
    lines.append("| Dataset | Participants | Patterns | Rules | Unique patterns |")
    lines.append("|---------|--------------|----------|-------|-----------------|")

    for ds_name, ds_data in datasets.items():
        unique = ds_data.get('unique_patterns', 0)
        total = ds_data.get('total_patterns', 1)
        pct = (unique / total * 100) if total > 0 else 0

        lines.append(f"| **{ds_name}** | {ds_data.get('num_clients', 'N/A')} | "
                    f"{ds_data.get('total_patterns', 'N/A')} | "
                    f"{ds_data.get('total_rules', 'N/A')} | "
                    f"{unique} ({pct:.1f}%) |")

    lines.append("")
    lines.append("### Datasets description")
    lines.append("")
    lines.append("- **CASE**: Continuous Annotation of Self-reported Emotions - continuous annotations via joystick")
    lines.append("- **K-EmoCon**: K-Emotion Convention - Empatica E4 data with self-assessments every 5s")
    lines.append("- **CEAP**: Continuous Emotion Annotation Protocol - continuous annotations in 360VR")
    lines.append("- **EmoWorker_v2**: Emotion annotation protocol for worker environments")
    lines.append("")

    return "\n".join(lines)


def generate_similarity_section(summary: dict) -> str:
    """Generates the common patterns and rules section."""
    comparison = summary.get('comparison', {})
    lines = []
    lines.append("## Common patterns and rules")
    lines.append("")
    lines.append(f"### Patterns common to all datasets: **{comparison.get('common_all_patterns', 0)}**")
    lines.append(f"### Rules common to all datasets: **{comparison.get('common_all_rules', 0)}**")
    lines.append("")
    return "\n".join(lines)


def generate_common_patterns_section(patterns_df: pd.DataFrame) -> str:
    """Generates the common patterns detailed section."""

    lines = []
    lines.append("## Common patterns")
    lines.append("")

    if len(patterns_df) == 0:
        lines.append("*No patterns common to all datasets.*")
        lines.append("")
        return "\n".join(lines)

    lines.append(f"Found **{len(patterns_df)}** patterns common to all datasets.")
    lines.append("")
    lines.append("### Top 15 patterns (by average support)")
    lines.append("")
    lines.append("| # | Pattern | Avg. Support | CASE | K-emoCon | CEAP |")
    lines.append("|---|---------|--------------|------|----------|------|")

    for i, (_, row) in enumerate(patterns_df.iterrows(), 1):
        pattern = row.get('pattern', 'N/A')
        avg_sup = row.get('avg_support', 'N/A')
        case_sup = row.get('CASE_support', 'N/A')
        kemo_sup = row.get('K-emoCon_support', 'N/A')
        ceap_sup = row.get('CEAP_support', 'N/A')

        # Formatuj wartości
        if isinstance(avg_sup, float):
            avg_sup = f"{avg_sup:.3f}"
        if isinstance(case_sup, float):
            case_sup = f"{case_sup:.3f}"
        if isinstance(kemo_sup, float):
            kemo_sup = f"{kemo_sup:.3f}"
        if isinstance(ceap_sup, float):
            ceap_sup = f"{ceap_sup:.3f}"

        lines.append(f"| {i} | `{pattern}` | {avg_sup} | {case_sup} | {kemo_sup} | {ceap_sup} |")

    lines.append("")

    return "\n".join(lines)


def generate_common_rules_section(rules_df: pd.DataFrame) -> str:
    """Generates the common rules detailed section."""

    lines = []
    lines.append("## Common rules")
    lines.append("")

    if len(rules_df) == 0:
        lines.append("*No rules common to all datasets.*")
        lines.append("")
        return "\n".join(lines)

    lines.append(f"Found **{len(rules_df)}** rules common to all datasets.")
    lines.append("")
    lines.append("### All common rules")
    lines.append("")

    for i, (_, row) in enumerate(rules_df.iterrows(), 1):
        rule = row.get('rule', 'N/A')
        avg_conf = row.get('avg_confidence', 'N/A')
        avg_sup = row.get('avg_support', 'N/A')

        if isinstance(avg_conf, float):
            avg_conf = f"{avg_conf:.3f}"
        if isinstance(avg_sup, float):
            avg_sup = f"{avg_sup:.3f}"

        lines.append(f"#### Rule {i}")
        lines.append("")
        lines.append(f"```")
        lines.append(f"{rule}")
        lines.append(f"```")
        lines.append("")
        lines.append(f"- **Avg. confidence**: {avg_conf}")
        lines.append(f"- **Avg. support**: {avg_sup}")
        lines.append("")

        # Interpretation
        lines.append(f"**Interpretation**: {interpret_rule(rule)}")
        lines.append("")

    return "\n".join(lines)


def generate_conclusions_section(summary: dict, patterns_df: pd.DataFrame, rules_df: pd.DataFrame) -> str:
    """Generates the conclusions section."""

    comparison = summary.get('comparison', {})
    common_patterns = comparison.get('common_all_patterns', 0)
    common_rules = comparison.get('common_all_rules', 0)

    lines = []
    lines.append("## Conclusions")
    lines.append("")

    # Main conclusion
    if common_patterns > 0:
        lines.append("### ✅ Patterns are universal")
        lines.append("")
        lines.append(f"**YES** - found **{common_patterns}** patterns common to all three ")
        lines.append("datasets. This means that certain dependencies between emotional ")
        lines.append("states and physiological signals are universal and independent of:")
        lines.append("")
        lines.append("- Research protocol (continuous annotations vs self-assessments)")
        lines.append("- Subject population")
        lines.append("- Measurement equipment")
        lines.append("")
    else:
        lines.append("### ❌ No universal patterns")
        lines.append("")
        lines.append("Found no patterns common to all datasets. ")
        lines.append("This may be due to:")
        lines.append("")
        lines.append("- Differences in research protocols")
        lines.append("- Differences in populations")
        lines.append("- Overly restrictive parameters (minsup, minconf)")
        lines.append("")

    # Rules summary
    if common_rules > 0:
        lines.append("### Predictive rules")
        lines.append("")
        lines.append(f"Found **{common_rules}** predictive rules that can be used for:")
        lines.append("")
        lines.append("1. Automatic recognition of emotional states")
        lines.append("2. Prediction of emotional changes based on physiological signals")
        lines.append("3. Validation of affective computing systems")
        lines.append("")


    return "\n".join(lines)


def generate_readme():
    """Main function generating README."""

    print("Generating README.md...")

    # Load data
    summary, patterns_df, rules_df = load_experiment_data()

    # Generate sections
    sections = []

    # Header
    sections.append("# ARMADA Experiment Results")
    sections.append("")
    sections.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*")
    sections.append("")
    sections.append("---")
    sections.append("")
    sections.append("## Experiment Goal")
    sections.append("")
    sections.append("The goal of the experiment was to check whether the temporal patterns discovered by the")
    sections.append("ARMADA algorithm are **universal** - i.e., whether they re-occur in different affective")
    sections.append("datasets coming from different research protocols and populations.")
    sections.append("")

    # Add sections
    sections.append(generate_methodology_section(summary))
    sections.append(generate_datasets_section(summary))
    sections.append(generate_similarity_section(summary))
    sections.append(generate_common_patterns_section(patterns_df))
    sections.append(generate_common_rules_section(rules_df))
    sections.append(generate_conclusions_section(summary, patterns_df, rules_df))

    # Footer
    sections.append("---")
    sections.append("")
    sections.append("## Output files")
    sections.append("")
    sections.append("- `experiment_summary.json` - JSON format summary")
    sections.append("- `common_patterns_details.csv` - common patterns details")
    sections.append("- `common_rules_details.csv` - common rules details")
    sections.append("- `patterns_comparison.png` - pattern comparison visualization")
    sections.append("- `comparison_report.md` - detailed comparison report")
    sections.append("")

    # Save README
    readme_content = "\n".join(sections)

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(readme_content)

    print(f"Saved: {OUTPUT_FILE}")
    print(f"Size: {len(readme_content)} characters")

    return OUTPUT_FILE


if __name__ == "__main__":
    try:
        output_file = generate_readme()
        print(f"\n✅ README generated successfully: {output_file}")
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("Run first: python compare_datasets.py")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        raise

