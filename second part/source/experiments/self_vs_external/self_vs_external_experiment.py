#!/usr/bin/env python3
"""
Experiment: Self-Annotations vs External Annotations Comparison

Compares ARMADA rules found using two annotation sources on K-emoCon:
- K-emoCon (self-annotations): participants' own arousal/valence ratings
- K-emoCon (external annotations): aggregated external observer ratings

Both use the same physiological data (Empatica E4), only annotation
source differs. This isolates the effect of annotation perspective
on discovered temporal patterns.

Analysis:
1. Runs ARMADA on each annotation variant separately.
2. Compares rule sets: shared, self-only, external-only.
3. Jaccard similarity of rule sets.
4. Per-rule confidence/support comparison for shared rules.
5. Generates detailed Markdown report.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Set, Tuple

SCRIPT_DIR = Path(__file__).parent
EXPERIMENTS_DIR = SCRIPT_DIR.parent
PROJECT_DIR = EXPERIMENTS_DIR.parent.parent

sys.path.insert(0, str(PROJECT_DIR / "source" / "processing" / "armada"))
sys.path.insert(0, str(EXPERIMENTS_DIR))

# ============================================================================
# EXPERIMENT PARAMETERS (same as external_annotations_experiment)
# ============================================================================
MINSUP = 0.4
MINCONF = 0.4
MAXGAP = 20
MAX_PATTERN_SIZE = 2

# ============================================================================
# RULE FILTERS
# ============================================================================
FILTER_BVP_ONLY = True
FILTER_EDA_ONLY = True
FILTER_PHYSIO_CROSS = True
FILTER_SINGLE_FEATURE = True

try:
    from experiment_utils import (
        run_armada_on_df,
        extract_rule_signatures,
        filter_rules,
        jaccard_similarity
    )
except ImportError as e:
    print(f"Error importing experiment_utils: {e}")
    sys.exit(1)

try:
    from armada_algorithm import ARMADA
except ImportError as e:
    print(f"Error importing armada_algorithm: {e}")
    sys.exit(1)


def run_armada_on_dataset(data_file: Path) -> Tuple[ARMADA, List, List]:
    """Runs ARMADA on a single dataset file."""
    print(f"  Loading CSV: {data_file}")
    df = pd.read_csv(data_file)
    return run_armada_on_df(df, MINSUP, MINCONF, MAXGAP, MAX_PATTERN_SIZE)


def get_rule_details(rules: List) -> Dict[str, dict]:
    """Builds a dict rule_signature -> {confidence, support, count}."""
    details = {}
    for r in rules:
        sig = f"{r.antecedent.get_relation_description()} => {r.consequent.get_relation_description()}"
        details[sig] = {
            'confidence': r.confidence,
            'support': r.support,
            'count': r.consequent.support_count
        }
    return details


def generate_report(
    self_armada: ARMADA, self_rules: List,
    ext_armada: ARMADA, ext_rules: List,
    self_filtered: Set[str], ext_filtered: Set[str],
    output_dir: Path
) -> None:
    """Generates a detailed comparison report."""

    shared = self_filtered & ext_filtered
    self_only = self_filtered - ext_filtered
    ext_only = ext_filtered - self_filtered
    jaccard = jaccard_similarity(self_filtered, ext_filtered)

    self_details = get_rule_details(self_rules)
    ext_details = get_rule_details(ext_rules)

    lines = []
    lines.append("# Self vs External Annotations — Rule Comparison")
    lines.append("")
    lines.append("Comparison of ARMADA rules discovered using the **same physiological data**")
    lines.append("(K-emoCon, Empatica E4) but different annotation sources:")
    lines.append("- **Self**: participants' own arousal/valence ratings (1–5 scale, every 5s)")
    lines.append("- **External**: aggregated external observer ratings (1–5 scale, every 5s)")
    lines.append("")

    # Params
    lines.append("## Parameters")
    lines.append("")
    lines.append(f"| Parameter | Value |")
    lines.append(f"|-----------|-------|")
    lines.append(f"| minsup | {MINSUP} ({MINSUP*100:.0f}%) |")
    lines.append(f"| minconf | {MINCONF} ({MINCONF*100:.0f}%) |")
    lines.append(f"| maxgap | {MAXGAP}s |")
    lines.append(f"| max_pattern_size | {MAX_PATTERN_SIZE} |")
    lines.append(f"| FILTER_BVP_ONLY | {FILTER_BVP_ONLY} |")
    lines.append(f"| FILTER_EDA_ONLY | {FILTER_EDA_ONLY} |")
    lines.append(f"| FILTER_PHYSIO_CROSS | {FILTER_PHYSIO_CROSS} |")
    lines.append(f"| FILTER_SINGLE_FEATURE | {FILTER_SINGLE_FEATURE} |")
    lines.append("")

    # Overview
    lines.append("## Overview")
    lines.append("")
    lines.append("| Metric | Self-annotations | External annotations |")
    lines.append("|--------|-----------------|---------------------|")
    lines.append(f"| Participants | {self_armada.num_clients} | {ext_armada.num_clients} |")
    lines.append(f"| Raw rules | {len(self_rules)} | {len(ext_rules)} |")
    lines.append(f"| Filtered rules | {len(self_filtered)} | {len(ext_filtered)} |")
    lines.append("")
    lines.append(f"**Jaccard similarity**: {jaccard:.3f}")
    lines.append("")
    lines.append(f"| Set | Count |")
    lines.append(f"|-----|-------|")
    lines.append(f"| Shared (both) | **{len(shared)}** |")
    lines.append(f"| Self-only | {len(self_only)} |")
    lines.append(f"| External-only | {len(ext_only)} |")
    lines.append(f"| Union | {len(self_filtered | ext_filtered)} |")
    lines.append("")

    # Shared rules — detailed comparison
    if shared:
        lines.append("## Shared Rules (present in both annotation variants)")
        lines.append("")
        lines.append(f"Found **{len(shared)}** rules common to both self and external annotations.")
        lines.append("")
        lines.append("| Rule | Self Conf | Ext Conf | Self Sup | Ext Sup |")
        lines.append("|------|----------|---------|---------|---------|")

        shared_rows = []
        for sig in sorted(shared):
            s = self_details.get(sig, {})
            e = ext_details.get(sig, {})
            s_conf = s.get('confidence', 0)
            e_conf = e.get('confidence', 0)
            s_sup = s.get('support', 0)
            e_sup = e.get('support', 0)
            delta = e_conf - s_conf
            shared_rows.append((sig, s_conf, e_conf, delta, s_sup, e_sup))  # delta kept for sorting

        # Sort by absolute delta (most different first)
        shared_rows.sort(key=lambda x: -abs(x[3]))

        for sig, s_conf, e_conf, delta, s_sup, e_sup in shared_rows:
            lines.append(
                f"| `{sig}` | {s_conf:.3f} | {e_conf:.3f} | "
                f"{s_sup:.3f} | {e_sup:.3f} |"
            )
        lines.append("")

        # Summary of confidence shifts
        deltas = [row[3] for row in shared_rows]
        avg_delta = np.mean(deltas)
        higher_ext = sum(1 for d in deltas if d > 0.05)
        higher_self = sum(1 for d in deltas if d < -0.05)
        similar = len(deltas) - higher_ext - higher_self

        lines.append("### Confidence Shift Summary")
        lines.append("")
        lines.append(f"- Mean Δ confidence (ext − self): **{avg_delta:+.3f}**")
        lines.append(f"- Rules with higher confidence in external: {higher_ext}")
        lines.append(f"- Rules with higher confidence in self: {higher_self}")
        lines.append(f"- Rules with similar confidence (|Δ| ≤ 0.05): {similar}")
        lines.append("")

    # Self-only rules
    if self_only:
        lines.append("## Self-Only Rules")
        lines.append("")
        lines.append(f"**{len(self_only)}** rules found only with self-annotations:")
        lines.append("")
        for sig in sorted(self_only):
            s = self_details.get(sig, {})
            lines.append(f"- `{sig}` (conf={s.get('confidence', 0):.3f}, sup={s.get('support', 0):.3f})")
        lines.append("")

    # External-only rules
    if ext_only:
        lines.append("## External-Only Rules")
        lines.append("")
        lines.append(f"**{len(ext_only)}** rules found only with external annotations:")
        lines.append("")
        for sig in sorted(ext_only):
            e = ext_details.get(sig, {})
            lines.append(f"- `{sig}` (conf={e.get('confidence', 0):.3f}, sup={e.get('support', 0):.3f})")
        lines.append("")

    # Save
    report_text = "\n".join(lines)
    report_path = output_dir / "self_vs_external_report.md"
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"Report saved: {report_path}")

    # Save shared rules CSV
    if shared:
        rows = []
        for sig in sorted(shared):
            s = self_details.get(sig, {})
            e = ext_details.get(sig, {})
            rows.append({
                'rule': sig,
                'self_confidence': round(s.get('confidence', 0), 4),
                'ext_confidence': round(e.get('confidence', 0), 4),
                'delta_confidence': round(e.get('confidence', 0) - s.get('confidence', 0), 4),
                'self_support': round(s.get('support', 0), 4),
                'ext_support': round(e.get('support', 0), 4),
                'self_count': s.get('count', 0),
                'ext_count': e.get('count', 0),
            })
        pd.DataFrame(rows).to_csv(
            output_dir / "shared_rules_comparison.csv", index=False
        )
        print(f"Shared rules CSV saved: {output_dir / 'shared_rules_comparison.csv'}")


def main():
    DATA_DIR = PROJECT_DIR / "data" / "armada_ready"
    OUTPUT_DIR = SCRIPT_DIR / "results"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("EXPERIMENT: SELF vs EXTERNAL ANNOTATIONS COMPARISON")
    print("=" * 80)
    print(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Results: {OUTPUT_DIR}")
    print()

    # Dataset paths
    self_file = DATA_DIR / "armada_k_emocon.csv"
    ext_file = DATA_DIR / "armada_k_emocon_ext.csv"

    for name, path in [("Self", self_file), ("External", ext_file)]:
        if not path.exists():
            print(f"ERROR: {name} file not found: {path}")
            sys.exit(1)

    # Run ARMADA on self-annotations
    print(f"\n{'='*60}")
    print("Processing: K-emoCon SELF-annotations")
    print(f"{'='*60}")
    self_armada, self_patterns, self_rules = run_armada_on_dataset(self_file)
    self_raw_sigs = extract_rule_signatures(self_rules)
    self_filtered = filter_rules(
        self_raw_sigs, FILTER_BVP_ONLY, FILTER_EDA_ONLY,
        FILTER_PHYSIO_CROSS, FILTER_SINGLE_FEATURE
    )
    print(f"  Raw rules: {len(self_rules)}")
    print(f"  Filtered: {len(self_filtered)}")

    # Run ARMADA on external annotations
    print(f"\n{'='*60}")
    print("Processing: K-emoCon EXTERNAL annotations")
    print(f"{'='*60}")
    ext_armada, ext_patterns, ext_rules = run_armada_on_dataset(ext_file)
    ext_raw_sigs = extract_rule_signatures(ext_rules)
    ext_filtered = filter_rules(
        ext_raw_sigs, FILTER_BVP_ONLY, FILTER_EDA_ONLY,
        FILTER_PHYSIO_CROSS, FILTER_SINGLE_FEATURE
    )
    print(f"  Raw rules: {len(ext_rules)}")
    print(f"  Filtered: {len(ext_filtered)}")

    # Compare
    print(f"\n{'='*80}")
    print("COMPARISON RESULTS")
    print(f"{'='*80}")

    shared = self_filtered & ext_filtered
    self_only = self_filtered - ext_filtered
    ext_only = ext_filtered - self_filtered
    jaccard = jaccard_similarity(self_filtered, ext_filtered)

    print(f"  Self filtered rules:     {len(self_filtered)}")
    print(f"  External filtered rules: {len(ext_filtered)}")
    print(f"  Shared:                  {len(shared)}")
    print(f"  Self-only:               {len(self_only)}")
    print(f"  External-only:           {len(ext_only)}")
    print(f"  Jaccard similarity:      {jaccard:.3f}")

    # Generate report
    generate_report(
        self_armada, self_rules,
        ext_armada, ext_rules,
        self_filtered, ext_filtered,
        OUTPUT_DIR
    )

    if shared:
        self_details = get_rule_details(self_rules)
        ext_details = get_rule_details(ext_rules)
        print(f"\nTOP SHARED RULES (by avg confidence):")
        for sig in sorted(shared,
                          key=lambda s: -(self_details.get(s, {}).get('confidence', 0)
                                          + ext_details.get(s, {}).get('confidence', 0)) / 2)[:10]:
            s_c = self_details.get(sig, {}).get('confidence', 0)
            e_c = ext_details.get(sig, {}).get('confidence', 0)
            print(f"  {sig}")
            print(f"    self_conf={s_c:.3f}, ext_conf={e_c:.3f}, Δ={e_c - s_c:+.3f}")

    print(f"\n{'='*80}")
    print("EXPERIMENT FINISHED")
    print(f"Results saved in: {OUTPUT_DIR}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
