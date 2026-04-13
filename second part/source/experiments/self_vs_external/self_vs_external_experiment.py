#!/usr/bin/env python3
"""
Experiment: Self-Annotations vs External Annotations Comparison

Compares ARMADA rules found using two annotation paradigms:

SELF-ANNOTATED datasets (participants rate their own emotions):
  - CASE, K-emoCon (self), CEAP-360VR, EmoWorker_v2

EXTERNAL-ANNOTATED datasets (observers rate participants' emotions):
  - K-emoCon (external), EMBOA

Analysis:
1. Runs ARMADA on each dataset independently.
2. Computes per-dataset rule sets after semantic filtering.
3. Compares union of self-rules vs union of external-rules.
4. Special focus on K-emoCon (same physio data, both annotation types).
5. Generates detailed Markdown report.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Set, Tuple
from datetime import datetime

SCRIPT_DIR = Path(__file__).parent
EXPERIMENTS_DIR = SCRIPT_DIR.parent
PROJECT_DIR = EXPERIMENTS_DIR.parent.parent

sys.path.insert(0, str(PROJECT_DIR / "source" / "processing" / "armada"))
sys.path.insert(0, str(EXPERIMENTS_DIR))

MINSUP = 0.5
MINCONF = 0.5
MAXGAP = 5
MAX_PATTERN_SIZE = 2

FILTER_BVP_ONLY = True
FILTER_EDA_ONLY = True
FILTER_PHYSIO_CROSS = True
FILTER_SINGLE_FEATURE = True

DATA_DIR = PROJECT_DIR / "data" / "armada_ready"

SELF_DATASETS = {
    "CASE":         DATA_DIR / "armada_case.csv",
    "K-emoCon":     DATA_DIR / "armada_k_emocon.csv",
    "CEAP":         DATA_DIR / "armada_ceap.csv",
    "EmoWorker_v2": DATA_DIR / "armada_emoworker_v2.csv",
}

EXTERNAL_DATASETS = {
    "K-emoCon (ext)": DATA_DIR / "armada_k_emocon_ext.csv",
    "EMBOA":          DATA_DIR / "armada_emboa.csv",
}

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
    """Builds a dict rule_signature -> {confidence, lift, support, count}."""
    details = {}
    for r in rules:
        sig = f"{r.antecedent.get_relation_description()} => {r.consequent.get_relation_description()}"
        details[sig] = {
            'confidence': r.confidence,
            'lift': r.lift,
            'support': r.support,
            'count': r.consequent.support_count
        }
    return details


def process_dataset_group(datasets: dict) -> dict:
    """
    Runs ARMADA on each dataset in the group.
    Returns dict: {name: {armada, rules, raw_sigs, filtered_sigs, details, n_participants}}
    """
    results = {}
    for name, path in datasets.items():
        if not path.exists():
            print(f"  WARNING: {name} file not found: {path}, skipping")
            continue

        print(f"\n{'='*60}")
        print(f"Processing: {name}")
        print(f"{'='*60}")

        armada, patterns, rules = run_armada_on_dataset(path)
        raw_sigs = extract_rule_signatures(rules)
        filtered = filter_rules(
            raw_sigs, FILTER_BVP_ONLY, FILTER_EDA_ONLY,
            FILTER_PHYSIO_CROSS, FILTER_SINGLE_FEATURE
        )
        details = get_rule_details(rules)

        results[name] = {
            'armada': armada,
            'rules': rules,
            'raw_sigs': raw_sigs,
            'filtered': filtered,
            'details': details,
            'n_participants': armada.num_clients,
            'n_raw': len(rules),
            'n_filtered': len(filtered),
        }

        print(f"  Participants: {armada.num_clients}")
        print(f"  Raw rules: {len(rules)}")
        print(f"  Filtered: {len(filtered)}")

    return results


def generate_report(self_results: dict, ext_results: dict, output_dir: Path) -> None:

    all_self_rules = set()
    for r in self_results.values():
        all_self_rules |= r['filtered']

    all_ext_rules = set()
    for r in ext_results.values():
        all_ext_rules |= r['filtered']

    shared_all = all_self_rules & all_ext_rules
    self_only_all = all_self_rules - all_ext_rules
    ext_only_all = all_ext_rules - all_self_rules
    jaccard_all = jaccard_similarity(all_self_rules, all_ext_rules)

    lines = []
    lines.append("# Self vs External Annotations — Rule Comparison")
    lines.append("")
    lines.append("Comparison of ARMADA rules discovered across **self-annotated** and")
    lines.append("**externally-annotated** physiological datasets.")
    lines.append("")
    lines.append("**Self-annotated datasets**: " + ", ".join(self_results.keys()))
    lines.append("")
    lines.append("**External-annotated datasets**: " + ", ".join(ext_results.keys()))
    lines.append("")

    lines.append("## Parameters")
    lines.append("")
    lines.append(f"| Parameter | Value |")
    lines.append(f"|-----------|-------|")
    lines.append(f"| minsup | {MINSUP} ({MINSUP*100:.0f}%) |")
    lines.append(f"| minconf | {MINCONF} ({MINCONF*100:.0f}%) |")
    lines.append(f"| maxgap | {MAXGAP}s |")
    lines.append(f"| max_pattern_size | {MAX_PATTERN_SIZE} |")
    lines.append("")

    lines.append("## Per-Dataset Statistics")
    lines.append("")
    lines.append("### Self-Annotated Datasets")
    lines.append("")
    lines.append("| Dataset | N | Raw Rules | Filtered Rules |")
    lines.append("|---------|---|-----------|----------------|")
    for name, r in self_results.items():
        lines.append(f"| {name} | {r['n_participants']} | {r['n_raw']} | {r['n_filtered']} |")
    lines.append("")

    lines.append("### External-Annotated Datasets")
    lines.append("")
    lines.append("| Dataset | N | Raw Rules | Filtered Rules |")
    lines.append("|---------|---|-----------|----------------|")
    for name, r in ext_results.items():
        lines.append(f"| {name} | {r['n_participants']} | {r['n_raw']} | {r['n_filtered']} |")
    lines.append("")

    lines.append("## Overall Comparison (Union of All Rules)")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Total unique self rules | {len(all_self_rules)} |")
    lines.append(f"| Total unique external rules | {len(all_ext_rules)} |")
    lines.append(f"| Shared rules | **{len(shared_all)}** |")
    lines.append(f"| Self-only rules | {len(self_only_all)} |")
    lines.append(f"| External-only rules | {len(ext_only_all)} |")
    lines.append(f"| Jaccard similarity | {jaccard_all:.3f} |")
    lines.append("")

    if "K-emoCon" in self_results and "K-emoCon (ext)" in ext_results:
        kemo_self = self_results["K-emoCon"]['filtered']
        kemo_ext = ext_results["K-emoCon (ext)"]['filtered']
        kemo_shared = kemo_self & kemo_ext
        kemo_jaccard = jaccard_similarity(kemo_self, kemo_ext)

        kemo_self_details = self_results["K-emoCon"]['details']
        kemo_ext_details = ext_results["K-emoCon (ext)"]['details']

        lines.append("## K-emoCon Controlled Comparison")
        lines.append("")
        lines.append("Same physiological data (Empatica E4), different annotation source.")
        lines.append("")
        lines.append("| Metric | Self | External |")
        lines.append("|--------|------|----------|")
        lines.append(f"| Filtered rules | {len(kemo_self)} | {len(kemo_ext)} |")
        lines.append(f"| Shared | \\multicolumn{{2}}{{c|}}{{{len(kemo_shared)}}} |")
        lines.append(f"| Self-only | {len(kemo_self - kemo_ext)} | — |")
        lines.append(f"| External-only | — | {len(kemo_ext - kemo_self)} |")
        lines.append(f"| Jaccard similarity | \\multicolumn{{2}}{{c|}}{{{kemo_jaccard:.3f}}} |")
        lines.append("")

        if kemo_shared:
            lines.append("### K-emoCon Shared Rules")
            lines.append("")
            lines.append("| Rule | Self Conf | Self Lift | Ext Conf | Ext Lift | Δ Conf |")
            lines.append("|------|----------|-----------|---------|----------|---|")

            shared_rows = []
            for sig in kemo_shared:
                s_conf = kemo_self_details.get(sig, {}).get('confidence', 0)
                s_lift = kemo_self_details.get(sig, {}).get('lift', 0)
                e_conf = kemo_ext_details.get(sig, {}).get('confidence', 0)
                e_lift = kemo_ext_details.get(sig, {}).get('lift', 0)
                delta = e_conf - s_conf
                shared_rows.append((sig, s_conf, s_lift, e_conf, e_lift, delta))

            shared_rows.sort(key=lambda x: -abs(x[5]))
            for sig, s_conf, s_lift, e_conf, e_lift, delta in shared_rows:
                lines.append(f"| `{sig}` | {s_conf:.3f} | {s_lift:.3f} | {e_conf:.3f} | {e_lift:.3f} | {delta:+.3f} |")
            lines.append("")

            deltas = [row[5] for row in shared_rows]
            avg_delta = np.mean(deltas)
            higher_ext = sum(1 for d in deltas if d > 0.05)
            higher_self = sum(1 for d in deltas if d < -0.05)
            similar = len(deltas) - higher_ext - higher_self

            lines.append(f"- Mean Δ confidence (ext − self): **{avg_delta:+.3f}**")
            lines.append(f"- Rules with higher confidence in external: {higher_ext}")
            lines.append(f"- Rules with higher confidence in self: {higher_self}")
            lines.append(f"- Rules with similar confidence (|Δ| ≤ 0.05): {similar}")
            lines.append("")

    lines.append("## Pairwise Jaccard Similarity Matrix")
    lines.append("")
    all_datasets = {}
    for name, r in self_results.items():
        all_datasets[f"{name} (self)"] = r['filtered']
    for name, r in ext_results.items():
        all_datasets[name] = r['filtered']

    names = list(all_datasets.keys())
    lines.append("| | " + " | ".join(names) + " |")
    lines.append("|" + "---|" * (len(names) + 1))
    for n1 in names:
        row = f"| **{n1}** |"
        for n2 in names:
            if n1 == n2:
                row += " 1.000 |"
            else:
                j = jaccard_similarity(all_datasets[n1], all_datasets[n2])
                row += f" {j:.3f} |"
        lines.append(row)
    lines.append("")

    if shared_all:
        lines.append("## Shared Rules (present in both self and external unions)")
        lines.append("")
        lines.append(f"Found **{len(shared_all)}** rules in common.")
        lines.append("")
        lines.append("| Rule | Present in (self) | Present in (ext) |")
        lines.append("|------|-------------------|------------------|")
        for sig in sorted(shared_all):
            self_present = [n for n, r in self_results.items() if sig in r['filtered']]
            ext_present = [n for n, r in ext_results.items() if sig in r['filtered']]
            lines.append(f"| `{sig}` | {', '.join(self_present)} | {', '.join(ext_present)} |")
        lines.append("")

    if self_only_all:
        lines.append("## Self-Only Rules")
        lines.append("")
        lines.append(f"**{len(self_only_all)}** rules found only in self-annotated datasets.")
        lines.append("")
        for sig in sorted(self_only_all):
            present_in = [n for n, r in self_results.items() if sig in r['filtered']]
            lines.append(f"- `{sig}` (in: {', '.join(present_in)})")
        lines.append("")

    if ext_only_all:
        lines.append("## External-Only Rules")
        lines.append("")
        lines.append(f"**{len(ext_only_all)}** rules found only in externally-annotated datasets.")
        lines.append("")
        for sig in sorted(ext_only_all):
            present_in = [n for n, r in ext_results.items() if sig in r['filtered']]
            lines.append(f"- `{sig}` (in: {', '.join(present_in)})")
        lines.append("")

    report_text = "\n".join(lines)
    report_path = output_dir / "self_vs_external_report.md"
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"Report saved: {report_path}")


def main():
    OUTPUT_DIR = SCRIPT_DIR / "results"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("EXPERIMENT: SELF vs EXTERNAL ANNOTATIONS COMPARISON")
    print("=" * 80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Results: {OUTPUT_DIR}")
    print()

    print("Self-annotated datasets:")
    for name, path in SELF_DATASETS.items():
        status = "OK" if path.exists() else "MISSING"
        print(f"  [{status}] {name}: {path.name}")

    print("\nExternal-annotated datasets:")
    for name, path in EXTERNAL_DATASETS.items():
        status = "OK" if path.exists() else "MISSING"
        print(f"  [{status}] {name}: {path.name}")

    print("\n" + "=" * 80)
    print("PROCESSING SELF-ANNOTATED DATASETS")
    print("=" * 80)
    self_results = process_dataset_group(SELF_DATASETS)

    print("\n" + "=" * 80)
    print("PROCESSING EXTERNAL-ANNOTATED DATASETS")
    print("=" * 80)
    ext_results = process_dataset_group(EXTERNAL_DATASETS)

    all_self = set()
    for r in self_results.values():
        all_self |= r['filtered']

    all_ext = set()
    for r in ext_results.values():
        all_ext |= r['filtered']

    shared = all_self & all_ext
    jaccard = jaccard_similarity(all_self, all_ext)

    print(f"\n{'='*80}")
    print("OVERALL COMPARISON")
    print(f"{'='*80}")
    print(f"  Self-annotated datasets:     {len(self_results)} ({', '.join(self_results.keys())})")
    print(f"  External-annotated datasets: {len(ext_results)} ({', '.join(ext_results.keys())})")
    print(f"  Total unique self rules:     {len(all_self)}")
    print(f"  Total unique ext rules:      {len(all_ext)}")
    print(f"  Shared:                      {len(shared)}")
    print(f"  Self-only:                   {len(all_self - all_ext)}")
    print(f"  External-only:               {len(all_ext - all_self)}")
    print(f"  Jaccard similarity:          {jaccard:.3f}")

    generate_report(self_results, ext_results, OUTPUT_DIR)

    if shared:
        print(f"\nTOP SHARED RULES (alphabetical, max 15):")
        for sig in sorted(shared)[:15]:
            self_present = [n for n, r in self_results.items() if sig in r['filtered']]
            ext_present = [n for n, r in ext_results.items() if sig in r['filtered']]
            print(f"  {sig}")
            print(f"    self: {', '.join(self_present)} | ext: {', '.join(ext_present)}")

    print(f"\n{'='*80}")
    print("EXPERIMENT FINISHED")
    print(f"Results saved in: {OUTPUT_DIR}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
