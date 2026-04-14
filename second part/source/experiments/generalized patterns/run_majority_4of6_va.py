#!/usr/bin/env python3
"""
Experiment: rules present in at least 4 of 6 datasets (valence-arousal setup).

Parameters:
- minsup = 0.3
- minconf = 0.5
- maxgap = 5
- max_pattern_size = 4
"""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = SCRIPT_DIR.parent
PROJECT_DIR = EXPERIMENTS_DIR.parent.parent

sys.path.insert(0, str(PROJECT_DIR / "source" / "processing" / "armada"))
sys.path.insert(0, str(EXPERIMENTS_DIR))

from experiment_utils import run_armada_on_df, filter_rules  # noqa: E402


MIN_SUP = 0.3
MIN_CONF = 0.5
MAX_GAP = 5
MAX_PATTERN_SIZE = 4

DATA_DIR = PROJECT_DIR / "data" / "armada_ready"
DATASETS = {
    "CASE": DATA_DIR / "armada_case.csv",
    "K-emoCon": DATA_DIR / "armada_k_emocon.csv",
    "CEAP": DATA_DIR / "armada_ceap.csv",
    "EmoWorker": DATA_DIR / "armada_emoworker_v2.csv",
    "K-emoCon (ext)": DATA_DIR / "armada_k_emocon_ext.csv",
    "EMBOA": DATA_DIR / "armada_emboa.csv",
}

OUT_DIR = SCRIPT_DIR / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "majority_4of6_va_rules.csv"


def signature_from_rule(rule) -> str:
    return (
        f"{rule.antecedent.get_relation_description()}"
        f" => {rule.consequent.get_relation_description()}"
    )


def run_armada_silent(df: pd.DataFrame):
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink):
        _, _, rules = run_armada_on_df(
            df,
            minsup=MIN_SUP,
            minconf=MIN_CONF,
            maxgap=MAX_GAP,
            max_pattern_size=MAX_PATTERN_SIZE,
        )
    return rules


def main() -> None:
    dataset_rule_signatures: Dict[str, set[str]] = {}
    dataset_rule_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}

    for dataset_name, csv_path in DATASETS.items():
        df = pd.read_csv(csv_path)
        rules = run_armada_silent(df)

        metrics_for_dataset: Dict[str, Dict[str, float]] = {}
        raw_signatures = set()

        for rule in rules:
            sig = signature_from_rule(rule)
            raw_signatures.add(sig)
            metrics_for_dataset[sig] = {
                "confidence": float(rule.confidence),
                "lift": float(rule.lift),
                "support": float(rule.support),
            }

        filtered_signatures = filter_rules(
            raw_signatures,
            filter_bvp_only=True,
            filter_eda_only=True,
            filter_physio_cross=True,
            filter_single_feature=True,
        )

        dataset_rule_signatures[dataset_name] = filtered_signatures
        dataset_rule_metrics[dataset_name] = {
            sig: metrics_for_dataset[sig] for sig in filtered_signatures if sig in metrics_for_dataset
        }

    all_signatures = set().union(*dataset_rule_signatures.values())
    selected = []

    for sig in sorted(all_signatures):
        present_in: List[str] = [
            name for name, sigs in dataset_rule_signatures.items() if sig in sigs
        ]
        if len(present_in) < 4:
            continue

        confs = [dataset_rule_metrics[name][sig]["confidence"] for name in present_in]
        lifts = [dataset_rule_metrics[name][sig]["lift"] for name in present_in]
        sups = [dataset_rule_metrics[name][sig]["support"] for name in present_in]

        row = {
            "rule": sig,
            "present_in_count": len(present_in),
            "present_in": ", ".join(present_in),
            "avg_confidence": sum(confs) / len(confs),
            "avg_lift": sum(lifts) / len(lifts),
            "avg_support": sum(sups) / len(sups),
        }

        for name in DATASETS:
            m = dataset_rule_metrics[name].get(sig)
            row[f"{name}_confidence"] = m["confidence"] if m else None
            row[f"{name}_lift"] = m["lift"] if m else None
            row[f"{name}_support"] = m["support"] if m else None

        selected.append(row)

    out_df = pd.DataFrame(selected)
    if not out_df.empty:
        out_df = out_df.sort_values(
            by=["present_in_count", "avg_confidence", "avg_lift"],
            ascending=[False, False, False],
        )
    out_df.to_csv(OUT_CSV, index=False)

    print(f"Saved: {OUT_CSV}")
    print(f"Rules present in >=4/6 datasets: {len(out_df)}")
    if not out_df.empty:
        print("Breakdown by dataset support count:")
        print(out_df["present_in_count"].value_counts().sort_index(ascending=False))


if __name__ == "__main__":
    main()
