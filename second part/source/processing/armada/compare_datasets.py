#!/usr/bin/env python3
"""
Porównanie wzorców ARMADA pomiędzy zbiorami danych.

Ten skrypt:
1. Uruchamia ARMADA na każdym zbiorze danych osobno (CASE, K-emoCon, CEAP)
2. Porównuje odkryte wzorce i reguły
3. Znajduje wzorce wspólne dla wszystkich zbiorów
4. Generuje raport porównawczy
"""

import sys
from pathlib import Path
import pandas as pd
import json
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import matplotlib.pyplot as plt

# Dodaj ścieżkę do modułów
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

from armada_algorithm import ARMADA


def run_armada_on_dataset(
    data_file: Path,
    minsup: float = 0.4,
    minconf: float = 0.5,
    maxgap: float = 30,
    max_pattern_size: int = 3
) -> Tuple[ARMADA, List, List]:
    """Uruchamia ARMADA na pojedynczym zbiorze danych."""
    armada = ARMADA(
        minsup=minsup,
        minconf=minconf,
        maxgap=maxgap,
        max_pattern_size=max_pattern_size
    )
    patterns, rules = armada.run(filepath=data_file)
    return armada, patterns, rules


def extract_pattern_signatures(patterns: List) -> Set[str]:
    """
    Ekstrahuje 'sygnatury' wzorców do porównania.
    Sygnatura to opis relacji bez konkretnych czasów.
    """
    signatures = set()
    for p in patterns:
        sig = p.get_relation_description()
        signatures.add(sig)
    return signatures


def extract_rule_signatures(rules: List) -> Set[str]:
    """Ekstrahuje sygnatury reguł do porównania."""
    signatures = set()
    for r in rules:
        sig = f"{r.antecedent.get_relation_description()} => {r.consequent.get_relation_description()}"
        signatures.add(sig)
    return signatures


def compare_pattern_sets(
    patterns_dict: Dict[str, Set[str]]
) -> Dict[str, Set[str]]:
    """
    Porównuje zbiory wzorców.

    Returns:
        Słownik z:
        - 'common_all': wzorce wspólne dla wszystkich
        - 'common_pairs': wzorce wspólne dla par zbiorów
        - 'unique_X': wzorce unikalne dla zbioru X
    """
    datasets = list(patterns_dict.keys())
    result = {}

    # Wspólne dla wszystkich
    if len(datasets) >= 2:
        common_all = patterns_dict[datasets[0]].copy()
        for ds in datasets[1:]:
            common_all &= patterns_dict[ds]
        result['common_all'] = common_all

    # Wspólne dla par
    for i, ds1 in enumerate(datasets):
        for ds2 in datasets[i+1:]:
            key = f"common_{ds1}_{ds2}"
            common = patterns_dict[ds1] & patterns_dict[ds2]
            result[key] = common

    # Unikalne dla każdego
    for ds in datasets:
        others = set()
        for other_ds in datasets:
            if other_ds != ds:
                others |= patterns_dict[other_ds]
        result[f"unique_{ds}"] = patterns_dict[ds] - others

    return result


def generate_comparison_report(
    patterns_comparison: Dict[str, Set[str]],
    rules_comparison: Dict[str, Set[str]],
    output_dir: Path
) -> None:
    """Generuje raport porównawczy."""

    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("RAPORT PORÓWNAWCZY WZORCÓW ARMADA")
    report_lines.append("=" * 80)
    report_lines.append("")

    # Wzorce
    report_lines.append("## WZORCE TEMPORALNE")
    report_lines.append("-" * 40)

    for key, patterns in sorted(patterns_comparison.items()):
        report_lines.append(f"\n### {key}: {len(patterns)} wzorców")
        if len(patterns) <= 20:
            for p in sorted(patterns):
                report_lines.append(f"  - {p}")
        else:
            for p in sorted(patterns)[:10]:
                report_lines.append(f"  - {p}")
            report_lines.append(f"  ... i {len(patterns) - 10} więcej")

    # Reguły
    report_lines.append("\n" + "=" * 80)
    report_lines.append("## REGUŁY TEMPORALNE")
    report_lines.append("-" * 40)

    for key, rules in sorted(rules_comparison.items()):
        report_lines.append(f"\n### {key}: {len(rules)} reguł")
        if len(rules) <= 15:
            for r in sorted(rules):
                report_lines.append(f"  - {r}")
        else:
            for r in sorted(rules)[:10]:
                report_lines.append(f"  - {r}")
            report_lines.append(f"  ... i {len(rules) - 10} więcej")

    # Zapisz raport
    report_text = "\n".join(report_lines)
    with open(output_dir / "comparison_report.txt", "w") as f:
        f.write(report_text)

    print(report_text)


def create_venn_diagram_data(
    patterns_dict: Dict[str, Set[str]],
    output_dir: Path
) -> None:
    """Tworzy dane do diagramu Venna."""
    datasets = list(patterns_dict.keys())

    if len(datasets) != 3:
        print("Diagram Venna wymaga dokładnie 3 zbiorów danych")
        return

    ds1, ds2, ds3 = datasets
    p1, p2, p3 = patterns_dict[ds1], patterns_dict[ds2], patterns_dict[ds3]

    # Oblicz rozmiary sekcji
    only_1 = len(p1 - p2 - p3)
    only_2 = len(p2 - p1 - p3)
    only_3 = len(p3 - p1 - p2)

    only_12 = len((p1 & p2) - p3)
    only_13 = len((p1 & p3) - p2)
    only_23 = len((p2 & p3) - p1)

    all_three = len(p1 & p2 & p3)

    venn_data = {
        f"only_{ds1}": only_1,
        f"only_{ds2}": only_2,
        f"only_{ds3}": only_3,
        f"only_{ds1}_{ds2}": only_12,
        f"only_{ds1}_{ds3}": only_13,
        f"only_{ds2}_{ds3}": only_23,
        "all_three": all_three,
        "total_unique": len(p1 | p2 | p3)
    }

    with open(output_dir / "venn_data.json", "w") as f:
        json.dump(venn_data, f, indent=2)

    # Prosty wykres słupkowy
    fig, ax = plt.subplots(figsize=(12, 6))

    labels = [
        f"Tylko {ds1}",
        f"Tylko {ds2}",
        f"Tylko {ds3}",
        f"{ds1} ∩ {ds2}",
        f"{ds1} ∩ {ds3}",
        f"{ds2} ∩ {ds3}",
        "Wszystkie 3"
    ]
    values = [only_1, only_2, only_3, only_12, only_13, only_23, all_three]
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#e67e22', '#1abc9c', '#f1c40f']

    bars = ax.bar(labels, values, color=colors)
    ax.set_ylabel('Liczba wzorców')
    ax.set_title('Porównanie wzorców ARMADA między zbiorami danych')
    ax.tick_params(axis='x', rotation=45)

    # Dodaj etykiety wartości
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(val), ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / "patterns_comparison.png", dpi=150)
    plt.close()

    print(f"\nDane diagramu Venna zapisane w {output_dir / 'venn_data.json'}")
    print(f"Wykres zapisany w {output_dir / 'patterns_comparison.png'}")


def save_common_patterns_details(
    common_patterns: Set[str],
    all_results: Dict[str, Tuple],
    output_dir: Path
) -> None:
    """Zapisuje szczegóły wspólnych wzorców z metrykami per zbiór."""

    details = []

    for pattern_sig in sorted(common_patterns):
        entry = {"pattern": pattern_sig}

        for ds_name, (armada, patterns, rules) in all_results.items():
            # Znajdź ten wzorzec w wynikach
            for p in patterns:
                if p.get_relation_description() == pattern_sig:
                    entry[f"{ds_name}_support"] = p.support
                    entry[f"{ds_name}_count"] = p.support_count
                    break

        details.append(entry)

    df = pd.DataFrame(details)
    df.to_csv(output_dir / "common_patterns_details.csv", index=False)

    print(f"\nSzczegóły wspólnych wzorców zapisane w {output_dir / 'common_patterns_details.csv'}")


def main():
    """Główna funkcja porównawcza."""

    # Ścieżki
    BASE_DIR = Path(__file__).parent.parent.parent.parent
    DATA_DIR = BASE_DIR / "data" / "armada_ready"
    OUTPUT_DIR = BASE_DIR / "data" / "armada_results" / "experiments"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Parametry ARMADA (te same dla wszystkich zbiorów)
    MINSUP = 0.6  # 60% uczestników - wyższe dla szybkości
    MINCONF = 0.5
    MAXGAP = 30  # 30 sekund
    MAX_PATTERN_SIZE = 2  # maksymalnie 2-wzorce dla szybkości

    print("=" * 80)
    print("PORÓWNANIE WZORCÓW ARMADA MIĘDZY ZBIORAMI DANYCH")
    print("=" * 80)
    print(f"Parametry: minsup={MINSUP}, minconf={MINCONF}, maxgap={MAXGAP}, max_size={MAX_PATTERN_SIZE}")
    print()

    # Zbiory danych
    datasets = {
        'CASE': DATA_DIR / "armada_sequences_case.txt",
        'K-emoCon': DATA_DIR / "armada_sequences_k_emocon.txt",
        'CEAP': DATA_DIR / "armada_sequences_ceap.txt"
    }

    # Uruchom ARMADA na każdym zbiorze
    all_results = {}
    patterns_signatures = {}
    rules_signatures = {}

    for ds_name, data_file in datasets.items():
        if not data_file.exists():
            print(f"POMINIĘTO: {data_file} nie istnieje")
            continue

        print(f"\n{'='*60}")
        print(f"Przetwarzanie: {ds_name}")
        print(f"{'='*60}")

        armada, patterns, rules = run_armada_on_dataset(
            data_file,
            minsup=MINSUP,
            minconf=MINCONF,
            maxgap=MAXGAP,
            max_pattern_size=MAX_PATTERN_SIZE
        )

        all_results[ds_name] = (armada, patterns, rules)
        patterns_signatures[ds_name] = extract_pattern_signatures(patterns)
        rules_signatures[ds_name] = extract_rule_signatures(rules)

        print(f"  Wzorców: {len(patterns)}")
        print(f"  Reguł: {len(rules)}")

        # Zapisz wyniki per zbiór
        ds_output = OUTPUT_DIR / ds_name.lower().replace('-', '_')
        ds_output.mkdir(exist_ok=True)
        armada.save_results(ds_output)

    if len(all_results) < 2:
        print("\nZa mało zbiorów danych do porównania!")
        return

    # Porównaj wzorce
    print("\n" + "=" * 80)
    print("PORÓWNANIE WZORCÓW")
    print("=" * 80)

    patterns_comparison = compare_pattern_sets(patterns_signatures)
    rules_comparison = compare_pattern_sets(rules_signatures)

    # Statystyki
    print(f"\nWzorce wspólne dla WSZYSTKICH zbiorów: {len(patterns_comparison.get('common_all', set()))}")
    print(f"Reguły wspólne dla WSZYSTKICH zbiorów: {len(rules_comparison.get('common_all', set()))}")

    for ds in patterns_signatures:
        print(f"\nWzorce unikalne dla {ds}: {len(patterns_comparison.get(f'unique_{ds}', set()))}")

    # Generuj raport
    generate_comparison_report(patterns_comparison, rules_comparison, OUTPUT_DIR)

    # Diagram porównawczy
    if len(patterns_signatures) == 3:
        create_venn_diagram_data(patterns_signatures, OUTPUT_DIR)

    # Szczegóły wspólnych wzorców
    common_all = patterns_comparison.get('common_all', set())
    if common_all:
        save_common_patterns_details(common_all, all_results, OUTPUT_DIR)

    # Zapisz podsumowanie JSON
    summary = {
        "parameters": {
            "minsup": MINSUP,
            "minconf": MINCONF,
            "maxgap": MAXGAP,
            "max_pattern_size": MAX_PATTERN_SIZE
        },
        "datasets": {},
        "comparison": {
            "common_all_patterns": len(patterns_comparison.get('common_all', set())),
            "common_all_rules": len(rules_comparison.get('common_all', set()))
        }
    }

    for ds_name in patterns_signatures:
        summary["datasets"][ds_name] = {
            "total_patterns": len(patterns_signatures[ds_name]),
            "total_rules": len(rules_signatures[ds_name]),
            "unique_patterns": len(patterns_comparison.get(f'unique_{ds_name}', set()))
        }

    with open(OUTPUT_DIR / "comparison_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*80}")
    print("ZAKOŃCZONO")
    print(f"Wyniki zapisane w: {OUTPUT_DIR}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

