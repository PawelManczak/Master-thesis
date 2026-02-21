#!/usr/bin/env python3
"""
Porównanie wzorców ARMADA między zbiorami danych.

Ten skrypt:
1. Uruchamia ARMADA na każdym zbiorze danych osobno (CASE, K-emoCon, CEAP)
2. Porównuje odkryte wzorce i reguły
3. Znajduje wzorce wspólne dla wszystkich zbiorów
4. Generuje raport porównawczy

Parametry:
- minsup: 0.5 (50% uczestników)
- minconf: 0.5 (50% ufność)
- maxgap: 30 sekund
- max_pattern_size: 3
"""

import sys
from pathlib import Path
import pandas as pd
import json
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import matplotlib.pyplot as plt

# Dodaj ścieżkę do modułów
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_DIR / "source" / "processing" / "armada"))

from armada_algorithm import ARMADA


# ============================================================================
# PARAMETRY EKSPERYMENTU
# ============================================================================
MINSUP = 0.5         # 50% uczestników musi mieć wzorzec
MINCONF = 0.5        # 50% ufność dla reguł
MAXGAP = 20           # 30 sekund maksymalna przerwa
MAX_PATTERN_SIZE = 3  # wzorce do 3 stanów

# ============================================================================
# FILTRY REGUŁ
# ============================================================================
# Jeśli True - odrzuca reguły gdzie WSZYSTKIE stany są związane z BVP (wszystkie metryki bvp_*)
FILTER_BVP_ONLY = True

# Jeśli True - odrzuca reguły gdzie wszystkie stany dotyczą tej samej cechy (np. tylko arousal_low, arousal_high)
FILTER_SINGLE_FEATURE = True

# Wszystkie prefiksy metryk BVP (pochodne z Blood Volume Pulse)
BVP_PREFIXES = (
    'bvp_sdnn',      # odchylenie standardowe IBI
    'bvp_rmssd',     # root mean square of successive differences
    'bvp_pnn50',     # procent różnic IBI > 50ms
    'bvp_mean_hr',   # średnie tętno
    'bvp_mean_ibi',  # średnie IBI
    'bvp_lf_power',  # moc w paśmie LF
    'bvp_hf_power',  # moc w paśmie HF
    'bvp_lf_hf_ratio'  # stosunek LF/HF
)


def is_bvp_only_rule(rule_signature: str) -> bool:
    """
    Sprawdza czy reguła zawiera TYLKO stany BVP (wszystkie pochodne).

    Przykład reguły BVP-only:
    (bvp_sdnn_low) => bvp_sdnn_low before bvp_rmssd_high
    """
    # Wyodrębnij wszystkie stany z reguły
    # Format: "state1 relation state2 AND ... => state3 relation state4 ..."

    # Usuń relacje i operatory
    clean_sig = rule_signature.replace('=>', ' ').replace('AND', ' ')
    clean_sig = clean_sig.replace('equals', ' ').replace('before', ' ')
    clean_sig = clean_sig.replace('meets', ' ').replace('overlaps', ' ')
    clean_sig = clean_sig.replace('contains', ' ').replace('starts', ' ')
    clean_sig = clean_sig.replace('is-finished-by', ' ')
    clean_sig = clean_sig.replace('(', '').replace(')', '')

    # Wyodrębnij stany
    tokens = [t.strip() for t in clean_sig.split() if t.strip()]
    states = [t for t in tokens if '_' in t]  # stany mają format "feature_level"

    if not states:
        return False

    # Sprawdź czy wszystkie stany zaczynają się od bvp_
    for state in states:
        is_bvp = any(state.startswith(prefix) for prefix in BVP_PREFIXES)
        if not is_bvp:
            return False

    return True


def is_single_feature_rule(rule_signature: str) -> bool:
    """
    Sprawdza czy reguła zawiera tylko jedną cechę (np. tylko arousal).

    Przykład reguły single-feature:
    (arousal_low) => arousal_low meets arousal_high
    """
    # Wyodrębnij wszystkie stany z reguły
    clean_sig = rule_signature.replace('=>', ' ').replace('AND', ' ')
    clean_sig = clean_sig.replace('equals', ' ').replace('before', ' ')
    clean_sig = clean_sig.replace('meets', ' ').replace('overlaps', ' ')
    clean_sig = clean_sig.replace('contains', ' ').replace('starts', ' ')
    clean_sig = clean_sig.replace('is-finished-by', ' ')
    clean_sig = clean_sig.replace('(', '').replace(')', '')

    tokens = [t.strip() for t in clean_sig.split() if t.strip()]
    states = [t for t in tokens if '_' in t]

    if not states:
        return False

    # Wyodrębnij bazowe cechy (np. arousal z arousal_low)
    features = set()
    for state in states:
        # Format: feature_level (np. arousal_low, hr_high, bvp_rmssd_medium)
        parts = state.rsplit('_', 1)
        if len(parts) == 2:
            feature = parts[0]
            features.add(feature)

    # Jeśli tylko jedna cecha - to single-feature rule
    return len(features) == 1


def filter_rules(
    rules: Set[str],
    filter_bvp_only: bool = FILTER_BVP_ONLY,
    filter_single_feature: bool = FILTER_SINGLE_FEATURE
) -> Set[str]:
    """
    Filtruje reguły według zadanych kryteriów.

    Args:
        rules: Zbiór sygnatur reguł
        filter_bvp_only: Czy odrzucać reguły tylko z BVP
        filter_single_feature: Czy odrzucać reguły z jedną cechą

    Returns:
        Przefiltrowany zbiór reguł
    """
    filtered = set()

    for rule in rules:
        # Sprawdź filtry
        if filter_bvp_only and is_bvp_only_rule(rule):
            continue
        if filter_single_feature and is_single_feature_rule(rule):
            continue

        filtered.add(rule)

    return filtered


def run_armada_on_dataset(
    data_file: Path,
    minsup: float = MINSUP,
    minconf: float = MINCONF,
    maxgap: float = MAXGAP,
    max_pattern_size: int = MAX_PATTERN_SIZE
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
    """Ekstrahuje sygnatury wzorców do porównania."""
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


def compare_pattern_sets(patterns_dict: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
    """Porównuje zbiory wzorców."""
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


def create_comparison_visualizations(
    patterns_dict: Dict[str, Set[str]],
    comparison: Dict[str, Set[str]],
    output_dir: Path
) -> None:
    """Tworzy wykresy porównawcze."""
    datasets = list(patterns_dict.keys())

    # Wykres 1: Liczba wzorców per zbiór
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1a. Słupkowy - wzorce per zbiór
    ax1 = axes[0]
    ds_names = list(patterns_dict.keys())
    counts = [len(patterns_dict[ds]) for ds in ds_names]
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    bars = ax1.bar(ds_names, counts, color=colors)
    ax1.set_ylabel('Liczba wzorców')
    ax1.set_title('Wzorce per zbiór danych')
    for bar, count in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                str(count), ha='center', va='bottom', fontweight='bold')

    # 1b. Porównanie wspólnych/unikalnych
    ax2 = axes[1]
    if len(datasets) == 3:
        ds1, ds2, ds3 = datasets
        p1, p2, p3 = patterns_dict[ds1], patterns_dict[ds2], patterns_dict[ds3]

        labels = ['Tylko\n' + ds1, 'Tylko\n' + ds2, 'Tylko\n' + ds3,
                  f'{ds1}∩{ds2}', f'{ds1}∩{ds3}', f'{ds2}∩{ds3}', 'Wszystkie 3']
        values = [
            len(p1 - p2 - p3),
            len(p2 - p1 - p3),
            len(p3 - p1 - p2),
            len((p1 & p2) - p3),
            len((p1 & p3) - p2),
            len((p2 & p3) - p1),
            len(p1 & p2 & p3)
        ]
        colors2 = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#e67e22', '#1abc9c', '#f1c40f']
        bars2 = ax2.bar(range(len(labels)), values, color=colors2)
        ax2.set_xticks(range(len(labels)))
        ax2.set_xticklabels(labels, fontsize=8, rotation=45, ha='right')
        ax2.set_ylabel('Liczba wzorców')
        ax2.set_title('Rozkład wzorców')
        for bar, val in zip(bars2, values):
            if val > 0:
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        str(val), ha='center', va='bottom', fontsize=9)

    # 1c. Procent wspólnych wzorców
    ax3 = axes[2]
    common_all = len(comparison.get('common_all', set()))
    total_unique = len(set().union(*patterns_dict.values()))

    sizes = [common_all, total_unique - common_all]
    labels_pie = [f'Wspólne\n({common_all})', f'Pozostałe\n({total_unique - common_all})']
    colors_pie = ['#f1c40f', '#95a5a6']
    ax3.pie(sizes, labels=labels_pie, colors=colors_pie, autopct='%1.1f%%', startangle=90)
    ax3.set_title('Udział wspólnych wzorców')

    plt.tight_layout()
    plt.savefig(output_dir / 'patterns_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()


    print(f"Zapisano wykresy w {output_dir}")


def save_common_patterns_details(
    common_patterns: Set[str],
    all_results: Dict[str, Tuple],
    output_dir: Path
) -> pd.DataFrame:
    """Zapisuje szczegóły wspólnych wzorców z metrykami per zbiór."""
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

        # Średnie wsparcie
        if supports:
            entry["avg_support"] = round(sum(supports) / len(supports), 4)
            entry["min_support"] = round(min(supports), 4)

        details.append(entry)

    df = pd.DataFrame(details)

    # Sortuj po średnim wsparciu
    if 'avg_support' in df.columns:
        df = df.sort_values('avg_support', ascending=False)

    df.to_csv(output_dir / "common_patterns_details.csv", index=False)

    return df


def save_common_rules_details(
    common_rules: Set[str],
    all_results: Dict[str, Tuple],
    output_dir: Path
) -> pd.DataFrame:
    """Zapisuje szczegóły wspólnych reguł."""
    details = []

    for rule_sig in sorted(common_rules):
        entry = {"rule": rule_sig}
        confidences = []
        supports = []

        for ds_name, (armada, patterns, rules) in all_results.items():
            for r in rules:
                sig = f"{r.antecedent.get_relation_description()} => {r.consequent.get_relation_description()}"
                if sig == rule_sig:
                    entry[f"{ds_name}_confidence"] = round(r.confidence, 4)
                    entry[f"{ds_name}_support"] = round(r.support, 4)
                    confidences.append(r.confidence)
                    supports.append(r.support)
                    break

        if confidences:
            entry["avg_confidence"] = round(sum(confidences) / len(confidences), 4)
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
    """Generuje raport w formacie Markdown."""

    lines = []
    lines.append("# Porównanie wzorców ARMADA między zbiorami danych")
    lines.append("")
    lines.append("## Parametry eksperymentu")
    lines.append("")
    lines.append(f"- **minsup**: {MINSUP} ({MINSUP*100:.0f}% uczestników)")
    lines.append(f"- **minconf**: {MINCONF} ({MINCONF*100:.0f}% ufność)")
    lines.append(f"- **maxgap**: {MAXGAP} sekund")
    lines.append(f"- **max_pattern_size**: {MAX_PATTERN_SIZE}")
    lines.append("")
    lines.append("## Filtry reguł")
    lines.append("")
    lines.append(f"- **FILTER_BVP_ONLY**: {FILTER_BVP_ONLY} - {'odrzuca reguły zawierające tylko stany BVP' if FILTER_BVP_ONLY else 'wyłączony'}")
    lines.append(f"- **FILTER_SINGLE_FEATURE**: {FILTER_SINGLE_FEATURE} - {'odrzuca reguły z jedną cechą (np. tylko arousal)' if FILTER_SINGLE_FEATURE else 'wyłączony'}")
    lines.append("")

    # Statystyki per zbiór
    lines.append("## Statystyki zbiorów danych")
    lines.append("")
    lines.append("| Zbiór | Uczestników | Wzorców | Reguł | Unikalne wzorce |")
    lines.append("|-------|-------------|---------|-------|-----------------|")

    for ds_name, (armada, patterns, rules) in all_results.items():
        unique = len(patterns_comparison.get(f'unique_{ds_name}', set()))
        unique_pct = unique / len(patterns) * 100 if patterns else 0
        lines.append(f"| **{ds_name}** | {armada.num_clients} | {len(patterns)} | {len(rules)} | {unique} ({unique_pct:.1f}%) |")

    lines.append("")

    # Porównanie
    lines.append("## Porównanie zbiorów")
    lines.append("")
    common_all = len(patterns_comparison.get('common_all', set()))
    common_rules_all = len(rules_comparison.get('common_all', set()))
    lines.append(f"- **Wzorce wspólne dla WSZYSTKICH zbiorów**: {common_all}")
    lines.append(f"- **Reguły wspólne dla WSZYSTKICH zbiorów**: {common_rules_all}")
    lines.append("")

    # Wspólne wzorce
    lines.append("## Wspólne wzorce (wszystkie 3 zbiory)")
    lines.append("")

    if len(common_patterns_df) > 0:
        lines.append("### Top 20 wspólnych wzorców (według średniego wsparcia)")
        lines.append("")
        lines.append("| Wzorzec | Śr. Support | CASE | K-emoCon | CEAP |")
        lines.append("|---------|-------------|------|----------|------|")

        for _, row in common_patterns_df.head(20).iterrows():
            case_sup = row.get('CASE_support', 'N/A')
            kemo_sup = row.get('K-emoCon_support', 'N/A')
            ceap_sup = row.get('CEAP_support', 'N/A')
            avg_sup = row.get('avg_support', 'N/A')

            if isinstance(case_sup, float):
                case_sup = f"{case_sup:.2f}"
            if isinstance(kemo_sup, float):
                kemo_sup = f"{kemo_sup:.2f}"
            if isinstance(ceap_sup, float):
                ceap_sup = f"{ceap_sup:.2f}"
            if isinstance(avg_sup, float):
                avg_sup = f"{avg_sup:.2f}"

            lines.append(f"| `{row['pattern']}` | {avg_sup} | {case_sup} | {kemo_sup} | {ceap_sup} |")

    lines.append("")

    # Wspólne reguły
    lines.append("## Wspólne reguły (wszystkie 3 zbiory)")
    lines.append("")

    if len(common_rules_df) > 0:
        lines.append("### Wszystkie wspólne reguły (według średniej ufności)")
        lines.append("")

        for _, row in common_rules_df.iterrows():
            rule = row['rule']
            avg_conf = row.get('avg_confidence', 'N/A')
            avg_sup = row.get('avg_support', 'N/A')

            if isinstance(avg_conf, float):
                avg_conf = f"{avg_conf:.2f}"
            if isinstance(avg_sup, float):
                avg_sup = f"{avg_sup:.2f}"

            lines.append(f"- `{rule}` (conf={avg_conf}, sup={avg_sup})")

    lines.append("")

    # Wnioski
    lines.append("## Wnioski")
    lines.append("")
    lines.append("### Wzorce powtarzające się między zbiorami")
    lines.append("")

    if common_all > 0:
        lines.append(f"**TAK** - {common_all} wzorców jest wspólnych dla wszystkich trzech zbiorów danych, czyli sa uniwersalne")
    else:
        lines.append("**NIE** - Nie znaleziono wzorców wspólnych dla wszystkich zbiorów.")

    lines.append("")

    # Zapisz raport
    report_text = "\n".join(lines)
    with open(output_dir / "comparison_report.md", "w") as f:
        f.write(report_text)

    print(f"Zapisano raport: {output_dir / 'comparison_report.md'}")


def main():
    """Główna funkcja eksperymentu."""

    # Ścieżki
    DATA_DIR = PROJECT_DIR / "data" / "armada_ready"
    OUTPUT_DIR = SCRIPT_DIR / "results"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("EKSPERYMENT: PORÓWNANIE WZORCÓW ARMADA MIĘDZY ZBIORAMI DANYCH")
    print("=" * 80)
    print(f"Data: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Parametry: minsup={MINSUP}, minconf={MINCONF}, maxgap={MAXGAP}, max_size={MAX_PATTERN_SIZE}")
    print(f"Filtry reguł: FILTER_BVP_ONLY={FILTER_BVP_ONLY}, FILTER_SINGLE_FEATURE={FILTER_SINGLE_FEATURE}")
    print(f"Wyniki: {OUTPUT_DIR}")
    print()

    # Zbiory danych
    datasets = {
        'CASE': DATA_DIR / "armada_sequences_case.txt",
        'K-emoCon': DATA_DIR / "armada_sequences_k_emocon.txt",
        'CEAP': DATA_DIR / "armada_sequences_ceap.txt"
    }

    # Sprawdź czy pliki istnieją
    for ds_name, data_file in datasets.items():
        if not data_file.exists():
            print(f"BŁĄD: Plik {data_file} nie istnieje!")
            print("Uruchom najpierw: python prepare_armada_data.py && python validate_armada_data.py")
            return

    # Uruchom ARMADA na każdym zbiorze
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

        # Zapisz wyniki per zbiór
        ds_output = OUTPUT_DIR / ds_name.lower().replace('-', '_')
        ds_output.mkdir(exist_ok=True)
        armada.save_results(ds_output)

    # Porównaj wzorce i reguły
    print("\n" + "=" * 80)
    print("PORÓWNANIE WZORCÓW")
    print("=" * 80)

    patterns_comparison = compare_pattern_sets(patterns_signatures)

    # Filtrowanie reguł przed porównaniem
    filtered_rules_signatures = {}
    for ds_name, rules_set in rules_signatures.items():
        original_count = len(rules_set)
        filtered = filter_rules(rules_set, FILTER_BVP_ONLY, FILTER_SINGLE_FEATURE)
        filtered_rules_signatures[ds_name] = filtered
        removed = original_count - len(filtered)
        if removed > 0:
            print(f"  {ds_name}: odfiltrowano {removed} reguł (z {original_count})")

    rules_comparison = compare_pattern_sets(filtered_rules_signatures)

    common_patterns = patterns_comparison.get('common_all', set())
    common_rules = rules_comparison.get('common_all', set())

    print(f"\nWzorce wspólne dla WSZYSTKICH zbiorów: {len(common_patterns)}")
    print(f"Reguły wspólne dla WSZYSTKICH zbiorów: {len(common_rules)}")

    for ds in patterns_signatures:
        unique = len(patterns_comparison.get(f'unique_{ds}', set()))
        print(f"Wzorce unikalne dla {ds}: {unique}")

    # Zapisz szczegóły wspólnych wzorców i reguł
    common_patterns_df = save_common_patterns_details(common_patterns, all_results, OUTPUT_DIR)
    common_rules_df = save_common_rules_details(common_rules, all_results, OUTPUT_DIR)

    # Wizualizacje
    create_comparison_visualizations(patterns_signatures, patterns_comparison, OUTPUT_DIR)

    # Raport Markdown
    generate_markdown_report(
        patterns_comparison, rules_comparison, all_results,
        common_patterns_df, common_rules_df, OUTPUT_DIR
    )

    # Zapisz podsumowanie JSON
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

    # Wyświetl wspólne wzorce
    print("\n" + "=" * 80)
    print("WSPÓLNE WZORCE (wszystkie 3 zbiory)")
    print("=" * 80)

    for _, row in common_patterns_df.head(15).iterrows():
        print(f"  {row['pattern']} (avg_sup={row.get('avg_support', 'N/A')})")

    if len(common_patterns_df) > 15:
        print(f"  ... i {len(common_patterns_df) - 15} więcej")

    print("\n" + "=" * 80)
    print("WSPÓLNE REGUŁY (wszystkie 3 zbiory)")
    print("=" * 80)

    for _, row in common_rules_df.head(10).iterrows():
        print(f"  {row['rule']}")
        print(f"    avg_conf={row.get('avg_confidence', 'N/A')}, avg_sup={row.get('avg_support', 'N/A')}")

    if len(common_rules_df) > 10:
        print(f"  ... i {len(common_rules_df) - 10} więcej")

    print("\n" + "=" * 80)
    print("EKSPERYMENT ZAKOŃCZONY")
    print(f"Wyniki zapisane w: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()

