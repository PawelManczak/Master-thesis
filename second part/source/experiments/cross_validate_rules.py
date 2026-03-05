#!/usr/bin/env python3
"""
Eksperyment: Walidacja krzyżowa reguł ARMADA między zbiorami danych.

Dla każdej kombinacji dwóch zbiorów treningowych i jednego walidacyjnego:
1. Znajdź reguły wspólne dla dwóch zbiorów treningowych
2. Sprawdź ile z nich pojawia się w zbiorze walidacyjnym
3. Porównaj ufność i wsparcie

Kombinacje:
  - Trening: CASE + K-emoCon    → Walidacja: CEAP
  - Trening: CASE + CEAP        → Walidacja: K-emoCon
  - Trening: K-emoCon + CEAP    → Walidacja: CASE
"""

import sys
from pathlib import Path
import pandas as pd
import json
from collections import defaultdict
from typing import Dict, List, Set, Tuple

# Dodaj ścieżkę do modułów
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_DIR / "source" / "processing" / "armada"))
sys.path.insert(0, str(SCRIPT_DIR))

from armada_algorithm import ARMADA, TemporalRule
from compare_datasets import (
    run_armada_on_dataset,
    extract_rule_signatures,
    filter_rules,
    MINSUP, MINCONF, MAXGAP, MAX_PATTERN_SIZE,
    FILTER_BVP_ONLY, FILTER_EDA_ONLY, FILTER_PHYSIO_CROSS, FILTER_SINGLE_FEATURE,
)


# ============================================================================
# KONFIGURACJA
# ============================================================================
DATA_DIR = PROJECT_DIR / "data" / "armada_ready"
OUTPUT_DIR = SCRIPT_DIR / "results" / "cross_validation"

DATASETS = {
    'CASE': DATA_DIR / "armada_sequences_case.txt",
    'K-emoCon': DATA_DIR / "armada_sequences_k_emocon.txt",
    'CEAP': DATA_DIR / "armada_sequences_ceap.txt",
}


def get_rule_details(rules: List[TemporalRule]) -> Dict[str, Dict]:
    """
    Mapuje sygnaturę reguły na jej szczegóły (confidence, support).

    Returns:
        dict: {sygnatura: {'confidence': float, 'support': float}}
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
    """Główna funkcja eksperymentu walidacji krzyżowej."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("EKSPERYMENT: WALIDACJA KRZYŻOWA REGUŁ ARMADA")
    print("=" * 90)
    print(f"Data: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Parametry: minsup={MINSUP}, minconf={MINCONF}, maxgap={MAXGAP}, max_size={MAX_PATTERN_SIZE}")
    print(f"Filtry: BVP_ONLY={FILTER_BVP_ONLY}, EDA_ONLY={FILTER_EDA_ONLY}, "
          f"PHYSIO_CROSS={FILTER_PHYSIO_CROSS}, SINGLE_FEAT={FILTER_SINGLE_FEATURE}")
    print()

    # Sprawdź pliki
    for ds_name, path in DATASETS.items():
        if not path.exists():
            print(f"BŁĄD: Brak pliku {path}")
            return

    # ================================================================
    # 1. Uruchom ARMADA na każdym zbiorze
    # ================================================================
    all_results = {}  # {name: (armada, patterns, rules)}
    all_rule_sigs = {}  # {name: set(signatures)}
    all_rule_details = {}  # {name: {sig: {conf, sup}}}

    for ds_name, data_file in DATASETS.items():
        print(f"Przetwarzanie: {ds_name}...", end=" ", flush=True)
        armada, patterns, rules = run_armada_on_dataset(data_file)
        all_results[ds_name] = (armada, patterns, rules)

        # Filtruj reguły
        raw_sigs = extract_rule_signatures(rules)
        filtered_sigs = filter_rules(raw_sigs, FILTER_BVP_ONLY, FILTER_EDA_ONLY,
                                     FILTER_PHYSIO_CROSS, FILTER_SINGLE_FEATURE)
        all_rule_sigs[ds_name] = filtered_sigs
        all_rule_details[ds_name] = get_rule_details(rules)

        n_clients = armada.num_clients
        print(f"{n_clients} uczestników, {len(patterns)} wzorców, "
              f"{len(rules)} reguł ({len(filtered_sigs)} po filtracji)")

    # ================================================================
    # 2. Walidacja krzyżowa: 2 treningowe → 1 walidacyjny
    # ================================================================
    ds_names = list(DATASETS.keys())
    combinations = [
        (ds_names[0], ds_names[1], ds_names[2]),  # CASE+K-emoCon → CEAP
        (ds_names[0], ds_names[2], ds_names[1]),  # CASE+CEAP → K-emoCon
        (ds_names[1], ds_names[2], ds_names[0]),  # K-emoCon+CEAP → CASE
    ]

    cv_results = []

    print()
    print("=" * 90)
    print("WALIDACJA KRZYŻOWA")
    print("=" * 90)

    for train1, train2, val in combinations:
        print(f"\n--- Trening: {train1} + {train2}  →  Walidacja: {val} ---")

        # Reguły wspólne dla pary treningowej
        common_train = all_rule_sigs[train1] & all_rule_sigs[train2]
        print(f"  Reguły wspólne ({train1} ∩ {train2}): {len(common_train)}")

        # Ile z nich jest w zbiorze walidacyjnym
        validated = common_train & all_rule_sigs[val]
        print(f"  Potwierdzone w {val}: {len(validated)}")

        hit_rate = len(validated) / len(common_train) * 100 if common_train else 0
        print(f"  Trafność: {hit_rate:.1f}%")

        # Zbierz szczegóły potwierdzonych reguł
        validated_details = []
        for sig in sorted(validated):
            d1 = all_rule_details[train1].get(sig, {})
            d2 = all_rule_details[train2].get(sig, {})
            dv = all_rule_details[val].get(sig, {})

            conf_train_avg = (d1.get('confidence', 0) + d2.get('confidence', 0)) / 2
            sup_train_avg = (d1.get('support', 0) + d2.get('support', 0)) / 2

            validated_details.append({
                'rule': sig,
                f'{train1}_conf': round(d1.get('confidence', 0), 3),
                f'{train1}_sup': round(d1.get('support', 0), 3),
                f'{train2}_conf': round(d2.get('confidence', 0), 3),
                f'{train2}_sup': round(d2.get('support', 0), 3),
                f'{val}_conf': round(dv.get('confidence', 0), 3),
                f'{val}_sup': round(dv.get('support', 0), 3),
                'train_avg_conf': round(conf_train_avg, 3),
                'val_conf': round(dv.get('confidence', 0), 3),
                'conf_diff': round(dv.get('confidence', 0) - conf_train_avg, 3),
            })

        cv_results.append({
            'train1': train1,
            'train2': train2,
            'val': val,
            'common_train': len(common_train),
            'validated': len(validated),
            'hit_rate': hit_rate,
            'details': validated_details,
        })

    # ================================================================
    # 3. Reguły potwierdzone we WSZYSTKICH kombinacjach
    # ================================================================
    all_validated = None
    for cv in cv_results:
        validated_set = {d['rule'] for d in cv['details']}
        if all_validated is None:
            all_validated = validated_set
        else:
            all_validated &= validated_set

    print()
    print("=" * 90)
    print(f"REGUŁY POTWIERDZONE WE WSZYSTKICH 3 KOMBINACJACH: {len(all_validated)}")
    print("=" * 90)

    # ================================================================
    # 4. Generuj tabelę podsumowującą
    # ================================================================
    summary_rows = []
    for cv in cv_results:
        summary_rows.append({
            'Trening': f"{cv['train1']} + {cv['train2']}",
            'Walidacja': cv['val'],
            'Reguły wspólne (trening)': cv['common_train'],
            'Potwierdzone w walidacji': cv['validated'],
            'Trafność (%)': round(cv['hit_rate'], 1),
        })

    summary_df = pd.DataFrame(summary_rows)
    print()
    print(summary_df.to_string(index=False))

    # ================================================================
    # 5. Szczegółowa tabela potwierdzonych reguł
    # ================================================================
    # Dla każdej reguły — w ilu kombinacjach została potwierdzona
    rule_counts = defaultdict(lambda: {
        'count': 0,
        'combinations': [],
        'confs': [],
        'sups': [],
    })

    for cv in cv_results:
        combo_label = f"{cv['train1']}+{cv['train2']}→{cv['val']}"
        for d in cv['details']:
            sig = d['rule']
            rule_counts[sig]['count'] += 1
            rule_counts[sig]['combinations'].append(combo_label)
            rule_counts[sig]['confs'].append(d['val_conf'])
            rule_counts[sig]['sups'].append(d[f"{cv['val']}_sup"])

    # Zbierz ogólne conf/sup ze wszystkich 3 datasetów
    universal_rules = []
    for sig, info in sorted(rule_counts.items(), key=lambda x: -x[1]['count']):
        all_confs = []
        all_sups = []
        per_ds = {}
        for ds in ds_names:
            dd = all_rule_details[ds].get(sig, {})
            c = dd.get('confidence', None)
            s = dd.get('support', None)
            if c is not None:
                all_confs.append(c)
                per_ds[f'{ds}_conf'] = round(c, 3)
            else:
                per_ds[f'{ds}_conf'] = '-'
            if s is not None:
                all_sups.append(s)
                per_ds[f'{ds}_sup'] = round(s, 3)
            else:
                per_ds[f'{ds}_sup'] = '-'

        universal_rules.append({
            'Reguła': sig,
            'Potwierdzenia (z 3)': info['count'],
            **{f'{ds} conf': per_ds[f'{ds}_conf'] for ds in ds_names},
            **{f'{ds} sup': per_ds[f'{ds}_sup'] for ds in ds_names},
            'Śr. conf': round(sum(all_confs) / len(all_confs), 3) if all_confs else '-',
            'Śr. sup': round(sum(all_sups) / len(all_sups), 3) if all_sups else '-',
        })

    universal_df = pd.DataFrame(universal_rules)
    if len(universal_df) > 0 and 'Potwierdzenia (z 3)' in universal_df.columns:
        universal_df = universal_df.sort_values(
            by=['Potwierdzenia (z 3)', 'Śr. conf'],
            ascending=[False, False]
        )

    # ================================================================
    # 6. Zapisz wyniki
    # ================================================================
    # Tabela podsumowująca
    summary_df.to_csv(OUTPUT_DIR / "cross_validation_summary.csv", index=False)

    # Tabela uniwersalnych reguł
    universal_df.to_csv(OUTPUT_DIR / "cross_validated_rules.csv", index=False)

    # Szczegóły per kombinacja
    for cv in cv_results:
        if cv['details']:
            df = pd.DataFrame(cv['details']).sort_values('train_avg_conf', ascending=False)
            fname = f"validated_{cv['train1']}_{cv['train2']}_to_{cv['val']}.csv"
            fname = fname.replace('-', '').lower()
            df.to_csv(OUTPUT_DIR / fname, index=False)

    # ================================================================
    # 7. Generuj raport Markdown
    # ================================================================
    lines = []
    lines.append("# Walidacja krzyżowa reguł ARMADA")
    lines.append("")
    lines.append(f"Data: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")
    lines.append("## Metodologia")
    lines.append("")
    lines.append("Dla każdej kombinacji dwóch zbiorów treningowych:")
    lines.append("1. Znajdź reguły wspólne dla pary treningowej")
    lines.append("2. Sprawdź ile z nich występuje w trzecim zbiorze (walidacyjnym)")
    lines.append("3. Oblicz trafność (% reguł potwierdzonych)")
    lines.append("")
    lines.append("## Parametry")
    lines.append("")
    lines.append(f"- minsup: {MINSUP}, minconf: {MINCONF}, maxgap: {MAXGAP}, max_pattern_size: {MAX_PATTERN_SIZE}")
    lines.append(f"- Filtry: BVP_ONLY={FILTER_BVP_ONLY}, EDA_ONLY={FILTER_EDA_ONLY}, "
                 f"PHYSIO_CROSS={FILTER_PHYSIO_CROSS}, SINGLE_FEATURE={FILTER_SINGLE_FEATURE}")
    lines.append("")

    # Statystyki zbiorów
    lines.append("## Statystyki zbiorów")
    lines.append("")
    lines.append("| Zbiór | Uczestników | Reguł (surowe) | Reguł (po filtracji) |")
    lines.append("|-------|-------------|----------------|----------------------|")
    for ds in ds_names:
        armada, patterns, rules = all_results[ds]
        lines.append(f"| {ds} | {armada.num_clients} | {len(rules)} | {len(all_rule_sigs[ds])} |")
    lines.append("")

    # Tabela podsumowująca
    lines.append("## Wyniki walidacji krzyżowej")
    lines.append("")
    lines.append("| Trening | Walidacja | Reguły wspólne (trening) | Potwierdzone | Trafność |")
    lines.append("|---------|-----------|--------------------------|--------------|----------|")
    for cv in cv_results:
        lines.append(f"| {cv['train1']} + {cv['train2']} | {cv['val']} | "
                     f"{cv['common_train']} | {cv['validated']} | **{cv['hit_rate']:.1f}%** |")
    lines.append("")

    # Średnia trafność
    avg_hit = sum(cv['hit_rate'] for cv in cv_results) / len(cv_results)
    lines.append(f"**Średnia trafność walidacji: {avg_hit:.1f}%**")
    lines.append("")

    # Reguły potwierdzone we wszystkich kombinacjach
    lines.append("## Reguły potwierdzone we wszystkich 3 kombinacjach")
    lines.append("")
    n_all = len([r for r in universal_rules if r['Potwierdzenia (z 3)'] == 3])
    lines.append(f"Liczba reguł potwierdzonych w każdej kombinacji: **{n_all}**")
    lines.append("")

    if n_all > 0:
        lines.append("| # | Reguła | Śr. ufność | Śr. wsparcie | CASE conf | K-emoCon conf | CEAP conf |")
        lines.append("|---|--------|------------|--------------|-----------|---------------|-----------|")
        i = 1
        for r in universal_rules:
            if r['Potwierdzenia (z 3)'] == 3:
                lines.append(f"| {i} | `{r['Reguła']}` | {r['Śr. conf']} | {r['Śr. sup']} | "
                             f"{r.get('CASE conf', '-')} | {r.get('K-emoCon conf', '-')} | {r.get('CEAP conf', '-')} |")
                i += 1
        lines.append("")

    # Reguły potwierdzone w 2 z 3 kombinacji
    n_two = len([r for r in universal_rules if r['Potwierdzenia (z 3)'] == 2])
    lines.append(f"## Reguły potwierdzone w 2 z 3 kombinacji: {n_two}")
    lines.append("")
    if n_two > 0:
        lines.append("| # | Reguła | Śr. ufność | Śr. wsparcie | CASE conf | K-emoCon conf | CEAP conf |")
        lines.append("|---|--------|------------|--------------|-----------|---------------|-----------|")
        i = 1
        for r in universal_rules:
            if r['Potwierdzenia (z 3)'] == 2:
                lines.append(f"| {i} | `{r['Reguła']}` | {r['Śr. conf']} | {r['Śr. sup']} | "
                             f"{r.get('CASE conf', '-')} | {r.get('K-emoCon conf', '-')} | {r.get('CEAP conf', '-')} |")
                i += 1
        lines.append("")

    # Podsumowanie
    lines.append("## Podsumowanie")
    lines.append("")
    total_unique_validated = len(rule_counts)
    lines.append(f"- Łącznie unikalnych reguł potwierdzonych w przynajmniej 1 kombinacji: **{total_unique_validated}**")
    lines.append(f"- Potwierdzone we wszystkich 3 kombinacjach: **{n_all}**")
    lines.append(f"- Potwierdzone w 2 z 3 kombinacji: **{n_two}**")
    lines.append(f"- Potwierdzone w 1 z 3 kombinacji: **{total_unique_validated - n_all - n_two}**")
    lines.append(f"- Średnia trafność walidacji: **{avg_hit:.1f}%**")
    lines.append("")

    # Zapisz raport
    report_path = OUTPUT_DIR / "cross_validation_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nZapisano raport: {report_path}")

    # Zapisz JSON z pełnymi wynikami
    json_summary = {
        'parameters': {
            'minsup': MINSUP, 'minconf': MINCONF,
            'maxgap': MAXGAP, 'max_pattern_size': MAX_PATTERN_SIZE,
        },
        'filters': {
            'bvp_only': FILTER_BVP_ONLY, 'eda_only': FILTER_EDA_ONLY,
            'physio_cross': FILTER_PHYSIO_CROSS, 'single_feature': FILTER_SINGLE_FEATURE,
        },
        'datasets': {
            ds: {
                'n_clients': all_results[ds][0].num_clients,
                'n_rules_raw': len(all_results[ds][2]),
                'n_rules_filtered': len(all_rule_sigs[ds]),
            } for ds in ds_names
        },
        'cross_validation': [
            {
                'train': [cv['train1'], cv['train2']],
                'val': cv['val'],
                'common_train': cv['common_train'],
                'validated': cv['validated'],
                'hit_rate_pct': round(cv['hit_rate'], 2),
            } for cv in cv_results
        ],
        'avg_hit_rate_pct': round(avg_hit, 2),
        'rules_confirmed_in_all_3': n_all,
        'rules_confirmed_in_2_of_3': n_two,
    }
    with open(OUTPUT_DIR / "cross_validation_summary.json", "w") as f:
        json.dump(json_summary, f, indent=2)

    print(f"Zapisano CSV: {OUTPUT_DIR / 'cross_validation_summary.csv'}")
    print(f"Zapisano CSV: {OUTPUT_DIR / 'cross_validated_rules.csv'}")
    print(f"Zapisano JSON: {OUTPUT_DIR / 'cross_validation_summary.json'}")


if __name__ == "__main__":
    run_cross_validation()


