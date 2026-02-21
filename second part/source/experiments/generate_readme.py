#!/usr/bin/env python3
"""
Generator README dla wyników eksperymentu ARMADA.

Ten skrypt:
1. Wczytuje wyniki z experiment_summary.json
2. Wczytuje wspólne wzorce i reguły z plików CSV
3. Generuje kompleksowy README.md z interpretacją wyników
"""

import json
from pathlib import Path
from datetime import datetime
import pandas as pd

# Ścieżki
SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results"
OUTPUT_FILE = RESULTS_DIR / "README.md"


def load_experiment_data():
    """Wczytuje dane eksperymentu."""
    # Wczytaj podsumowanie JSON
    summary_file = RESULTS_DIR / "experiment_summary.json"
    if not summary_file.exists():
        raise FileNotFoundError(f"Brak pliku {summary_file}. Uruchom najpierw compare_datasets.py")

    with open(summary_file) as f:
        summary = json.load(f)

    # Wczytaj wspólne wzorce
    patterns_file = RESULTS_DIR / "common_patterns_details.csv"
    patterns_df = pd.read_csv(patterns_file) if patterns_file.exists() else pd.DataFrame()

    # Wczytaj wspólne reguły
    rules_file = RESULTS_DIR / "common_rules_details.csv"
    rules_df = pd.read_csv(rules_file) if rules_file.exists() else pd.DataFrame()

    return summary, patterns_df, rules_df


def interpret_pattern(pattern: str) -> str:
    """Interpretuje wzorzec w języku naturalnym."""

    # Mapowanie stanów na opisy
    state_descriptions = {
        # Arousal
        'arousal_low': 'niskie pobudzenie emocjonalne (relaks, senność)',
        'arousal_medium': 'umiarkowane pobudzenie (normalna czujność)',
        'arousal_high': 'wysokie pobudzenie (ekscytacja lub stres)',

        # Valence
        'valence_low': 'negatywna walencja (nieprzyjemne emocje)',
        'valence_medium': 'neutralna walencja',
        'valence_high': 'pozytywna walencja (przyjemne emocje)',

        # EDA
        'eda_low': 'niskie przewodnictwo skórne (niski stres)',
        'eda_medium': 'średnie przewodnictwo skórne',
        'eda_high': 'wysokie przewodnictwo skórne (wysoki stres)',

        # HR
        'hr_low': 'niskie tętno',
        'hr_medium': 'normalne tętno',
        'hr_high': 'przyspieszone tętno',

        # TEMP
        'temp_low': 'niska temperatura skóry',
        'temp_medium': 'normalna temperatura skóry',
        'temp_high': 'podwyższona temperatura skóry',

        # HRV
        'hrv_sdnn_low': 'niska zmienność rytmu serca (SDNN)',
        'hrv_sdnn_medium': 'średnia zmienność rytmu serca (SDNN)',
        'hrv_sdnn_high': 'wysoka zmienność rytmu serca (SDNN)',
        'hrv_rmssd_low': 'niska zmienność rytmu serca (RMSSD)',
        'hrv_rmssd_medium': 'średnia zmienność rytmu serca (RMSSD)',
        'hrv_rmssd_high': 'wysoka zmienność rytmu serca (RMSSD)',
    }

    # Mapowanie relacji
    relation_descriptions = {
        'equals': 'współwystępuje z',
        'before': 'poprzedza',
        'meets': 'bezpośrednio poprzedza',
        'overlaps': 'nakłada się z',
        'contains': 'zawiera',
        'starts': 'rozpoczyna się z',
        'is-finished-by': 'kończy się wraz z'
    }

    # Parsuj wzorzec
    interpretation = pattern

    for state, desc in state_descriptions.items():
        if state in interpretation:
            interpretation = interpretation.replace(state, f"**{state}** ({desc})")

    for rel, desc in relation_descriptions.items():
        if rel in interpretation:
            interpretation = interpretation.replace(rel, f"*{desc}*")

    return interpretation


def interpret_rule(rule: str) -> str:
    """Interpretuje regułę w języku naturalnym."""

    # Rozdziel na poprzednik i następnik
    if '=>' not in rule:
        return rule

    parts = rule.split('=>')
    antecedent = parts[0].strip()
    consequent = parts[1].strip()

    # Interpretuj obie części
    ant_interp = interpret_pattern(antecedent)
    cons_interp = interpret_pattern(consequent)

    return f"Jeśli {ant_interp}, to {cons_interp}"


def generate_methodology_section(summary: dict) -> str:
    """Generuje sekcję metodologii."""

    params = summary.get('parameters', {})
    filters = summary.get('filters', {})

    lines = []
    lines.append("## Metodologia")
    lines.append("")
    lines.append("### Algorytm ARMADA")
    lines.append("")
    lines.append("ARMADA (Association Rule Mining for Anomaly Detection in Affective data) to algorytm")
    lines.append("do wykrywania wzorców temporalnych w danych afektywnych. Wykorzystuje relacje czasowe")
    lines.append("Allena (equals, before, meets, overlaps, contains, starts, is-finished-by) do opisu")
    lines.append("zależności między stanami emocjonalnymi i fizjologicznymi.")
    lines.append("")
    lines.append("### Parametry eksperymentu")
    lines.append("")
    lines.append("| Parametr | Wartość | Opis |")
    lines.append("|----------|---------|------|")
    lines.append(f"| minsup | {params.get('minsup', 'N/A')} | Minimalny odsetek uczestników z wzorcem |")
    lines.append(f"| minconf | {params.get('minconf', 'N/A')} | Minimalna ufność reguły |")
    lines.append(f"| maxgap | {params.get('maxgap', 'N/A')}s | Maksymalna przerwa między stanami |")
    lines.append(f"| max_pattern_size | {params.get('max_pattern_size', 'N/A')} | Maksymalna liczba stanów we wzorcu |")
    lines.append("")
    lines.append("### Filtry reguł")
    lines.append("")

    if filters.get('filter_bvp_only', False):
        lines.append("- ✅ **Odfiltrowano reguły tylko z BVP** - reguły zawierające wyłącznie metryki HRV (bvp_*) zostały usunięte")
    else:
        lines.append("- ❌ Filtr BVP-only wyłączony")

    if filters.get('filter_single_feature', False):
        lines.append("- ✅ **Odfiltrowano reguły jednocechowe** - reguły opisujące tylko jedną cechę (np. arousal) zostały usunięte")
    else:
        lines.append("- ❌ Filtr single-feature wyłączony")

    lines.append("")
    lines.append("### Dyskretyzacja zmiennych")
    lines.append("")
    lines.append("#### Arousal i Valence (skala SAM)")
    lines.append("")
    lines.append("Zgodnie z literaturą (Ahmad et al.), wartości SAM 1-9 pogrupowano w trzy poziomy:")
    lines.append("")
    lines.append("| Poziom | Zakres (0-1) | SAM (1-9) | Interpretacja |")
    lines.append("|--------|--------------|-----------|---------------|")
    lines.append("| low | [0.00, 0.25] | 1-3 | Negatywne/niskie |")
    lines.append("| medium | (0.25, 0.75) | 4-6 | Neutralne/umiarkowane |")
    lines.append("| high | [0.75, 1.00] | 7-9 | Pozytywne/wysokie |")
    lines.append("")
    lines.append("#### EDA (przewodnictwo skórne)")
    lines.append("")
    lines.append("Zgodnie z zaleceniami psychofizjologicznymi (Boucsein, Horvers et al.):")
    lines.append("")
    lines.append("1. **Normalizacja osobnicza**: `EDA_norm = (EDA - EDA_min) / (EDA_max - EDA_min)`")
    lines.append("2. **Progi**: low [0, 0.33], medium (0.33, 0.66], high (0.66, 1.00]")
    lines.append("")
    lines.append("#### Pozostałe zmienne fizjologiczne")
    lines.append("")
    lines.append("HR, TEMP, HRV: tercyle per uczestnik (33% i 67% percentyl)")
    lines.append("")

    return "\n".join(lines)


def generate_datasets_section(summary: dict) -> str:
    """Generuje sekcję o zbiorach danych."""

    datasets = summary.get('datasets', {})

    lines = []
    lines.append("## Zbiory danych")
    lines.append("")
    lines.append("| Zbiór | Uczestników | Wzorców | Reguł | Unikalne wzorce |")
    lines.append("|-------|-------------|---------|-------|-----------------|")

    for ds_name, ds_data in datasets.items():
        unique = ds_data.get('unique_patterns', 0)
        total = ds_data.get('total_patterns', 1)
        pct = (unique / total * 100) if total > 0 else 0

        lines.append(f"| **{ds_name}** | {ds_data.get('num_clients', 'N/A')} | "
                    f"{ds_data.get('total_patterns', 'N/A')} | "
                    f"{ds_data.get('total_rules', 'N/A')} | "
                    f"{unique} ({pct:.1f}%) |")

    lines.append("")
    lines.append("### Opis zbiorów")
    lines.append("")
    lines.append("- **CASE**: Continuous Annotation of Self-reported Emotions - adnotacje ciągłe joystickiem")
    lines.append("- **K-EmoCon**: K-Emotion Convention - dane z Empatica E4 z samoocenami co 5s")
    lines.append("- **CEAP**: Continuous Emotion Annotation Protocol - adnotacje ciągłe 360VR")
    lines.append("")

    return "\n".join(lines)


def generate_similarity_section(summary: dict) -> str:
    """Generuje sekcję o wspólnych wzorcach i regułach."""
    comparison = summary.get('comparison', {})
    lines = []
    lines.append("## Wspólne wzorce i reguły")
    lines.append("")
    lines.append(f"### Wzorce wspólne dla wszystkich zbiorów: **{comparison.get('common_all_patterns', 0)}**")
    lines.append(f"### Reguły wspólne dla wszystkich zbiorów: **{comparison.get('common_all_rules', 0)}**")
    lines.append("")
    return "\n".join(lines)


def generate_common_patterns_section(patterns_df: pd.DataFrame) -> str:
    """Generuje sekcję o wspólnych wzorcach."""

    lines = []
    lines.append("## Wspólne wzorce")
    lines.append("")

    if len(patterns_df) == 0:
        lines.append("*Brak wspólnych wzorców dla wszystkich trzech zbiorów.*")
        lines.append("")
        return "\n".join(lines)

    lines.append(f"Znaleziono **{len(patterns_df)}** wzorców wspólnych dla wszystkich trzech zbiorów.")
    lines.append("")
    lines.append("### Top 15 wzorców (według średniego wsparcia)")
    lines.append("")
    lines.append("| # | Wzorzec | Śr. Support | CASE | K-emoCon | CEAP |")
    lines.append("|---|---------|-------------|------|----------|------|")

    for i, (_, row) in enumerate(patterns_df.head(15).iterrows(), 1):
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

    if len(patterns_df) > 15:
        lines.append(f"| ... | *{len(patterns_df) - 15} więcej wzorców* | | | | |")

    lines.append("")

    return "\n".join(lines)


def generate_common_rules_section(rules_df: pd.DataFrame) -> str:
    """Generuje sekcję o wspólnych regułach."""

    lines = []
    lines.append("## Wspólne reguły")
    lines.append("")

    if len(rules_df) == 0:
        lines.append("*Brak wspólnych reguł dla wszystkich trzech zbiorów.*")
        lines.append("")
        return "\n".join(lines)

    lines.append(f"Znaleziono **{len(rules_df)}** reguł wspólnych dla wszystkich trzech zbiorów.")
    lines.append("")
    lines.append("### Wszystkie wspólne reguły")
    lines.append("")

    for i, (_, row) in enumerate(rules_df.iterrows(), 1):
        rule = row.get('rule', 'N/A')
        avg_conf = row.get('avg_confidence', 'N/A')
        avg_sup = row.get('avg_support', 'N/A')

        if isinstance(avg_conf, float):
            avg_conf = f"{avg_conf:.3f}"
        if isinstance(avg_sup, float):
            avg_sup = f"{avg_sup:.3f}"

        lines.append(f"#### Reguła {i}")
        lines.append("")
        lines.append(f"```")
        lines.append(f"{rule}")
        lines.append(f"```")
        lines.append("")
        lines.append(f"- **Średnia ufność**: {avg_conf}")
        lines.append(f"- **Średnie wsparcie**: {avg_sup}")
        lines.append("")

        # Interpretacja
        lines.append(f"**Interpretacja**: {interpret_rule(rule)}")
        lines.append("")

    return "\n".join(lines)


def generate_conclusions_section(summary: dict, patterns_df: pd.DataFrame, rules_df: pd.DataFrame) -> str:
    """Generuje sekcję wniosków."""

    comparison = summary.get('comparison', {})
    common_patterns = comparison.get('common_all_patterns', 0)
    common_rules = comparison.get('common_all_rules', 0)

    lines = []
    lines.append("## Wnioski")
    lines.append("")

    # Główny wniosek
    if common_patterns > 0:
        lines.append("### ✅ Wzorce są uniwersalne")
        lines.append("")
        lines.append(f"**TAK** - znaleziono **{common_patterns}** wzorców wspólnych dla wszystkich trzech ")
        lines.append("zbiorów danych. Oznacza to, że pewne zależności między stanami emocjonalnymi ")
        lines.append("a sygnałami fizjologicznymi są uniwersalne i niezależne od:")
        lines.append("")
        lines.append("- Protokołu badawczego (ciągłe adnotacje vs samooceny)")
        lines.append("- Populacji badanych")
        lines.append("- Sprzętu pomiarowego")
        lines.append("")
    else:
        lines.append("### ❌ Brak uniwersalnych wzorców")
        lines.append("")
        lines.append("Nie znaleziono wzorców wspólnych dla wszystkich zbiorów. ")
        lines.append("Może to wynikać z:")
        lines.append("")
        lines.append("- Różnic w protokołach badawczych")
        lines.append("- Różnic w populacjach")
        lines.append("- Zbyt restrykcyjnych parametrów (minsup, minconf)")
        lines.append("")

    # Podsumowanie reguł
    if common_rules > 0:
        lines.append("### Reguły predykcyjne")
        lines.append("")
        lines.append(f"Znaleziono **{common_rules}** reguł predykcyjnych, które mogą być wykorzystane do:")
        lines.append("")
        lines.append("1. Automatycznego rozpoznawania stanów emocjonalnych")
        lines.append("2. Predykcji zmian emocjonalnych na podstawie sygnałów fizjologicznych")
        lines.append("3. Walidacji systemów affective computing")
        lines.append("")


    return "\n".join(lines)


def generate_readme():
    """Główna funkcja generująca README."""

    print("Generowanie README.md...")

    # Wczytaj dane
    summary, patterns_df, rules_df = load_experiment_data()

    # Generuj sekcje
    sections = []

    # Nagłówek
    sections.append("# Wyniki eksperymentu ARMADA")
    sections.append("")
    sections.append(f"*Wygenerowano: {datetime.now().strftime('%Y-%m-%d %H:%M')}*")
    sections.append("")
    sections.append("---")
    sections.append("")
    sections.append("## Cel eksperymentu")
    sections.append("")
    sections.append("Celem eksperymentu było sprawdzenie, czy wzorce temporalne odkryte przez algorytm ARMADA")
    sections.append("są **uniwersalne** - czyli czy powtarzają się w różnych zbiorach danych afektywnych")
    sections.append("pochodzących z różnych protokołów badawczych i populacji.")
    sections.append("")

    # Dodaj sekcje
    sections.append(generate_methodology_section(summary))
    sections.append(generate_datasets_section(summary))
    sections.append(generate_similarity_section(summary))
    sections.append(generate_common_patterns_section(patterns_df))
    sections.append(generate_common_rules_section(rules_df))
    sections.append(generate_conclusions_section(summary, patterns_df, rules_df))

    # Stopka
    sections.append("---")
    sections.append("")
    sections.append("## Pliki wynikowe")
    sections.append("")
    sections.append("- `experiment_summary.json` - podsumowanie w formacie JSON")
    sections.append("- `common_patterns_details.csv` - szczegóły wspólnych wzorców")
    sections.append("- `common_rules_details.csv` - szczegóły wspólnych reguł")
    sections.append("- `patterns_comparison.png` - wizualizacja porównania wzorców")
    sections.append("- `comparison_report.md` - szczegółowy raport porównawczy")
    sections.append("")

    # Zapisz README
    readme_content = "\n".join(sections)

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(readme_content)

    print(f"Zapisano: {OUTPUT_FILE}")
    print(f"Rozmiar: {len(readme_content)} znaków")

    return OUTPUT_FILE


if __name__ == "__main__":
    try:
        output_file = generate_readme()
        print(f"\n✅ README wygenerowany pomyślnie: {output_file}")
    except FileNotFoundError as e:
        print(f"\n❌ Błąd: {e}")
        print("Uruchom najpierw: python compare_datasets.py")
    except Exception as e:
        print(f"\n❌ Nieoczekiwany błąd: {e}")
        raise

