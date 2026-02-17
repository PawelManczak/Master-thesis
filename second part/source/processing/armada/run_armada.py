#!/usr/bin/env python3
"""
Uruchomienie algorytmu ARMADA na danych emocjonalnych.

Ten skrypt uruchamia algorytm ARMADA na przygotowanych danych z trzech zbiorów:
- CASE
- K-emoCon
- CEAP

Pozwala na różne konfiguracje parametrów i analizę wyników.
"""

import sys
from pathlib import Path
import pandas as pd
import argparse
import json
from datetime import datetime

# Dodaj ścieżkę do modułów
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

from armada_algorithm import ARMADA, TemporalPattern, TemporalRule


def run_armada_analysis(
    data_file: Path,
    output_dir: Path,
    minsup: float = 0.3,
    minconf: float = 0.5,
    maxgap: float = 20,
    max_pattern_size: int = 4,
    name: str = "default"
) -> dict:
    """
    Uruchamia analizę ARMADA z danymi parametrami.

    Args:
        data_file: Ścieżka do pliku z danymi
        output_dir: Katalog na wyniki
        minsup: Minimalne wsparcie
        minconf: Minimalna ufność
        maxgap: Maksymalna przerwa czasowa
        max_pattern_size: Maksymalny rozmiar wzorca
        name: Nazwa eksperymentu

    Returns:
        Słownik z wynikami
    """
    print(f"\n{'='*60}")
    print(f"ANALIZA: {name}")
    print(f"{'='*60}")
    print(f"Plik danych: {data_file}")
    print(f"Parametry: minsup={minsup}, minconf={minconf}, maxgap={maxgap}, max_pattern_size={max_pattern_size}")

    # Utwórz katalog wyjściowy
    experiment_dir = output_dir / name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Uruchom ARMADA
    armada = ARMADA(minsup=minsup, minconf=minconf, maxgap=maxgap, max_pattern_size=max_pattern_size)
    patterns, rules = armada.run(filepath=data_file)

    # Wyświetl podsumowanie
    armada.print_summary()

    # Zapisz wyniki
    armada.save_results(experiment_dir)

    # Przygotuj statystyki
    results = {
        'name': name,
        'data_file': str(data_file),
        'minsup': minsup,
        'minconf': minconf,
        'maxgap': maxgap,
        'num_clients': armada.num_clients,
        'num_patterns': len(patterns),
        'num_rules': len(rules),
        'patterns_by_dim': {},
        'timestamp': datetime.now().isoformat()
    }

    # Zlicz wzorce według wymiaru
    for p in patterns:
        dim = p.dim
        if dim not in results['patterns_by_dim']:
            results['patterns_by_dim'][dim] = 0
        results['patterns_by_dim'][dim] += 1

    # Zapisz metadane eksperymentu
    with open(experiment_dir / 'experiment_info.json', 'w') as f:
        json.dump(results, f, indent=2)

    return results


def analyze_emotional_patterns(armada: ARMADA) -> dict:
    """
    Analizuje wzorce związane z emocjami (arousal, valence).

    Returns:
        Słownik z analizą emocjonalną
    """
    emotional_patterns = []
    physio_patterns = []
    mixed_patterns = []

    emotional_states = {'arousal', 'valence'}
    physio_states = {'hr', 'eda', 'temp', 'hrv_rmssd', 'hrv_sdnn'}

    for pattern in armada.frequent_patterns:
        states_set = set()
        for state in pattern.states:
            # Wyodrębnij bazową nazwę stanu (np. arousal z arousal_high)
            base_state = state.rsplit('_', 1)[0]
            states_set.add(base_state)

        has_emotional = bool(states_set & emotional_states)
        has_physio = bool(states_set & physio_states)

        if has_emotional and has_physio:
            mixed_patterns.append(pattern)
        elif has_emotional:
            emotional_patterns.append(pattern)
        elif has_physio:
            physio_patterns.append(pattern)

    return {
        'emotional_only': emotional_patterns,
        'physio_only': physio_patterns,
        'mixed': mixed_patterns
    }


def find_emotion_physio_rules(armada: ARMADA) -> list:
    """
    Znajduje reguły łączące stany fizjologiczne z emocjonalnymi.

    Szuka reguł typu: physio => emotion (predykcja emocji na podstawie fizjologii)
    """
    emotional_states = {'arousal', 'valence'}
    interesting_rules = []

    for rule in armada.temporal_rules:
        # Sprawdź czy antecedent zawiera fizjologię
        ante_states = set(s.rsplit('_', 1)[0] for s in rule.antecedent.states)
        cons_states = set(s.rsplit('_', 1)[0] for s in rule.consequent.states)

        # Reguły gdzie fizjologia przewiduje emocje
        if not (ante_states & emotional_states) and (cons_states & emotional_states):
            interesting_rules.append(rule)

    return interesting_rules


def main():
    parser = argparse.ArgumentParser(description='Uruchom algorytm ARMADA na danych emocjonalnych')
    parser.add_argument('--minsup', type=float, default=0.4, help='Minimalne wsparcie (0-1)')
    parser.add_argument('--minconf', type=float, default=0.5, help='Minimalna ufność (0-1)')
    parser.add_argument('--maxgap', type=float, default=30, help='Maksymalna przerwa czasowa (sekundy)')
    parser.add_argument('--max_pattern_size', type=int, default=4, help='Maksymalny rozmiar wzorca')
    parser.add_argument('--dataset', type=str, default='ceap',
                        choices=['all', 'case', 'k_emocon', 'ceap'],
                        help='Zbiór danych do analizy')
    args = parser.parse_args()

    # Ścieżki
    BASE_DIR = Path(__file__).parent.parent.parent.parent
    DATA_DIR = BASE_DIR / "data" / "armada_ready"
    OUTPUT_DIR = BASE_DIR / "data" / "armada_results"

    # Wybierz plik danych
    if args.dataset == 'all':
        data_file = DATA_DIR / "armada_sequences.txt"
        name = f"all_minsup{args.minsup}_maxgap{args.maxgap}_size{args.max_pattern_size}"
    else:
        data_file = DATA_DIR / f"armada_sequences_{args.dataset}.txt"
        name = f"{args.dataset}_minsup{args.minsup}_maxgap{args.maxgap}_size{args.max_pattern_size}"

    if not data_file.exists():
        print(f"Błąd: Plik {data_file} nie istnieje!")
        print("Uruchom najpierw prepare_armada_data.py i validate_armada_data.py")
        return

    # Uruchom analizę
    results = run_armada_analysis(
        data_file=data_file,
        output_dir=OUTPUT_DIR,
        minsup=args.minsup,
        minconf=args.minconf,
        maxgap=args.maxgap,
        max_pattern_size=args.max_pattern_size,
        name=name
    )

    print("\n" + "=" * 60)
    print("ANALIZA ZAKOŃCZONA")
    print("=" * 60)
    print(f"Znaleziono {results['num_patterns']} wzorców")
    print(f"Wygenerowano {results['num_rules']} reguł")
    print(f"Wyniki zapisano w: {OUTPUT_DIR / name}")


def run_multiple_experiments():
    """Uruchamia serię eksperymentów z różnymi parametrami."""
    BASE_DIR = Path(__file__).parent.parent.parent.parent
    DATA_DIR = BASE_DIR / "data" / "armada_ready"
    OUTPUT_DIR = BASE_DIR / "data" / "armada_results"

    experiments = [
        # Różne minsup
        {'minsup': 0.5, 'minconf': 0.5, 'maxgap': 60, 'dataset': 'all'},
        {'minsup': 0.3, 'minconf': 0.5, 'maxgap': 60, 'dataset': 'all'},
        {'minsup': 0.2, 'minconf': 0.5, 'maxgap': 60, 'dataset': 'all'},

        # Różne maxgap
        {'minsup': 0.3, 'minconf': 0.5, 'maxgap': 30, 'dataset': 'all'},
        {'minsup': 0.3, 'minconf': 0.5, 'maxgap': 120, 'dataset': 'all'},

        # Per dataset
        {'minsup': 0.3, 'minconf': 0.5, 'maxgap': 60, 'dataset': 'case'},
        {'minsup': 0.3, 'minconf': 0.5, 'maxgap': 60, 'dataset': 'k_emocon'},
        {'minsup': 0.3, 'minconf': 0.5, 'maxgap': 60, 'dataset': 'ceap'},
    ]

    all_results = []

    for exp in experiments:
        dataset = exp['dataset']
        if dataset == 'all':
            data_file = DATA_DIR / "armada_sequences.txt"
        else:
            data_file = DATA_DIR / f"armada_sequences_{dataset}.txt"

        if not data_file.exists():
            print(f"Pominięto: {data_file} nie istnieje")
            continue

        name = f"{dataset}_minsup{exp['minsup']}_maxgap{exp['maxgap']}"

        try:
            results = run_armada_analysis(
                data_file=data_file,
                output_dir=OUTPUT_DIR,
                minsup=exp['minsup'],
                minconf=exp['minconf'],
                maxgap=exp['maxgap'],
                name=name
            )
            all_results.append(results)
        except Exception as e:
            print(f"Błąd w eksperymencie {name}: {e}")

    # Zapisz podsumowanie wszystkich eksperymentów
    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv(OUTPUT_DIR / "experiments_summary.csv", index=False)
    print(f"\nPodsumowanie zapisano w: {OUTPUT_DIR / 'experiments_summary.csv'}")

    return all_results


if __name__ == "__main__":
    # Sprawdź czy uruchomić pojedynczy eksperyment czy serię
    if len(sys.argv) > 1 and sys.argv[1] == '--batch':
        run_multiple_experiments()
    else:
        main()

