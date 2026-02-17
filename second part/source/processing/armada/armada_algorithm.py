#!/usr/bin/env python3
"""
ARMADA - Association Rule Mining Algorithm for Temporal Databases

Implementacja algorytmu ARMADA do odkrywania bogatszych reguł asocjacyjnych
z danych temporalnych (interval-based time series).

Oparte na:
Winarko, E., & Roddick, J. F. (2007). ARMADA–An algorithm for discovering
richer relative temporal association rules from interval-based data.
Data & Knowledge Engineering, 63(1), 76-90.

Relacje Allena (znormalizowane - 7 z 13):
- before (b): A kończy się przed rozpoczęciem B
- meets (m): A kończy się dokładnie gdy B się zaczyna
- overlaps (o): A zaczyna się przed B, ale kończy w trakcie B
- is-finished-by (fi): A zaczyna się przed B i kończy gdy B się kończy
- contains (c): A zaczyna się przed B i kończy po B
- equals (=): A i B mają te same czasy
- starts (s): A zaczyna się gdy B, ale kończy wcześniej
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set, Optional, Any
from collections import defaultdict
from pathlib import Path
import json
from copy import deepcopy

# Relacje Allena (znormalizowane)
ALLEN_RELATIONS = {
    'b': 'before',
    'm': 'meets',
    'o': 'overlaps',
    'fi': 'is-finished-by',
    'c': 'contains',
    '=': 'equals',
    's': 'starts'
}


@dataclass
class StateInterval:
    """Reprezentuje interwał stanu (b, s, f) gdzie b=start, s=state, f=end."""
    state: str
    start_time: float
    end_time: float

    def __hash__(self):
        return hash((self.state, self.start_time, self.end_time))

    def __eq__(self, other):
        if not isinstance(other, StateInterval):
            return False
        return (self.state == other.state and
                self.start_time == other.start_time and
                self.end_time == other.end_time)

    def __lt__(self, other):
        """Porządkowanie: start_time, end_time, state."""
        if self.start_time != other.start_time:
            return self.start_time < other.start_time
        if self.end_time != other.end_time:
            return self.end_time < other.end_time
        return self.state < other.state


@dataclass
class TemporalPattern:
    """
    Wzorzec czasowy definiowany jako para (s, M) gdzie:
    - s: mapowanie indeksów do stanów
    - M: macierz relacji między interwałami
    """
    states: List[str]  # Lista stanów w porządku
    relations_matrix: List[List[str]]  # Macierz relacji n x n
    support: float = 0.0
    support_count: int = 0

    @property
    def dim(self) -> int:
        """Wymiar wzorca (liczba interwałów)."""
        return len(self.states)

    def __hash__(self):
        # Konwertuj macierz do tuple dla hashowania
        matrix_tuple = tuple(tuple(row) for row in self.relations_matrix)
        return hash((tuple(self.states), matrix_tuple))

    def __eq__(self, other):
        if not isinstance(other, TemporalPattern):
            return False
        return (self.states == other.states and
                self.relations_matrix == other.relations_matrix)

    def to_string(self) -> str:
        """Konwertuje wzorzec do czytelnej postaci tekstowej."""
        if self.dim == 1:
            return f"<{self.states[0]}>"

        parts = [f"States: {self.states}"]
        parts.append("Relations:")
        for i in range(self.dim):
            row = []
            for j in range(self.dim):
                if i < j:
                    row.append(self.relations_matrix[i][j])
                elif i == j:
                    row.append("=")
                else:
                    row.append("-")
            parts.append(f"  {row}")
        return "\n".join(parts)

    def get_relation_description(self) -> str:
        """Zwraca opis relacji w formie czytelnej."""
        if self.dim == 1:
            return f"({self.states[0]})"

        descriptions = []
        for i in range(self.dim):
            for j in range(i + 1, self.dim):
                rel = self.relations_matrix[i][j]
                rel_name = ALLEN_RELATIONS.get(rel, rel)
                descriptions.append(f"{self.states[i]} {rel_name} {self.states[j]}")

        return " AND ".join(descriptions)


@dataclass
class TemporalRule:
    """Reguła temporalna X => Y gdzie X jest podwzorcem Y."""
    antecedent: TemporalPattern
    consequent: TemporalPattern
    confidence: float
    support: float

    def to_string(self) -> str:
        return f"{self.antecedent.get_relation_description()} => {self.consequent.get_relation_description()}"


@dataclass
class IndexElement:
    """Element indeksu dla algorytmu ARMADA."""
    client_id: str
    intervals: List[StateInterval]  # a_intv - lista interwałów tworzących wzorzec
    pos: int  # Pozycja pierwszego wystąpienia stanu stem w sekwencji klienta


class ClientSequence:
    """Sekwencja klienta - seria interwałów stanów."""

    def __init__(self, client_id: str, intervals: List[StateInterval] = None):
        self.client_id = client_id
        self.intervals = sorted(intervals or [])

    def add_interval(self, interval: StateInterval):
        self.intervals.append(interval)
        self.intervals.sort()

    def __len__(self):
        return len(self.intervals)

    def __getitem__(self, idx):
        return self.intervals[idx]


class ARMADA:
    """
    Implementacja algorytmu ARMADA do odkrywania wzorców czasowych.

    Algorytm działa w trzech krokach:
    1. Wczytanie bazy danych i znalezienie częstych 1-wzorców
    2. Konstrukcja zbiorów indeksowych
    3. Odkrywanie wzorców metodą find-then-index
    """

    def __init__(self, minsup: float = 0.1, minconf: float = 0.5, maxgap: float = -1, max_pattern_size: int = 5):
        """
        Args:
            minsup: Minimalne wsparcie (0-1)
            minconf: Minimalna ufność dla reguł (0-1)
            maxgap: Maksymalna przerwa czasowa między interwałami (-1 = brak ograniczenia)
            max_pattern_size: Maksymalny rozmiar wzorca (limit głębokości rekurencji)
        """
        self.minsup = minsup
        self.minconf = minconf
        self.maxgap = maxgap
        self.max_pattern_size = max_pattern_size

        # Baza danych w pamięci
        self.client_sequences: Dict[str, ClientSequence] = {}
        self.num_clients = 0

        # Wyniki
        self.frequent_patterns: List[TemporalPattern] = []
        self.temporal_rules: List[TemporalRule] = []

        # Cache dla wsparcia stanów
        self._state_support: Dict[str, int] = defaultdict(int)
        self._frequent_states: Set[str] = set()

        # Liczniki do monitorowania postępu
        self._patterns_found = 0
        self._depth_stats = defaultdict(int)

    def load_data(self, filepath: Path) -> None:
        """
        Wczytuje dane z pliku w formacie ARMADA.

        Format pliku:
        SEQUENCE client_id
        state start_time end_time
        ...
        """
        self.client_sequences = {}
        current_client = None

        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                if line.startswith("SEQUENCE"):
                    client_id = line.split(" ", 1)[1]
                    current_client = ClientSequence(client_id)
                    self.client_sequences[client_id] = current_client
                else:
                    parts = line.split()
                    if len(parts) >= 3 and current_client is not None:
                        state = parts[0]
                        start_time = float(parts[1])
                        end_time = float(parts[2])
                        interval = StateInterval(state, start_time, end_time)
                        current_client.add_interval(interval)

        self.num_clients = len(self.client_sequences)
        print(f"Wczytano {self.num_clients} sekwencji klientów")

    def load_from_dataframe(self, df: pd.DataFrame) -> None:
        """
        Wczytuje dane z DataFrame.

        Wymagane kolumny: client_id, state, start_time, end_time
        """
        self.client_sequences = {}

        for client_id in df['client_id'].unique():
            client_df = df[df['client_id'] == client_id].sort_values(['start_time', 'end_time'])
            client_seq = ClientSequence(client_id)

            for _, row in client_df.iterrows():
                interval = StateInterval(
                    state=row['state'],
                    start_time=row['start_time'],
                    end_time=row['end_time']
                )
                client_seq.add_interval(interval)

            self.client_sequences[client_id] = client_seq

        self.num_clients = len(self.client_sequences)
        print(f"Wczytano {self.num_clients} sekwencji klientów")

    def _compute_allen_relation(self, a: StateInterval, b: StateInterval) -> str:
        """
        Oblicza znormalizowaną relację Allena między dwoma interwałami.
        Zakłada że a < b w porządku (a.start <= b.start).
        """
        # a musi być przed lub równo z b w sensie start_time
        if a.start_time > b.start_time:
            a, b = b, a

        a_start, a_end = a.start_time, a.end_time
        b_start, b_end = b.start_time, b.end_time

        # Relacje gdy start_time różne
        if a_start < b_start:
            if a_end < b_start:
                return 'b'  # before
            elif a_end == b_start:
                return 'm'  # meets
            elif a_end < b_end:
                return 'o'  # overlaps
            elif a_end == b_end:
                return 'fi'  # is-finished-by
            else:  # a_end > b_end
                return 'c'  # contains

        # Relacje gdy start_time równe
        else:  # a_start == b_start
            if a_end == b_end:
                return '='  # equals
            elif a_end < b_end:
                return 's'  # starts
            else:
                return 'fi'  # is-finished-by (odwrotność starts)

        return '?'  # nieznana relacja

    def _check_gap_constraint(self, intervals: List[StateInterval]) -> bool:
        """Sprawdza czy interwały spełniają ograniczenie maxgap."""
        if self.maxgap < 0:
            return True

        for i in range(len(intervals)):
            for j in range(i + 1, len(intervals)):
                gap = intervals[j].start_time - intervals[i].end_time
                if gap > self.maxgap:
                    return False

        return True

    def _find_frequent_1_patterns(self) -> Dict[str, int]:
        """
        Krok 1: Znajdź wszystkie częste stany (1-wzorce).

        Returns:
            Słownik {stan: liczba_wsparcia}
        """
        state_support = defaultdict(int)

        for client_seq in self.client_sequences.values():
            # Zlicz unikalne stany w sekwencji klienta
            client_states = set()
            for interval in client_seq.intervals:
                client_states.add(interval.state)

            for state in client_states:
                state_support[state] += 1

        # Filtruj stany z wystarczającym wsparciem
        min_support_count = int(self.minsup * self.num_clients)
        self._state_support = {
            state: count
            for state, count in state_support.items()
            if count >= min_support_count
        }
        self._frequent_states = set(self._state_support.keys())

        # Twórz 1-wzorce
        for state, count in self._state_support.items():
            pattern = TemporalPattern(
                states=[state],
                relations_matrix=[['=']],
                support=count / self.num_clients,
                support_count=count
            )
            self.frequent_patterns.append(pattern)

        print(f"Znaleziono {len(self._frequent_states)} częstych 1-wzorców")
        return self._state_support

    def _create_index_set(
        self,
        stem: str,
        prefix_pattern: Optional[TemporalPattern],
        range_set: List[IndexElement]
    ) -> List[IndexElement]:
        """
        Krok 2: Tworzy zbiór indeksowy dla wzorca utworzonego z prefix i stem.

        Args:
            stem: Stan do dodania
            prefix_pattern: Wzorzec prefiksowy (None dla 1-wzorców)
            range_set: Zbiór sekwencji do przeszukania

        Returns:
            Zbiór indeksowy dla nowego wzorca
        """
        index_set = []

        for elem in range_set:
            client_seq = self.client_sequences[elem.client_id]
            start_pos = elem.pos if prefix_pattern else -1

            # Szukaj pierwszego wystąpienia stem po pozycji start_pos
            for pos in range(start_pos + 1, len(client_seq)):
                interval = client_seq[pos]

                if interval.state == stem:
                    # Sprawdź ograniczenie maxgap
                    new_intervals = elem.intervals + [interval]
                    if self._check_gap_constraint(new_intervals):
                        new_elem = IndexElement(
                            client_id=elem.client_id,
                            intervals=new_intervals,
                            pos=pos
                        )
                        index_set.append(new_elem)
                    break  # Bierzemy tylko pierwsze wystąpienie

        return index_set

    def _create_pattern_from_intervals(
        self,
        intervals: List[StateInterval]
    ) -> TemporalPattern:
        """Tworzy wzorzec z listy interwałów."""
        n = len(intervals)
        states = [iv.state for iv in intervals]

        # Buduj macierz relacji
        relations = [['=' for _ in range(n)] for _ in range(n)]

        for i in range(n):
            for j in range(i + 1, n):
                rel = self._compute_allen_relation(intervals[i], intervals[j])
                relations[i][j] = rel
                # Dolna część macierzy zostaje pusta (-)

        return TemporalPattern(states=states, relations_matrix=relations)

    def _mine_index_set(
        self,
        prefix_pattern: TemporalPattern,
        index_set: List[IndexElement],
        depth: int = 1
    ) -> None:
        """
        Krok 3: Odkrywa wzorce z zestawu indeksowego metodą rekurencyjną.

        Args:
            prefix_pattern: Wzorzec prefiksowy
            index_set: Zbiór indeksowy do przeszukania
            depth: Aktualna głębokość rekurencji
        """
        # Sprawdź limit głębokości
        if depth >= self.max_pattern_size:
            return

        self._depth_stats[depth] += 1

        # Zlicz potencjalne stemy (stany)
        stem_support = defaultdict(int)
        stem_patterns = defaultdict(list)  # stem -> lista interwałów dla każdego klienta

        for elem in index_set:
            client_seq = self.client_sequences[elem.client_id]
            counted_stems = set()  # Zlicz każdy stem raz per klient

            for pos in range(elem.pos + 1, len(client_seq)):
                interval = client_seq[pos]
                state = interval.state

                if state in self._frequent_states and state not in counted_stems:
                    # Sprawdź ograniczenie maxgap
                    new_intervals = elem.intervals + [interval]
                    if self._check_gap_constraint(new_intervals):
                        counted_stems.add(state)
                        stem_support[state] += 1
                        stem_patterns[state].append((elem, interval, pos))

        # Znajdź częste stemy
        min_support_count = int(self.minsup * self.num_clients)
        frequent_stems = {
            stem: count
            for stem, count in stem_support.items()
            if count >= min_support_count
        }

        # Dla każdego częstego stem, utwórz nowy wzorzec i kontynuuj rekurencyjnie
        for stem, count in frequent_stems.items():
            # Znajdź wszystkie wystąpienia tego stemu
            stem_index_set = []
            seen_clients = set()

            for elem, interval, pos in stem_patterns[stem]:
                if elem.client_id not in seen_clients:
                    seen_clients.add(elem.client_id)
                    new_intervals = elem.intervals + [interval]

                    stem_index_set.append(IndexElement(
                        client_id=elem.client_id,
                        intervals=new_intervals,
                        pos=pos
                    ))

            if stem_index_set:
                # Dodaj wzorzec do wyników
                sample_pattern = self._create_pattern_from_intervals(stem_index_set[0].intervals)
                sample_pattern.support_count = count
                sample_pattern.support = count / self.num_clients
                self.frequent_patterns.append(sample_pattern)

                self._patterns_found += 1
                if self._patterns_found % 100 == 0:
                    print(f"  Znaleziono {self._patterns_found} wzorców (głębokość {depth+1})...")

                # Rekurencyjnie szukaj dłuższych wzorców
                self._mine_index_set(sample_pattern, stem_index_set, depth + 1)

    def mine_patterns(self) -> List[TemporalPattern]:
        """
        Główna funkcja odkrywania wzorców.

        Returns:
            Lista częstych wzorców temporalnych
        """
        print("=" * 60)
        print("ARMADA - Mining Temporal Patterns")
        print("=" * 60)
        print(f"Parametry: minsup={self.minsup}, minconf={self.minconf}, maxgap={self.maxgap}, max_pattern_size={self.max_pattern_size}")

        self.frequent_patterns = []
        self._patterns_found = 0
        self._depth_stats = defaultdict(int)

        # Krok 1: Znajdź częste 1-wzorce
        print("\nKrok 1: Szukanie częstych 1-wzorców...")
        self._find_frequent_1_patterns()

        # Krok 2-3: Dla każdego częstego stanu, utwórz indeks i kopaj
        print("\nKrok 2-3: Odkrywanie wzorców n-wymiarowych...")

        for idx, state in enumerate(sorted(self._frequent_states)):
            print(f"  Przetwarzanie stanu {idx+1}/{len(self._frequent_states)}: {state}")

            # Utwórz początkowy wzorzec
            pattern = TemporalPattern(
                states=[state],
                relations_matrix=[['=']],
                support=self._state_support[state] / self.num_clients,
                support_count=self._state_support[state]
            )

            # Utwórz początkowy zbiór indeksowy
            initial_index_set = []
            for client_id, client_seq in self.client_sequences.items():
                for pos, interval in enumerate(client_seq.intervals):
                    if interval.state == state:
                        initial_index_set.append(IndexElement(
                            client_id=client_id,
                            intervals=[interval],
                            pos=pos
                        ))
                        break  # Tylko pierwsze wystąpienie

            # Rekurencyjnie odkrywaj wzorce
            self._mine_index_set(pattern, initial_index_set, depth=1)

        # Usuń duplikaty
        unique_patterns = []
        seen = set()
        for p in self.frequent_patterns:
            key = (tuple(p.states), tuple(tuple(row) for row in p.relations_matrix))
            if key not in seen:
                seen.add(key)
                unique_patterns.append(p)

        self.frequent_patterns = unique_patterns

        print(f"\nZnaleziono {len(self.frequent_patterns)} unikalnych wzorców")
        return self.frequent_patterns

    def generate_rules(self) -> List[TemporalRule]:
        """
        Generuje reguły temporalne z częstych wzorców.

        Dla każdego częstego n-wzorca Y (n > 1), znajdujemy wszystkie
        podwzorce X i generujemy regułę X => Y jeśli confidence >= minconf.

        Returns:
            Lista reguł temporalnych
        """
        print("\nGenerowanie reguł temporalnych...")
        self.temporal_rules = []

        # Mapowanie wzorców dla szybkiego wyszukiwania
        pattern_support = {}
        for p in self.frequent_patterns:
            key = (tuple(p.states), tuple(tuple(row) for row in p.relations_matrix))
            pattern_support[key] = p.support

        for pattern in self.frequent_patterns:
            if pattern.dim <= 1:
                continue

            # Generuj podwzorce (usuwając po jednym stanie od końca)
            for i in range(1, pattern.dim):
                # Podwzorzec z pierwszych i stanów
                sub_states = pattern.states[:i]
                sub_matrix = [row[:i] for row in pattern.relations_matrix[:i]]

                sub_key = (tuple(sub_states), tuple(tuple(row) for row in sub_matrix))

                if sub_key in pattern_support:
                    sub_support = pattern_support[sub_key]
                    confidence = pattern.support / sub_support if sub_support > 0 else 0

                    if confidence >= self.minconf:
                        antecedent = TemporalPattern(
                            states=sub_states,
                            relations_matrix=sub_matrix,
                            support=sub_support
                        )

                        rule = TemporalRule(
                            antecedent=antecedent,
                            consequent=pattern,
                            confidence=confidence,
                            support=pattern.support
                        )
                        self.temporal_rules.append(rule)

        print(f"Wygenerowano {len(self.temporal_rules)} reguł")
        return self.temporal_rules

    def run(self, filepath: Optional[Path] = None, df: Optional[pd.DataFrame] = None) -> Tuple[List[TemporalPattern], List[TemporalRule]]:
        """
        Uruchamia pełny pipeline ARMADA.

        Args:
            filepath: Ścieżka do pliku z danymi
            df: DataFrame z danymi (alternatywnie)

        Returns:
            Tuple (lista wzorców, lista reguł)
        """
        if filepath:
            self.load_data(filepath)
        elif df is not None:
            self.load_from_dataframe(df)
        else:
            raise ValueError("Musisz podać filepath lub df")

        patterns = self.mine_patterns()
        rules = self.generate_rules()

        return patterns, rules

    def save_results(self, output_dir: Path) -> None:
        """Zapisuje wyniki do plików."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Zapisz wzorce
        patterns_data = []
        for p in self.frequent_patterns:
            patterns_data.append({
                'dimension': p.dim,
                'states': p.states,
                'relations': p.relations_matrix,
                'support': p.support,
                'support_count': p.support_count,
                'description': p.get_relation_description()
            })

        with open(output_dir / 'patterns.json', 'w') as f:
            json.dump(patterns_data, f, indent=2)

        # Zapisz wzorce w formacie CSV
        patterns_df = pd.DataFrame([
            {
                'dimension': p['dimension'],
                'description': p['description'],
                'support': p['support'],
                'support_count': p['support_count']
            }
            for p in patterns_data
        ])
        patterns_df.to_csv(output_dir / 'patterns.csv', index=False)

        # Zapisz reguły - tylko top 1000 według confidence * support
        rules_data = []
        for r in self.temporal_rules:
            rules_data.append({
                'antecedent': r.antecedent.get_relation_description(),
                'consequent': r.consequent.get_relation_description(),
                'confidence': r.confidence,
                'support': r.support,
                'score': r.confidence * r.support  # metryka do sortowania
            })

        # Sortuj według score i weź top 1000
        rules_data = sorted(rules_data, key=lambda x: -x['score'])[:1000]

        # Usuń score przed zapisem
        for r in rules_data:
            del r['score']

        with open(output_dir / 'rules.json', 'w') as f:
            json.dump(rules_data, f, indent=2)

        rules_df = pd.DataFrame(rules_data)
        rules_df.to_csv(output_dir / 'rules.csv', index=False)

        print(f"Wyniki zapisano w {output_dir}")
        print(f"  Wzorce: {len(patterns_data)}")
        print(f"  Reguły (top 1000): {len(rules_data)}")

    def print_summary(self) -> None:
        """Wyświetla podsumowanie wyników."""
        print("\n" + "=" * 60)
        print("PODSUMOWANIE WYNIKÓW")
        print("=" * 60)

        # Grupuj wzorce według wymiaru
        by_dim = defaultdict(list)
        for p in self.frequent_patterns:
            by_dim[p.dim].append(p)

        print("\nWzorce według wymiaru:")
        for dim in sorted(by_dim.keys()):
            patterns = by_dim[dim]
            print(f"  {dim}-wzorce: {len(patterns)}")

        print(f"\nŁącznie wzorców: {len(self.frequent_patterns)}")
        print(f"Łącznie reguł: {len(self.temporal_rules)}")

        # Top 10 wzorców według wsparcia
        print("\nTop 10 wzorców (według wsparcia):")
        sorted_patterns = sorted(self.frequent_patterns, key=lambda x: -x.support)
        for i, p in enumerate(sorted_patterns[:10]):
            print(f"  {i+1}. {p.get_relation_description()} (sup={p.support:.3f})")

        # Top 10 reguł według ufności
        if self.temporal_rules:
            print("\nTop 10 reguł (według ufności):")
            sorted_rules = sorted(self.temporal_rules, key=lambda x: -x.confidence)
            for i, r in enumerate(sorted_rules[:10]):
                print(f"  {i+1}. {r.to_string()} (conf={r.confidence:.3f}, sup={r.support:.3f})")


def main():
    """Przykład użycia algorytmu ARMADA."""
    # Ścieżki
    BASE_DIR = Path(__file__).parent.parent.parent.parent
    DATA_DIR = BASE_DIR / "data" / "armada_ready"
    OUTPUT_DIR = BASE_DIR / "data" / "armada_results"

    # Uruchom ARMADA na danych
    # max_pattern_size=4 oznacza wzorce do 4 stanów (ogranicza eksplozję kombinatoryczną)
    armada = ARMADA(minsup=0.4, minconf=0.5, maxgap=30, max_pattern_size=4)

    # Wczytaj dane - użyj mniejszego zbioru do testu
    data_file = DATA_DIR / "armada_sequences_ceap.txt"
    if data_file.exists():
        patterns, rules = armada.run(filepath=data_file)
        armada.print_summary()
        armada.save_results(OUTPUT_DIR / "ceap_test")
    else:
        print(f"Plik {data_file} nie istnieje!")


if __name__ == "__main__":
    main()

