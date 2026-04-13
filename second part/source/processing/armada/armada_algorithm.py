#!/usr/bin/env python3
"""
ARMADA - Association Rule Mining Algorithm for Temporal Databases

Implementation of the ARMADA algorithm for discovering richer association rules
from temporal data (interval-based time series).

Based on:
Winarko, E., & Roddick, J. F. (2007). ARMADA–An algorithm for discovering
richer relative temporal association rules from interval-based data.
Data & Knowledge Engineering, 63(1), 76-90.

Allen's relations (normalized - 7 out of 13):
- before (b): A ends before B starts
- meets (m): A ends exactly when B starts
- overlaps (o): A starts before B, but ends during B
- is-finished-by (fi): A starts before B and ends when B ends
- contains (c): A starts before B and ends after B
- equals (=): A and B have the same times
- starts (s): A starts when B starts, but ends earlier
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set, Optional, Any
from collections import defaultdict
from pathlib import Path
import json
from copy import deepcopy

# Allen's relations (normalized)
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
    """Represents a state interval (b, s, f) where b=start, s=state, f=end."""
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
        """Ordering: start_time, end_time, state."""
        if self.start_time != other.start_time:
            return self.start_time < other.start_time
        if self.end_time != other.end_time:
            return self.end_time < other.end_time
        return self.state < other.state


@dataclass
class TemporalPattern:
    """
    Temporal pattern defined as a pair (s, M) where:
    - s: index to states mapping
    - M: interval relation matrix
    """
    states: List[str]  # List of states in order
    relations_matrix: List[List[str]]  # n x n relation matrix
    support: float = 0.0
    support_count: int = 0

    @property
    def dim(self) -> int:
        """Pattern dimension (number of intervals)."""
        return len(self.states)

    def __hash__(self):
        # Convert matrix to tuple for hashing
        matrix_tuple = tuple(tuple(row) for row in self.relations_matrix)
        return hash((tuple(self.states), matrix_tuple))

    def __eq__(self, other):
        if not isinstance(other, TemporalPattern):
            return False
        return (self.states == other.states and
                self.relations_matrix == other.relations_matrix)

    def to_string(self) -> str:
        """Converts pattern to human-readable text."""
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
        """Returns relation description in human-readable form."""
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
    """Temporal rule X => Y where X is a subpattern of Y."""
    antecedent: TemporalPattern
    consequent: TemporalPattern
    confidence: float
    support: float
    lift: float = 0.0

    def to_string(self) -> str:
        return f"{self.antecedent.get_relation_description()} => {self.consequent.get_relation_description()}"


@dataclass
class IndexElement:
    """Index element for ARMADA algorithm."""
    client_id: str
    intervals: List[StateInterval]  # a_intv - list of intervals forming the pattern
    pos: int  # Position of the first occurrence of stem state in client sequence


class ClientSequence:
    """Client sequence - series of state intervals."""

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
    Implementation of the ARMADA algorithm for discovering temporal patterns.

    The algorithm operates in three steps:
    1. Read database and find frequent 1-patterns
    2. Construct index sets
    3. Discover patterns using find-then-index method
    """

    def __init__(self, minsup: float = 0.1, minconf: float = 0.5, maxgap: float = -1, max_pattern_size: int = 5):
        """
        Args:
            minsup: Minimum support (0-1)
            minconf: Minimum rule confidence (0-1)
            maxgap: Maximum time gap between intervals (-1 = no limit)
            max_pattern_size: Maximum pattern size (recursion depth limit)
        """
        self.minsup = minsup
        self.minconf = minconf
        self.maxgap = maxgap
        self.max_pattern_size = max_pattern_size

        # In-memory database
        self.client_sequences: Dict[str, ClientSequence] = {}
        self.num_clients = 0

        # Results
        self.frequent_patterns: List[TemporalPattern] = []
        self.temporal_rules: List[TemporalRule] = []

        # Cache for state support
        self._state_support: Dict[str, int] = defaultdict(int)
        self._frequent_states: Set[str] = set()

        # Counters for progress monitoring
        self._patterns_found = 0
        self._depth_stats = defaultdict(int)

    def load_data(self, filepath: Path) -> None:
        """
        Loads data from file in ARMADA format.

        File format:
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
        print(f"Loaded {self.num_clients} client sequences")

    def load_from_dataframe(self, df: pd.DataFrame) -> None:
        """
        Loads data from DataFrame.

        Required columns: client_id, state, start_time, end_time
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
        print(f"Loaded {self.num_clients} client sequences")

    def _compute_allen_relation(self, a: StateInterval, b: StateInterval) -> str:
        """
        Computes normalized Allen relation between two intervals.
        Assumes a < b in order (a.start <= b.start).
        """
        # a must be before or equal to b in terms of start_time
        if a.start_time > b.start_time:
            a, b = b, a

        a_start, a_end = a.start_time, a.end_time
        b_start, b_end = b.start_time, b.end_time

        # Relations when start_time differs
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

        # Relations when start_time equals
        else:  # a_start == b_start
            if a_end == b_end:
                return '='  # equals
            elif a_end < b_end:
                return 's'  # starts
            else:
                return 'fi'  # is-finished-by (inverse of starts)

        return '?'  # unknown relation

    def _check_gap_constraint(self, intervals: List[StateInterval]) -> bool:
        """Checks if intervals satisfy maxgap constraint."""
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
        Step 1: Find all frequent states (1-patterns).

        Returns:
            Dictionary {state: support_count}
        """
        state_support = defaultdict(int)

        for client_seq in self.client_sequences.values():
            # Count unique states in client sequence
            client_states = set()
            for interval in client_seq.intervals:
                client_states.add(interval.state)

            for state in client_states:
                state_support[state] += 1

        # Filter states with sufficient support
        min_support_count = int(self.minsup * self.num_clients)
        self._state_support = {
            state: count
            for state, count in state_support.items()
            if count >= min_support_count
        }
        self._frequent_states = set(self._state_support.keys())

        # Create 1-patterns
        for state, count in self._state_support.items():
            pattern = TemporalPattern(
                states=[state],
                relations_matrix=[['=']],
                support=count / self.num_clients,
                support_count=count
            )
            self.frequent_patterns.append(pattern)

        print(f"Found {len(self._frequent_states)} frequent 1-patterns")
        return self._state_support

    def _create_index_set(
        self,
        stem: str,
        prefix_pattern: Optional[TemporalPattern],
        range_set: List[IndexElement]
    ) -> List[IndexElement]:
        """
        Step 2: Creates index set for pattern formed from prefix and stem.

        Args:
            stem: State to add
            prefix_pattern: Prefix pattern (None for 1-patterns)
            range_set: Set of sequences to search in

        Returns:
            Index set for new pattern
        """
        index_set = []

        for elem in range_set:
            client_seq = self.client_sequences[elem.client_id]
            start_pos = elem.pos if prefix_pattern else -1

            # Search for first occurrence of stem after pos start_pos
            for pos in range(start_pos + 1, len(client_seq)):
                interval = client_seq[pos]

                if interval.state == stem:
                    # Check maxgap constraint
                    new_intervals = elem.intervals + [interval]
                    if self._check_gap_constraint(new_intervals):
                        new_elem = IndexElement(
                            client_id=elem.client_id,
                            intervals=new_intervals,
                            pos=pos
                        )
                        index_set.append(new_elem)
                    break  # Take only first occurrence

        return index_set

    def _create_pattern_from_intervals(
        self,
        intervals: List[StateInterval]
    ) -> TemporalPattern:
        """Creates pattern from list of intervals."""
        n = len(intervals)
        states = [iv.state for iv in intervals]

        # Build relations matrix
        relations = [['=' for _ in range(n)] for _ in range(n)]

        for i in range(n):
            for j in range(i + 1, n):
                rel = self._compute_allen_relation(intervals[i], intervals[j])
                relations[i][j] = rel
                # Lower part of matrix remains empty (-)

        return TemporalPattern(states=states, relations_matrix=relations)

    def _mine_index_set(
        self,
        prefix_pattern: TemporalPattern,
        index_set: List[IndexElement],
        depth: int = 1
    ) -> None:
        """
        Step 3: Discovers patterns from index set using recursive method.

        Args:
            prefix_pattern: Prefix pattern
            index_set: Index set to search
            depth: Current recursion depth
        """
        # Check depth limit
        if depth >= self.max_pattern_size:
            return

        self._depth_stats[depth] += 1

        # Count potential stems (states)
        stem_support = defaultdict(int)
        stem_patterns = defaultdict(list)  # stem -> list of intervals for each client

        for elem in index_set:
            client_seq = self.client_sequences[elem.client_id]
            counted_stems = set()  # Count each stem once per client

            for pos in range(elem.pos + 1, len(client_seq)):
                interval = client_seq[pos]
                state = interval.state

                if state in self._frequent_states and state not in counted_stems:
                    # Check maxgap constraint
                    new_intervals = elem.intervals + [interval]
                    if self._check_gap_constraint(new_intervals):
                        counted_stems.add(state)
                        stem_support[state] += 1
                        stem_patterns[state].append((elem, interval, pos))

        # Find frequent stems
        min_support_count = int(self.minsup * self.num_clients)
        frequent_stems = {
            stem: count
            for stem, count in stem_support.items()
            if count >= min_support_count
        }

        # For each frequent stem, create new pattern and continue recursively
        for stem, count in frequent_stems.items():
            # Find all occurrences of this stem
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
                # Add pattern to results
                sample_pattern = self._create_pattern_from_intervals(stem_index_set[0].intervals)
                sample_pattern.support_count = count
                sample_pattern.support = count / self.num_clients
                self.frequent_patterns.append(sample_pattern)

                self._patterns_found += 1
                if self._patterns_found % 100 == 0:
                    print(f"  Found {self._patterns_found} patterns (depth {depth+1})...")

                # Recursively search for longer patterns
                self._mine_index_set(sample_pattern, stem_index_set, depth + 1)

    def mine_patterns(self) -> List[TemporalPattern]:
        """
        Main function for discovering patterns.

        Returns:
            List of frequent temporal patterns
        """
        print("=" * 60)
        print("ARMADA - Mining Temporal Patterns")
        print("=" * 60)
        print(f"Parameters: minsup={self.minsup}, minconf={self.minconf}, maxgap={self.maxgap}, max_pattern_size={self.max_pattern_size}")

        self.frequent_patterns = []
        self._patterns_found = 0
        self._depth_stats = defaultdict(int)

        # Step 1: Find frequent 1-patterns
        print("\nStep 1: Finding frequent 1-patterns...")
        self._find_frequent_1_patterns()

        # Step 2-3: For each frequent state, create index and mine
        print("\nStep 2-3: Discovering n-dimensional patterns...")

        for idx, state in enumerate(sorted(self._frequent_states)):
            print(f"  Processing state {idx+1}/{len(self._frequent_states)}: {state}")

            # Create initial pattern
            pattern = TemporalPattern(
                states=[state],
                relations_matrix=[['=']],
                support=self._state_support[state] / self.num_clients,
                support_count=self._state_support[state]
            )

            # Create initial index set
            initial_index_set = []
            for client_id, client_seq in self.client_sequences.items():
                for pos, interval in enumerate(client_seq.intervals):
                    if interval.state == state:
                        initial_index_set.append(IndexElement(
                            client_id=client_id,
                            intervals=[interval],
                            pos=pos
                        ))
                        break  # Only first occurrence

            # Recursively discover patterns
            self._mine_index_set(pattern, initial_index_set, depth=1)

        # Remove duplicates
        unique_patterns = []
        seen = set()
        for p in self.frequent_patterns:
            key = (tuple(p.states), tuple(tuple(row) for row in p.relations_matrix))
            if key not in seen:
                seen.add(key)
                unique_patterns.append(p)

        self.frequent_patterns = unique_patterns

        print(f"\nFound {len(self.frequent_patterns)} unique patterns")
        return self.frequent_patterns

    def generate_rules(self) -> List[TemporalRule]:
        """
        Generates temporal rules from frequent patterns.

        For each frequent n-pattern Y (n > 1), we find all
        subpatterns X and generate rule X => Y if confidence >= minconf.

        Returns:
            List of temporal rules
        """
        print("\nGenerating temporal rules...")
        self.temporal_rules = []

        # Mapping patterns for fast lookup
        pattern_support = {}
        for p in self.frequent_patterns:
            key = (tuple(p.states), tuple(tuple(row) for row in p.relations_matrix))
            pattern_support[key] = p.support

        for pattern in self.frequent_patterns:
            if pattern.dim <= 1:
                continue

            # Generate subpatterns (removing one state from the end)
            for i in range(1, pattern.dim):
                # Subpattern from first i states
                sub_states = pattern.states[:i]
                sub_matrix = [row[:i] for row in pattern.relations_matrix[:i]]

                sub_key = (tuple(sub_states), tuple(tuple(row) for row in sub_matrix))

                if sub_key in pattern_support:
                    sub_support = pattern_support[sub_key]
                    confidence = pattern.support / sub_support if sub_support > 0 else 0

                    end_states = pattern.states[i:]
                    end_matrix = [row[i:] for row in pattern.relations_matrix[i:]]
                    end_key = (tuple(end_states), tuple(tuple(row) for row in end_matrix))
                    
                    end_support = pattern_support.get(end_key, 0)
                    lift = (pattern.support / (sub_support * end_support)) if (sub_support > 0 and end_support > 0) else 0.0

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
                            support=pattern.support,
                            lift=lift
                        )
                        self.temporal_rules.append(rule)

        print(f"Generated {len(self.temporal_rules)} rules")
        return self.temporal_rules

    def run(self, filepath: Optional[Path] = None, df: Optional[pd.DataFrame] = None) -> Tuple[List[TemporalPattern], List[TemporalRule]]:
        """
        Runs the full ARMADA pipeline.

        Args:
            filepath: Path to data file
            df: DataFrame with data (alternative)

        Returns:
            Tuple (list of patterns, list of rules)
        """
        if filepath:
            self.load_data(filepath)
        elif df is not None:
            self.load_from_dataframe(df)
        else:
            raise ValueError("Must provide either filepath or df")

        patterns = self.mine_patterns()
        rules = self.generate_rules()

        return patterns, rules

    def save_results(self, output_dir: Path) -> None:
        """Saves results to files."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save patterns
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

        # Save patterns in CSV format
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

        # Save rules - only top 1000 by confidence * support
        rules_data = []
        for r in self.temporal_rules:
            rules_data.append({
                'antecedent': r.antecedent.get_relation_description(),
                'consequent': r.consequent.get_relation_description(),
                'confidence': r.confidence,
                'lift': r.lift,
                'support': r.support,
                'score': r.confidence * r.support  # sorting metric
            })

        # Sort by score and take top 1000
        rules_data = sorted(rules_data, key=lambda x: -x['score'])[:1000]

        # Remove score before saving
        for r in rules_data:
            del r['score']

        with open(output_dir / 'rules.json', 'w') as f:
            json.dump(rules_data, f, indent=2)

        rules_df = pd.DataFrame(rules_data)
        rules_df.to_csv(output_dir / 'rules.csv', index=False)

        print(f"Results saved to {output_dir}")
        print(f"  Patterns: {len(patterns_data)}")
        print(f"  Rules (top 1000): {len(rules_data)}")

    def print_summary(self) -> None:
        """Prints results summary."""
        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)

        # Group patterns by dimension
        by_dim = defaultdict(list)
        for p in self.frequent_patterns:
            by_dim[p.dim].append(p)

        print("\nPatterns by dimension:")
        for dim in sorted(by_dim.keys()):
            patterns = by_dim[dim]
            print(f"  {dim}-patterns: {len(patterns)}")

        print(f"\nTotal patterns: {len(self.frequent_patterns)}")
        print(f"Total rules: {len(self.temporal_rules)}")

        # Top 10 patterns by support
        print("\nTop 10 patterns (by support):")
        sorted_patterns = sorted(self.frequent_patterns, key=lambda x: -x.support)
        for i, p in enumerate(sorted_patterns[:10]):
            print(f"  {i+1}. {p.get_relation_description()} (sup={p.support:.3f})")

        # Top 10 rules by confidence
        if self.temporal_rules:
            print("\nTop 10 rules (by confidence):")
            sorted_rules = sorted(self.temporal_rules, key=lambda x: -x.confidence)
            for i, r in enumerate(sorted_rules[:10]):
                print(f"  {i+1}. {r.to_string()} (conf={r.confidence:.3f}, lift={r.lift:.3f}, sup={r.support:.3f})")


def main():
    """Example usage of ARMADA algorithm."""
    # Paths
    BASE_DIR = Path(__file__).parent.parent.parent.parent
    DATA_DIR = BASE_DIR / "data" / "armada_ready"
    OUTPUT_DIR = BASE_DIR / "data" / "armada_results"

    # Run ARMADA on data
    # max_pattern_size=4 limits patterns to 4 states (prevents combinatorial explosion)
    armada = ARMADA(minsup=0.4, minconf=0.5, maxgap=30, max_pattern_size=4)

    # Load data - use smaller dataset for test
    data_file = DATA_DIR / "armada_sequences_ceap.txt"
    if data_file.exists():
        patterns, rules = armada.run(filepath=data_file)
        armada.print_summary()
        armada.save_results(OUTPUT_DIR / "ceap_test")
    else:
        print(f"File {data_file} does not exist!")


if __name__ == "__main__":
    main()

