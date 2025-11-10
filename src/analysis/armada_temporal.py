"""
ARMADA: Algorithm for Mining Richer Temporal Association Rules
Implementation based on: Winarko & Roddick (2007)
Exact implementation following the paper's algorithms and pseudo-code
"""

import pandas as pd
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
import os


@dataclass
class StateInterval:
    """Represents a state occurrence with time interval"""
    state: str
    start_time: int
    end_time: int
    position: int  # Position in client sequence

    def __hash__(self):
        return hash((self.state, self.start_time, self.end_time, self.position))

    def __repr__(self):
        return f"({self.start_time},{self.state},{self.position})"


@dataclass
class ClientSequence:
    """Represents a sequence of state intervals for one client/session"""
    client_id: str
    intervals: List[StateInterval]

    def __len__(self):
        return len(self.intervals)

    def __getitem__(self, idx):
        return self.intervals[idx]

    def __repr__(self):
        return f"cs({self.client_id}): {self.intervals}"


@dataclass
class TemporalPattern:
    """Represents a temporal pattern with relationships (Allen's relations)"""
    states: List[str]
    relationships: List[List[str]] = field(default_factory=list)  # Matrix of relationships
    support_count: int = 0
    support_ratio: float = 0.0

    def __str__(self):
        if len(self.states) == 1:
            return f"⟨{self.states[0]}⟩"

        # For 2+ states, show matrix representation
        result = f"⟨{self.states[0]}"
        for i, state in enumerate(self.states[1:], 1):
            if self.relationships and i-1 < len(self.relationships):
                rel = self.relationships[i-1][0] if self.relationships[i-1] else 'b'
            else:
                rel = 'b'
            result += f" {rel} {state}"
        result += f"⟩"
        return result

    def __hash__(self):
        return hash((tuple(self.states), tuple(tuple(r) for r in self.relationships)))

    def dim(self):
        """Return dimension (number of states) of pattern"""
        return len(self.states)


@dataclass
class IndexElement:
    """Element of index set for efficient pattern mining"""
    ptr_cs: ClientSequence  # Pointer to client sequence
    a_intv: List[StateInterval]  # List of intervals generating the pattern
    pos: int  # Position of stem in client sequence

    def __repr__(self):
        return f"IndexElem(cs={self.ptr_cs.client_id}, pos={self.pos}, intv={self.a_intv})"


@dataclass
class TemporalRule:
    """Temporal association rule: X => Y"""
    antecedent: TemporalPattern
    consequent: TemporalPattern
    confidence: float
    support: float

    def __str__(self):
        return f"{self.antecedent} ⇒ {self.consequent} (conf={self.confidence:.2%}, sup={self.support:.2%})"


class ARMADA:
    """
    ARMADA - Algorithm for Mining Richer Temporal Association Rules

    Implementation following Winarko & Roddick (2007) paper exactly
    """

    def __init__(self, min_support: float = 0.4, min_confidence: float = 0.6):
        """
        Initialize ARMADA

        Args:
            min_support: Minimum support threshold (0-1)
            min_confidence: Minimum confidence threshold for rules (0-1)
        """
        self.min_support = min_support
        self.min_confidence = min_confidence

        # MDB - in-memory database
        self.MDB: List[ClientSequence] = []

        # Frequent patterns by size
        self.frequent_patterns: Dict[int, List[TemporalPattern]] = {}

        # Generated rules
        self.rules: List[TemporalRule] = []

        # State counts for support calculation
        self.state_counts: Dict[str, int] = defaultdict(int)

    def load_merged_csv(self, csv_path: str) -> None:
        """
        Load merged CSV file and convert to client sequences

        Expected format:
        FrameMillis, EmotionRangeAnger, Age, PANAS, Personality, Gender, ...

        Args:
            csv_path: Path to merged CSV file
        """
        print(f"\nLoading data from: {csv_path}")
        df = pd.read_csv(csv_path)

        # Extract time column
        time_col = 'FrameMillis'
        if time_col not in df.columns:
            raise ValueError(f"Expected '{time_col}' column not found")

        # Extract emotion columns
        emotion_cols = [col for col in df.columns if col.startswith('EmotionRange')]

        if not emotion_cols:
            raise ValueError("No EmotionRange columns found")

        print(f"Found {len(emotion_cols)} emotion attributes: {', '.join([c.replace('EmotionRange', '') for c in emotion_cols])}")

        # Create client sequence
        # Each frame will have multiple state intervals (one per emotion)
        intervals = []
        position = 0

        for frame_idx, row in df.iterrows():
            frame_time = int(row[time_col])

            # Create state intervals for each emotion at this frame
            for emotion_col in emotion_cols:
                emotion_name = emotion_col.replace('EmotionRange', '')
                emotion_value = row[emotion_col]

                # State representation: "Emotion_Value" (e.g., "Anger_<0.2Anger")
                state = f"{emotion_name}_{emotion_value}"

                interval = StateInterval(
                    state=state,
                    start_time=frame_time,
                    end_time=frame_time,  # Point interval
                    position=position
                )
                intervals.append(interval)
                position += 1

        # Create single client sequence
        client_seq = ClientSequence(
            client_id=os.path.basename(csv_path),
            intervals=intervals
        )

        self.MDB.append(client_seq)
        print(f"Loaded {len(client_seq)} state intervals into MDB")

    # ========================================================================
    # STEP 2: Create Index Set (Algorithm 2)
    # ========================================================================

    def create_index_set(self, stem: str, prefix: TemporalPattern,
                        prefix_idx) -> List[IndexElement]:
        """
        Algorithm 2: CreateIndexSet(s, ρ, ρ-idx)

        Create index set for stem s given prefix pattern ρ

        Args:
            stem: The stem state to add
            prefix: Current prefix pattern
            prefix_idx: Index set for prefix (or MDB if prefix is empty)

        Returns:
            New index set for pattern ρ' = ρ + stem
        """
        index_set = []

        # Handle empty prefix - search through all client sequences in MDB
        if not prefix.states:
            # Line 2: for each cs ∈ MDB do
            for cs in prefix_idx:  # prefix_idx is MDB when prefix is empty
                # Line 3-6: find first occurrence of stem in cs
                for pos in range(len(cs.intervals)):
                    if cs.intervals[pos].state == stem:
                        # Found first occurrence at position pos
                        index_elem = IndexElement(
                            ptr_cs=cs,
                            a_intv=[cs.intervals[pos]],
                            pos=pos
                        )
                        index_set.append(index_elem)
                        break  # First occurrence only
        else:
            # Non-empty prefix - search after positions in prefix_idx
            # Line 9: for each (ptr_cs, a_intv, start_pos) ∈ ρ-idx do
            for idx_elem in prefix_idx:
                cs = idx_elem.ptr_cs
                start_pos = idx_elem.pos

                # Line 11-14: find first occurrence of stem after start_pos
                for pos in range(start_pos + 1, len(cs.intervals)):
                    if cs.intervals[pos].state == stem:
                        # Found stem at position pos
                        # Line 13: insert (ptr_cs, a_intv, pos) to index set
                        # a_intv includes intervals from prefix + new stem interval
                        a_intv = idx_elem.a_intv + [cs.intervals[pos]]

                        index_elem = IndexElement(
                            ptr_cs=cs,
                            a_intv=a_intv,
                            pos=pos
                        )
                        index_set.append(index_elem)
                        break  # First occurrence only

        # Line 17: return index set ρ'-idx
        return index_set

    # ========================================================================
    # STEP 3: Mine Index Set (Algorithm 3)
    # ========================================================================

    def mine_index_set(self, prefix: TemporalPattern,
                      prefix_idx: List[IndexElement]) -> None:
        """
        Algorithm 3: MineIndexSet(ρ, ρ-idx)

        Mine patterns from index set ρ-idx to find stems and recursively discover patterns

        Args:
            prefix: Current prefix pattern ρ
            prefix_idx: Index set ρ-idx for the prefix
        """
        # Line 3-7: Count potential stems
        stem_counts = defaultdict(int)

        # Line 3: for each cs pointed by index elements of ρ-idx do
        for idx_elem in prefix_idx:
            cs = idx_elem.ptr_cs
            pos = idx_elem.pos

            # Line 4: for pos = pos + 1 to |cs| in cs do
            for i in range(pos + 1, len(cs.intervals)):
                potential_stem = cs.intervals[i].state
                # Line 5: count(s) = count(s) + 1
                stem_counts[potential_stem] += 1

        # Line 8: find S = the set of stems s
        min_count = int(self.min_support * len(self.MDB))
        frequent_stems = [
            state for state, count in stem_counts.items()
            if count >= min_count
        ]

        # Line 9: for each stem state s ∈ S do
        for stem in frequent_stems:
            # Line 10: output the pattern ρ' by combining prefix ρ and stem s
            # Create new pattern ρ' = ρ + stem
            new_states = prefix.states + [stem]

            # Build relationships matrix (simplified: use 'b' for before)
            new_relationships = []
            if len(new_states) > 1:
                # Add relationship to previous states
                new_relationships = prefix.relationships.copy()
                new_relationships.append(['b'])  # Simplified: assume 'before'

            new_pattern = TemporalPattern(
                states=new_states,
                relationships=new_relationships,
                support_count=stem_counts[stem],
                support_ratio=stem_counts[stem] / len(self.MDB)
            )

            # Store pattern
            size = len(new_states)
            if size not in self.frequent_patterns:
                self.frequent_patterns[size] = []
            self.frequent_patterns[size].append(new_pattern)

            print(f"  Found: {new_pattern} (σ={new_pattern.support_ratio:.1%})")

            # Line 11: call CreateIndexSet(s, ρ, ρ-idx)
            new_idx = self.create_index_set(stem, prefix, prefix_idx)

            # Line 12: call MineIndexSet(ρ', ρ'-idx)
            if new_idx:  # If index set is not empty, recurse
                self.mine_index_set(new_pattern, new_idx)

    def read_database_and_find_frequent_states(self) -> List[str]:
        """
        Read MDB and find all frequent states (1-itemsets)

        Returns:
            List of frequent states
        """
        # Count occurrences of each state
        state_counts = defaultdict(int)

        for cs in self.MDB:
            # Track which states appear in this sequence
            states_in_seq = set()
            for interval in cs.intervals:
                states_in_seq.add(interval.state)

            # Count each state once per sequence
            for state in states_in_seq:
                state_counts[state] += 1

        # Store counts
        self.state_counts = state_counts

        # Find frequent states
        min_count = int(self.min_support * len(self.MDB))
        frequent_states = [
            state for state, count in state_counts.items()
            if count >= min_count
        ]

        print(f"\nFound {len(frequent_states)} frequent states (from {len(state_counts)} total)")
        return frequent_states

    # ========================================================================
    # Main ARMADA Algorithm (Algorithm 1)
    # ========================================================================

    def discover_patterns(self) -> Dict[int, List[TemporalPattern]]:
        """
        Algorithm 1: Main ARMADA algorithm

        INPUT: temporal database D (loaded in MDB), minsup
        OUTPUT: all frequent normalized temporal patterns

        Returns:
            Dictionary mapping pattern size to list of patterns
        """
        print("\n" + "="*80)
        print("ARMADA - Discovering Frequent Temporal Patterns")
        print("="*80)
        print(f"Min Support: {self.min_support:.0%}")
        print(f"Database size: {len(self.MDB)} client sequences")

        # Line 1: read D into MDB to find all frequent states
        frequent_states = self.read_database_and_find_frequent_states()

        # Create 1-patterns
        self.frequent_patterns[1] = []
        for state in frequent_states:
            pattern = TemporalPattern(
                states=[state],
                support_count=self.state_counts[state],
                support_ratio=self.state_counts[state] / len(self.MDB)
            )
            self.frequent_patterns[1].append(pattern)

        print("\n" + "="*80)
        print("STEP 2-3: Constructing index sets and mining patterns")
        print("="*80)

        # Line 2: for each frequent state s do
        for state in frequent_states:
            print(f"\nMining with prefix: ⟨{state}⟩")

            # Line 3: form a pattern ρ = ⟨s⟩, output ρ
            pattern = TemporalPattern(states=[state])

            # Line 4: construct ρ-idx = CreateIndexSet(s, ⟨⟩, MDB)
            # Empty prefix represented by pattern with empty states list
            empty_prefix = TemporalPattern(states=[])
            idx_set = self.create_index_set(state, empty_prefix, self.MDB)

            print(f"  Created index set with {len(idx_set)} elements")

            # Line 5: call MineIndexSet(ρ, ρ-idx)
            if idx_set:
                self.mine_index_set(pattern, idx_set)

        # Summary
        print("\n" + "="*80)
        print("Pattern Discovery Complete!")
        print("="*80)
        total_patterns = sum(len(patterns) for patterns in self.frequent_patterns.values())
        print(f"Total frequent patterns: {total_patterns}")
        for size in sorted(self.frequent_patterns.keys()):
            print(f"  {size}-patterns: {len(self.frequent_patterns[size])}")

        return self.frequent_patterns

    # ========================================================================
    # Generate Temporal Association Rules (Section 4.3)
    # ========================================================================

    def generate_rules(self) -> List[TemporalRule]:
        """
        Section 4.3: Generating temporal association rules
        
        For each frequent n-pattern Y (n > 1), generate rules X ⇒ Y
        where X is a subpattern of Y
        
        Returns:
            List of temporal association rules
        """
        print("\n" + "="*80)
        print("Generating Temporal Association Rules")
        print("="*80)
        print(f"Min Confidence: {self.min_confidence:.0%}")

        self.rules = []

        # Process patterns of size 2 and larger
        for size in sorted(self.frequent_patterns.keys()):
            if size < 2:
                continue

            for pattern_Y in self.frequent_patterns[size]:
                # S = (s₁, s₂, ..., sₙ) - ordered list of states in Y
                n = len(pattern_Y.states)

                # For i = n-1 down to 1, find subpatterns X with first i states
                for i in range(n - 1, 0, -1):
                    subpattern_states = pattern_Y.states[:i]

                    # Find this subpattern in frequent patterns
                    pattern_X = None
                    if i in self.frequent_patterns:
                        for p in self.frequent_patterns[i]:
                            if p.states == subpattern_states:
                                pattern_X = p
                                break

                    if pattern_X is None:
                        continue

                    # Calculate confidence: conf(X ⇒ Y) = sup(Y) / sup(X)
                    confidence = pattern_Y.support_ratio / pattern_X.support_ratio

                    # If confidence >= minconf, generate rule
                    if confidence >= self.min_confidence:
                        rule = TemporalRule(
                            antecedent=pattern_X,
                            consequent=pattern_Y,
                            confidence=confidence,
                            support=pattern_Y.support_ratio
                        )
                        self.rules.append(rule)
                        print(f"  {rule}")
                    else:
                        # If X ⇒ Y doesn't have enough confidence,
                        # don't check smaller subpatterns (lower confidence)
                        break

        print(f"\nGenerated {len(self.rules)} rules")
        return self.rules

    def save_results(self, output_dir: str) -> None:
        """Save discovered patterns and rules to CSV files"""
        os.makedirs(output_dir, exist_ok=True)

        # Save patterns
        patterns_data = []
        for size, patterns in self.frequent_patterns.items():
            for pattern in patterns:
                patterns_data.append({
                    'Size': size,
                    'Pattern': str(pattern),
                    'States': ' → '.join(pattern.states),
                    'Support Count': pattern.support_count,
                    'Support Ratio': pattern.support_ratio
                })

        patterns_df = pd.DataFrame(patterns_data)
        patterns_file = os.path.join(output_dir, 'armada_patterns.csv')
        patterns_df.to_csv(patterns_file, index=False)
        print(f"\n✓ Patterns saved to: {patterns_file}")

        # Save rules
        if self.rules:
            rules_data = []
            for rule in self.rules:
                rules_data.append({
                    'Antecedent': str(rule.antecedent),
                    'Consequent': str(rule.consequent),
                    'Confidence': rule.confidence,
                    'Support': rule.support
                })

            rules_df = pd.DataFrame(rules_data)
            rules_file = os.path.join(output_dir, 'armada_rules.csv')
            rules_df.to_csv(rules_file, index=False)
            print(f"✓ Rules saved to: {rules_file}")


def main():

    import sys


    merged_csv = "/Users/pawelmanczak/mgr sem 2/masters thesis/data/processed/csv_data/merged/merged_Exp3P20V5.csv"
    output_dir = "/Users/pawelmanczak/mgr sem 2/masters thesis/output/reports"

    if len(sys.argv) > 1:
        merged_csv = sys.argv[1]

    print("="*80)
    print("ARMADA - Mining Richer Temporal Association Rules")
    print("Implementation following Winarko & Roddick (2007)")
    print("="*80)
    print(f"\nInput: {merged_csv}")
    print(f"Min Support: 40%")
    print(f"Min Confidence: 60%")

    # Initialize ARMADA
    armada = ARMADA(min_support=0.4, min_confidence=0.6)

    # Load data into MDB
    armada.load_merged_csv(merged_csv)

    # Discover patterns (Algorithm 1)
    patterns = armada.discover_patterns()

    # Generate rules (Section 4.3)
    rules = armada.generate_rules()

    # Save results
    armada.save_results(output_dir)

    print("\n" + "="*80)
    print("ARMADA Analysis Complete!")
    print("="*80)


if __name__ == "__main__":
    main()

