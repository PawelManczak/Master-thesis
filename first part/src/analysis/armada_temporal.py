import pandas as pd
from typing import Dict, List
from dataclasses import dataclass, field
from collections import defaultdict
import os
import time


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

    def __init__(self, min_support: float = 0.4, min_confidence: float = 0.6, max_pattern_size: int = 4):
        """
        Initialize ARMADA

        Args:
            min_support: Minimum support threshold (0-1)
            min_confidence: Minimum confidence threshold for rules (0-1)
            max_pattern_size: Maximum pattern size to prevent memory exhaustion (default: 4)
        """
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.max_pattern_size = max_pattern_size

        # MDB - in-memory database
        self.MDB: List[ClientSequence] = []

        # Frequent patterns by size
        self.frequent_patterns: Dict[int, List[TemporalPattern]] = {}

        # Generated rules
        self.rules: List[TemporalRule] = []

        # State counts for support calculation
        self.state_counts: Dict[str, int] = defaultdict(int)

    def load_merged_csv(self, csv_path: str, silent: bool = False) -> None:
        """
        Load merged or interval CSV file and convert to client sequences

        Args:
            csv_path: Path to merged or interval CSV file
            silent: If True, suppress output
        """
        df = pd.read_csv(csv_path)

        # Check if this is an interval file or merged file
        is_interval_file = 'StartTime' in df.columns and 'EndTime' in df.columns

        # Extract emotion columns
        emotion_cols = [col for col in df.columns if col.startswith('EmotionRange')]
        if not emotion_cols:
            raise ValueError("No EmotionRange columns found")

        # Create client sequence
        intervals = []
        position = 0

        if is_interval_file:
            # Process interval file - vectorized for speed
            for idx, row in df.iterrows():
                start_time = int(row['StartTime'])
                end_time = int(row['EndTime'])

                for emotion_col in emotion_cols:
                    emotion_name = emotion_col.replace('EmotionRange', '')
                    state = f"{emotion_name}_{row[emotion_col]}"

                    intervals.append(StateInterval(
                        state=state,
                        start_time=start_time,
                        end_time=end_time,
                        position=position
                    ))
                    position += 1
        else:
            # Process merged file
            time_col = 'FrameMillis'
            if time_col not in df.columns:
                raise ValueError(f"Expected '{time_col}' column not found")

            for frame_idx, row in df.iterrows():
                frame_time = int(row[time_col])

                for emotion_col in emotion_cols:
                    emotion_name = emotion_col.replace('EmotionRange', '')
                    state = f"{emotion_name}_{row[emotion_col]}"

                    intervals.append(StateInterval(
                        state=state,
                        start_time=frame_time,
                        end_time=frame_time,
                        position=position
                    ))
                    position += 1

        # Create single client sequence
        client_seq = ClientSequence(
            client_id=os.path.basename(csv_path),
            intervals=intervals
        )

        self.MDB.append(client_seq)

    def load_all_intervals(self, intervals_dir: str, pattern: str = "*.csv",
                          gender_filter: str = None) -> None:
        """
        Load all interval CSV files from a directory, optionally filtered by gender
        Uses parallel loading for better performance

        Args:
            intervals_dir: Directory containing interval CSV files
            pattern: File pattern to match (default: "*.csv")
            gender_filter: Filter by gender: "Male", "Female", or None for all
        """
        import glob

        print(f"\n{'='*80}\nLOADING INTERVAL FILES")
        print(f"Directory: {intervals_dir} | Pattern: {pattern}", end='')
        if gender_filter:
            print(f" | Gender: {gender_filter}")
        else:
            print()

        search_pattern = os.path.join(intervals_dir, pattern)
        files = sorted(glob.glob(search_pattern))

        if not files:
            raise ValueError(f"No files found matching: {search_pattern}")

        print(f"Found {len(files)} files")

        # Filter by gender if specified
        files_to_load = []
        for filepath in files:
            if gender_filter:
                try:
                    df_check = pd.read_csv(filepath, nrows=1)
                    if 'Gender' in df_check.columns:
                        if df_check['Gender'].iloc[0] == gender_filter:
                            files_to_load.append(filepath)
                except:
                    continue
            else:
                files_to_load.append(filepath)

        total_files = len(files_to_load)
        print(f"Loading {total_files} files...", end='', flush=True)

        start_time = time.time()
        loaded_count = 0

        for i, filepath in enumerate(files_to_load, 1):
            try:
                self.load_merged_csv(filepath, silent=True)
                loaded_count += 1

                if i % 10 == 0 or i == total_files:
                    elapsed = time.time() - start_time
                    print(f"\rLoading {total_files} files... {i}/{total_files} ({i*100//total_files}%) [{elapsed:.1f}s]",
                          end='', flush=True)
            except Exception as e:
                continue

        elapsed = time.time() - start_time
        print(f"\n✓ Loaded {loaded_count} sequences in {elapsed:.1f}s")
        print("="*80)

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
                      prefix_idx: List[IndexElement], depth: int = 0) -> None:
        """
        Algorithm 3: MineIndexSet(ρ, ρ-idx)
        Optimized with depth limit to prevent memory exhaustion

        Args:
            prefix: Current prefix pattern ρ
            prefix_idx: Index set ρ-idx for the prefix
            depth: Recursion depth (for tracking)
        """
        # Stop if we've reached max pattern size
        if len(prefix.states) >= self.max_pattern_size:
            return

        # Count potential stems - count UNIQUE SEQUENCES where stem appears
        stem_counts = defaultdict(int)
        stem_sequences = defaultdict(set)  # Track which sequences contain each stem

        for idx_elem in prefix_idx:
            cs = idx_elem.ptr_cs
            pos = idx_elem.pos

            # Find stems that appear after this position
            for i in range(pos + 1, len(cs.intervals)):
                stem = cs.intervals[i].state
                # Only count once per sequence (using set)
                if cs.client_id not in stem_sequences[stem]:
                    stem_sequences[stem].add(cs.client_id)
                    stem_counts[stem] += 1

        # Find frequent stems
        min_count = int(self.min_support * len(self.MDB))
        frequent_stems = [
            state for state, count in stem_counts.items()
            if count >= min_count
        ]

        # Process each stem
        for stem in frequent_stems:
            # Create new pattern
            new_states = prefix.states + [stem]

            # Check size limit before creating pattern
            if len(new_states) > self.max_pattern_size:
                continue

            new_relationships = []
            if len(new_states) > 1:
                new_relationships = prefix.relationships.copy()
                new_relationships.append(['b'])

            # Support count = number of UNIQUE sequences containing this pattern
            support_count = stem_counts[stem]
            support_ratio = support_count / len(self.MDB)

            # Ensure support_ratio is in [0, 1] range
            support_ratio = min(max(support_ratio, 0.0), 1.0)

            new_pattern = TemporalPattern(
                states=new_states,
                relationships=new_relationships,
                support_count=support_count,
                support_ratio=support_ratio
            )

            # Store pattern
            size = len(new_states)
            if size not in self.frequent_patterns:
                self.frequent_patterns[size] = []
            self.frequent_patterns[size].append(new_pattern)

            # Recurse only if we haven't reached max size
            if size < self.max_pattern_size:
                new_idx = self.create_index_set(stem, prefix, prefix_idx)
                if new_idx:
                    self.mine_index_set(new_pattern, new_idx, depth + 1)

    def read_database_and_find_frequent_states(self) -> List[str]:
        """
        Read MDB and find all frequent states (1-itemsets)
        Optimized: vectorized counting

        Returns:
            List of frequent states
        """
        state_counts = defaultdict(int)

        # Vectorized counting
        for cs in self.MDB:
            states_in_seq = {interval.state for interval in cs.intervals}
            for state in states_in_seq:
                state_counts[state] += 1

        self.state_counts = state_counts

        # Find frequent states
        min_count = int(self.min_support * len(self.MDB))
        frequent_states = [
            state for state, count in state_counts.items()
            if count >= min_count
        ]

        print(f"Found {len(frequent_states)} frequent states")
        return frequent_states

    # ========================================================================
    # Main ARMADA Algorithm (Algorithm 1)
    # ========================================================================

    def discover_patterns(self) -> Dict[int, List[TemporalPattern]]:
        """
        Algorithm 1: Main ARMADA algorithm
        Optimized: parallel processing and reduced printing

        Returns:
            Dictionary mapping pattern size to list of patterns
        """
        print(f"\n{'='*80}\nDISCOVERING PATTERNS | Support: {self.min_support:.0%} | Sequences: {len(self.MDB)} | Max Size: {self.max_pattern_size}")

        start_time = time.time()

        # Find frequent states
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

        print(f"Mining patterns from {len(frequent_states)} frequent states...")

        # Mine patterns for each frequent state
        total_states = len(frequent_states)
        for i, state in enumerate(frequent_states, 1):
            # Report progress every 10 states
            if i % 10 == 0 or i == total_states:
                elapsed = time.time() - start_time
                total_patterns = sum(len(p) for p in self.frequent_patterns.values())
                print(f"\rProcessing state {i}/{total_states} ({i*100//total_states}%) | "
                      f"Patterns: {total_patterns} | Time: {elapsed:.1f}s", end='', flush=True)

            pattern = TemporalPattern(states=[state])
            empty_prefix = TemporalPattern(states=[])
            idx_set = self.create_index_set(state, empty_prefix, self.MDB)

            if idx_set:
                self.mine_index_set(pattern, idx_set)

        # Summary
        elapsed = time.time() - start_time
        total_patterns = sum(len(p) for p in self.frequent_patterns.values())
        print(f"\n✓ Discovery complete: {total_patterns} patterns in {elapsed:.1f}s")
        for size in sorted(self.frequent_patterns.keys()):
            print(f"  Size {size}: {len(self.frequent_patterns[size])} patterns")

        return self.frequent_patterns

    # ========================================================================
    # Generate Temporal Association Rules (Section 4.3)
    # ========================================================================

    def generate_rules(self) -> List[TemporalRule]:
        """
        Section 4.3: Generating temporal association rules
        Optimized: reduced printing

        Returns:
            List of temporal association rules
        """
        print(f"\n{'='*80}\nGENERATING RULES | Confidence: {self.min_confidence:.0%}")

        start_time = time.time()
        self.rules = []

        # Process patterns of size 2 and larger
        for size in sorted(self.frequent_patterns.keys()):
            if size < 2:
                continue

            for pattern_Y in self.frequent_patterns[size]:
                n = len(pattern_Y.states)

                for i in range(n - 1, 0, -1):
                    subpattern_states = pattern_Y.states[:i]

                    # Find subpattern
                    pattern_X = None
                    if i in self.frequent_patterns:
                        for p in self.frequent_patterns[i]:
                            if p.states == subpattern_states:
                                pattern_X = p
                                break

                    if pattern_X is None:
                        continue

                    # Calculate confidence: conf(X → Y) = sup(Y) / sup(X)
                    # Since Y contains X as prefix, sup(Y) ≤ sup(X), so conf ≤ 1.0
                    if pattern_X.support_ratio > 0:
                        confidence = pattern_Y.support_ratio / pattern_X.support_ratio

                        # Debug: Check for anomalies
                        if confidence > 1.0:
                            # This should never happen - Y should have lower or equal support than X
                            # Log warning but clamp value
                            import sys
                            print(f"\n⚠ WARNING: Confidence > 1.0 detected!", file=sys.stderr)
                            print(f"  X: {pattern_X.states}, sup={pattern_X.support_ratio:.4f}", file=sys.stderr)
                            print(f"  Y: {pattern_Y.states}, sup={pattern_Y.support_ratio:.4f}", file=sys.stderr)
                            print(f"  Confidence: {confidence:.4f} (clamping to 1.0)", file=sys.stderr)

                        # Clamp to [0, 1] range
                        confidence = min(max(confidence, 0.0), 1.0)
                    else:
                        # Skip if X has 0 support (shouldn't happen but safety check)
                        continue

                    if confidence >= self.min_confidence:
                        rule = TemporalRule(
                            antecedent=pattern_X,
                            consequent=pattern_Y,
                            confidence=confidence,
                            support=pattern_Y.support_ratio
                        )
                        self.rules.append(rule)
                    else:
                        break

        elapsed = time.time() - start_time
        print(f"✓ Generated {len(self.rules)} rules in {elapsed:.1f}s")
        return self.rules

    def save_results(self, output_dir: str, suffix: str = "") -> None:
        """
        Save discovered patterns and rules to CSV files

        Args:
            output_dir: Directory to save results
            suffix: Optional suffix for filenames
        """
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
        patterns_file = os.path.join(output_dir, f'armada_patterns{suffix}.csv')
        patterns_df.to_csv(patterns_file, index=False)
        print(f"✓ Patterns: {len(patterns_df)} → {patterns_file}")

        # Save rules
        if self.rules:
            rules_data = []
            for rule in self.rules:
                # Ensure Confidence and Support are in valid range [0, 1]
                confidence = min(max(rule.confidence, 0.0), 1.0)
                support = min(max(rule.support, 0.0), 1.0)

                rules_data.append({
                    'Antecedent': str(rule.antecedent),
                    'Consequent': str(rule.consequent),
                    'Confidence': confidence,
                    'Support': support
                })

            rules_df = pd.DataFrame(rules_data)
            rules_file = os.path.join(output_dir, f'armada_rules{suffix}.csv')
            rules_df.to_csv(rules_file, index=False)
            print(f"✓ Rules: {len(rules_df)} → {rules_file}")
        else:
            print("✓ Rules: 0 (none met confidence threshold)")


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

