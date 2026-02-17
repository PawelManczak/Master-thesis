
if __name__ == "__main__":
    import sys
    import os

    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, project_root)

    from src.analysis.armada_temporal import ARMADA


    min_support = 0.3
    min_confidence = 0.4
    max_pattern_size = 2


    intervals_dir = "data/processed/csv_data/intervals"
    output_dir = "output/reports"

    mode = "all"  # Options: "all", "exp3", "fr"

    if mode == "all":
        pattern = "intervals_*.csv"
        base_suffix = "_all_intervals_V5"
    elif mode == "exp3":
        pattern = "intervals_Exp3*.csv"
        base_suffix = "_Exp3_intervals_V5"
    elif mode == "fr":
        pattern = "intervals_FR*.csv"
        base_suffix = "_FR_intervals_V5"
    else:
        pattern = "intervals_*.csv"
        base_suffix = "_all_intervals_V5"

    print(f"\nConfiguration:")
    print(f"  Min Support: {min_support:.0%}")
    print(f"  Min Confidence: {min_confidence:.0%}")
    print(f"  Max Pattern Size: {max_pattern_size} (prevents memory exhaustion)")
    print(f"  Mode: {mode}")
    print(f"  Pattern: {pattern}")
    print(f"  Intervals directory: {intervals_dir}")
    print(f"  Output directory: {output_dir}")
    print(f"\n  Processing by gender to reduce computational complexity")

    all_results = {}

    for gender in ["Male", "Female"]:
        print("\n" + "="*80)
        print(f"PROCESSING: {gender.upper()}")
        print("="*80)

        try:
            armada = ARMADA(min_support=min_support, min_confidence=min_confidence,
                          max_pattern_size=max_pattern_size)

            suffix = f"{base_suffix}_{gender}"
            armada.load_all_intervals(intervals_dir, pattern=pattern, gender_filter=gender)

            if len(armada.MDB) == 0:
                print(f"\n⚠ No data found for {gender}, skipping...")
                continue

            print("\n" + "="*80)
            print(f"DISCOVERING FREQUENT TEMPORAL PATTERNS - {gender.upper()}")
            print("="*80)
            patterns = armada.discover_patterns()

            rules = armada.generate_rules()

            print("\n" + "="*80)
            print(f"SAVING RESULTS - {gender.upper()}")
            print("="*80)
            armada.save_results(output_dir, suffix=suffix)

            total_patterns = sum(len(p) for p in patterns.values())
            all_results[gender] = {
                'sequences': len(armada.MDB),
                'patterns': total_patterns,
                'rules': len(rules),
                'suffix': suffix
            }

        except Exception as e:
            print(f"Error processing {gender}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if all_results:

        for gender, results in all_results.items():
            print(f"\n{gender.upper()}:")
            print(f"  Sequences: {results['sequences']}")
            print(f"  Patterns: {results['patterns']}")
            print(f"  Rules: {results['rules']}")
            print(f"  Files: {output_dir}/armada_patterns{results['suffix']}.csv")
            print(f"         {output_dir}/armada_rules{results['suffix']}.csv")

