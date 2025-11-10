#!/usr/bin/env python
"""Run ARMADA - Implementation following Winarko & Roddick (2007)"""

if __name__ == "__main__":
    import sys
    import os

    # Add project to path
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, project_root)

    from src.analysis.armada_temporal import ARMADA

    print("\nARMADA - Temporal Association Rule Mining")
    print("Implementation: Winarko & Roddick (2007)")

    # Configuration
    min_support = 0.4  # 40% as in the paper example
    min_confidence = 0.6  # 60%
    input_file = "data/processed/csv_data/merged/merged_Exp3P20V5.csv"
    output_dir = "output/reports"

    print(f"\nConfiguration: min_support={min_support:.0%}, min_confidence={min_confidence:.0%}")
    print(f"Input: {input_file}")

    # Run ARMADA
    try:
        # Initialize ARMADA
        armada = ARMADA(min_support=min_support, min_confidence=min_confidence)
        
        # Load data into MDB
        armada.load_merged_csv(input_file)

        # Algorithm 1: Discover frequent temporal patterns
        patterns = armada.discover_patterns()

        # Section 4.3: Generate temporal association rules
        rules = armada.generate_rules()

        # Save results
        armada.save_results(output_dir)

        print(f"\nARMADA COMPLETE: {sum(len(p) for p in patterns.values())} patterns, {len(rules)} rules")
        print(f"Output: {output_dir}/armada_patterns.csv, armada_rules.csv\n")

    except FileNotFoundError as e:
        print(f"\nError: File not found: {e}")
        print("Make sure you run this from the project root directory")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

