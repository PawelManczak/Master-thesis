#!/usr/bin/env python3
"""
Master script: Re-run the full pipeline after normalization change.

Steps:
1. Regenerate ARMADA-ready data (prepare_armada_data.py + prepare_external_annotations_armada.py)
2. Run RQ1 experiment (self_vs_external)
3. Run RQ1.2 experiment (emotion_labels)
4. Run RQ2 experiment (universal_rules)
5. Run RQ2.1/2.2 experiments (gender + age)
"""

import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent.parent
ARMADA_DIR = PROJECT_DIR / "source" / "processing" / "armada"
EXPERIMENTS_DIR = PROJECT_DIR / "source" / "experiments"

PYTHON = sys.executable


def run_script(script_path: Path, label: str):
    print(f"\n{'=' * 80}")
    print(f"  RUNNING: {label}")
    print(f"  Script:  {script_path}")
    print(f"{'=' * 80}\n")

    result = subprocess.run(
        [PYTHON, str(script_path)],
        cwd=str(script_path.parent),
        capture_output=False,
    )
    if result.returncode != 0:
        print(f"\n  ERROR: {label} failed with return code {result.returncode}")
        return False
    return True


def main():
    print("=" * 80)
    print("  FULL PIPELINE RE-RUN (with all-personal normalization)")
    print("=" * 80)

    steps = [
        (ARMADA_DIR / "prepare_armada_data.py", "Step 1/6: Prepare self-annotated ARMADA data"),
        (ARMADA_DIR / "prepare_external_annotations_armada.py", "Step 2/6: Prepare external-annotated ARMADA data"),
        (EXPERIMENTS_DIR / "self_vs_external" / "self_vs_external_experiment.py",
         "Step 3/6: RQ1.1 — Self vs External"),
        (EXPERIMENTS_DIR / "emotion_labels" / "emotion_labels_experiment.py",
         "Step 4/6: RQ1.2 — Dimensional vs Discrete"),
        (EXPERIMENTS_DIR / "universal_rules" / "universal_rules.py", "Step 5/6: RQ2 — Universal Rules"),
    ]

    # Check for demographic scripts
    demo_analysis_script = EXPERIMENTS_DIR / "demographics" / "demographic_analysis.py"
    demo_universal_script = EXPERIMENTS_DIR / "demographics" / "demographic_universal_rules.py"
    
    if demo_analysis_script.exists():
        steps.append((demo_analysis_script, "Step 6a/6: RQ2.1/2.2 — Demographic Global Analysis"))
    if demo_universal_script.exists():
        steps.append((demo_universal_script, "Step 6b/6: RQ2.1/2.2 — Demographic Universal Rules"))

    for script, label in steps:
        if not script.exists():
            print(f"\n  SKIP: {label} — script not found: {script}")
            continue
        success = run_script(script, label)
        if not success:
            print(f"\n  Pipeline stopped at: {label}")
            sys.exit(1)

    print("\n" + "=" * 80)
    print("  ALL EXPERIMENTS COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
