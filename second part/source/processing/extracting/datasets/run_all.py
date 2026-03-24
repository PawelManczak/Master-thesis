"""
Run all dataset processing scripts sequentially.

Usage:
    python datasets/run_all.py
    python datasets/run_all.py --only CASE K_emoCon
"""

import sys
import argparse
import traceback
from pathlib import Path

# Make all dataset modules importable
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'utils'))

DATASETS = {
    'CASE':          ('CASE',          'CASE.py'),
    'K_emoCon':      ('K_emoCon',      'K_emoCon.py'),
    'CEAP':          ('CEAP',          'CEAP.py'),
    'EmoWorker_v2':  ('EmoWorker_v2',  'EmoWorker_v2.py'),
}


def run_dataset(name: str) -> bool:
    print(f"\n{'='*60}")
    print(f"PROCESSING: {name}")
    print(f"{'='*60}")
    try:
        module = __import__(name)
        module.main()
        print(f"✓ {name} completed successfully")
        return True
    except Exception as e:
        print(f"✗ {name} failed: {e}")
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='Process all physiological datasets')
    parser.add_argument(
        '--only', nargs='+',
        choices=list(DATASETS.keys()),
        help='Process only specified datasets (default: all)'
    )
    args = parser.parse_args()

    to_run = args.only if args.only else list(DATASETS.keys())

    results = {}
    for name in to_run:
        results[name] = run_dataset(name)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, ok in results.items():
        status = "✓ OK" if ok else "✗ FAILED"
        print(f"  {status}  {name}")

    failed = [n for n, ok in results.items() if not ok]
    if failed:
        print(f"\n{len(failed)} dataset(s) failed: {', '.join(failed)}")
        sys.exit(1)
    else:
        print(f"\nAll {len(results)} datasets processed successfully.")


if __name__ == "__main__":
    main()
