import sys
from pathlib import Path
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = SCRIPT_DIR.parent
PROJECT_DIR = EXPERIMENTS_DIR.parent.parent

sys.path.insert(0, str(PROJECT_DIR / "source" / "processing" / "armada"))
sys.path.insert(0, str(EXPERIMENTS_DIR))

from experiment_utils import run_armada_on_df, extract_rule_signatures, filter_rules
from generate_heatmap import DATASETS

for name, path in DATASETS.items():
    df = pd.read_csv(path)
    armada, patterns, rules = run_armada_on_df(df, 0.3, 0.5, 5, 4)
    raw_sigs = extract_rule_signatures(rules)
    filtered = filter_rules(raw_sigs, True, True, True, True)
    print(f"{name}: {len(rules)} raw, {len(filtered)} filtered")
