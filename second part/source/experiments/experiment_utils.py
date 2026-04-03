#!/usr/bin/env python3
"""
Common utility functions for ARMADA experiments (compare_datasets, demographic_analysis).
"""

from typing import Set, List, Tuple
import pandas as pd
from armada_algorithm import ARMADA

# ============================================================================
# CONSTANTS AND PREFIXES
# ============================================================================

# All EDA metric prefixes
EDA_PREFIXES = (
    'eda_scr_amp',    # SCR amplitude
    'eda_scr_auc',    # SCR AUC
    'eda_std',        # phasic variability
    'eda_max',        # maximum signal in window
    'eda_peaks',      # number of SCR peaks
    'eda',            # SCL
)

# All BVP/HRV and HR metric prefixes
BVP_PREFIXES = (
    'bvp_std', 'bvp_peak_to_peak', 'bvp_spectral_power',
    'hrv_sdnn', 'hrv_rmssd', 'hrv_pnn50', 'hrv_pnn20',
    'hrv_cvnn', 'hrv_cvsd', 'hrv_lf_hf', 'hrv_lfn', 'hrv_hfn',
    'hr_',
)

# ============================================================================
# CORE ARMADA FUNCTIONS
# ============================================================================

def run_armada_on_df(df: pd.DataFrame, minsup: float, minconf: float, maxgap: int, max_pattern_size: int) -> Tuple[ARMADA, List, List]:
    """Runs the ARMADA algorithm on a given DataFrame."""
    armada = ARMADA(
        minsup=minsup,
        minconf=minconf,
        maxgap=maxgap,
        max_pattern_size=max_pattern_size
    )
    patterns, rules = armada.run(df=df)
    return armada, patterns, rules

def extract_rule_signatures(rules: List) -> Set[str]:
    """Extracts unique signatures (e.g. stateA => stateB) from a list of rules."""
    signatures = set()
    for r in rules:
        sig = f"{r.antecedent.get_relation_description()} => {r.consequent.get_relation_description()}"
        signatures.add(sig)
    return signatures

def extract_pattern_signatures(patterns: List) -> Set[str]:
    """Extracts pattern signatures (frequent sequences)."""
    signatures = set()
    for p in patterns:
        signatures.add(p.get_relation_description())
    return signatures

def jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    """Calculates Jaccard index (similarity) for two sets."""
    if not set_a and not set_b:
        return 1.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0

# ============================================================================
# RULE FILTERING
# ============================================================================

def _clean_rule_signature(rule_signature: str) -> List[str]:
    """Cleans a rule signature from temporal relations and returns a list of states."""
    clean_sig = rule_signature.replace('=>', ' ').replace('AND', ' ')
    clean_sig = clean_sig.replace('equals', ' ').replace('before', ' ').replace('meets', ' ')
    clean_sig = clean_sig.replace('overlaps', ' ').replace('contains', ' ').replace('starts', ' ')
    clean_sig = clean_sig.replace('is-finished-by', ' ').replace('(', '').replace(')', '')
    tokens = [t.strip() for t in clean_sig.split() if t.strip()]
    return [t for t in tokens if '_' in t]  # states are e.g. "arousal_high"

def is_bvp_only_rule(rule_signature: str) -> bool:
    """Checks if a rule contains ONLY states derived from BVP/HRV/HR."""
    states = _clean_rule_signature(rule_signature)
    if not states: return False
    for state in states:
        if not any(state.startswith(prefix) for prefix in BVP_PREFIXES):
            return False
    return True

def is_eda_only_rule(rule_signature: str) -> bool:
    """Checks if a rule contains ONLY states derived from EDA."""
    states = _clean_rule_signature(rule_signature)
    if not states: return False
    for state in states:
        if not any(state.startswith(prefix) for prefix in EDA_PREFIXES):
            return False
    return True

def is_physio_cross_rule(rule_signature: str) -> bool:
    """
    Checks if a rule should be filtered out based on emotion/temp criteria.
    Returns True (= reject) if the rule:
      - does NOT contain any arousal/valence state, OR
      - contains any temperature state.
    In other words, only rules that include arousal/valence AND exclude temp will pass.
    """
    states = _clean_rule_signature(rule_signature)
    if not states: return False

    has_emotion = any(
        state.startswith('arousal') or state.startswith('valence')
        for state in states
    )
    has_temp = any(state.startswith('temp') for state in states)

    # Reject if no emotion component or if temp is present
    if not has_emotion or has_temp:
        return True
    return False

def is_single_feature_rule(rule_signature: str) -> bool:
    """Checks if a rule concerns only one feature (e.g. only arousal_* states)."""
    states = _clean_rule_signature(rule_signature)
    if not states: return False
    # Feature is everything up to the last _, e.g. "eda_mean" from "eda_mean_high"
    features = set(state.rsplit('_', 1)[0] for state in states)
    return len(features) == 1

def filter_rules(
    rules: Set[str],
    filter_bvp_only: bool = False,
    filter_eda_only: bool = False,
    filter_physio_cross: bool = False,
    filter_single_feature: bool = False
) -> Set[str]:
    """Rejects rules based on the given flags."""
    filtered_rules = set()
    for sig in rules:
        if filter_bvp_only and is_bvp_only_rule(sig):
            continue
        if filter_eda_only and is_eda_only_rule(sig):
            continue
        if filter_physio_cross and is_physio_cross_rule(sig):
            continue
        if filter_single_feature and is_single_feature_rule(sig):
            continue
        filtered_rules.add(sig)
    return filtered_rules
