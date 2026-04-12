# ARMADA Data Preparation Module
"""
Moduł do przygotowania danych dla algorytmu ARMADA
(Association Rule Mining in Temporal Databases).
"""

from .prepare_armada_data import (
    normalize_value,
    discretize_value,
    extract_state_intervals,
    process_participant_data,
    process_dataset,
)

