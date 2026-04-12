# Master's Thesis - Part 2: Multimodal Emotion Recognition & Temporal Rule Mining

This directory forms the core of the Master's Thesis research. It explores the extraction of generalized, interpretative temporal association rules linking physiological signals (HRV, EDA, BVP, SKT) to human emotional states (Arousal, Valence) using the ARMADA algorithm.

## Purpose

The aim of this part is to apply rule mining (data mining techniques) to multiple, continuously-annotated, multimodal emotion datasets to find universally applicable physiological patterns that are distinct from those specific to demographic factors (Age, Gender) or annotation paradigms (Self-annotated vs. External).

## Datasets Covered

The analysis works across five state-of-the-art affective datasets:
- **CASE**
- **CEAP-360VR**
- **EMBOA**
- **EmoWorker_v2**
- **K-emoCon**

## Directory Structure

*   **`data/`**: Storage for raw and processed physiological data.
    *   `armada_ready/`: Fully processed, synchronized, discretised CSV files ready to be ingested by the ARMADA algorithm.
    *   `CASE/`, `CEAP/`, `EMBOA/`, `EmoWorker_v2/`, `K-emoCon/`: Native directory structure of each independent dataset.
*   **`papers/`**: Source literature, experimental notes, and readmes regarding emotional dimensions (3 classes) and annotations.
*   **`source/`**: The main Python codebase driving the experimental design.
    *   `data/`: Data-related processors.
    *   `processing/`: General algorithms including `armada` (the temporal mining algorithm), normalizers, synchronizers, and extractors mapping continuous data to discrete variables.
    *   `experiments/`: Dedicated validation scripts checking hypotheses (Research Questions). Includes:
        *   `cross_validation/`: Machine Learning-style Leave-One-Out validation (testing generalization capability).
        *   `dataset_comparison/`: Comparisons of rule structures and overlaps between pairs of datasets.
        *   `demographics/`: Scripts examining the influence of Age and Gender subgroups on physiological baselines (RQ 2.1 & 2.2).
        *   `emotion_labels/`: Checks the influence of using dimensional vs. discrete emotion models on rule discovery (RQ 1.2).
        *   `external_annotations/`: Analyzing patterns explicitly derived from observer-based external mood labels.
        *   `heatmap_analysis/`: Generating visual pairwise overlap heatmaps to show dataset similarities.
        *   `multimodality_advantage/`: Investigates whether combining multiple physiological modalities (rule size 3 or 4) improves rule confidence (RQ 2.3).
        *   `self_vs_external/`: Compares rules found in self-annotated vs. externally annotated data to check annotation resilience (RQ 1.1).
        *   `state_distribution/`: Computes statistical distributions of discretised physiological and emotional states across windows.
        *   `universal_rules/`: Data mining checking for universally appearing baseline rules across multiple datasets simultaneously.
        *   `run_all_experiments.py`: Master script orchestrating the sequential execution of all above modules.
    *   `latex_paper/`: LaTeX source files building the final master's thesis PDF document (`isd.tex`). Contains images, references, and build instructions.
