# Master's Thesis: Interpretable Temporal Rule Mining in Multimodal Emotion Recognition

This repository contains the complete source code, datasets processing pipelines, experiments, and LaTeX documentation for the Master's Thesis. The research focuses on moving away from black-box affective computing models to an Explainable AI (XAI) paradigm by applying Sequential Rule Mining (the ARMADA algorithm) to discover universally readable temporal patterns between physiological signals (HRV, EDA, SKT, BVP) and structural emotional state transitions (Arousal, Valence).

## Repository Structure

The project is organizationally divided into two distinct historical and functional folders:

### 1. `first part/` (Preliminary Research)
This module acts as the early experimental branch of the thesis. It was originally focused on:
*   Parsing and cleaning semantic data formats (ontology, RDF files).
*   Extracting temporal intervals using custom logic.
*   Preliminary rule mining with the ARMADA algorithm applied to structural patterns (Graph Neural Networks datasets).
*   *For details, see [`first part/README.md`](first part/README.md).*

### 2. `second part/` (Core Master's Thesis Research)
This is the primary implementation and evaluation environment for the final thesis. It performs cross-dataset rule mining to extract psychological and affective associations. 
*   **Datasets utilized:** CASE, CEAP-360VR, EMBOA, EmoWorker_v2, K-emoCon.
*   **Pipeline:** Feature extraction, static multi-resolution sliding windows synchronization, emotion modeling mappings (discrete vs. dimensional), and discretization into categorical states.
*   **Experiments:** In-depth validation covering Research Questions (RQs) on Demographic Influences (Age, Gender), Multimodality advantages, Annotation Paradigms (Self-report vs. External), Objectivity via Cross-Validation, and Universal Baselines.
*   **Thesis Document:** Contains the entire LaTeX source code (`isd.tex`) used to build the final thesis PDF.
*   *For details, see [`second part/README.md`](second part/README.md).*

## Setup & Installation

It is recommended to run this project within a dedicated Python virtual environment (e.g., `venv` or `conda`).

1. **Clone the repository.**
2. **Navigate to the root directory.**
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Acquire Datasets:** Since physiological datasets often require user agreements (e.g., EmoWorker, CASE), raw signals are kept offline or within strict data folders according to the structure defined in `second part/README.md`. Ensure `second part/data/` is properly populated before running scripts.

## Author
*Paweł Mańczak*