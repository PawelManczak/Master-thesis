# Multimodality Advantage

An experiment analyzing the potential advantage of combining multiple physiological modalities simultaneously. The script examines and compares rules formed by short, single-signal patterns against those utilizing aggregated sequential signals of sizes 2, 3, or 4 to assess how multimodality impacts confidence scores and rule detectability.


### Datasets & Annotations
This experiment comprehensively processes multiple types of data and annotations across predefined groups:
- **Self Dimensional:** CASE, K-emoCon, CEAP, EmoWorker (subjective continuous participant ratings).
- **Self Discrete:** mapped 3x3 label matrices of the aforementioned sets.
- **External Dimensional:** K-emoCon ext, EMBOA (external observers' ratings).
- **External Discrete:** mapped 3x3 label matrices of the external sets.
It subsequently forms major agglomerated baseline evaluations: All Self, All External, and All Combined completely mixing all paradigms.
