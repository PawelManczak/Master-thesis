# Emotion Labels Experiment

This folder contains experiments investigating the impact of the emotion categorization model (e.g., dimensional *Arousal/Valence* vs. discrete/categorical model) on the stability and reproducibility of physiological rules detected by the ARMADA algorithm.


### Datasets & Annotations
- **Datasets:** CASE, K-emoCon, CEAP-360VR, EmoWorker_v2 (self-annotated) and K-emoCon, EMBOA (external-annotated)
- **Annotations:** Both Self-annotations and External-annotations. The dimensional continuous ratings (arousal/valence) are mapped to categorical/discrete emotional tags (e.g. from EMBOA external raters, K-emoCon self/external raters, etc.) in order to compare with the basic dimensional model. 

### Architecture Overview

![Architecture Diagram](emotion_labels_diagram.png)

### Emotion Label Mapping
The continuous dimensional ratings (Arousal and Valence) are categorized into a 3x3 grid matching Russell's circumplex model. The mapping merges `arousal` and `valence` into a single `emotion` label:

- **(High Arousal, High Valence)** -> `excited`
- **(High Arousal, Medium Valence)** -> `alert`
- **(High Arousal, Low Valence)** -> `stressed_nervous`
- **(Medium Arousal, High Valence)** -> `happy`
- **(Medium Arousal, Medium Valence)** -> `neutral`
- **(Medium Arousal, Low Valence)** -> `sad`
- **(Low Arousal, High Valence)** -> `relaxed`
- **(Low Arousal, Medium Valence)** -> `calm`
- **(Low Arousal, Low Valence)** -> `depressed_bored`

This experiment replaces the separate arousal/valence features with a specific emotion tag and then executes ARMADA to see how the extracted physiological rules differ.
