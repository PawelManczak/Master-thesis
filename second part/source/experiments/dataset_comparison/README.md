# Dataset Comparison Experiment

This folder serves as the core experimental hub for large-scale association rule mining. 
It contains logic used to independently build models for each psychological dataset and compute exact set intersections resulting in "Universal Rules". 

The core scripts inside are:
- `compare_datasets.py` which executes ARMADA rules generation and plots multidimensional comparisons.
- `generate_readme.py` which translates the structural rules (`beats`, `contains`) and generated JSON outputs into human readable markdown text.


### Datasets & Annotations
- **Datasets:** CASE, K-emoCon, CEAP, EmoWorker
- **Annotations:** Self-annotations (subjective participant ratings)
