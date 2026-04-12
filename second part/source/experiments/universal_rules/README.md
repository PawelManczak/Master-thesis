# Universal Rules Extraction

The core of the analysis on identifying ARMADA patterns that remain consistent across multiple research datasets (i.e., universal psychophysiological rules common to CASE, K-emoCon, CEAP, and EmoWorker). The scripts utilize methods ranging from strict logical intersections (rules present in every dataset) to *leave-one-out* approaches, outlining generalization statistics and outliers.


### Architecture Overview

![Architecture Overview](./universal%20rules.png)

### Datasets & Annotations
- **Datasets:** CASE, K-emoCon, CEAP, EmoWorker, EMBOA
- **Annotations:** Self-annotations (subjective participant ratings), External annotations (observer ratings), and Combined (Self + External).

### Scenarios
The scripts generate results for three combinations:
1. **`self`**: Only datasets with self-annotations (CASE, K-emoCon, CEAP, EmoWorker)
2. **`external`**: Only datasets with external annotations (K-emoCon External, EMBOA)
3. **`external_self`**: The combined superset of all 6 datasets.

### Scripts Overview (When to run what)
- **`universal_rules.py`**: Run this to find rules that are strictly present across **all** datasets in a given scenario. Best for finding the most robust, absolute physiological truths.
- **`universal_rules_leave_one_out.py`**: Run this to find rules present in **all but one** dataset. Best for identifying outliers or understanding what specific physiological reactions a specific dataset might be missing.
- **`universal_rules_leave_one_out_combined.py`**: Run this to evaluate **model generalization**. It concatenates N-1 datasets into a large "train pool", extracts rules, and checks if those rules independently survive in the Nth "test" dataset.
