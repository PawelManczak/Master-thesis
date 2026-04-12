# Demographics Experiment

This module contains analytical scripts that evaluate the physiological ARMADA patterns through the lens of demographic groupings, specifically focusing on **gender (Male vs Female)** and **age (Young ≤ 25 vs Old > 25)**.

The overarching goal is to understand how demographic traits modify or dictate physiological emotional responses. It approaches this problem in two distinct scales:

## 1. Global Demographic Analysis (`demographic_analysis.py`)

This mega-analysis aggregates participants from all four available datasets (CASE, K-emoCon, CEAP, EmoWorker) into one massive pool (the *Combined* dataset). 
It then splits this aggregated data strictly by demographic criteria and analyzes the statistical significance of physiological variations between men and women, or young and old participants. It performs:
- Broad similarity calculations across demographic splits (e.g. jaccard index between male and female general rule sets).
- Analysis of pattern overlaps and gender x age interaction matrices.
- Post-hoc support scaling (demographic breakdown of individual highly supported rules).

## 2. Demographic Universal Rules (`demographic_universal_rules.py`)

Rather than pooling datasets together, this script preserves the laboratory contexts. It operates by breaking down *each distinct dataset* by gender or age groups.
It then looks for rules that are so fundamental to a specific demographic that they appear as "Universal" independently across all 4 isolated datasets. 
This strictly tests physiological generalization, aiming to find absolute true-positive physiological behaviors inherent to "Men", "Women", "Young", or "Old", undeterred by environmental or situational factors.


### Datasets & Annotations
- **Datasets:** CASE, K-emoCon, CEAP, EmoWorker
- **Annotations:** Self-annotations (subjective participant ratings)
