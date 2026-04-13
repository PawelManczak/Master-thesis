# Multimodality Advantage Report

**Parameters:** minsup=0.3, minconf=0.5, maxgap=5s

| Group | Dataset | Size 2 (N) | Size 2 (Avg Conf) | Size 2 (Avg Lift) | Size 3 (N) | Size 3 (Avg Conf) | Size 3 (Avg Lift) | Size 4 (N) | Size 4 (Avg Conf) | Size 4 (Avg Lift) |
|---|---|---|---|---|---|---|---|---|---|---|
| Self Dimensional | CASE | 390 | 0.675 | 0.655 | 204 | 0.692 | 0.721 | 472 | 0.702 | 0.727 |
| Self Dimensional | K-emoCon | 345 | 0.656 | 0.636 | 132 | 0.649 | 0.642 | 237 | 0.637 | 0.686 |
| Self Dimensional | CEAP | 979 | 0.648 | 0.642 | 94 | 0.663 | 0.623 | 389 | 0.600 | 0.590 |
| Self Dimensional | EmoWorker | 181 | 0.645 | 0.648 | 94 | 0.638 | 0.661 | 213 | 0.629 | 0.663 |
| Self Discrete | CASE | 139 | 0.635 | 0.622 | 70 | 0.667 | 0.680 | 114 | 0.661 | 0.660 |
| Self Discrete | K-emoCon | 60 | 0.657 | 0.635 | 37 | 0.687 | 0.671 | 20 | 0.694 | 0.741 |
| Self Discrete | CEAP | 175 | 0.646 | 0.651 | 29 | 0.646 | 0.697 | 74 | 0.614 | 0.602 |
| Self Discrete | EmoWorker | 43 | 0.626 | 0.632 | 40 | 0.596 | 0.614 | 37 | 0.614 | 0.613 |
| External Dimensional | K-emoCon ext | 265 | 0.673 | 0.658 | 158 | 0.687 | 0.679 | 353 | 0.683 | 0.673 |
| External Dimensional | EMBOA | 857 | 0.693 | 0.631 | 139 | 0.634 | 0.590 | 294 | 0.610 | 0.522 |
| External Discrete | K-emoCon ext | 78 | 0.625 | 0.584 | 72 | 0.614 | 0.612 | 75 | 0.603 | 0.608 |
| External Discrete | EMBOA | 127 | 0.681 | 0.617 | 61 | 0.631 | 0.601 | 186 | 0.596 | 0.431 |
| Combined baseline | ALL_SELF | 10 | 0.724 | 0.893 | 10 | 0.681 | 0.770 | 0 | - | - |
| Combined baseline | ALL_EXTERNAL | 8 | 0.729 | 1.110 | 4 | 0.749 | 1.125 | 1 | 0.692 | 0.692 |
| Combined baseline | ALL_COMBINED | 7 | 0.706 | 0.975 | 5 | 0.646 | 0.858 | 0 | - | - |
