import re

content = open('generate_heatmap.py').read()

replacements = {
    '"Pairs -> Third Dataset"': '"Generalization of Pairwise Shared Rules\\nto Third Datasets"',
    '"Dataset Pairs"': '"Intersecting Pairs"',
    '"Triplets -> Fourth Dataset"': '"Generalization of Triplet Intersections\\nto Fourth Datasets"',
    '"Dataset Triplets"': '"Intersecting Triplets"',
    '"Quadruplets -> Fifth Dataset"': '"Generalization of Quadruplet Shared Rules\\nto Fifth Datasets"',
    '"Dataset Quadruplets"': '"Intersecting Quadruplets"',
    '"Leave-One-Out Quintuplets"': '"Leave-one-out Quintuplets Reliability\\n(Generalization to Sixth Dataset)"',
    '"Quintuplet (5 Datasets)"': '"Intersecting Quintuplets"'
}

for old, new in replacements.items():
    content = content.replace(old, new)

open('generate_heatmap.py', 'w').write(content)
