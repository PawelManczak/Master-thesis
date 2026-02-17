"""Emotion Merger - Merges all emotion time series into single wide-format CSV"""

import pandas as pd
from typing import Dict, List, Optional
import os
import re


class EmotionMerger:
    """Merge multiple emotion CSV files into single wide-format file"""

    def __init__(self, csv_dir: str, output_dir: Optional[str] = None):
        self.csv_dir = csv_dir
        if output_dir is None:
            self.output_dir = os.path.join(csv_dir, "merged")
        else:
            self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def get_emotion_files(self, participant: str, video: str, method: str, variant: str = "") -> Dict[str, str]:
        """Find all emotion files for a specific participant/video/method/variant in subdirectories"""
        emotion_files = {}

        if variant:
            pattern = f"ts{method}P{participant}V{video}(.+){variant}\\.csv"
        else:
            # Match files without GTE02 or LE02 suffix
            pattern = f"ts{method}P{participant}V{video}([a-z]+)\\.csv"

        if method == 'FR':
            search_dir = os.path.join(self.csv_dir, 'extracted', 'FR')
        elif method == 'Exp3':
            search_dir = os.path.join(self.csv_dir, 'extracted', 'Exp3')
        else:
            search_dir = self.csv_dir

        if not os.path.exists(search_dir):
            search_dir = self.csv_dir

        if os.path.exists(search_dir):
            for filename in os.listdir(search_dir):
                match = re.match(pattern, filename)
                if match:
                    emotion_part = match.group(1)

                    # For non-variant files, make sure they don't have GTE02/LE02
                    if not variant:
                        if 'GTE02' in filename or 'LE02' in filename:
                            continue

                    if emotion_part:
                        emotion_files[emotion_part] = os.path.join(search_dir, filename)

        return emotion_files

    def merge_emotions(self, participant: str, video: str, method: str, variant: str = "") -> Optional[pd.DataFrame]:
        """Merge all emotions for a participant/video/method/variant into one DataFrame"""
        emotion_files = self.get_emotion_files(participant, video, method, variant)

        if not emotion_files:
            return None

        variant_label = f" ({variant})" if variant else " (All)"
        print(f"  Found {len(emotion_files)} emotions{variant_label}: {', '.join(emotion_files.keys())}")

        # Fixed emotion order for consistency
        emotion_order = ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']

        # Sort emotion files by the standard order
        sorted_emotions = [(e, emotion_files[e]) for e in emotion_order if e in emotion_files]

        if not sorted_emotions:
            return None

        # Start with first emotion
        first_emotion, first_filepath = sorted_emotions[0]
        df_merged = pd.read_csv(first_filepath)
        df_merged = df_merged.rename(columns={'EmotionRange': f'EmotionRange{first_emotion.capitalize()}'})

        # Merge other emotions in fixed order
        for emotion, filepath in sorted_emotions[1:]:
            df_emotion = pd.read_csv(filepath)
            df_emotion = df_emotion[['FrameMillis', 'EmotionRange']]
            df_emotion = df_emotion.rename(columns={'EmotionRange': f'EmotionRange{emotion.capitalize()}'})
            df_merged = df_merged.merge(df_emotion, on='FrameMillis', how='outer')

        df_merged = df_merged.sort_values('FrameMillis').reset_index(drop=True)

        return df_merged

    def find_all_combinations(self) -> List[tuple]:
        """Find all unique participant/video/method combinations in main and subdirectories"""
        combinations = set()

        # Check main directory
        if os.path.exists(self.csv_dir):
            for filename in os.listdir(self.csv_dir):
                match = re.match(r'ts(FR|Exp3)P(\d+)V(\d+)', filename)
                if match:
                    method = match.group(1)
                    participant = match.group(2)
                    video = match.group(3)
                    combinations.add((participant, video, method))

        # Check extracted/FR subdirectory
        fr_dir = os.path.join(self.csv_dir, 'extracted', 'FR')
        if os.path.exists(fr_dir):
            for filename in os.listdir(fr_dir):
                match = re.match(r'ts(FR|Exp3)P(\d+)V(\d+)', filename)
                if match:
                    method = match.group(1)
                    participant = match.group(2)
                    video = match.group(3)
                    combinations.add((participant, video, method))

        # Check extracted/Exp3 subdirectory
        exp3_dir = os.path.join(self.csv_dir, 'extracted', 'Exp3')
        if os.path.exists(exp3_dir):
            for filename in os.listdir(exp3_dir):
                match = re.match(r'ts(FR|Exp3)P(\d+)V(\d+)', filename)
                if match:
                    method = match.group(1)
                    participant = match.group(2)
                    video = match.group(3)
                    combinations.add((participant, video, method))

        return sorted(combinations)

    def merge_all(self) -> Dict[str, str]:
        """Merge all emotion files for all combinations (base files only, no GTE02/LE02)"""
        combinations = self.find_all_combinations()

        print(f"\n{'='*60}")
        print(f"Merging emotion files (base files only)")
        print(f"Found {len(combinations)} participant/video/method combinations")
        print(f"{'='*60}\n")

        merged_files = {}
        success_count = 0

        for participant, video, method in combinations:
            print(f"Processing: {method}P{participant}V{video}")

            df_merged = self.merge_emotions(participant, video, method, variant='')

            if df_merged is not None:
                output_filename = f"merged_{method}P{participant}V{video}.csv"
                output_path = os.path.join(self.output_dir, output_filename)

                df_merged.to_csv(output_path, index=False)
                merged_files[f"{method}P{participant}V{video}"] = output_path
                success_count += 1

                print(f"  Saved: {output_filename}")
                print(f"  Shape: {df_merged.shape} (rows, columns)")
            else:
                print(f"  Skipped: No files found")

        print(f"\n{'='*60}")
        print(f"Merge complete!")
        print(f"{'='*60}")
        print(f"  Success: {success_count}")
        print(f"  Total: {len(combinations)}")
        print(f"\nMerged files saved to: {self.output_dir}/")

        return merged_files


def main():
    import sys

    csv_dir = "/Users/pawelmanczak/mgr sem 2/masters thesis/data/processed/csv_data"
    output_dir = None  # Will default to csv_dir/merged

    if len(sys.argv) > 1:
        csv_dir = sys.argv[1]
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]

    print("="*60)
    print("EMOTION MERGER - Merge Emotions into Wide Format")
    print("="*60)
    print(f"\nInput directory: {csv_dir}")
    print(f"Output directory: {output_dir}\n")

    merger = EmotionMerger(csv_dir, output_dir)
    merged_files = merger.merge_all()

    if merged_files:
        print("\nExample of merged data:")
        first_file = list(merged_files.values())[0]
        df = pd.read_csv(first_file)
        print(f"\nFile: {os.path.basename(first_file)}")
        print(df.head(5))


if __name__ == "__main__":
    main()

