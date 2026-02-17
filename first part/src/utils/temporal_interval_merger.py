"""Temporal Interval Merger - Consolidates consecutive frames with same state into intervals"""

import pandas as pd
import os
from typing import List, Tuple

from config import CSV_DATA_DIR


class TemporalIntervalMerger:
    """
    Consolidates consecutive frames with the same emotion state into single temporal intervals.
    This reduces data size and improves ARMADA performance significantly.
    
    Example:
        Frame 0-100ms: Happiness_GTE02
        Frame 100-200ms: Happiness_GTE02
        Frame 200-300ms: Happiness_GTE02
        
        Becomes:
        Interval [0-300ms]: Happiness_GTE02
    """

    def __init__(self, input_dir: str, output_dir: str = None):
        self.input_dir = input_dir
        if output_dir is None:
            self.output_dir = os.path.join(input_dir, "intervals")
        else:
            self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def consolidate_intervals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Consolidate consecutive frames with same state into intervals
        
        Input format:
            FrameMillis, EmotionRangeAnger, EmotionRangeHappiness, ..., Age, PANAS, etc.
        
        Output format:
            StartTime, EndTime, EmotionRangeAnger, EmotionRangeHappiness, ..., Age, PANAS, etc.
        
        Args:
            df: Input DataFrame with individual frames
            
        Returns:
            DataFrame with consolidated intervals
        """
        if df.empty:
            return df

        # Identify emotion columns and context columns
        time_col = 'FrameMillis'
        emotion_cols = [col for col in df.columns if col.startswith('EmotionRange')]
        context_cols = [col for col in df.columns if col not in emotion_cols and col != time_col]
        
        # Sort by time
        df = df.sort_values(time_col).reset_index(drop=True)
        
        # Build intervals
        intervals = []
        current_start = df[time_col].iloc[0]
        current_state = tuple(df[emotion_cols + context_cols].iloc[0])
        
        for idx in range(1, len(df)):
            row_state = tuple(df[emotion_cols + context_cols].iloc[idx])
            
            # If state changed, save previous interval and start new one
            if row_state != current_state:
                # Save interval
                interval_dict = {
                    'StartTime': current_start,
                    'EndTime': df[time_col].iloc[idx - 1]
                }
                
                # Add emotion and context values
                for col_idx, col in enumerate(emotion_cols + context_cols):
                    interval_dict[col] = current_state[col_idx]
                
                intervals.append(interval_dict)
                
                # Start new interval
                current_start = df[time_col].iloc[idx]
                current_state = row_state
        
        # Save last interval
        interval_dict = {
            'StartTime': current_start,
            'EndTime': df[time_col].iloc[-1]
        }
        for col_idx, col in enumerate(emotion_cols + context_cols):
            interval_dict[col] = current_state[col_idx]
        intervals.append(interval_dict)
        
        result_df = pd.DataFrame(intervals)
        
        # Reorder columns: StartTime, EndTime, emotions, context
        col_order = ['StartTime', 'EndTime'] + emotion_cols + context_cols
        result_df = result_df[col_order]
        
        return result_df

    def process_merged_file(self, input_file: str) -> Tuple[str, int, int]:
        """
        Process a single merged CSV file and consolidate into intervals
        
        Args:
            input_file: Path to merged CSV file
            
        Returns:
            Tuple of (output_path, original_rows, consolidated_rows)
        """
        filename = os.path.basename(input_file)
        print(f"\nProcessing: {filename}")
        
        df = pd.read_csv(input_file)
        original_rows = len(df)
        print(f"  Original: {original_rows} frames")
        
        df_intervals = self.consolidate_intervals(df)
        consolidated_rows = len(df_intervals)
        reduction = 100 * (1 - consolidated_rows / original_rows) if original_rows > 0 else 0
        
        print(f"  Consolidated: {consolidated_rows} intervals ({reduction:.1f}% reduction)")
        
        output_filename = filename.replace('merged_', 'intervals_')
        output_path = os.path.join(self.output_dir, output_filename)
        df_intervals.to_csv(output_path, index=False)
        print(f"  Saved: {output_filename}")
        
        return output_path, original_rows, consolidated_rows

    def process_all_merged_files(self, merged_dir: str = None) -> List[Tuple[str, int, int]]:
        """
        Process all merged CSV files in directory
        
        Args:
            merged_dir: Directory containing merged CSV files (default: input_dir/merged)
            
        Returns:
            List of tuples (output_path, original_rows, consolidated_rows)
        """
        if merged_dir is None:
            merged_dir = os.path.join(self.input_dir, "merged")
        
        print("="*70)
        print("TEMPORAL INTERVAL MERGER - Consolidate Consecutive Frames")
        print("="*70)
        print(f"\nInput directory: {merged_dir}")
        print(f"Output directory: {self.output_dir}")
        
        merged_files = [f for f in os.listdir(merged_dir) if f.startswith('merged_') and f.endswith('.csv')]
        
        if not merged_files:
            print("\nNo merged CSV files found!")
            return []
        
        print(f"\nFound {len(merged_files)} merged files to process")
        
        results = []
        total_original = 0
        total_consolidated = 0
        
        for filename in sorted(merged_files):
            input_path = os.path.join(merged_dir, filename)
            output_path, orig_rows, consol_rows = self.process_merged_file(input_path)
            results.append((output_path, orig_rows, consol_rows))
            total_original += orig_rows
            total_consolidated += consol_rows
        
        # Summary
        total_reduction = 100 * (1 - total_consolidated / total_original) if total_original > 0 else 0
        print("\n" + "="*70)
        print("CONSOLIDATION COMPLETE")
        print("="*70)
        print(f"Total original frames: {total_original:,}")
        print(f"Total consolidated intervals: {total_consolidated:,}")
        print(f"Overall reduction: {total_reduction:.1f}%")
        print(f"\nOutput saved to: {self.output_dir}/")
        
        return results


def main():
    import sys
    
    input_dir = CSV_DATA_DIR
    output_dir = None  # Will default to input_dir/intervals
    
    if len(sys.argv) > 1:
        input_dir = sys.argv[1]
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    
    merger = TemporalIntervalMerger(input_dir, output_dir)
    results = merger.process_all_merged_files()
    
    if results:
        print(f"\nSuccessfully processed {len(results)} files")
    else:
        print("\nNo files were processed")


if __name__ == "__main__":
    main()
