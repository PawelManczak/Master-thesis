"""
Complete Pipeline Example
Shows full data processing workflow from RDF to merged CSV for all participants in video 20
"""
from config import PROCESSED_DATA_DIR, RDF_DATA_DIR, CSV_DATA_DIR, PROCESSED_RDF_FILE
from src.utils.rdf_filter_flexible import generate_all_participants_for_video
from src.utils.csv_extractor import CSVExtractor
from src.utils.context_enricher import ContextEnricher
from src.utils.emotion_merger import EmotionMerger
from src.utils.temporal_interval_merger import TemporalIntervalMerger
import pandas as pd
import os


def run_complete_pipeline():
    """Execute complete data processing pipeline for all participants in video 20"""

    print("\n" + "="*70)
    print(" "*15 + "COMPLETE DATA PROCESSING PIPELINE - VIDEO 20")
    print("="*70)

    # Configuration
    video = 5
    input_rdf = PROCESSED_RDF_FILE
    output_rdf_dir = RDF_DATA_DIR
    csv_dir = CSV_DATA_DIR

    # Step 0: Generate RDF files for all participants in video 20
    print("\n" + "="*70)
    print("STEP 0: Generate RDF files for all participants in video 20")
    print("="*70)

    results = generate_all_participants_for_video(input_rdf, output_rdf_dir, video)
    
    successful_participants = [p for p, _, success in results if success]
    
    if not successful_participants:
        print("ERROR: No RDF files were generated successfully")
        return
    
    print(f"\nSuccessfully generated RDF files for {len(successful_participants)} participants")
    print(f"Participants: {', '.join([f'P{p}' for p in successful_participants])}")

    # Process each participant through the complete pipeline
    all_merged_files = {}
    
    for participant in successful_participants:
        print("\n" + "="*70)
        print(f"PROCESSING PARTICIPANT P{participant}V{video}")
        print("="*70)
        
        rdf_file = f"{output_rdf_dir}/GraphNeuralNetwork_P{participant}V{video}.rdf"
        
        # Step 1: Extract CSVs from RDF
        print("\n" + "-"*70)
        print(f"STEP 1: Extract CSV files from RDF (P{participant}V{video})")
        print("-"*70)

        extractor = CSVExtractor(rdf_file, csv_dir)
        extractor.extract_and_process_all(skip_existing=True)

        # Step 2: Add participant context
        print("\n" + "-"*70)
        print(f"STEP 2: Add participant context (P{participant}V{video})")
        print("-"*70)

        enricher = ContextEnricher(rdf_file)
        enricher.extract_all_participants()
        enricher.enrich_all_csvs(csv_dir)

        # Step 3: Merge emotions into wide format
        print("\n" + "-"*70)
        print(f"STEP 3: Merge emotions into wide format (P{participant}V{video})")
        print("-"*70)

        merger = EmotionMerger(csv_dir)
        merged_files = merger.merge_all()
        
        # Store merged files for this participant
        for name, path in merged_files.items():
            all_merged_files[f"P{participant}_{name}"] = path

        # Step 4: Consolidate into temporal intervals
        print("\n" + "-"*70)
        print(f"STEP 4: Consolidate into temporal intervals (P{participant}V{video})")
        print("-"*70)

        interval_merger = TemporalIntervalMerger(csv_dir)

        # Process each merged file for this participant
        for name, merged_path in merged_files.items():
            if os.path.exists(merged_path):
                interval_path, orig_rows, interval_rows = interval_merger.process_merged_file(merged_path)
                reduction = 100 * (1 - interval_rows / orig_rows) if orig_rows > 0 else 0
                print(f"  {name}: {orig_rows} frames → {interval_rows} intervals ({reduction:.1f}% reduction)")

    # Final Summary
    print("\n" + "="*70)
    print("PIPELINE COMPLETE FOR ALL PARTICIPANTS!")
    print("="*70)

    print(f"\nProcessed {len(successful_participants)} participants for video {video}")
    print(f"Total merged files generated: {len(all_merged_files)}")
    
    # Check for interval files
    intervals_dir = os.path.join(csv_dir, "intervals")
    if os.path.exists(intervals_dir):
        interval_files = [f for f in os.listdir(intervals_dir) if f.startswith('intervals_') and f.endswith('.csv')]
        print(f"Total interval files generated: {len(interval_files)}")

    print(f"\nGenerated Files by Participant:")
    for participant in successful_participants:
        participant_files = {k: v for k, v in all_merged_files.items() if k.startswith(f"P{participant}_")}
        print(f"\n  P{participant}V{video}:")
        for name, path in participant_files.items():
            if os.path.exists(path):
                df = pd.read_csv(path)
                print(f"    {name}: {df.shape[0]} rows × {df.shape[1]} columns")

    # Show example from first participant
    if all_merged_files:
        print("\n" + "="*70)
        print("EXAMPLE DATA (First Participant)")
        print("="*70)

        first_file = list(all_merged_files.values())[0]
        first_name = list(all_merged_files.keys())[0]
        df = pd.read_csv(first_file)

        print(f"\nFile: {first_name}")
        print(f"Path: {first_file}")
        print(f"Shape: {df.shape}")
        print(f"\nColumns:")
        for i, col in enumerate(df.columns, 1):
            print(f"  {i:2d}. {col}")

        print(f"\nFirst 3 rows:")
        print(df.head(3).to_string(index=False))

        # Show emotion distribution at frame 0
        print(f"\nEmotion snapshot at Frame 0:")
        emotion_cols = [c for c in df.columns if 'EmotionRange' in c]
        if emotion_cols:
            for col in emotion_cols[:5]:  # Show first 5 emotions
                print(f"  {col:30s} = {df[col].iloc[0]}")
    
    print("\n" + "="*70)
    print("OUTPUT LOCATIONS:")
    print("="*70)
    print(f"  Extracted CSVs: {csv_dir}/extracted/FR/ and {csv_dir}/extracted/Exp3/")
    print(f"  Merged files: {csv_dir}/")
    print(f"  Interval files: {csv_dir}/intervals/")
    print("="*70)


if __name__ == "__main__":
    run_complete_pipeline()


