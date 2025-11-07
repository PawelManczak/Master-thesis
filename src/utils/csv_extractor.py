"""CSV Extractor - Processes time series CSV files from RDF metadata"""

import pandas as pd
from rdflib import Graph, Namespace
from typing import Dict, Optional
import os


class CSVExtractor:
    """Extract and process time series CSV files from RDF graph"""

    def __init__(self, rdf_file: str, output_dir: str = "data/processed/csv_data"):
        self.rdf_file = rdf_file
        self.output_dir = output_dir

        os.makedirs(output_dir, exist_ok=True)

        print(f"Loading RDF file: {rdf_file}")
        self.g = Graph()
        self.g.parse(rdf_file, format="xml")

        self.CO = Namespace("http://www.semanticweb.org/GRISERA/contextualOntology#")
        self.PC = Namespace("http://www.semanticweb.org/GRISERA/contextualOntology/propertyConcept#")
        self.BASE = Namespace("http://www.semanticweb.org/ontologies/2021/8/untitled-ontology-3065#")

        self.time_series_metadata: Dict = {}
        self.processed_files: Dict[str, str] = {}

    def extract_metadata(self) -> Dict:
        print("\nExtracting time series metadata...")

        for s, p, o in self.g.triples((None, self.CO.timeSeriesSource, None)):
            ts_id = str(s).split("#")[-1]
            source_url = str(o)

            measure = None
            for _, _, measure_obj in self.g.triples((s, self.CO.hasMeasure, None)):
                measure = str(measure_obj).split("#")[-1].replace("Measure", "")

            method = "unknown"
            for _, _, prop in self.g.triples((s, self.PC.hasProperty, None)):
                prop_name = str(prop).split("#")[-1]
                if "method" in prop_name.lower():
                    method = prop_name.replace("method", "")

            self.time_series_metadata[ts_id] = {
                'measure': measure or 'unknown',
                'method': method,
                'source_url': source_url
            }

        print(f"Found {len(self.time_series_metadata)} time series")
        return self.time_series_metadata

    def process_csv(self, ts_id: str, emotion_name: str, source_url: str) -> Optional[pd.DataFrame]:
        try:
            print(f"  Loading: {source_url}")
            df = pd.read_csv(source_url)

            if 'FrameMillis' not in df.columns or 'Value' not in df.columns:
                print(f"  Warning: Missing required columns in {ts_id}")
                return None

            if 'EmotionRange' in df.columns:
                emotion_capitalized = emotion_name.capitalize()
                df['EmotionRange'] = df['EmotionRange'].astype(str) + emotion_capitalized

            required_cols = ['FrameMillis']
            if 'EmotionRange' in df.columns:
                required_cols.append('EmotionRange')

            df_processed = df[required_cols].copy()
            return df_processed

        except Exception as e:
            print(f"  Error processing {ts_id}: {e}")
            return None

    def save_processed_csv(self, ts_id: str, df: pd.DataFrame) -> str:
        output_path = os.path.join(self.output_dir, f"{ts_id}.csv")
        df.to_csv(output_path, index=False)
        print(f"  Saved: {output_path}")
        return output_path

    def extract_and_process_all(self, skip_existing: bool = True) -> Dict[str, str]:
        if not self.time_series_metadata:
            self.extract_metadata()

        print(f"\n{'='*60}")
        print(f"Processing {len(self.time_series_metadata)} CSV files")
        print(f"Output directory: {self.output_dir}")
        print(f"{'='*60}\n")

        success_count = 0
        skip_count = 0
        error_count = 0

        for ts_id, metadata in self.time_series_metadata.items():

            output_path = os.path.join(self.output_dir, f"{ts_id}.csv")
            if skip_existing and os.path.exists(output_path):
                print(f"  Skipped: File already exists")
                self.processed_files[ts_id] = output_path
                skip_count += 1
                continue

            df = self.process_csv(ts_id, metadata['measure'], metadata['source_url'])

            if df is not None:
                saved_path = self.save_processed_csv(ts_id, df)
                self.processed_files[ts_id] = saved_path
                success_count += 1
            else:
                error_count += 1

        print(f"\n{'='*60}")
        print(f"Processing complete!")
        print(f"  Success: {success_count}")
        print(f"  Skipped: {skip_count}")
        print(f"  Errors: {error_count}")
        print(f"  Total: {len(self.time_series_metadata)}")
        print(f"\nProcessed files saved to: {self.output_dir}/")

        return self.processed_files

    def get_processed_file_path(self, ts_id: str) -> Optional[str]:
        return self.processed_files.get(ts_id)

    def load_processed_csv(self, ts_id: str) -> Optional[pd.DataFrame]:
        file_path = self.get_processed_file_path(ts_id)
        if file_path and os.path.exists(file_path):
            return pd.read_csv(file_path)
        return None



def main():
    import sys

    rdf_file = "/Users/pawelmanczak/mgr sem 2/masters thesis/data/processed/GraphNeuralNetwork_P20V5.rdf"
    output_dir = "/Users/pawelmanczak/mgr sem 2/masters thesis/data/processed/csv_data"

    if len(sys.argv) > 1:
        rdf_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]

    print(f"\nRDF file: {rdf_file}")
    print(f"Output directory: {output_dir}")

    extractor = CSVExtractor(rdf_file, output_dir)
    processed_files = extractor.extract_and_process_all(skip_existing=False)


    if processed_files:
        example_ts = list(processed_files.keys())[0]
        print(f"\n{'='*60}")
        print(f"EXAMPLE: {example_ts}")
        print(f"{'='*60}")

        df_example = extractor.load_processed_csv(example_ts)
        if df_example is not None:
            print("\nFirst 10 rows:")
            print(df_example.head(10).to_string(index=False))

            if 'EmotionRange' in df_example.columns:
                print("\nUnique EmotionRange values:")
                print(df_example['EmotionRange'].unique())


if __name__ == "__main__":
    main()

