"""Context Enricher - Adds participant contextual information to CSV files"""

import pandas as pd
from rdflib import Graph, Namespace
from typing import Dict, List, Optional
import os


class ContextEnricher:
    """Extract participant context from RDF and add to CSV files"""

    def __init__(self, rdf_file: str):
        self.rdf_file = rdf_file

        print(f"Loading RDF file: {rdf_file}")
        self.g = Graph()
        self.g.parse(rdf_file, format="xml")

        self.CO = Namespace("http://www.semanticweb.org/GRISERA/contextualOntology#")
        self.PC = Namespace("http://www.semanticweb.org/GRISERA/contextualOntology/propertyConcept#")
        self.BASE = Namespace("http://www.semanticweb.org/ontologies/2021/8/untitled-ontology-3065#")

        self.participant_context: Dict[str, Dict] = {}

    def extract_participant_context(self, participant_id: str) -> Dict[str, str]:
        """Extract contextual properties for a participant"""
        participant_state = self.BASE[f"P{participant_id}State"]
        participant = self.BASE[f"P{participant_id}"]

        context = {
            'age': None,
            'panas': None,
            'personality': None,
            'gender': None
        }

        for s, p, o in self.g.triples((participant_state, self.PC.hasProperty, None)):
            prop_name = str(o).split("#")[-1]

            if prop_name.startswith('age'):
                context['age'] = prop_name.replace('age', '')
            elif 'PANAS' in prop_name:
                context['panas'] = prop_name.replace('dominantPANAS', '')
            elif 'Personality' in prop_name:
                context['personality'] = prop_name.replace('dominantPersonality', '')

        # Extract gender from Participant (hasSex property)
        for s, p, o in self.g.triples((participant, self.CO.hasSex, None)):
            sex_value = str(o).split("#")[-1]
            if 'Male' in sex_value:
                context['gender'] = 'Male'
            elif 'Female' in sex_value:
                context['gender'] = 'Female'

        return context

    def extract_all_participants(self) -> Dict[str, Dict]:
        """Extract context for all participants in RDF"""
        print("\nExtracting participant contexts...")

        for s, p, o in self.g.triples((None, self.CO.hasParticipant, None)):
            state_uri = str(s)
            if 'State' in state_uri:
                participant_match = state_uri.split("#")[-1]
                if participant_match.endswith('State'):
                    participant_id = participant_match.replace('State', '').replace('P', '')
                    context = self.extract_participant_context(participant_id)
                    self.participant_context[participant_id] = context
                    print(f"  P{participant_id}: {context}")

        print(f"\nFound context for {len(self.participant_context)} participants")
        return self.participant_context

    def enrich_csv(self, csv_path: str, participant_id: str, output_path: Optional[str] = None) -> pd.DataFrame:
        """Add context columns to CSV file"""
        df = pd.read_csv(csv_path)

        if participant_id not in self.participant_context:
            print(f"Warning: No context found for P{participant_id}")
            return df

        context = self.participant_context[participant_id]

        df['Age'] = context['age']
        df['PANAS'] = context['panas']
        df['Personality'] = context['personality']
        df['Gender'] = context['gender']

        if output_path:
            df.to_csv(output_path, index=False)
            print(f"Saved enriched CSV to: {output_path}")

        return df

    def enrich_all_csvs(self, csv_dir: str, output_dir: Optional[str] = None):
        """Enrich all CSV files in directory with participant context, including subdirectories"""
        if not self.participant_context:
            self.extract_all_participants()

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        else:
            output_dir = csv_dir

        print(f"\n{'='*60}")
        print(f"Enriching CSV files with context")
        print(f"Input directory: {csv_dir}")
        print(f"Output directory: {output_dir}")
        print(f"{'='*60}\n")

        # Collect CSV files from main directory and subdirectories
        csv_files_to_process = []

        # Check main directory
        if os.path.exists(csv_dir):
            for f in os.listdir(csv_dir):
                if f.endswith('.csv') and f.startswith('ts'):
                    csv_files_to_process.append((csv_dir, f))

        # Check extracted/FR subdirectory
        fr_dir = os.path.join(csv_dir, 'extracted', 'FR')
        if os.path.exists(fr_dir):
            for f in os.listdir(fr_dir):
                if f.endswith('.csv') and f.startswith('ts'):
                    csv_files_to_process.append((fr_dir, f))

        # Check extracted/Exp3 subdirectory
        exp3_dir = os.path.join(csv_dir, 'extracted', 'Exp3')
        if os.path.exists(exp3_dir):
            for f in os.listdir(exp3_dir):
                if f.endswith('.csv') and f.startswith('ts'):
                    csv_files_to_process.append((exp3_dir, f))

        enriched_count = 0

        for file_dir, csv_file in csv_files_to_process:
            import re
            match = re.search(r'P(\d+)V\d+', csv_file)
            if not match:
                continue

            participant_id = match.group(1)

            print(f"Processing: {csv_file} (in {os.path.basename(file_dir)})")
            csv_path = os.path.join(file_dir, csv_file)
            output_path = os.path.join(file_dir, csv_file)  # Save in same directory

            df = self.enrich_csv(csv_path, participant_id, output_path)

            if df is not None and 'Age' in df.columns:
                enriched_count += 1
                gender_val = df['Gender'].iloc[0] if 'Gender' in df.columns else 'N/A'
                print(f"  Added: Age={df['Age'].iloc[0]}, PANAS={df['PANAS'].iloc[0]}, Personality={df['Personality'].iloc[0]}, Gender={gender_val}")

        print(f"\n{'='*60}")
        print(f"Enrichment complete!")
        print(f"{'='*60}")
        print(f"  Enriched: {enriched_count} files")
        print(f"  Total: {len(csv_files_to_process)} files")


def main():
    import sys

    rdf_file = "/Users/pawelmanczak/mgr sem 2/masters thesis/data/processed/rdf/GraphNeuralNetwork_P20V5.rdf"
    csv_dir = "/Users/pawelmanczak/mgr sem 2/masters thesis/data/processed/csv_data"

    if len(sys.argv) > 1:
        rdf_file = sys.argv[1]
    if len(sys.argv) > 2:
        csv_dir = sys.argv[2]

    print("="*60)
    print("CONTEXT ENRICHER - Add Participant Context to CSV")
    print("="*60)
    print(f"\nRDF file: {rdf_file}")
    print(f"CSV directory: {csv_dir}\n")

    enricher = ContextEnricher(rdf_file)
    enricher.extract_all_participants()
    enricher.enrich_all_csvs(csv_dir)

    print("\nExample of enriched data:")
    example_file = os.path.join(csv_dir, "tsFRP20V5happiness.csv")
    if os.path.exists(example_file):
        df = pd.read_csv(example_file)
        print(df.head(5))


if __name__ == "__main__":
    main()

