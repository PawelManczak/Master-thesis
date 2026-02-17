#!/usr/bin/env python3
import rdflib
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import re
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


class GNNRDFAnalyzer:

    def __init__(self, rdf_file_path):
        self.rdf_file_path = Path(rdf_file_path).resolve()
        self.graph = None
        self.time_series_data = {}
        self.emotions = ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']

    def load_rdf(self):
        print("Ładowanie pliku RDF...")
        self.graph = rdflib.Graph()
        try:
            if not self.rdf_file_path.exists():
                raise FileNotFoundError(f"RDF file not found: {self.rdf_file_path}")

            self.graph.parse(str(self.rdf_file_path), format="xml")
            print(f"Załadowano {len(self.graph)} tripletów RDF")
            return True
        except Exception as e:
            print(f"Błąd podczas ładowania RDF: {e}")
            return False

    def extract_time_series_info(self):
        print("Wyciąganie informacji o szeregach czasowych...")

        query = """
        PREFIX co: <http://www.semanticweb.org/GRISERA/contextualOntology#>
        PREFIX pc: <http://www.semanticweb.org/GRISERA/contextualOntology/propertyConcept#>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        
        SELECT ?timeSeries ?source ?property ?key ?value ?measure ?measureName
        WHERE {
            ?timeSeries rdf:type co:TimeSeries .
            OPTIONAL { ?timeSeries co:timeSeriesSource ?source }
            OPTIONAL {
                ?timeSeries pc:hasProperty ?property .
                ?property pc:hasKey ?key .
                ?property pc:hasValue ?value
            }
            OPTIONAL {
                ?timeSeries co:hasMeasure ?measure .
                ?measure co:hasMeasureName ?measureName
            }
        }
        """

        results = self.graph.query(query)
        ts_data = defaultdict(lambda: {'properties': {}, 'measures': []})

        for row in results:
            ts_id = str(row.timeSeries)
            if row.source:
                ts_data[ts_id]['source'] = str(row.source)
            if row.key and row.value:
                ts_data[ts_id]['properties'][str(row.key)] = str(row.value)
            if row.measureName:
                ts_data[ts_id]['measures'].append(str(row.measureName))

        self.time_series_data = dict(ts_data)
        print(f"Znaleziono {len(self.time_series_data)} szeregów czasowych")

    def analyze_emotions_distribution(self):
        print("\nAnaliza rozkładu emocji...")

        emotion_counts = Counter()
        method_counts = Counter()
        quadrant_counts = Counter()

        for ts_id, data in self.time_series_data.items():
            for emotion in self.emotions:
                if emotion in ts_id:
                    emotion_counts[emotion] += 1
                    break

            properties = data.get('properties', {})
            if 'method' in properties:
                method_counts[properties['method']] += 1
            if 'quadrant' in properties:
                quadrant_counts[properties['quadrant']] += 1

        return emotion_counts, method_counts, quadrant_counts

    def create_visualizations(self, emotion_counts, method_counts, quadrant_counts):
        print("Tworzenie wizualizacji...")

        plt.rcParams['font.family'] = 'DejaVu Sans'
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Analiza danych Graph Neural Network - Emocje', fontsize=16)

        emotions = list(emotion_counts.keys())
        counts = list(emotion_counts.values())

        axes[0, 0].bar(emotions, counts, color='skyblue')
        axes[0, 0].set_title('Rozkład emocji w dataset')
        axes[0, 0].set_xlabel('Emocje')
        axes[0, 0].set_ylabel('Liczba próbek')
        axes[0, 0].tick_params(axis='x', rotation=45)

        axes[0, 1].pie(counts, labels=emotions, autopct='%1.1f%%', startangle=90)
        axes[0, 1].set_title('Procentowy rozkład emocji')

        if quadrant_counts:
            quads = list(quadrant_counts.keys())
            quad_counts = list(quadrant_counts.values())
            axes[1, 0].bar(quads, quad_counts, color='lightcoral')
            axes[1, 0].set_title('Rozkład kwadrantów emocjonalnych\n(Valence-Arousal)')
            axes[1, 0].set_xlabel('Kwadrany (HV/LV + HA/LA)')
            axes[1, 0].set_ylabel('Liczba próbek')

        if method_counts:
            methods = list(method_counts.keys())
            method_vals = list(method_counts.values())
            axes[1, 1].bar(methods, method_vals, color='lightgreen')
            axes[1, 1].set_title('Metody zbierania danych')
            axes[1, 1].set_xlabel('Metoda')
            axes[1, 1].set_ylabel('Liczba próbek')

        plt.tight_layout()

        project_root = Path(__file__).parent.parent.parent.resolve()
        output_path = project_root / "output" / "plots" / "gnn_analysis.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Wykres zapisano jako '{output_path}'")
        plt.show()

    def analyze_data_sources(self):
        print("\nAnaliza źródeł danych...")

        sources = [data['source'] for data in self.time_series_data.values() if 'source' in data]
        print(f"Znaleziono {len(sources)} źródeł danych")

        file_patterns = defaultdict(int)
        emotion_pattern = r'(anger|disgust|fear|happiness|neutral|sadness|surprise)\.csv$'

        for source in sources:
            filename = source.split('/')[-1]
            pattern = re.sub(emotion_pattern, 'EMOTION.csv', filename)
            file_patterns[pattern] += 1

        print("\nWzorce nazw plików:")
        for pattern, count in file_patterns.items():
            print(f"  {pattern}: {count} plików")

    def generate_statistics(self):
        print("\n" + "=" * 50)
        print("SZCZEGÓŁOWE STATYSTYKI DATASETU")
        print("=" * 50)

        total_ts = len(self.time_series_data)
        print(f"Całkowita liczba szeregów czasowych: {total_ts}")

        emotion_counts, method_counts, quadrant_counts = self.analyze_emotions_distribution()

        print(f"\nEmocje (model Ekmana):")
        for emotion, count in sorted(emotion_counts.items()):
            percentage = (count / total_ts) * 100 if total_ts > 0 else 0
            print(f"  {emotion.capitalize()}: {count} ({percentage:.1f}%)")

        if method_counts:
            print(f"\nMetody:")
            for method, count in method_counts.items():
                percentage = (count / total_ts) * 100 if total_ts > 0 else 0
                print(f"  {method}: {count} ({percentage:.1f}%)")

        if quadrant_counts:
            print(f"\nKwadrany emocjonalne:")
            for quad, count in quadrant_counts.items():
                percentage = (count / total_ts) * 100 if total_ts > 0 else 0
                print(f"  {quad}: {count} ({percentage:.1f}%)")

        return emotion_counts, method_counts, quadrant_counts

    def run_full_analysis(self):
        if not self.load_rdf():
            return False

        self.extract_time_series_info()
        self.analyze_data_sources()
        emotion_counts, method_counts, quadrant_counts = self.generate_statistics()
        self.create_visualizations(emotion_counts, method_counts, quadrant_counts)

        print("Wykres zapisano jako 'gnn_analysis.png'")

        return True


def main():
    project_root = Path(__file__).parent.parent.parent.resolve()
    rdf_file = project_root / "data" / "raw" / "GraphNeuralNetwork.rdf"

    print("Graph Neural Network - Analyzer danych RDF")
    print("=" * 60)

    analyzer = GNNRDFAnalyzer(rdf_file)
    success = analyzer.run_full_analysis()

    if not success:
        print("error occurred")


if __name__ == '__main__':
    main()
