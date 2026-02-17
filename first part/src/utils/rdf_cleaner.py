#!/usr/bin/env python3
"""
RDF Cleaner - removes TimeSeries entries that reference CSV files not present in HTML
"""

from rdflib import Graph, Namespace, RDF, URIRef
from pathlib import Path
import re


def extract_csv_links_from_html():
    try:
        project_root = Path(__file__).parent.parent.parent.resolve()
        html_file = project_root / "data" / "raw" / "Index of _datasets_GraphNeuralNetwork.html"
        print(f"Reading HTML file: {html_file}")

        if not html_file.exists():
            raise FileNotFoundError(f"HTML file not found at: {html_file}")

        csv_pattern = re.compile(r'href="([^"]*\.csv)"')
        csv_links = set()

        with open(html_file, 'r', encoding='utf-8', errors='ignore') as file:
            for line in file:
                matches = csv_pattern.finditer(line)
                for match in matches:
                    href = match.group(1)
                    if href.startswith('https://'):
                        full_url = href
                    else:
                        full_url = f"https://road.affectivese.org/datasets/GraphNeuralNetwork/{href}"
                    csv_links.add(full_url)

        print(f"Found {len(csv_links)} CSV links in HTML file")
        return csv_links
    except Exception as e:
        print(f"Error reading HTML file: {e}")
        return set()


def clean_rdf_file():
    """Remove TimeSeries entries that reference CSV files not in HTML"""
    project_root = Path(__file__).parent.parent.parent.resolve()
    rdf_file = project_root / "data" / "raw" / "GraphNeuralNetwork.rdf"
    output_file = project_root / "data" / "processed" / "GraphNeuralNetwork_cleaned.rdf"

    print(f"\nReading RDF file: {rdf_file}")

    # Load the RDF graph
    g = Graph()
    g.parse(str(rdf_file))

    print(f"Loaded {len(g)} triples from RDF file")

    # Get CSV links from HTML
    html_links = extract_csv_links_from_html()

    if not html_links:
        print("Error: No CSV links found in HTML file")
        return False

    # Define namespaces
    CO = Namespace("http://www.semanticweb.org/GRISERA/contextualOntology#")

    # Find all TimeSeries entities
    time_series_list = list(g.subjects(RDF.type, CO.TimeSeries))
    print(f"\nFound {len(time_series_list)} TimeSeries entries in RDF")

    # Track TimeSeries to remove
    to_remove = []
    to_keep = []

    for ts in time_series_list:
        # Get the timeSeriesSource for this TimeSeries
        sources = list(g.objects(ts, CO.timeSeriesSource))

        if sources:
            source_url = str(sources[0])

            if source_url not in html_links:
                to_remove.append(ts)
            else:
                to_keep.append(ts)
        else:
            to_keep.append(ts)

    print(f"\nTimeSeries to remove: {len(to_remove)}")
    print(f"TimeSeries to keep: {len(to_keep)}")

    if to_remove:
        print("\nSample of TimeSeries being removed:")
        for ts in to_remove[:5]:
            sources = list(g.objects(ts, CO.timeSeriesSource))
            if sources:
                print(f"  - {ts} -> {sources[0]}")

    new_g = Graph()

    for prefix, namespace in g.namespaces():
        new_g.bind(prefix, namespace)

    # Copy all triples that don't involve removed TimeSeries
    removed_count = 0
    kept_count = 0

    for s, p, o in g:
        # Skip triples where subject is a removed TimeSeries
        if s in to_remove:
            removed_count += 1
            continue

        # Skip triples where object is a removed TimeSeries
        if o in to_remove:
            removed_count += 1
            continue

        # Keep this triple
        new_g.add((s, p, o))
        kept_count += 1

    print(f"\nTriples removed: {removed_count}")
    print(f"Triples kept: {kept_count}")
    print(f"New graph size: {len(new_g)} triples")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving cleaned RDF to: {output_file}")
    new_g.serialize(destination=str(output_file), format='xml')

    print("✓ RDF file cleaned successfully!")

    # Generate summary report
    summary_file = output_file.parent / "cleaning_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("RDF Cleaning Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Original file: {rdf_file}\n")
        f.write(f"Cleaned file: {output_file}\n\n")
        f.write(f"Original triples: {len(g)}\n")
        f.write(f"Cleaned triples: {len(new_g)}\n")
        f.write(f"Removed triples: {removed_count}\n\n")
        f.write(f"Original TimeSeries: {len(time_series_list)}\n")
        f.write(f"Removed TimeSeries: {len(to_remove)}\n")
        f.write(f"Kept TimeSeries: {len(to_keep)}\n\n")
        f.write(f"CSV files in HTML: {len(html_links)}\n\n")

        if to_remove:
            f.write("Sample of removed TimeSeries:\n")
            for ts in to_remove[:20]:
                sources = list(g.objects(ts, CO.timeSeriesSource))
                if sources:
                    f.write(f"  - {sources[0]}\n")

    print(f"Summary saved to: {summary_file}")

    return True


def main():
    print("RDF Cleaner - Removing TimeSeries with missing CSV sources")
    print("=" * 60)

    success = clean_rdf_file()

    if success:
        print("\n" + "=" * 60)
        print("Cleaning completed successfully!")
        print("\nYou can now use the cleaned RDF file:")
        print("  data/processed/GraphNeuralNetwork_cleaned.rdf")
    else:
        print("\nCleaning failed!")


if __name__ == "__main__":
    main()

