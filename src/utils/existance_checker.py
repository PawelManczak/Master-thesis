from rdflib import Graph
import re

def extract_csv_links_from_rdf():
    try:
        from pathlib import Path
        g = Graph()

        project_root = Path(__file__).parent.parent.parent.resolve()
        rdf_file = project_root / "data" / "raw" / "GraphNeuralNetwork.rdf"
        print(f"Reading RDF file: {rdf_file}")

        if not rdf_file.exists():
            raise FileNotFoundError(f"RDF file not found at: {rdf_file}")

        g.parse(str(rdf_file))

        # unique CSV links
        csv_links = set()

        # Query for all objects in the graph
        for s, p, o in g:
            obj_str = str(o)
            if obj_str.lower().endswith('.csv'):
                csv_links.add(obj_str)

        print(f"Found {len(csv_links)} CSV links in RDF file")
        return csv_links
    except Exception as e:
        print(f"Error reading RDF file: {e}")
        return set()

def extract_csv_links_from_html():
    try:
        from pathlib import Path
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

def compare_csv_links():
    print("\nStarting comparison of CSV links...")

    rdf_links = extract_csv_links_from_rdf()
    html_links = extract_csv_links_from_html()

    if not rdf_links and not html_links:
        print("Error: Could not extract links from either file")
        return

    # Find common links
    common_links = rdf_links.intersection(html_links)

    print(f"\nSummary:")
    print(f"Total links found in RDF: {len(rdf_links)}")
    print(f"Total links found in HTML: {len(html_links)}")
    print(f"Links present in both files: {len(common_links)}")

    if common_links:
        print("\nFirst 10 common links as sample:")
        for link in sorted(list(common_links))[:10]:
            print(f"- {link}")
        if len(common_links) > 10:
            print(f"... and {len(common_links) - 10} more links")
    else:
        print("\nNo common links found between the files.")

    if rdf_links - html_links:
        print("\nSample of links only in RDF file:")
        for link in sorted(list(rdf_links - html_links))[:5]:
            print(f"- {link}")

    if html_links - rdf_links:
        print("\nSample of links only in HTML file:")
        for link in sorted(list(html_links - rdf_links))[:5]:
            print(f"- {link}")

if __name__ == "__main__":
    try:
        compare_csv_links()
    except Exception as e:
        print(f"Unexpected error: {e}")
