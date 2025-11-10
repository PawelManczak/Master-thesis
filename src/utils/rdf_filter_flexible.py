import sys
import re
import os
from rdflib import Graph, Namespace
from collections import deque


def extract_subgraph(input_file, output_file, participant, video):
    """
    Extract a subgraph for a specific participant and video combination.

    Args:
        input_file: Path to the input RDF file
        output_file: Path to save the filtered RDF file
        participant: Participant ID (int)
        video: Video ID (int)

    Returns:
        Graph: The filtered RDF graph, or None if the node doesn't exist
    """
    print(f"Loading RDF file: {input_file}")
    g = Graph()
    g.parse(input_file, format="xml")

    print(f"Original graph has {len(g)} triples")

    CO = Namespace("http://www.semanticweb.org/GRISERA/contextualOntology#")
    PC = Namespace("http://www.semanticweb.org/GRISERA/contextualOntology/propertyConcept#")
    BASE = Namespace("http://www.semanticweb.org/ontologies/2021/8/untitled-ontology-3065#")

    target = f"P{participant}V{video}"
    actExec_name = f"actExecP{participant}V{video}"
    start_node = BASE[actExec_name]

    print(f"\n{'='*60}")
    print(f"Extracting data for: {target}")
    print(f"Activity Execution: {actExec_name}")
    print(f"{'='*60}")

    if (start_node, None, None) not in g:
        print(f"\nERROR: Node {actExec_name} not found in graph!")
        return None

    visited = set()
    to_visit = deque([start_node])
    new_g = Graph()

    for prefix, namespace in g.namespaces():
        new_g.bind(prefix, namespace)

    print(f"\nExtracting subgraph for {actExec_name}...")

    depth = {start_node: 0}
    max_depth = 2

    while to_visit:
        current = to_visit.popleft()

        if current in visited:
            continue

        visited.add(current)
        current_depth = depth.get(current, 0)

        for s, p, o in g.triples((current, None, None)):
            if str(o).endswith("#exp"):
                new_g.add((s, p, o))
                continue

            new_g.add((s, p, o))
            if not isinstance(o, str) and o not in visited and current_depth < max_depth:
                to_visit.append(o)
                depth[o] = current_depth + 1

        for s, p, o in g.triples((None, None, current)):
            if str(s).endswith("#exp"):
                new_g.add((s, p, o))
                continue

            new_g.add((s, p, o))
            if s not in visited and current_depth < max_depth:
                to_visit.append(s)
                depth[s] = current_depth + 1

    print(f"\nAdding nodes with {target} in name...")
    pattern = f"P{participant}V{video}(?![0-9])"

    for s, p, o in g:
        s_str = str(s)
        if target in s_str:
            parts = s_str.split("#")
            if len(parts) > 1:
                node_name = parts[-1]
                if target in node_name and re.search(pattern, node_name):
                    for triple in g.triples((s, None, None)):
                        new_g.add(triple)
                        obj = triple[2]
                        if not isinstance(obj, str):
                            for sub_triple in g.triples((obj, None, None)):
                                new_g.add(sub_triple)

                    for triple in g.triples((None, None, s)):
                        new_g.add(triple)
                        subj = triple[0]
                        for sub_triple in g.triples((subj, None, None)):
                            new_g.add(sub_triple)

    print(f"\nAdding time series for obsInfP{participant}V{video}...")
    obsInf = BASE[f"obsInfP{participant}V{video}"]
    ts_count = 0

    for ts_s, ts_p, ts_o in g.triples((None, CO.hasObservableInformation, obsInf)):
        ts_count += 1
        for triple in g.triples((ts_s, None, None)):
            new_g.add(triple)
            obj = triple[2]
            if not isinstance(obj, str):
                for sub_triple in g.triples((obj, None, None)):
                    new_g.add(sub_triple)

    if ts_count > 0:
        print(f"  Found {ts_count} time series")
    else:
        print(f"  No time series found for {target}")

    print(f"\nAdding ParticipantState for P{participant}V{video}...")
    participant_state = BASE[f"P{participant}State"]
    participant_node = BASE[f"P{participant}"]

    ps_triples_added = 0
    p_triples_added = 0

    for triple in g.triples((participant_state, None, None)):
        new_g.add(triple)
        ps_triples_added += 1
        obj = triple[2]
        if not isinstance(obj, str):
            for sub_triple in g.triples((obj, None, None)):
                new_g.add(sub_triple)

    for triple in g.triples((None, None, participant_state)):
        subj_str = str(triple[0])
        if f"P{participant}V{video}" in subj_str:
            if re.search(pattern, subj_str):
                new_g.add(triple)
                ps_triples_added += 1
                for sub_triple in g.triples((triple[0], None, None)):
                    new_g.add(sub_triple)

    for triple in g.triples((participant_node, None, None)):
        new_g.add(triple)
        p_triples_added += 1
        obj = triple[2]
        if not isinstance(obj, str):
            for sub_triple in g.triples((obj, None, None)):
                new_g.add(sub_triple)

    for triple in g.triples((None, None, participant_node)):
        subj_str = str(triple[0])
        if f"P{participant}V{video}" in subj_str:
            if re.search(pattern, subj_str):
                new_g.add(triple)
                p_triples_added += 1

    if ps_triples_added > 0:
        print(f"  Added {ps_triples_added} triples for P{participant}State (V{video} only)")
    else:
        print(f"  No ParticipantState found for P{participant}")

    if p_triples_added > 0:
        print(f"  Added {p_triples_added} triples for participant P{participant} (V{video} only)")
    else:
        print(f"  No data found for participant P{participant}")

    print(f"\nCleaning references to other participants...")
    exp_node = BASE.exp
    removed_count = 0

    triples_to_remove = []
    for s, p, o in new_g.triples((exp_node, CO.hasScenario, None)):
        o_str = str(o)
        if f"{participant}V{video}" not in o_str:
            triples_to_remove.append((s, p, o))
            removed_count += 1

    for triple in triples_to_remove:
        new_g.remove(triple)

    if removed_count > 0:
        print(f"  Removed {removed_count} references to other participants")

    print(f"\nFiltered graph has {len(new_g)} triples")
    print(f"Saving to: {output_file}")

    new_g.serialize(destination=output_file, format="xml")

    print(f"\n{'='*60}")
    print(f"Graph successfully filtered!")
    print(f"{'='*60}")
    print(f"  Original: {len(g)} triples")
    print(f"  Filtered: {len(new_g)} triples")
    print(f"  Reduction: {100 - (len(new_g)/len(g)*100):.1f}%")
    print(f"  Time series: {ts_count}")
    print(f"{'='*60}\n")

    return new_g


def list_available_combinations(input_file):
    """Display available participant and video combinations"""
    print(f"\nLoading RDF file: {input_file}")
    g = Graph()
    g.parse(input_file, format="xml")

    actExecs = set()
    for s, p, o in g:
        s_str = str(s)
        match = re.search(r'actExecP(\d+)V(\d+)', s_str)
        if match:
            participant = match.group(1)
            video = match.group(2)
            actExecs.add((int(participant), int(video)))

    actExecs = sorted(actExecs)

    print(f"\n{'='*60}")
    print(f"AVAILABLE COMBINATIONS (Participant, Video)")
    print(f"{'='*60}")

    by_participant = {}
    for p, v in actExecs:
        if p not in by_participant:
            by_participant[p] = []
        by_participant[p].append(v)

    for p in sorted(by_participant.keys()):
        videos = by_participant[p]
        print(f"  P{p}: V{', V'.join(map(str, sorted(videos)))}")

    print(f"\nTotal: {len(actExecs)} combinations")
    print(f"Participants: {len(by_participant)}")
    print(f"{'='*60}\n")

    return by_participant


def generate_all_participants_for_video(input_file, output_dir, video):
    """
    Generate RDF files for all participants for a specific video.

    Args:
        input_file: Path to the input RDF file
        output_dir: Directory to save the filtered RDF files
        video: Video ID (int)

    Returns:
        list: List of tuples (participant_id, output_file_path, success)
    """
    print(f"\n{'='*60}")
    print(f"GENERATING FILES FOR ALL PARTICIPANTS - VIDEO {video}")
    print(f"{'='*60}\n")

    # Get all available combinations
    print(f"Analyzing available combinations...")
    g = Graph()
    g.parse(input_file, format="xml")

    # Find all participants for this video
    participants = set()
    for s, p, o in g:
        s_str = str(s)
        match = re.search(rf'actExecP(\d+)V{video}(?![0-9])', s_str)
        if match:
            participant = int(match.group(1))
            participants.add(participant)

    participants = sorted(participants)

    if not participants:
        print(f"ERROR: No participants found for video V{video}")
        return []

    print(f"Found {len(participants)} participants for video V{video}:")
    print(f"  Participants: {', '.join([f'P{p}' for p in participants])}\n")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate files for each participant
    results = []
    successful = 0
    failed = 0

    for participant in participants:
        output_file = os.path.join(output_dir, f"GraphNeuralNetwork_P{participant}V{video}.rdf")

        print(f"\n{'='*60}")
        print(f"Processing: P{participant}V{video} ({successful + failed + 1}/{len(participants)})")
        print(f"{'='*60}")

        try:
            result = extract_subgraph(input_file, output_file, participant, video)
            if result is not None:
                results.append((participant, output_file, True))
                successful += 1
                print(f"✓ Successfully generated: {output_file}")
            else:
                results.append((participant, output_file, False))
                failed += 1
                print(f"✗ Failed to generate: {output_file}")
        except Exception as e:
            results.append((participant, output_file, False))
            failed += 1
            print(f"✗ Error processing P{participant}V{video}: {str(e)}")

    # Summary
    print(f"\n{'='*60}")
    print(f"GENERATION COMPLETE - VIDEO {video}")
    print(f"{'='*60}")
    print(f"  Total participants: {len(participants)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Output directory: {output_dir}")
    print(f"{'='*60}\n")

    return results


if __name__ == "__main__":
    input_file = "/Users/pawelmanczak/mgr sem 2/masters thesis/data/processed/GraphNeuralNetwork_cleaned.rdf"
    output_dir = "/Users/pawelmanczak/mgr sem 2/masters thesis/data/processed/rdf"

    print("\n" + "="*60)
    print("RDF GRAPH FILTER - Select participant and video")
    print("="*60)
    print("\nUsage:")
    print("  python rdf_filter_flexible.py [participant] [video]")
    print("  python rdf_filter_flexible.py all [video]    # Generate for all participants")
    print("  python rdf_filter_flexible.py list           # List available combinations")
    print("\nExamples:")
    print("  python rdf_filter_flexible.py 31 10")
    print("  python rdf_filter_flexible.py 5 3")
    print("  python rdf_filter_flexible.py all 5          # Generate for all participants, video 5")
    print("  python rdf_filter_flexible.py list")

    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "list":
            list_available_combinations(input_file)
            sys.exit(0)
        elif sys.argv[1] == "all" and len(sys.argv) > 2:
            # Generate for all participants for a specific video
            video = int(sys.argv[2])
            results = generate_all_participants_for_video(input_file, output_dir, video)
            sys.exit(0)
        elif len(sys.argv) > 2:
            # Single participant and video
            participant = int(sys.argv[1])
            video = int(sys.argv[2])
            output_file = f"{output_dir}/GraphNeuralNetwork_P{participant}V{video}.rdf"
            result = extract_subgraph(input_file, output_file, participant, video)
            if result is None:
                sys.exit(1)
            print(f"File saved: {output_file}")
            sys.exit(0)

    # Default behavior - list combinations
    list_available_combinations(input_file)

    # Default example - single participant
    participant = 20
    video = 5
    output_file = f"{output_dir}/GraphNeuralNetwork_P{participant}V{video}.rdf"
    result = extract_subgraph(input_file, output_file, participant, video)

    if result is None:
        sys.exit(1)

    print(f"File saved: {output_file}")
    print(f"\nTo analyze the data, use:")
    print(f"  from rdflib import Graph")
    print(f"  g = Graph()")
    print(f"  g.parse('data/processed/rdf/GraphNeuralNetwork_P{participant}V{video}.rdf')")

