import spacy
from tqdm import tqdm

def convert_m2_to_text(m2_string, annotator_id=None):
    """
    Convert M2 formatted text to corrected sentence(s).
    Args:
        m2_string (str): String in M2 format
        annotator_id (int, optional): Specific annotator ID to use. If None, returns all annotators' corrections.
    Returns:
        dict: Dictionary with original and corrected sentences (or list of corrected sentences if multiple annotators)
    """
    if not m2_string:
        raise ValueError("Input string is empty")
    # Split the input into lines
    lines = m2_string.strip().split('\n')
    if len(lines) < 1 or not lines[0].startswith('S '):
        raise ValueError("Input string is not a valid M2 formatted string: {}".format(m2_string))
    # Get the original sentence (first line starting with S)
    original = lines[0][2:].strip()  # Remove 'S ' from the start

    # Group edits by annotator ID
    edits_by_annotator = {}

    # Process annotation lines
    for line in lines[1:]:
        if not line.startswith('A'):
            continue
        # Parse the annotation line
        parts = line[2:].split('|||')  # Remove 'A ' from the start
        if len(parts) < 6:
            continue
        # Get the position indices, replacement, and annotator ID
        try:
            start, end = map(int, parts[0].split())
            replacement = parts[2]
            ann_id = int(parts[5])

            # Filter by annotator_id if specified
            if annotator_id is not None and ann_id != annotator_id:
                continue

            if ann_id not in edits_by_annotator:
                edits_by_annotator[ann_id] = []
            edits_by_annotator[ann_id].append((start, end, replacement))
        except (ValueError, IndexError):
            continue

    # Apply edits for each annotator
    corrected_by_annotator = {}
    for ann_id, edits in edits_by_annotator.items():
        tokens = original.split()
        # Reverse edits to apply from end to start
        edits_reversed = reversed(edits)

        # Apply the edits
        for start, end, replacement in edits_reversed:
            if start < 0 or end < 0:
                continue
            if replacement == '':
                tokens = tokens[:start] + tokens[end:]
            else:
                tokens = tokens[:start] + [replacement] + tokens[end:]

        corrected = ' '.join(tokens).strip()
        corrected_by_annotator[ann_id] = corrected

    # If no edits were found, the sentence is already correct (for all annotators)
    # Return the original as the corrected version
    if not corrected_by_annotator:
        corrected_by_annotator[0] = original

    return {
        'original': original,
        'corrected': corrected_by_annotator
    }

def convert_m2_file(file_path: str, annotator_id=None):
    """
    Convert an M2 file to a list of dictionaries with original and corrected sentences.
    Args:
        file_path (str): Path to the M2 file
        annotator_id (int, optional): Specific annotator ID to use. If None, returns all annotators' corrections.
    Returns:
        list: List of dictionaries with original and corrected sentences by annotator
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    # Split the content into sentence blocks
    blocks = content.strip().split('\n\n')
    # Convert each block
    results = []
    for block in blocks:
        if block.strip():
            results.append(convert_m2_to_text(block, annotator_id))
    return results


def get_references_from_m2(file_path: str):
    """
    Extract sources and multiple references from M2 file for ERRANT evaluation.
    Args:
        file_path (str): Path to the M2 file
    Returns:
        tuple: (sources, references) where references is a list of reference lists (one per annotator)
    """
    results = convert_m2_file(file_path)

    sources = []
    references_by_annotator = {}

    # First, find all unique annotator IDs
    all_annotators = set()
    for result in results:
        all_annotators.update(result['corrected'].keys())

    # Initialize reference lists for all annotators
    for ann_id in all_annotators:
        references_by_annotator[ann_id] = []

    for result in results:
        sources.append(result['original'])

        # For each annotator, add their correction if they have one
        # Otherwise, use the original sentence (indicating no correction needed)
        for ann_id in all_annotators:
            if ann_id in result['corrected']:
                references_by_annotator[ann_id].append(result['corrected'][ann_id])
            else:
                # If this annotator didn't annotate this sentence, use the original
                references_by_annotator[ann_id].append(result['original'])

    # Convert to list of reference lists (sorted by annotator ID for consistency)
    references = [references_by_annotator[ann_id] for ann_id in sorted(all_annotators)]

    return sources, references


if __name__ == "__main__":
    from pathlib import Path

    # Extract sources and references from M2 file
    input_path = 'original_datasets/conll_2014/conll_m2_file.m2'
    sources, references = get_references_from_m2(input_path)

    # Create output directory
    output_path = Path('testing_datasets')
    output_path.mkdir(parents=True, exist_ok=True)

    # Save sources as conll_incorrect.txt
    with open(output_path / 'conll_incorrect.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(sources))

    # Save each annotator's references to separate files
    for i, ref_set in enumerate(references):
        with open(output_path / f'conll_reference_{i}.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(ref_set))

    print(f"Number of sources: {len(sources)}")
    print(f"Number of reference sets (annotators): {len(references)}")
    print(f"Saved to: testing_datasets/conll_incorrect.txt and testing_datasets/conll_reference_0.txt, conll_reference_1.txt, ...")
    print(f"\nFirst source: {sources[0]}")
    print(f"First sentence references:")
    for i, ref_set in enumerate(references):
        print(f"  Annotator {i}: {ref_set[0]}")
