import pandas as pd
from gec_metrics import get_metric
import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def extract_data_from_excel(excel_path, reference):
    # Read the Excel file
    # print(f"Reading Excel file from {excel_path}...")
    df = pd.read_excel(excel_path)

    # Check if the required columns exist
    required_cols = ['original', reference]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in the Excel file")

    # Convert all values to strings and handle NaN values
    df['original'] = df['original'].astype(str).replace('nan', '')
    df[reference] = df[reference].astype(str).replace('nan', '')

    # Extract the required columns
    sources = df['original'].tolist()
    references = df[reference].tolist()

    # Return the data as lists, ready to use with GEC metrics
    return sources, references

def extract_data_from_excel_2ref(excel_path, reference1, reference2):
    # Read the Excel file
    df = pd.read_excel(excel_path)

    # Check if the required columns exist
    required_cols = ['incorrect', reference1, reference2]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in the Excel file")

    # Convert all values to strings and handle NaN values
    df['incorrect'] = df['incorrect'].astype(str).replace('nan', '')
    df[reference1] = df[reference1].astype(str).replace('nan', '')
    df[reference2] = df[reference2].astype(str).replace('nan', '')

    # Extract the required columns
    sources = df['incorrect'].tolist()
    references1 = df[reference1].tolist()
    references2 = df[reference2].tolist()

    # Return the data as lists
    return sources, [references1, references2]

def extract_corrected(excel_file, corrected):
    # print(f"Reading Excel file from {excel_path}...")
    df = pd.read_excel(excel_file)

    # Convert all values to strings and handle NaN values
    df[corrected] = df[corrected].astype(str).replace('nan', '')

    # Extract the required columns
    corrected = df[corrected].tolist()

    # print(f"Extracted data:")
    # print(f"Hypotheses: {len(corrected)} sentences")

    # Return the data as lists, ready to use with GEC metrics
    return corrected

def calculate_errant_metrics(sources, references, hypotheses, beta=0.5, language='en'):

    metric_cls = get_metric('errant')
    metric = metric_cls(metric_cls.Config(
        beta=beta,
        language=language
    ))

    # Calculate corpus-level score
    corpus_score = metric.score_corpus(
        sources=sources,
        hypotheses=hypotheses,
        references=references
    )

    return corpus_score

def calculate_gleu_metrics(sources, references, hypotheses, iter=500, n=4, unit='word'):
    """
    Calculates GLEU metrics for GEC evaluation.

    Args:
        sources (list): List of original sentences with errors
        references (list): List of gold standard corrections
        hypotheses (list): List of system-corrected sentences
        iter (int): Number of iterations for GLEU calculation
        n (int): Maximum n-gram order
        unit (str): Unit for n-grams ('word' or 'char')

    Returns:
        float: Corpus-level GLEU score
    """
    # Initialize GLEU metric
    metric_cls = get_metric('gleu')
    metric = metric_cls(metric_cls.Config(
        iter=iter,
        n=n,
        unit=unit
    ))

    # Calculate corpus-level score
    # Note: GLEU expects references in the same format as ERRANT
    corpus_score = metric.score_corpus(
        sources=sources,
        hypotheses=hypotheses,
        references=references  # Format consistent with your working ERRANT code
    )

    return corpus_score

def calculate_pt_errant_metrics(sources, references, hypotheses, beta=0.5, language='en'):
    """
    Calculates PT-ERRANT metrics for GEC evaluation.

    Args:
        sources (list): Original sentences with errors
        references (list): Gold standard corrections
        hypotheses (list): System-corrected sentences
        beta (float): The beta value for F-beta score (default 0.5)
        language (str): Language for SpaCy (default 'en')

    Returns:
        float: Corpus-level PT-ERRANT score
    """
    # Initialize PT-ERRANT metric
    # First, get the weight model (BERTScore by default)

    # Initialize PT-ERRANT metric
    metric_cls = get_metric('pterrant')
    weight_model_id = 'bertscore'
    weight_model_cls = get_metric(weight_model_id)

    # Create the metric with proper configuration - Fix is here
    metric = metric_cls(metric_cls.Config(
        beta=beta,  # Pass as a named parameter, not in a dict
        weight_model_name=weight_model_id,
        weight_model_config=weight_model_cls.Config(
            score_type='f',
            rescale_with_baseline=True
        )
    ))

    # Calculate corpus-level score
    corpus_score = metric.score_corpus(
        sources=sources,
        hypotheses=hypotheses,
        references=references
    )

    return corpus_score

def calculate_impara_metrics(sources, hypotheses, model_qe='gotutiyan/IMPARA-QE',
                             model_se='bert-base-cased', threshold=0.9):
    """
    Calculates IMPARA metrics for GEC evaluation.
    IMPARA is a reference-free metric, so it only needs source and hypothesis.

    Args:
        sources (list): List of original sentences with errors
        hypotheses (list): List of system-corrected sentences
        model_qe (str): The model name or path for quality estimation
        model_se (str): The model name or path for similarity estimation
        threshold (float): The threshold for the similarity score

    Returns:
        float: Corpus-level IMPARA score
        list: Sentence-level IMPARA scores
    """

    # Initialize IMPARA metric
    metric_cls = get_metric('impara')
    metric = metric_cls(metric_cls.Config(
        model_qe=model_qe,
        model_se=model_se,
        threshold=threshold
    ))

    # Calculate corpus-level score
    # Note: IMPARA is reference-free, so we don't pass references
    corpus_score = metric.score_corpus(
        sources=sources,
        hypotheses=hypotheses
    )

    return corpus_score

def calculate_scribendi_metrics(sources, hypotheses, model='gpt2', threshold=0.8):
    """
    Calculates Scribendi metrics for GEC evaluation.
    Scribendi is a reference-free metric, so it only needs source and hypothesis.

    Args:
        sources (list): List of original sentences with errors
        hypotheses (list): List of system-corrected sentences
        model (str): The model name or path to the language model to compute perplexity
        threshold (float): The threshold for the maximum values of token-sort-ratio and levenshtein distance ratio

    Returns:
        float: Corpus-level Scribendi score
    """
    # Initialize Scribendi metric
    metric_cls = get_metric('scribendi')
    metric = metric_cls(metric_cls.Config(
        model=model,
        threshold=threshold
    ))

    # Calculate corpus-level score
    # Note: Scribendi is reference-free, so we don't pass references
    corpus_score = metric.score_corpus(
        sources=sources,
        hypotheses=hypotheses
    )

    return corpus_score

if __name__ == "__main__":
    # ===================================================================
    # OPTION 1: Evaluate on CoNLL dataset (2 references)
    # ===================================================================
    sources, references = extract_data_from_excel_2ref("../results/conll_outputs.xlsx", "reference_0", "reference_1")
    hypotheses = extract_corrected("../results/conll_outputs.xlsx", "heterogenous_ensemble")

    # ===================================================================
    # OPTION 2: Evaluate on BEA dataset (1 reference)
    # ===================================================================
    # sources, references_single = extract_data_from_excel("../results/bea_outputs.xlsx", "Gold")
    # hypotheses = extract_corrected("../results/bea_outputs.xlsx", "gpt4o_ft_corrected")
    # references = [references_single]  # Wrap single reference in a list for consistency

    # ===================================================================
    # Calculate Metrics
    # ===================================================================
    print("Calculating metrics...")
    print("="*50)

    # Calculate ERRANT metrics with references
    errant_score = calculate_errant_metrics(sources, references, hypotheses)
    print(f"ERRANT F0.5 Score: {errant_score:.4f}")

    # Calculate GLEU Score
    gleu_score = calculate_gleu_metrics(sources, references, hypotheses)
    print(f"GLEU Score: {gleu_score:.4f}")

    # Calculate PT-ERRANT Score
    pt_errant_score = calculate_pt_errant_metrics(sources, references, hypotheses)
    print(f"PT-ERRANT Score: {pt_errant_score:.4f}")

    # IMPARA Score (reference-free)
    impara_score = calculate_impara_metrics(sources, hypotheses)
    print(f"IMPARA Score: {impara_score:.4f}")

    # Scribendi Score (reference-free)
    scribendi_score = calculate_scribendi_metrics(sources, hypotheses)
    print(f"Scribendi Score: {scribendi_score/len(sources):.4f}")

    print("="*50)
