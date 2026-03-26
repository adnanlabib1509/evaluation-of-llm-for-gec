"""
Complete Statistical Significance Testing for GEC Systems

This script performs statistical significance testing for GEC evaluation:
- BEA-dev dataset: ERRANT F0.5, GLEU, and PT-ERRANT
- CoNLL dataset: ERRANT F0.5 and GLEU (no PT-ERRANT)

For each metric, it calculates:
- P-values (permutation test)
- Differences between systems
- 95% bootstrap confidence intervals
"""

import pandas as pd
from gec_metrics import get_metric
import numpy as np
import nltk
from tqdm import tqdm
import random

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_txt_file(file_path):
    """Load sentences from text file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f]

def extract_data_from_excel(excel_path, reference):
    """Extract source and reference from Excel file."""
    df = pd.read_excel(excel_path)

    required_cols = ['original', reference]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in the Excel file")

    df['original'] = df['original'].astype(str).replace('nan', '')
    df[reference] = df[reference].astype(str).replace('nan', '')

    sources = df['original'].tolist()
    references = df[reference].tolist()

    return sources, references

def extract_corrected(excel_path, corrected):
    """Extract corrected sentences from Excel file."""
    df = pd.read_excel(excel_path)
    df[corrected] = df[corrected].astype(str).replace('nan', '')
    return df[corrected].tolist()

# ============================================================================
# METRIC CALCULATION FUNCTIONS
# ============================================================================

def calculate_errant_metrics(sources, references, hypotheses, beta=0.5, language='en'):
    """Calculate ERRANT F0.5 score."""
    metric_cls = get_metric('errant')
    metric = metric_cls(metric_cls.Config(
        beta=beta,
        language=language
    ))

    # Handle both single and multiple references
    if isinstance(references[0], list):
        ref_list = references
    else:
        ref_list = [references]

    corpus_score = metric.score_corpus(
        sources=sources,
        hypotheses=hypotheses,
        references=ref_list
    )

    return corpus_score

def calculate_gleu_metrics(sources, references, hypotheses, iter=500, n=4, unit='word'):
    """Calculate GLEU score."""
    metric_cls = get_metric('gleu')
    metric = metric_cls(metric_cls.Config(
        iter=iter,
        n=n,
        unit=unit
    ))

    # Handle both single and multiple references
    if isinstance(references[0], list):
        ref_list = references
    else:
        ref_list = [references]

    corpus_score = metric.score_corpus(
        sources=sources,
        hypotheses=hypotheses,
        references=ref_list
    )

    return corpus_score

def calculate_pt_errant_metrics(sources, references, hypotheses, beta=0.5):
    """Calculate PT-ERRANT score."""
    metric_cls = get_metric('pterrant')
    weight_model_id = 'bertscore'
    weight_model_cls = get_metric(weight_model_id)

    metric = metric_cls(metric_cls.Config(
        beta=beta,
        weight_model_name=weight_model_id,
        weight_model_config=weight_model_cls.Config(
            score_type='f',
            rescale_with_baseline=True
        )
    ))

    # Handle both single and multiple references
    if isinstance(references[0], list):
        ref_list = references
    else:
        ref_list = [references]

    corpus_score = metric.score_corpus(
        sources=sources,
        hypotheses=hypotheses,
        references=ref_list
    )

    return corpus_score

# ============================================================================
# STATISTICAL TEST FUNCTIONS
# ============================================================================

def permutation_test(sources, references, hyp1, hyp2, metric_func, n_permutations=1000, **metric_kwargs):
    """
    Permutation test for statistical significance.

    Args:
        sources: Source sentences
        references: Reference sentences (single or multiple)
        hyp1: Hypotheses from system 1
        hyp2: Hypotheses from system 2
        metric_func: Function to calculate metric (e.g., calculate_errant_metrics)
        n_permutations: Number of permutations
        **metric_kwargs: Additional arguments for metric function

    Returns:
        tuple: (original_difference, p_value)
    """
    # Calculate original difference
    score1 = metric_func(sources, references, hyp1, **metric_kwargs)
    score2 = metric_func(sources, references, hyp2, **metric_kwargs)
    original_diff = score1 - score2

    # Generate random differences
    random_diffs = []
    n_sentences = len(sources)

    for i in tqdm(range(n_permutations), desc="Permutation test"):
        random.seed(i)
        perm_hyp1 = []
        perm_hyp2 = []

        for j in range(n_sentences):
            if random.random() < 0.5:
                perm_hyp1.append(hyp2[j])
                perm_hyp2.append(hyp1[j])
            else:
                perm_hyp1.append(hyp1[j])
                perm_hyp2.append(hyp2[j])

        perm_score1 = metric_func(sources, references, perm_hyp1, **metric_kwargs)
        perm_score2 = metric_func(sources, references, perm_hyp2, **metric_kwargs)
        random_diffs.append(perm_score1 - perm_score2)

    p_value = np.mean(np.abs(random_diffs) >= abs(original_diff))

    return original_diff, p_value

def bootstrap_difference_ci(sources, references, hyp1, hyp2, metric_func, n_bootstrap=1000, **metric_kwargs):
    """
    Bootstrap confidence interval for difference between systems.

    Args:
        sources: Source sentences
        references: Reference sentences (single or multiple)
        hyp1: Hypotheses from system 1
        hyp2: Hypotheses from system 2
        metric_func: Function to calculate metric
        n_bootstrap: Number of bootstrap samples
        **metric_kwargs: Additional arguments for metric function

    Returns:
        tuple: (lower_bound, upper_bound) of 95% CI
    """
    differences = []
    n_sentences = len(sources)

    for i in tqdm(range(n_bootstrap), desc="Bootstrap CI"):
        indices = np.random.choice(n_sentences, n_sentences, replace=True)
        boot_sources = [sources[idx] for idx in indices]

        # Handle multiple references
        if isinstance(references, list) and len(references) > 0 and isinstance(references[0], list):
            boot_references = [[ref[idx] for idx in indices] for ref in references]
        else:
            boot_references = [references[idx] for idx in indices]

        boot_hyp1 = [hyp1[idx] for idx in indices]
        boot_hyp2 = [hyp2[idx] for idx in indices]

        score1 = metric_func(boot_sources, boot_references, boot_hyp1, **metric_kwargs)
        score2 = metric_func(boot_sources, boot_references, boot_hyp2, **metric_kwargs)

        differences.append(score1 - score2)

    return np.percentile(differences, [2.5, 97.5])

# ============================================================================
# MAIN TESTING FUNCTIONS
# ============================================================================

def run_significance_tests(sources, references, systems_dict, comparisons,
                          metric_name, metric_func, n_permutations=1000,
                          n_bootstrap=1000, **metric_kwargs):
    """
    Run complete significance tests for a given metric.

    Args:
        sources: Source sentences
        references: Reference sentences
        systems_dict: Dictionary mapping system names to hypothesis lists
        comparisons: List of (system1_name, system2_name) tuples
        metric_name: Name of the metric (for display)
        metric_func: Function to calculate the metric
        n_permutations: Number of permutations for p-value
        n_bootstrap: Number of bootstrap samples for CI
        **metric_kwargs: Additional arguments for metric function

    Returns:
        pd.DataFrame: Results table with differences, p-values, and CIs
    """
    results = []

    print(f"\n{'='*80}")
    print(f"{metric_name} Statistical Significance Tests")
    print(f"{'='*80}\n")

    for system1_name, system2_name in comparisons:
        print(f"\nComparing: {system1_name} vs {system2_name}")

        hyp1 = systems_dict[system1_name]
        hyp2 = systems_dict[system2_name]

        # Permutation test for p-value
        diff, p_val = permutation_test(
            sources, references, hyp1, hyp2,
            metric_func, n_permutations, **metric_kwargs
        )

        # Bootstrap CI
        lower, upper = bootstrap_difference_ci(
            sources, references, hyp1, hyp2,
            metric_func, n_bootstrap, **metric_kwargs
        )

        # Check if significant by both methods
        boot_significant = not (lower <= 0 <= upper)
        both_significant = (p_val < 0.05) and boot_significant

        results.append({
            'System 1': system1_name,
            'System 2': system2_name,
            'Difference': diff,
            'P-value': p_val,
            'CI Lower': lower,
            'CI Upper': upper,
            'Significant': both_significant
        })

        print(f"  Difference: {diff:.4f}")
        print(f"  P-value: {p_val:.4f}")
        print(f"  95% CI: ({lower:.4f}, {upper:.4f})")
        print(f"  Significant: {both_significant}")
        print("-" * 80)

    return pd.DataFrame(results)

# ============================================================================
# BEA-DEV DATASET TESTS
# ============================================================================

def run_bea_dev_tests(n_permutations=1000, n_bootstrap=1000):
    """Run all significance tests for BEA-dev dataset."""

    print("\n" + "="*80)
    print("BEA-DEV DATASET - Statistical Significance Testing")
    print("="*80)

    # Load data
    excel_path = "../results/bea_outputs.xlsx"
    sources, references = extract_data_from_excel(excel_path, "Gold")

    # Define systems to test
    systems_dict = {
        'Fine-tuned GPT-4o': extract_corrected(excel_path, 'gpt4o_ft_corrected'),
        'GPT-4o': extract_corrected(excel_path, 'gpt4o_corrected'),
        'Fine-tuned LLaMA': extract_corrected(excel_path, 'llama_ft_corrected'),
        'LLaMA': extract_corrected(excel_path, 'llama_corrected'),
        'Ensemble Best 7': extract_corrected(excel_path, 'ensemble_best7_corrected'),
        'N-gram': extract_corrected(excel_path, 'ngram_selected_correction'),
        'Majority Voting': extract_corrected(excel_path, 'majority_voting'),
        'Perplexity': extract_corrected(excel_path, 'perplexity_selected_correction'),
        'Qwen': extract_corrected(excel_path, 'qwen_selected_corrected'),
        'DeepSeek': extract_corrected(excel_path, 'deepseek_corrected'),
        'EditScorer': extract_corrected(excel_path, 'editscorer_corrected'),
        'Fine-tuned LLaMA 13B': extract_corrected(excel_path, 'llama_2_13B_corrected'),
        'Fine-tuned T5 11B': extract_corrected(excel_path, 'T5_11B_corrected'),
    }

    # Define comparisons
    comparisons = [
        # Fine-tuning improvements
        ('Fine-tuned GPT-4o', 'GPT-4o'),
        ('Fine-tuned LLaMA', 'LLaMA'),

        # Fine-tuned GPT-4o vs other systems
        ('Fine-tuned GPT-4o', 'Ensemble Best 7'),
        ('Fine-tuned GPT-4o', 'EditScorer'),
        ('Fine-tuned GPT-4o', 'Fine-tuned LLaMA 13B'),
        ('Fine-tuned GPT-4o', 'Fine-tuned T5 11B'),

        # N-gram vs ensemble variants
        ('N-gram', 'Majority Voting'),
        ('N-gram', 'Qwen'),
        ('N-gram', 'Perplexity'),

        # N-gram vs baseline ensemble
        ('N-gram', 'Ensemble Best 7'),

        # N-gram vs fine-tuned GPT-4o
        ('N-gram', 'Fine-tuned GPT-4o'),
    ]

    # Run tests for each metric
    all_results = {}

    # ERRANT F0.5
    all_results['ERRANT'] = run_significance_tests(
        sources, references, systems_dict, comparisons,
        'ERRANT F0.5', calculate_errant_metrics,
        n_permutations, n_bootstrap, beta=0.5
    )

    # GLEU
    all_results['GLEU'] = run_significance_tests(
        sources, references, systems_dict, comparisons,
        'GLEU', calculate_gleu_metrics,
        n_permutations, n_bootstrap
    )

    # PT-ERRANT
    all_results['PT-ERRANT'] = run_significance_tests(
        sources, references, systems_dict, comparisons,
        'PT-ERRANT', calculate_pt_errant_metrics,
        n_permutations, n_bootstrap, beta=0.5
    )

    # Save results
    with pd.ExcelWriter('bea_dev_significance_tests.xlsx') as writer:
        for metric_name, results_df in all_results.items():
            results_df.to_excel(writer, sheet_name=metric_name, index=False)

    print("\n" + "="*80)
    print("BEA-dev results saved to: bea_dev_significance_tests.xlsx")
    print("="*80)

    return all_results

# ============================================================================
# CONLL DATASET TESTS
# ============================================================================

def run_conll_tests(n_permutations=1000, n_bootstrap=1000):
    """Run significance tests for CoNLL dataset (ERRANT and GLEU only)."""

    print("\n" + "="*80)
    print("CONLL DATASET - Statistical Significance Testing")
    print("="*80)

    # Load data from Excel file
    excel_path = "../results/conll_outputs.xlsx"
    df = pd.read_excel(excel_path)

    sources = df['incorrect'].astype(str).tolist()
    reference_0 = df['reference_0'].astype(str).tolist()
    reference_1 = df['reference_1'].astype(str).tolist()
    references = [reference_0, reference_1]

    # Define systems (only those available in the Excel file)
    systems_dict = {
        'Fine-tuned GPT-4o': df['gpt_corrected'].astype(str).tolist(),
        'Fine-tuned T5 11B': df['t5_corrected'].astype(str).tolist(),
        'EditScorer': df['editscorer_corrected'].astype(str).tolist(),
        'Heterogeneous Ensemble': df['heterogenous_ensemble'].astype(str).tolist(),
    }

    # Define comparisons
    comparisons = [
        ('Fine-tuned GPT-4o', 'Heterogeneous Ensemble'),
        ('Fine-tuned GPT-4o', 'EditScorer'),
        ('Fine-tuned GPT-4o', 'Fine-tuned T5 11B'),
        ('Heterogeneous Ensemble', 'EditScorer'),
        ('Heterogeneous Ensemble', 'Fine-tuned T5 11B'),
    ]

    # Run tests for ERRANT and GLEU only (no PT-ERRANT for CoNLL)
    all_results = {}

    # ERRANT F0.5
    all_results['ERRANT'] = run_significance_tests(
        sources, references, systems_dict, comparisons,
        'ERRANT F0.5', calculate_errant_metrics,
        n_permutations, n_bootstrap, beta=0.5
    )

    # GLEU
    all_results['GLEU'] = run_significance_tests(
        sources, references, systems_dict, comparisons,
        'GLEU', calculate_gleu_metrics,
        n_permutations, n_bootstrap
    )

    # Save results
    with pd.ExcelWriter('conll_significance_tests.xlsx') as writer:
        for metric_name, results_df in all_results.items():
            results_df.to_excel(writer, sheet_name=metric_name, index=False)

    print("\n" + "="*80)
    print("CoNLL results saved to: conll_significance_tests.xlsx")
    print("="*80)

    return all_results

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run statistical significance tests for GEC systems')
    parser.add_argument('--dataset', choices=['bea', 'conll', 'both'], default='both',
                       help='Which dataset to test (default: both)')
    parser.add_argument('--n_permutations', type=int, default=1000,
                       help='Number of permutations for p-value calculation (default: 1000)')
    parser.add_argument('--n_bootstrap', type=int, default=1000,
                       help='Number of bootstrap samples for CI (default: 1000)')

    args = parser.parse_args()

    if args.dataset in ['bea', 'both']:
        bea_results = run_bea_dev_tests(args.n_permutations, args.n_bootstrap)

    if args.dataset in ['conll', 'both']:
        conll_results = run_conll_tests(args.n_permutations, args.n_bootstrap)

    print("\n" + "="*80)
    print("ALL TESTS COMPLETED SUCCESSFULLY")
    print("="*80)
