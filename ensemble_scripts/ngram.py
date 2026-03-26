import pandas as pd
from collections import defaultdict

def ngram_overlap_selection(df, n_value):
    """
    Select corrections based on n-gram overlap analysis.
    """
    # Create results columns
    df['ngram_selected_correction'] = None
    df['ngram_selected_model'] = None
    
    # Process each row
    for idx, row in df.iterrows():
        # Skip if all models agree
        if row['comparison_result'] == '3 are same' or row['comparison_result'] == '2 are same':
            df.at[idx, 'ngram_selected_correction'] = row['selected_correction']
            df.at[idx, 'ngram_selected_model'] = 'all_same'
            continue
        
        # Get corrections from each model
        corrections = {
            'gpt4o': row['gpt4o_ft_corrected'],
            'llama': row['llama_ft_corrected'],
            'deepseek': row['deepseek_corrected']
        }
        
        # Calculate best model based on n-gram overlap
        best_model = get_best_by_ngram_overlap(corrections, n=n_value)
        
        # Update dataframe
        df.at[idx, 'ngram_selected_correction'] = corrections[best_model]
        df.at[idx, 'ngram_selected_model'] = best_model
    
    # # Calculate agreement percentage with gold standard
    # matches = 0
    # total = 0
    # for idx, row in df.iterrows():
    #     # If you have a gold_standard column, use it for comparison
    #     # Otherwise, you'll need to adjust this part based on how you evaluate correctness
    #     if 'gold_standard' in df.columns:
    #         if row['ngram_selected_correction'] == row['gold_standard']:
    #             matches += 1
    #     total += 1
    
    # if total > 0:
    #     agreement_percentage = (matches / total) * 100
    #     print(f"N-gram selection agreement with gold standard: {agreement_percentage:.2f}%")
    
    return df

def get_best_by_ngram_overlap(corrections, n=4):
    """
    Find the model whose correction has highest average n-gram overlap with other models.
    
    Args:
        corrections: Dictionary mapping model names to their corrections
        n: Size of n-grams to use (default: 2 for bigrams)
    
    Returns:
        Name of the model with highest average n-gram overlap
    """
    # Extract n-grams for each correction
    model_ngrams = {}
    for model, text in corrections.items():
        model_ngrams[model] = get_ngrams(text, n)
    
    # Calculate pairwise overlaps
    pairwise_overlaps = defaultdict(list)
    for model1 in corrections:
        for model2 in corrections:
            if model1 != model2:
                overlap = calculate_ngram_overlap(
                    model_ngrams[model1], 
                    model_ngrams[model2]
                )
                pairwise_overlaps[model1].append(overlap)
    
    # Calculate average overlap for each model
    average_overlaps = {}
    for model, overlaps in pairwise_overlaps.items():
        average_overlaps[model] = sum(overlaps) / len(overlaps) if overlaps else 0
    
    # Return the model with highest average overlap
    return max(average_overlaps, key=average_overlaps.get)

def get_ngrams(text, n):
    """
    Extract n-grams from text.
    
    Args:
        text: Input string
        n: Size of n-grams
    
    Returns:
        Set of n-grams
    """
    words = text.lower().split()
    if len(words) < n:
        return {' '.join(words)}  # Return the whole text as a single n-gram if too short
    
    return set(' '.join(words[i:i+n]) for i in range(len(words) - n + 1))

def calculate_ngram_overlap(ngrams1, ngrams2):
    """
    Calculate Jaccard similarity between two sets of n-grams.
    
    Args:
        ngrams1: First set of n-grams
        ngrams2: Second set of n-grams
    
    Returns:
        Jaccard similarity (intersection over union)
    """
    if not ngrams1 and not ngrams2:
        return 1.0  # Both empty means perfect match
    
    intersection = ngrams1.intersection(ngrams2)
    union = ngrams1.union(ngrams2)
    
    return len(intersection) / len(union)

# Example usage
df = pd.read_excel('../results/bea_outputs.xlsx')
# for i in range(100):
df_with_ngram = ngram_overlap_selection(df, 3) # update n value as needed
df_with_ngram.to_excel('corrections_with_ngram.xlsx', index=False)

# Calculate rows where Gold matches any of the three correction columns
matches_any = (
    # (df["Gold"] == df["gpt4o_ft_corrected"]) # |
    # (df["Gold"] == df["llama_ft_corrected"]) # |
    (df["Gold"] == df["ngram_selected_correction"])
)

# Count matches and calculate percentage
matching_rows = matches_any.sum()
total_rows = len(df)

if total_rows > 0:
    matching_percentage = (matching_rows / total_rows) * 100
else:
    matching_percentage = 0

print(f"Percentage of rows where Gold matches any of the three correction columns: {matching_percentage:.2f}%")
    
# n = 3 is the best