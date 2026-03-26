import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


def calculate_perplexity(text, model, tokenizer, device='cuda'):
    """
    Calculate perplexity using a causal language model.

    Args:
        text (str): The text to calculate perplexity for
        model: The causal language model
        tokenizer: The tokenizer for the model
        device (str): Device to run on ('cuda' or 'cpu')

    Returns:
        float: The perplexity score
    """
    # Tokenize input
    inputs = tokenizer(text, return_tensors='pt').to(device)

    # Get token IDs
    input_ids = inputs['input_ids']

    # Set target as input shifted right
    target_ids = input_ids.clone()

    # Calculate log likelihood
    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)
        neg_log_likelihood = outputs.loss

    # Calculate perplexity
    perplexity = torch.exp(neg_log_likelihood).item()

    return perplexity


def select_by_perplexity(df, model, tokenizer, device='cuda'):
    """
    Select the best correction based on perplexity (lowest = most fluent).

    Args:
        df (pd.DataFrame): DataFrame with correction columns
        model: The causal language model
        tokenizer: The tokenizer for the model
        device (str): Device to run on ('cuda' or 'cpu')

    Returns:
        pd.DataFrame: DataFrame with perplexity-based selections added
    """
    # Create new columns for perplexity-based correction
    df['perplexity_selected_correction'] = None
    df['min_perplexity_model'] = None

    # Process each row with tqdm progress bar
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Calculating perplexities"):
        # Skip if all models agree (all are same)
        if row['comparison_result'] == '3 are same' or row['comparison_result'] == '2 are same':
            df.at[idx, 'perplexity_selected_correction'] = row['selected_correction']
            df.at[idx, 'min_perplexity_model'] = 'all_same'
            continue

        # Get perplexity for each correction
        perplexities = {}
        perplexities['gpt4o'] = calculate_perplexity(row['gpt4o_ft_corrected'], model, tokenizer, device)
        perplexities['llama'] = calculate_perplexity(row['llama_ft_corrected'], model, tokenizer, device)
        perplexities['deepseek'] = calculate_perplexity(row['deepseek_corrected'], model, tokenizer, device)

        # Find model with lowest perplexity (most fluent)
        min_perplexity_model = min(perplexities, key=perplexities.get)

        # Get the correction with lowest perplexity
        if min_perplexity_model == 'gpt4o':
            correction = row['gpt4o_ft_corrected']
        elif min_perplexity_model == 'llama':
            correction = row['llama_ft_corrected']
        else:
            correction = row['deepseek_corrected']

        # Update dataframe
        df.at[idx, 'perplexity_selected_correction'] = correction
        df.at[idx, 'min_perplexity_model'] = min_perplexity_model

    return df


def main():
    """Main function to run perplexity-based model selection."""
    # Configuration
    cache_dir = "../cache"
    input_file = '../results/bea_outputs.xlsx'
    output_file = 'corrections_with_perplexity.xlsx'
    model_name = "Qwen/Qwen-7B"

    print("Loading model and tokenizer...")
    # Load Qwen model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir=cache_dir
    ).to('cuda')

    print(f"Loading data from {input_file}...")
    # Load the Excel file
    df = pd.read_excel(input_file)

    print("Selecting best corrections based on perplexity...")
    # Select by perplexity
    df = select_by_perplexity(df, model, tokenizer)

    # Calculate metrics to compare with original majority voting
    agreement = (df['selected_correction'] == df['perplexity_selected_correction']).mean() * 100
    print(f"\nAgreement between majority voting and perplexity selection: {agreement:.2f}%")

    # Calculate rows where Gold matches perplexity selection
    matches_any = (df["Gold"] == df["perplexity_selected_correction"])
    matching_rows = matches_any.sum()
    total_rows = len(df)

    if total_rows > 0:
        matching_percentage = (matching_rows / total_rows) * 100
    else:
        matching_percentage = 0

    print(f"Percentage where Gold matches perplexity selection: {matching_percentage:.2f}%")

    # Save results
    print(f"\nSaving results to {output_file}...")
    df.to_excel(output_file, index=False)

    print("\nProcessing complete!")


if __name__ == "__main__":
    main()
