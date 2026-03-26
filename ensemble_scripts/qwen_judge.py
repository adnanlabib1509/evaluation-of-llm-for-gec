import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import logging as transformers_logging
import json
import pandas as pd
from tqdm import tqdm
import time

# Suppress transformers warnings
transformers_logging.set_verbosity_error()


def evaluate_corrections(model, tokenizer, original_sentence, corrections, model_names):
    """
    Evaluate multiple corrections and select the best one without revealing model names.

    Args:
        model: The language model for evaluation
        tokenizer: The tokenizer for the model
        original_sentence (str): The original sentence with errors
        corrections (list): List of corrected sentences
        model_names (list): List of model names corresponding to corrections

    Returns:
        dict: Dictionary containing evaluation results
    """
    # Construct the prompt with anonymous correction options
    options_text = ""
    for i, correction in enumerate(corrections):
        options_text += f"Option {chr(65+i)}: \"{correction}\"\n"

    prompt = f"""Compare the sentences given below and tell me which one (A, B, or C) is the most grammatically correct version of the original given below.

Original: "{original_sentence}"

Sentences: "{options_text}"

Provide your response in JSON format as follows:
{{
  "best_option": "The letter of the best option (A, B, C, etc.)",
  "reasoning": "A brief explanation of why this is the best option",
}}"""

    # Create messages for chat format
    messages = [
        {"role": "system", "content": "You are a grammatical error correction expert."},
        {"role": "user", "content": prompt}
    ]

    # Apply chat template and generate response
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=1024,
        do_sample=False
    )

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # Parse the JSON response
    try:
        # Extract JSON from the response
        json_str = response
        if "```json" in response:
            json_str = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            json_str = response.split("```")[1].split("```")[0].strip()

        result = json.loads(json_str)

        # Get the selected best option
        best_option_letter = result.get("best_option", "")

        # Check if it's exactly "A", "B", or "C"
        if best_option_letter in ["A", "B", "C"]:
            best_idx = ord(best_option_letter) - ord('A')
            best_correction = corrections[best_idx]
            best_model = model_names[best_idx]
        else:
            # Invalid option - will be handled in the main function
            best_correction = None
            best_model = None
            best_idx = None

        return {
            "best_correction": best_correction,
            "best_model": best_model,
            "reasoning": result.get("reasoning", ""),
            "best_option": best_option_letter,
            "best_idx": best_idx,
            "raw_response": response
        }

    except json.JSONDecodeError:
        print("Failed to parse JSON from response")
        print("Raw response:", response)
        # Return None for all fields for manual review
        return {
            "best_correction": None,
            "best_model": None,
            "reasoning": "Failed to parse JSON from response",
            "best_option": None,
            "best_idx": None,
            "raw_response": response
        }


def process_excel_file(input_file_path, output_file_path, model, tokenizer):
    """
    Process an Excel file containing original sentences and corrections,
    evaluate only the disagreement cases with the Qwen model, and save results to a new file.

    Args:
        input_file_path (str): Path to input Excel file
        output_file_path (str): Path to output Excel file
        model: The language model for evaluation
        tokenizer: The tokenizer for the model

    Returns:
        pd.DataFrame: DataFrame with results
    """
    # Read the Excel file
    df = pd.read_excel(input_file_path)

    # Create a copy of the dataframe to preserve original data
    df_results = df.copy()

    # Add new columns for Qwen-based selection
    df_results['qwen_targeted_model'] = None
    df_results['qwen_targeted_correction'] = None
    df_results['selection_method'] = None

    # Initialize counter for invalid responses
    invalid_count = 0

    # Only process rows where all three models disagree
    disagreement_mask = df['comparison_result'] == 'none are same'

    # For majority cases, just use the existing selection
    majority_mask = ~disagreement_mask
    df_results.loc[majority_mask, 'qwen_targeted_correction'] = df.loc[majority_mask, 'selected_correction']
    df_results.loc[majority_mask, 'selection_method'] = 'majority_voting'

    # Process only disagreement cases with tqdm progress bar
    for index, row in tqdm(df[disagreement_mask].iterrows(), total=disagreement_mask.sum(),
                          desc="Processing disagreement cases with Qwen"):
        original = row['original']
        corrections = [
            row['gpt4o_ft_corrected'],
            row['llama_ft_corrected'],
            row['deepseek_corrected']
        ]
        model_names = ["gpt4o", "llama", "deepseek"]

        # Evaluate corrections using Qwen
        result = evaluate_corrections(model, tokenizer, original, corrections, model_names)

        # Check if result has invalid option
        if result["best_option"] not in ["A", "B", "C"]:
            invalid_count += 1
            print(f"Invalid option detected ({invalid_count}): '{result['best_option']}'")
            print(f"For original: '{original[:50]}...'")
            # Fall back to GPT-4o for invalid responses
            df_results.at[index, 'qwen_targeted_model'] = 'gpt4o (fallback)'
            df_results.at[index, 'qwen_targeted_correction'] = row['gpt4o_ft_corrected']
            df_results.at[index, 'selection_method'] = 'qwen_invalid_fallback'
        else:
            # Update the results dataframe
            df_results.at[index, 'qwen_targeted_model'] = result['best_model']
            df_results.at[index, 'qwen_targeted_correction'] = result['best_correction']
            df_results.at[index, 'selection_method'] = 'qwen_targeted'
            df_results.at[index, 'raw_response'] = result['raw_response']

    # Print statistics
    total_entries = len(df)
    disagreements = disagreement_mask.sum()

    print(f"\n{'='*50}")
    print("PROCESSING STATISTICS")
    print(f"{'='*50}")
    print(f"Total entries: {total_entries}")
    print(f"Entries with majority agreement: {total_entries - disagreements} ({(total_entries - disagreements)/total_entries:.2%})")
    print(f"Entries requiring Qwen judgment: {disagreements} ({disagreements/total_entries:.2%})")
    print(f"Invalid Qwen responses: {invalid_count} ({invalid_count/disagreements:.2%} of judgments)")

    # Save to Excel
    df_results.to_excel(output_file_path, index=False)

    print(f"\nResults saved to {output_file_path}")

    # Calculate match with gold standard for analysis
    if 'Gold' in df.columns:
        matches = (df_results['qwen_targeted_correction'] == df['Gold']).sum()
        match_percentage = (matches / len(df)) * 100
        print(f"Match with gold standard: {matches}/{len(df)} ({match_percentage:.2f}%)")

    print(f"{'='*50}\n")

    return df_results


def main():
    """Main function to run Qwen-based model selection."""
    # Configuration
    model_id = "Qwen/Qwen2.5-7B-Instruct"
    cache_dir = "../cache"
    input_file = "../results/bea_outputs.xlsx"
    output_file = "qwen_judge.xlsx"

    print("Loading model and tokenizer...")
    print(f"Model: {model_id}")

    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        use_fast=True,
        cache_dir=cache_dir
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        load_in_4bit=True,  # 4-bit quantization
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=cache_dir
    )

    print("Model loaded successfully!\n")

    # Process the Excel file
    print(f"Processing {input_file}...")
    df_results = process_excel_file(input_file, output_file, model, tokenizer)

    print("Processing complete!")


if __name__ == "__main__":
    main()
