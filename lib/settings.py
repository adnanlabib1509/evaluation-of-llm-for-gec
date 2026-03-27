train_files = {
    "original": "dataset/training_dataset/ABC_train_original_sentences.txt",
    "corrected": "dataset/training_dataset/ABC_train_corrected_sentences.txt",
}

test_files = {
    "original": "dataset/testing_datasets/bea_original_sentences.txt",
    "corrected": "dataset/testing_datasets/bea_corrected_sentences.txt",
}

################
# OpenAI models

# Add these configurations
dataset_train_filename = "dataset/train.jsonl"
dataset_val_filename = "dataset/val.jsonl"
dataset_test_filename = "dataset/test.jsonl"
dataset_test_openai_batch_filename = "dataset/test_openai_batch.jsonl"
dataset_test_result_gpt_4o_baseline_filename = "results/test_result_gpt_4o_baseline.jsonl"
dataset_test_result_gpt_4o_finetuned_filename = "results/test_result_gpt_4o_finetuned.jsonl"

train_rate = 0.8

# Job
run_id = "20250225"
file_id_filename = f"results/job/{run_id}/file_ids.json"
job_id_filename = f"results/job/{run_id}/job_id.json"

# Model
fine_tuning_base_model_id = "gpt-4o-2024-08-06"  # or your preferred base model
model_suffix = "grammar-correction"

inference_finetuned_model_temperature = 0

inference_base_model_id = "gpt-4o-2024-08-06"
inference_base_model_temperature = 0

DEFAULT_LOG_LEVEL = "INFO"

# Add these lines to your existing settings.py
excel_output_dir = "results/excel"
gpt_4o_baseline_results_excel = "results/excel/baseline_results_gpt_4o.xlsx"
gpt_4o_finetuned_results_excel = "results/excel/finetuned_results_gpt_4o.xlsx"

################
# Custom models

dataset_test_result_deepseek_baseline_filename = "results/test_result_deepseek_baseline.jsonl"
deepseek_baseline_results_excel = "results/excel/baseline_results_deepseek.xlsx"


inference_prompt_template = """You are an English linguist and your task is to correct the grammatical and mechanical errors in English sentences. 
Please make only necessary corrections to the extent that a sentence will be free from errors and comprehensible. 
Do not alter word choices unnecessarily (e.g., replacing words with synonyms) or make stylistic improvements. 
Also, the sentences are tokenized, which means punctuation marks are separated from the English words by spaces. 
When returning the corrected sentences, please use the same tokenized format. 
Please respond in the following JSON format:
{{
  "corrected": "..."
}}

The original sentence is:
{original}"""
