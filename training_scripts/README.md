# Training Scripts

Scripts for preparing datasets and fine-tuning models for grammar error correction.

## Files

### A01_gpt_prepare_dataset.py
Prepares JSONL datasets (train/val/test) from raw text files for GPT fine-tuning. Uses `DatasetPreparation` class.

### A02_gpt_finetune.py
Fine-tunes GPT-4o on OpenAI's servers. Uploads train/val datasets, creates fine-tuning job, and waits for completion.

### B01_prepare_dataset.py
Creates HuggingFace dataset from original/corrected text files and uploads to HuggingFace Hub.

### E01_llama_finetune.py
Fine-tunes Llama 3.x models locally using LoRA/QLoRA. Uses same JSONL format as GPT pipeline.

### Z01_add_sentence_id_to_dataset.py
Utility script to add missing sentence IDs to result files or check for duplicate sentences.
