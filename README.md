# GEC Ensemble Methods with Fine-Tuned LLMs

This repository contains code for Grammatical Error Correction (GEC) using fine-tuned large language models and ensemble methods.

## Dataset

**Training:** 34,308 sentences from the ABC train partition of the W&I+LOCNESS dataset (BEA 2019 Shared Task)

**Testing:**
- **BEA 2019 Dev:** ABCN development set (includes native English texts)
- **CoNLL-14 Test:** Essays from National University of Singapore students (primarily Asian L1 backgrounds)

The BEA 2019 dataset contains English essays written by non-native ESL students across A1 to C2 CEFR levels. These datasets enable direct comparison with baseline systems evaluated on the same benchmarks.

## Experimental Settings

### GPT-4o Fine-tuning
- **API:** OpenAI's fine-tuning API
- **Epochs:** 2
- **Cost:** $88.48 (including inference)
- **Note:** Reproducibility limited due to proprietary model

### LLaMA 3.3 Fine-tuning
- **Method:** LoRA via PEFT library
- **Model:** Fine-tuned LLaMA 3.3 70B ([judywq/llama-ft-gec](https://huggingface.co/judywq/llama-ft-gec))
- **Hyperparameters:**
  - Rank: 8
  - Alpha: 16
  - Learning rate: 2e-5
  - Optimizer: AdamW (betas=(0.9, 0.999), epsilon=1e-8)
  - Gradient accumulation steps: 8
  - Batch size: 32
- **Hardware:** 2x H200 GPUs (141GB VRAM, 188GB RAM each)
- **GPU Cost:** $3.47 per GPU-hour
- **Epochs:** 2
- **Total Cost:** $27.86 (including fine-tuning and inference)

## Repository Structure

```
AIED_Code/
├── dataset/                    # Dataset processing and raw data files
│   ├── training_dataset/       # ABC training data (original/corrected)
│   ├── testing_datasets/       # BEA and CoNLL test data
│   ├── bea_dataset_builder.py  # Convert BEA M2 to text format
│   └── conll_dataset_builder.py # Convert CoNLL M2 to text format
├── training_scripts/           # Model fine-tuning scripts
│   ├── A01_gpt_prepare_dataset.py  # Prepare JSONL for GPT fine-tuning
│   ├── A02_gpt_finetune.py         # Fine-tune GPT-4o via OpenAI API
│   ├── B01_prepare_dataset.py      # Create HuggingFace dataset
│   └── E01_llama_finetune.py       # Fine-tune Llama with LoRA
├── inference_scripts/          # Model inference and result formatting
│   ├── A03_gpt_inference.py        # Run GPT-4o inference
│   ├── A04_gpt_format_output.py    # Convert JSONL results to Excel
│   ├── C01_inference_parrallel.py  # Generic parallel inference (LiteLLM)
│   ├── D01_prepare_openai_batch.py # Prepare OpenAI batch jobs
│   ├── D02_run_openai_batch.py     # Run OpenAI batch processing
│   └── E02_llama_inference.py      # Run Llama inference
├── ensemble_scripts/           # Ensemble selection methods
├── error_type_analysis/        # Error type analysis using ERRANT
├── evaluation/                 # Automated metrics calculation
├── confidence_scores/          # Statistical significance testing
├── lib/                        # Shared utilities and helper classes
│   ├── settings.py             # Configuration (paths, models, prompts)
│   ├── dataset_preparation.py  # Dataset preparation logic
│   ├── finetuning_helper.py    # OpenAI fine-tuning helper
│   ├── model_runner.py         # Model inference runner
│   └── data_formatter.py       # Result formatting to Excel
└── results/                    # Output files (JSONL, Excel, job metadata)
```

## Quick Start

### 1. Process Datasets
```bash
cd dataset
python bea_dataset_builder.py    # Convert BEA M2 to text
python conll_dataset_builder.py  # Convert CoNLL M2 to text
```

### 2. Fine-tune Models
```bash
cd training_scripts

# Prepare datasets (train/val/test JSONL files)
python A01_gpt_prepare_dataset.py

# Fine-tune GPT-4o via OpenAI API
python A02_gpt_finetune.py

# Fine-tune LLaMA 3.3 with LoRA
python E01_llama_finetune.py
```

### 3. Run Inference
```bash
cd inference_scripts

# Run GPT-4o fine-tuned model inference
python A03_gpt_inference.py

# Run Llama fine-tuned model inference
python E02_llama_inference.py

# Run parallel inference with any model (via LiteLLM)
python C01_inference_parrallel.py --model openai/gpt-4o-mini

# Format results to Excel
python A04_gpt_format_output.py
```

### 4. Run Ensemble Methods
```bash
cd ensemble_scripts
python ngram.py                           # N-gram overlap selection
python perplexity.py                      # Perplexity-based selection
python qwen_judge.py                      # LLM-as-judge selection
python heterogenous_ensemble_bea.py       # Majority voting (BEA)
python heterogenous_ensemble_conll.py     # Majority voting (CoNLL)
```

### 5. Evaluate with Metrics
```bash
cd evaluation
python automated_metrics.py  # Calculate ERRANT, GLEU, PT-ERRANT, IMPARA, Scribendi
```

### 6. Statistical Significance Testing
```bash
cd confidence_scores
python confidence_score_script.py --dataset both  # Run permutation tests and bootstrap CI
```

### 7. Error Type Analysis
```bash
cd error_type_analysis
python extract_edit.py        # Extract error types with ERRANT
python calculate_rates.py     # Calculate correction/false insertion rates
python calculate_correlation.py  # Analyze model correlations
```

## Requirements

```bash
pip install pandas torch transformers gec_metrics errant nltk tqdm openpyxl scipy numpy
```

For GPU acceleration (recommended for ensemble methods):
```bash
pip install bitsandbytes accelerate peft
```

## Citation

If you use this code, please cite our paper:
```
[Citation information will be added after publication]
```
