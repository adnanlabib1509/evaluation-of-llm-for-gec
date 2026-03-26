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
├── dataset/                    # Dataset processing scripts
├── training_scripts/           # Model fine-tuning scripts (GPT-4o, LLaMA)
├── inference_scripts/          # Model inference scripts for generating corrections
├── ensemble_scripts/           # Ensemble selection methods
├── error_type_analysis/        # Error type analysis using ERRANT
├── evaluation/                 # Automated metrics calculation
├── confidence_scores/          # Statistical significance testing
└── results/                    # Output files (bea_outputs.xlsx, conll_outputs.xlsx)
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
# Fine-tune GPT-4o (via OpenAI API)
python train_gpt4o.py --epochs 2 --train_file ../dataset/training_dataset/ABC_train.jsonl

# Fine-tune LLaMA 3.3 (with LoRA)
python train_llama.py --rank 8 --alpha 16 --lr 2e-5 --epochs 2 --batch_size 32
```

### 3. Run Inference
```bash
cd inference_scripts
# Generate corrections with fine-tuned models
python inference_gpt4o.py --input ../results/bea_outputs.xlsx --output_column gpt4o_ft_corrected
python inference_llama.py --input ../results/bea_outputs.xlsx --output_column llama_ft_corrected
python inference_deepseek.py --input ../results/bea_outputs.xlsx --output_column deepseek_corrected
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
