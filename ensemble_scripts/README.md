# Ensemble Selection Scripts

This directory contains scripts for selecting the best GEC corrections from multiple models using different ensemble strategies.

## Files

### [ngram.py](ngram.py)
Selects corrections based on n-gram overlap analysis. Chooses the model whose correction has the highest average n-gram overlap with other models.

**Input:** `../results/bea_outputs.xlsx`
**Output:** `corrections_with_ngram.xlsx`
**Method:** N-gram Jaccard similarity (n=3 by default, can be changed)

**Usage:**
```bash
python ngram.py
```

### [perplexity.py](perplexity.py)
Selects corrections based on perplexity scores using Qwen-7B. Lower perplexity indicates more fluent text.

**Input:** `../results/bea_outputs.xlsx`
**Output:** `corrections_with_perplexity.xlsx`
**Method:** Perplexity-based selection (Qwen-7B model)

**Usage:**
```bash
python perplexity.py
```

### [qwen_judge.py](qwen_judge.py)
Uses Qwen-2.5-7B-Instruct as an LLM judge to select the best correction when models disagree.

**Input:** `../results/bea_outputs.xlsx`
**Output:** `qwen_judge.xlsx`
**Method:** LLM-as-judge with targeted prompting

**Usage:**
```bash
python qwen_judge.py
```

### [heterogenous_ensemble_bea.py](heterogenous_ensemble_bea.py)
Creates a heterogeneous ensemble for BEA dataset using majority voting among GPT-4o-Fine-tuned, EditScorer, and T5-11B.

**Input:** `../results/bea_outputs.xlsx`
**Output:** `heterogenous_ensembles.xlsx`
**Method:** Majority voting (2-out-of-3 agreement)

**Usage:**
```bash
python heterogenous_ensemble_bea.py
```

### [heterogenous_ensemble_conll.py](heterogenous_ensemble_conll.py)
Creates a heterogeneous ensemble for CoNLL dataset using majority voting among GPT-4o-Fine-tuned, EditScorer, and T5.

**Input:** `../results/conll_outputs.xlsx`
**Output:** `heterogenous_ensembles_conll.xlsx`
**Method:** Majority voting (2-out-of-3 agreement)

**Usage:**
```bash
python heterogenous_ensemble_conll.py
```

## Ensemble Strategies

- **N-gram Overlap:** Selects the correction most similar to others based on n-gram overlap
- **Perplexity:** Selects the most fluent correction based on language model perplexity
- **LLM Judge:** Uses an instruction-tuned LLM to judge which correction is best
- **Majority Voting:** Selects the correction agreed upon by the majority of models

## Requirements

```bash
pip install pandas torch transformers tqdm openpyxl
```

For perplexity and qwen_judge scripts, GPU is recommended.
