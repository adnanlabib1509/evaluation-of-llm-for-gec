# Evaluation Scripts

This directory contains scripts for evaluating GEC system outputs using automated metrics.

## Files

### [automated_metrics.py](automated_metrics.py)
Calculates multiple automated evaluation metrics for GEC systems.

**Input:** Configurable (default: CoNLL dataset)
**Output:** Console output with metric scores

**Metrics calculated:**
- **ERRANT F0.5:** Error-type weighted precision/recall with multiple references
- **GLEU:** Generalized Language Evaluation Understanding
- **PT-ERRANT:** Partial-credit ERRANT using BERTScore weighting
- **IMPARA:** Reference-free quality estimation metric
- **Scribendi:** Reference-free metric using perplexity and edit distance

**Usage:**

Default (evaluates CoNLL dataset):
```bash
python automated_metrics.py
```

**To switch datasets:**

**Option 1 - CoNLL (2 references):**
```python
# Lines 231-232 (already active)
sources, references = extract_data_from_excel_2ref("../results/conll_outputs.xlsx", "reference_0", "reference_1")
hypotheses = extract_corrected("../results/conll_outputs.xlsx", "heterogenous_ensemble")
```

**Option 2 - BEA (1 reference):**
```python
# Uncomment lines 237-239
sources, references_single = extract_data_from_excel("../results/bea_outputs.xlsx", "Gold")
hypotheses = extract_corrected("../results/bea_outputs.xlsx", "gpt4o_ft_corrected")
references = [references_single]  # Wrap single reference in a list
```

**To evaluate different model outputs:** Change the second parameter in `extract_corrected()`:
- BEA options: `"gpt4o_ft_corrected"`, `"llama_ft_corrected"`, `"ngram_selected_correction"`, etc.
- CoNLL options: `"gpt_corrected"`, `"t5_corrected"`, `"editscorer_corrected"`, `"heterogenous_ensemble"`

### Helper Functions

- `extract_data_from_excel()` - Extract sources and single reference
- `extract_data_from_excel_2ref()` - Extract sources and two references (for CoNLL)
- `extract_corrected()` - Extract hypothesis/corrections column
- `calculate_errant_metrics()` - Compute ERRANT F0.5
- `calculate_gleu_metrics()` - Compute GLEU score
- `calculate_pt_errant_metrics()` - Compute PT-ERRANT score
- `calculate_impara_metrics()` - Compute IMPARA score
- `calculate_scribendi_metrics()` - Compute Scribendi score

## Requirements

```bash
pip install pandas gec_metrics nltk openpyxl
```

## Notes

- ERRANT, GLEU, and PT-ERRANT are reference-based metrics
- IMPARA and Scribendi are reference-free metrics (only need source and hypothesis)
- For BEA dataset: use single reference (`extract_data_from_excel`)
- For CoNLL dataset: use two references (`extract_data_from_excel_2ref`)
