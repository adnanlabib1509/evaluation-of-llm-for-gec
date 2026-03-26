# Statistical Significance Testing for GEC Systems

This script performs comprehensive statistical significance testing for Grammatical Error Correction (GEC) systems using permutation tests and bootstrap confidence intervals.

## Requirements

```bash
pip install pandas gec_metrics numpy nltk tqdm openpyxl
```

## Input Data Format

Both datasets use Excel files (`.xlsx`) containing source sentences, references, and system outputs.

### BEA-dev Dataset

**File:** `../results/bea_outputs.xlsx`

**Columns:**
- `original` - Source sentences (with errors)
- `Gold` - Reference corrections
- System output columns: `gpt4o_ft_corrected`, `gpt4o_corrected`, `llama_ft_corrected`, `llama_corrected`, `ngram_selected_correction`, `ensemble_best7_corrected`, `editscorer_corrected`, etc.

### CoNLL Dataset

**File:** `../results/conll_outputs.xlsx`

**Columns:**
- `incorrect` - Source sentences (with errors)
- `reference_0` - First reference correction
- `reference_1` - Second reference correction
- System output columns: `gpt_corrected`, `t5_corrected`, `editscorer_corrected`, `heterogenous_ensemble`

## Usage

### Run tests on both datasets:
```bash
python confidence_score_script.py
```

### Run tests on specific dataset:
```bash
# BEA-dev only
python confidence_score_script.py --dataset bea

# CoNLL only
python confidence_score_script.py --dataset conll
```

### Adjust statistical parameters:
```bash
python confidence_score_script.py --n_permutations 10000 --n_bootstrap 10000
```

## Input File Paths

The script is pre-configured to use the existing data files:

**BEA-dev:**
- Line 307: `excel_path = "../results/bea_outputs.xlsx"`

**CoNLL:**
- Line 398: `excel_path = "../results/conll_outputs.xlsx"`

**Note:** The script should be run from the `confidence_scores/` directory for the relative paths to work correctly.

## Output

The script generates Excel files with statistical test results:

- `bea_dev_significance_tests.xlsx` - BEA-dev results (3 sheets: ERRANT F0.5, GLEU, PT-ERRANT)
- `conll_significance_tests.xlsx` - CoNLL results (2 sheets: ERRANT F0.5, GLEU)

Each sheet contains:
- System comparisons
- Difference scores
- P-values (permutation test)
- 95% Bootstrap confidence intervals
- Significance indicator (True/False)

