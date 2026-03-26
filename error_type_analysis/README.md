# Error Type Analysis

This directory contains scripts for analyzing grammatical error types and correction patterns using ERRANT (ERRor ANnotation Toolkit).

## Files

### [extract_edit.py](extract_edit.py)
Extracts and annotates grammatical errors from GEC model outputs using ERRANT.

**Input:** `../results/bea_outputs.xlsx`
**Output:** `error_types/deepseek.csv` (or gold.csv, gpt4o.csv, llama.csv depending on which column is analyzed)

**What it does:**
- Parses source and corrected sentences with ERRANT
- Identifies error types (e.g., VERB:SVA, NOUN:NUM, R:PREP)
- Extracts original and corrected text for each error
- Saves detailed error annotations to CSV

**Usage:**
```bash
python extract_edit.py
```

**Note:** Modify line 16 to change which model's corrections to analyze:
```python
hypothesis = row['deepseek_corrected']  # Change to 'Gold', 'gpt4o_ft_corrected', etc.
```

### [calculate_rates.py](calculate_rates.py)
Calculates correction rates and false insertion rates for each error type across different models.

**Input:**
- `error_types/gold.csv`
- `error_types/gpt4o.csv`
- `error_types/llama.csv`
- `error_types/deepseek.csv`

**Output:** Console output with tables and statistics

**What it does:**
- **Correction Rate:** Percentage of gold errors actually fixed by each model per error type
- **False Insertion Rate:** Percentage of model corrections not present in gold (over-corrections)
- Generates summary statistics and comparison tables

**Usage:**
```bash
python calculate_rates.py
```

### [calculate_correlation.py](calculate_correlation.py)
Calculates Spearman rank correlations between models' correction and false insertion rates.

**Input:** Correction and false insertion rate data
**Output:** Console output with correlation coefficients and p-values

**What it does:**
- Computes correlation between GPT-4o vs LLaMA correction patterns
- Computes correlation between GPT-4o vs DeepSeek correction patterns
- Computes correlation between LLaMA vs DeepSeek correction patterns
- Shows if models have similar strengths/weaknesses across error types

**Usage:**
```bash
python calculate_correlation.py
```

## Workflow

1. **Extract errors** for each model/reference:
   ```bash
   # Edit extract_edit.py to set hypothesis column, then run:
   python extract_edit.py  # Creates error_types/model.csv
   ```

2. **Calculate metrics** comparing models to gold standard:
   ```bash
   python calculate_rates.py  # Shows correction and false insertion rates
   ```

3. **Analyze correlations** between model behaviors:
   ```bash
   python calculate_correlation.py  # Shows model similarity patterns
   ```

## ERRANT Error Type Descriptions

ERRANT uses error types in the format: `[Operation]:[Error Category]:[Subcategory]`

### Operations
- `M:` Missing - word/element present in correction but absent in original
- `R:` Replacement - different word/form used in correction compared to original
- `U:` Unnecessary - word/element present in original but omitted in correction

### Complete Error Type List

#### Missing (M:) - Insertions Needed
| Code | Description |
|------|-------------|
| M:ADJ | Missing adjective |
| M:ADV | Missing adverb |
| M:CONJ | Missing conjunction |
| M:CONTR | Missing contraction |
| M:DET | Missing determiner |
| M:NOUN | Missing noun |
| M:NOUN:POSS | Missing possessive noun |
| M:OTHER | Other missing words |
| M:PART | Missing particle |
| M:PREP | Missing preposition |
| M:PRON | Missing pronoun |
| M:PUNCT | Missing punctuation |
| M:VERB | Missing verb |
| M:VERB:FORM | Missing verb form |
| M:VERB:TENSE | Missing verb tense |

#### Replacement (R:) - Substitutions Needed
| Code | Description |
|------|-------------|
| R:ADJ | Different adjective needed |
| R:ADJ:FORM | Different adjective form needed |
| R:ADV | Different adverb needed |
| R:CONJ | Different conjunction needed |
| R:CONTR | Different contraction needed |
| R:DET | Different determiner needed |
| R:MORPH | Different morphological form needed |
| R:NOUN | Different noun needed |
| R:NOUN:INFL | Different noun inflection needed |
| R:NOUN:NUM | Singular/plural correction needed |
| R:NOUN:POSS | Different possessive form needed |
| R:ORTH | Orthography/formatting correction needed |
| R:OTHER | Other replacement needed |
| R:PART | Different particle needed |
| R:PREP | Different preposition needed |
| R:PRON | Different pronoun needed |
| R:PUNCT | Different punctuation needed |
| R:SPELL | Spelling correction needed |
| R:VERB | Different verb needed |
| R:VERB:FORM | Different verb form needed |
| R:VERB:INFL | Different verb inflection needed |
| R:VERB:SVA | Subject-verb agreement correction needed |
| R:VERB:TENSE | Different verb tense needed |
| R:WO | Word order correction needed |

#### Unnecessary (U:) - Deletions Needed
| Code | Description |
|------|-------------|
| U:ADJ | Unnecessary adjective |
| U:ADV | Unnecessary adverb |
| U:CONJ | Unnecessary conjunction |
| U:CONTR | Unnecessary contraction |
| U:DET | Unnecessary determiner |
| U:NOUN | Unnecessary noun |
| U:NOUN:POSS | Unnecessary possessive noun |
| U:OTHER | Other unnecessary words |
| U:PART | Unnecessary particle |
| U:PREP | Unnecessary preposition |
| U:PRON | Unnecessary pronoun |
| U:PUNCT | Unnecessary punctuation |
| U:VERB | Unnecessary verb |
| U:VERB:FORM | Unnecessary verb form |
| U:VERB:TENSE | Unnecessary verb tense |

## Requirements

```bash
pip install pandas errant tqdm openpyxl
```
