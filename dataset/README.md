# Dataset Processing Scripts

This directory contains scripts to convert M2 formatted datasets into text files for GEC evaluation.

## Files

### [bea_dataset_builder.py](bea_dataset_builder.py)
Converts BEA-2019 M2 files to text format.

**Input:** `original_datasets/bea_2019_dev/*.m2`
**Output:** `testing_datasets/*.orig.txt` and `*.cor.txt`

**Usage:**
```bash
python bea_dataset_builder.py
```

### [conll_dataset_builder.py](conll_dataset_builder.py)
Converts CoNLL-2014 M2 file to text format with multiple references.

**Input:** `original_datasets/conll_2014/conll_m2_file.m2`
**Output:**
- `testing_datasets/conll_incorrect.txt` (source sentences)
- `testing_datasets/conll_reference_0.txt` (first annotator)
- `testing_datasets/conll_reference_1.txt` (second annotator)

**Usage:**
```bash
python conll_dataset_builder.py
```

## Directory Structure

```
dataset/
├── original_datasets/        # Raw M2 files
│   ├── bea_2019_dev/
│   └── conll_2014/
├── testing_datasets/         # Processed text files
└── training_dataset/         # Training data
```

## M2 Format

M2 is a standard format for GEC annotation with error spans and corrections. These scripts extract the original and corrected sentences for evaluation.
