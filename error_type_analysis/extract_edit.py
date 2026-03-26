import pandas as pd
import errant
import tqdm

# Initialize ERRANT
annotator = errant.load('en')

# Read input Excel file
df = pd.read_excel('../results/bea_outputs.xlsx')

# Prepare a list to store errors
error_data = []

for idx, row in df.iterrows():
    source = row['original']
    hypothesis = row['deepseek_corrected']
    
    # Parse sentences with ERRANT
    orig = annotator.parse(source)
    cor = annotator.parse(hypothesis)
    
    # Get edits (error annotations)
    edits = annotator.annotate(orig, cor)
    
    # Store each error's details
    for edit in edits:
        error_data.append({
            'row_id': idx,               # To track original row
            'error_type': edit.type,      # e.g., "VERB:SVA"
            'original_text': edit.o_str,  # e.g., "go"
            'corrected_text': edit.c_str,  # e.g., "goes"
            'source_sentence': source,    # Full original sentence
            'hypothesis_sentence': hypothesis  # Full corrected sentence
        })

# Convert to DataFrame and save as CSV
error_df = pd.DataFrame(error_data)
error_df.to_csv('error_types/deepseek.csv', index=False)

print(f"Saved error analysis to 'error_analysis.csv'")