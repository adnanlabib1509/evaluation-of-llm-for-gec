import pandas as pd

# Read CSV files
gold = pd.read_csv('error_types/gold.csv')
gpt4o = pd.read_csv('error_types/gpt4o.csv')
llama = pd.read_csv('error_types/llama.csv')
deepseek = pd.read_csv('error_types/deepseek.csv')

def calculate_correction_rate(gold_df, model_df):
    """Calculate what percentage of gold errors were actually corrected by the model"""
    correction_rates = {}
    gold_counts = {}
    
    # Create sets of (row_id, error_type) pairs for efficient lookup
    gold_errors = set(zip(gold_df['row_id'], gold_df['error_type']))
    model_corrections = set(zip(model_df['row_id'], model_df['error_type']))
    
    # For each error type in gold
    for error_type in gold_df['error_type'].unique():
        # Get all gold instances of this error type
        gold_instances = gold_df[gold_df['error_type'] == error_type]
        total_gold = len(gold_instances)
        gold_counts[error_type] = total_gold
        
        # Count how many of these were actually corrected by the model
        corrected_count = 0
        for _, row in gold_instances.iterrows():
            if (row['row_id'], row['error_type']) in model_corrections:
                corrected_count += 1
        
        correction_rates[error_type] = (corrected_count / total_gold * 100) if total_gold > 0 else 0
    
    return correction_rates, gold_counts

def calculate_false_insertion_rate(gold_df, model_df):
    """Calculate what percentage of model corrections were not in gold (false insertions)"""
    false_insertion_rates = {}
    false_insertion_counts = {}
    
    # Create sets of (row_id, error_type) pairs for efficient lookup
    gold_errors = set(zip(gold_df['row_id'], gold_df['error_type']))
    
    # For each error type in model corrections
    for error_type in model_df['error_type'].unique():
        # Get all model instances of this error type
        model_instances = model_df[model_df['error_type'] == error_type]
        total_model = len(model_instances)
        
        # Count how many of these were NOT in gold (false insertions)
        false_insertions = 0
        for _, row in model_instances.iterrows():
            if (row['row_id'], row['error_type']) not in gold_errors:
                false_insertions += 1
        
        false_insertion_counts[error_type] = false_insertions
        false_insertion_rates[error_type] = (false_insertions / total_model * 100) if total_model > 0 else 0
    
    return false_insertion_rates, false_insertion_counts

# Calculate metrics for each model
print("Calculating correction rates and false insertion rates...")
print()

# Correction rates (how many gold errors were actually fixed)
gpt4o_correction, gpt4o_gold_counts = calculate_correction_rate(gold, gpt4o)
llama_correction, llama_gold_counts = calculate_correction_rate(gold, llama)
deepseek_correction, deepseek_gold_counts = calculate_correction_rate(gold, deepseek)

# False insertion rates (how many model corrections were not in gold)
gpt4o_false, gpt4o_false_counts = calculate_false_insertion_rate(gold, gpt4o)
llama_false, llama_false_counts = calculate_false_insertion_rate(gold, llama)
deepseek_false, deepseek_false_counts = calculate_false_insertion_rate(gold, deepseek)

# Get all error types from all datasets
all_error_types = set(gold['error_type'].unique()) | set(gpt4o['error_type'].unique()) | \
                 set(llama['error_type'].unique()) | set(deepseek['error_type'].unique())

# Create correction rates table
correction_data = pd.DataFrame({
    'Count': [gpt4o_gold_counts.get(et, 0) for et in sorted(all_error_types)],
    'GPT-4o Correction %': [gpt4o_correction.get(et, 0) for et in sorted(all_error_types)],
    'Llama Correction %': [llama_correction.get(et, 0) for et in sorted(all_error_types)],
    'DeepSeek Correction %': [deepseek_correction.get(et, 0) for et in sorted(all_error_types)]
}, index=sorted(all_error_types))

# Create false insertion rates table  
false_insertion_data = pd.DataFrame({
    'Count': [max(gpt4o_false_counts.get(et, 0), llama_false_counts.get(et, 0), deepseek_false_counts.get(et, 0)) for et in sorted(all_error_types)],
    'GPT-4o False Insert %': [gpt4o_false.get(et, 0) for et in sorted(all_error_types)],
    'Llama False Insert %': [llama_false.get(et, 0) for et in sorted(all_error_types)],
    'DeepSeek False Insert %': [deepseek_false.get(et, 0) for et in sorted(all_error_types)]
}, index=sorted(all_error_types))

# Print results
print("=" * 100)
print("CORRECTION RATES: Percentage of Gold Errors Actually Fixed by Each Model")
print("Count = Total gold errors for each error type")
print("=" * 100)
print()
print(correction_data.to_string(float_format='%.1f'))
print()

print("=" * 100)
print("FALSE INSERTION RATES: Percentage of Model Corrections Not Present in Gold")
print("Count = Maximum false insertions made by any model for each error type")
print("=" * 100)
print()
print(false_insertion_data.to_string(float_format='%.1f'))
print()

# Summary statistics
print("=" * 90)
print("SUMMARY STATISTICS")
print("=" * 90)
print("Correction Rates (Average):")
print(f"  GPT-4o: {correction_data['GPT-4o Correction %'].mean():.1f}%")
print(f"  Llama: {correction_data['Llama Correction %'].mean():.1f}%")
print(f"  DeepSeek: {correction_data['DeepSeek Correction %'].mean():.1f}%")
print()
print("False Insertion Rates (Average):")
print(f"  GPT-4o: {false_insertion_data['GPT-4o False Insert %'].mean():.1f}%")
print(f"  Llama: {false_insertion_data['Llama False Insert %'].mean():.1f}%")
print(f"  DeepSeek: {false_insertion_data['DeepSeek False Insert %'].mean():.1f}%")
print()
print(f"Total gold errors: {len(gold)}")
print(f"Total error types: {len(all_error_types)}")
print("=" * 90)