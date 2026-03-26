import pandas as pd

# Read the Excel file
df = pd.read_excel('../results/bea_outputs.xlsx')

# Define the model columns
models = ['gpt4o_ft_corrected', 'editscorer_corrected', 'T5_11B_corrected']

def ensemble_vote(row):
    corrections = [row[m] for m in models]
    
    # Check agreement
    if corrections[0] == corrections[1] == corrections[2]:
        agreement = 'all_same'
        dissenting = None
        ensemble = corrections[0]
    elif corrections[0] == corrections[1]:
        agreement = 'two_same'
        dissenting = 't5_correction'
        ensemble = corrections[0]
    elif corrections[0] == corrections[2]:
        agreement = 'two_same'
        dissenting = 'editscorer_correction'
        ensemble = corrections[0]
    elif corrections[1] == corrections[2]:
        agreement = 'two_same'
        dissenting = 'gpt4o_ft_corrected'
        ensemble = corrections[1]
    else:
        agreement = 'none_same'
        dissenting = None
        ensemble = row['gpt4o_ft_corrected']
    
    return pd.Series([ensemble, agreement, dissenting])

# Apply ensemble logic
df[['diff_system_ensemble', 'agreement_status', 'dissenting_model']] = df.apply(ensemble_vote, axis=1)

# Save result
df.to_excel('heterogenous_ensembles.xlsx', index=False)

# Statistics
print(df['agreement_status'].value_counts(normalize=True) * 100)
print("\nDissenting model breakdown:")
print(df[df['agreement_status'] == 'two_same']['dissenting_model'].value_counts(normalize=True) * 100)