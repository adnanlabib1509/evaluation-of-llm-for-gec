import numpy as np
from scipy.stats import spearmanr
import pandas as pd

# Correction rates data
correction_data = {
    'Error_Type': ['M:ADJ', 'M:ADV', 'M:CONJ', 'M:CONTR', 'M:DET', 'M:NOUN', 'M:NOUN:POSS', 
                   'M:OTHER', 'M:PART', 'M:PREP', 'M:PRON', 'M:PUNCT', 'M:VERB', 'M:VERB:FORM',
                   'M:VERB:TENSE', 'R:ADJ', 'R:ADJ:FORM', 'R:ADV', 'R:CONJ', 'R:CONTR', 'R:DET',
                   'R:MORPH', 'R:NOUN', 'R:NOUN:INFL', 'R:NOUN:NUM', 'R:NOUN:POSS', 'R:ORTH',
                   'R:OTHER', 'R:PART', 'R:PREP', 'R:PRON', 'R:PUNCT', 'R:SPELL', 'R:VERB',
                   'R:VERB:FORM', 'R:VERB:INFL', 'R:VERB:SVA', 'R:VERB:TENSE', 'R:WO', 'U:ADJ',
                   'U:ADV', 'U:CONJ', 'U:CONTR', 'U:DET', 'U:NOUN', 'U:NOUN:POSS', 'U:OTHER',
                   'U:PART', 'U:PREP', 'U:PRON', 'U:PUNCT', 'U:VERB', 'U:VERB:FORM', 'U:VERB:TENSE'],
    'GPT4o': [40.0, 37.9, 8.3, 0.0, 82.0, 22.7, 73.1, 43.1, 50.0, 61.4, 63.8, 85.7, 55.9, 91.3,
              72.9, 36.6, 81.8, 36.5, 28.6, 52.4, 52.8, 73.9, 41.4, 100.0, 80.6, 76.2, 84.3, 53.1,
              68.9, 72.2, 54.1, 75.1, 96.0, 48.2, 77.1, 85.7, 84.5, 65.7, 62.5, 27.3, 40.0, 33.3,
              87.5, 73.9, 45.7, 70.0, 33.3, 66.7, 70.9, 68.2, 44.1, 39.0, 42.9, 57.4],
    'LLaMA': [40.0, 20.7, 4.2, 50.0, 76.7, 25.0, 73.1, 46.9, 40.0, 60.8, 60.3, 81.2, 49.2, 87.0,
              67.8, 39.8, 81.8, 34.9, 14.3, 52.4, 51.7, 65.3, 37.9, 100.0, 73.3, 71.4, 75.4, 50.7,
              64.4, 69.2, 56.5, 64.8, 94.7, 47.9, 78.1, 85.7, 84.5, 63.8, 59.1, 18.2, 32.0, 44.4,
              75.0, 65.8, 39.1, 50.0, 33.3, 66.7, 61.2, 65.9, 40.9, 48.8, 42.9, 61.7],
    'DeepSeek': [25.0, 37.9, 37.5, 0.0, 74.8, 27.3, 65.4, 35.6, 30.0, 55.7, 60.3, 78.2, 47.5, 87.0,
                 50.8, 25.8, 81.8, 20.6, 14.3, 9.5, 46.1, 69.3, 22.8, 100.0, 73.3, 71.4, 77.8, 42.8,
                 42.2, 52.0, 51.8, 34.1, 96.2, 26.9, 76.2, 100.0, 90.8, 45.8, 58.0, 18.2, 32.0, 55.6,
                 37.5, 54.9, 30.4, 60.0, 28.4, 50.0, 59.2, 56.8, 40.9, 34.1, 71.4, 48.9]
}

df_correction = pd.DataFrame(correction_data)

# Calculate Spearman correlations for correction rates
corr_gpt_llama, p_gpt_llama = spearmanr(df_correction['GPT4o'], df_correction['LLaMA'])
corr_gpt_deep, p_gpt_deep = spearmanr(df_correction['GPT4o'], df_correction['DeepSeek'])
corr_llama_deep, p_llama_deep = spearmanr(df_correction['LLaMA'], df_correction['DeepSeek'])

print("=== CORRECTION RATE CORRELATIONS ===")
print(f"GPT-4o vs LLaMA:    ρ = {corr_gpt_llama:.3f}, p = {p_gpt_llama:.4f}")
print(f"GPT-4o vs DeepSeek: ρ = {corr_gpt_deep:.3f}, p = {p_gpt_deep:.4f}")
print(f"LLaMA vs DeepSeek:  ρ = {corr_llama_deep:.3f}, p = {p_llama_deep:.4f}")

# False insertion rates data
false_insertion_data = {
    'Error_Type': ['M:ADJ', 'M:ADV', 'M:CONJ', 'M:CONTR', 'M:DET', 'M:NOUN', 'M:NOUN:POSS',
                   'M:OTHER', 'M:PART', 'M:PREP', 'M:PRON', 'M:PUNCT', 'M:VERB', 'M:VERB:FORM',
                   'M:VERB:TENSE', 'R:ADJ', 'R:ADJ:FORM', 'R:ADV', 'R:CONJ', 'R:CONTR', 'R:DET',
                   'R:MORPH', 'R:NOUN', 'R:NOUN:INFL', 'R:NOUN:NUM', 'R:NOUN:POSS', 'R:ORTH',
                   'R:OTHER', 'R:PART', 'R:PREP', 'R:PRON', 'R:PUNCT', 'R:SPELL', 'R:VERB',
                   'R:VERB:FORM', 'R:VERB:INFL', 'R:VERB:SVA', 'R:VERB:TENSE', 'R:WO', 'U:ADJ',
                   'U:ADV', 'U:CONJ', 'U:CONTR', 'U:DET', 'U:NOUN', 'U:NOUN:POSS', 'U:OTHER',
                   'U:PART', 'U:PREP', 'U:PRON', 'U:PUNCT', 'U:VERB', 'U:VERB:FORM', 'U:VERB:TENSE'],
    'GPT4o': [52.9, 52.2, 71.4, 100.0, 19.5, 41.2, 13.6, 34.6, 37.5, 26.2, 32.1, 21.1, 38.2, 28.6,
              38.6, 33.3, 30.8, 45.2, 0.0, 20.0, 27.8, 23.7, 30.2, 16.7, 24.0, 27.3, 17.6, 27.7,
              26.2, 20.1, 35.2, 22.0, 6.7, 19.3, 24.2, 25.0, 27.0, 22.8, 34.5, 40.0, 33.3, 40.0,
              0.0, 24.0, 53.5, 0.0, 38.1, 42.9, 27.7, 16.7, 41.2, 40.7, 40.0, 30.8],
    'LLaMA': [52.9, 73.9, 90.9, 50.0, 19.0, 52.2, 26.9, 34.2, 50.0, 25.0, 32.7, 18.1, 39.6, 24.0,
              39.4, 29.4, 0.0, 37.1, 50.0, 20.0, 28.0, 27.2, 32.2, 0.0, 27.1, 34.8, 17.5, 33.4,
              23.7, 18.9, 27.3, 23.2, 6.0, 18.9, 24.8, 14.3, 27.3, 22.9, 39.8, 75.0, 46.9, 42.9,
              0.0, 28.2, 60.0, 16.7, 43.5, 42.9, 34.3, 17.6, 42.2, 35.5, 57.1, 27.5],
    'DeepSeek': [61.5, 61.3, 84.7, 0.0, 26.5, 45.5, 5.6, 50.4, 25.0, 46.1, 40.7, 38.4, 56.9, 33.3,
                 50.8, 60.7, 30.8, 53.6, 66.7, 88.9, 42.1, 42.1, 53.4, 16.7, 35.3, 25.0, 54.6, 49.8,
                 32.1, 26.8, 48.9, 25.8, 11.1, 30.8, 38.6, 41.7, 38.6, 21.2, 42.0, 66.7, 51.6, 64.3,
                 0.0, 37.6, 53.8, 0.0, 56.0, 50.0, 34.4, 43.5, 70.8, 54.8, 78.3, 37.8]
}

df_false = pd.DataFrame(false_insertion_data)

# Calculate Spearman correlations for false insertion rates
corr_gpt_llama_fi, p_gpt_llama_fi = spearmanr(df_false['GPT4o'], df_false['LLaMA'])
corr_gpt_deep_fi, p_gpt_deep_fi = spearmanr(df_false['GPT4o'], df_false['DeepSeek'])
corr_llama_deep_fi, p_llama_deep_fi = spearmanr(df_false['LLaMA'], df_false['DeepSeek'])

print("\n=== FALSE INSERTION RATE CORRELATIONS ===")
print(f"GPT-4o vs LLaMA:    ρ = {corr_gpt_llama_fi:.3f}, p = {p_gpt_llama_fi:.4f}")
print(f"GPT-4o vs DeepSeek: ρ = {corr_gpt_deep_fi:.3f}, p = {p_gpt_deep_fi:.4f}")
print(f"LLaMA vs DeepSeek:  ρ = {corr_llama_deep_fi:.3f}, p = {p_llama_deep_fi:.4f}")

# Also show summary statistics
print("\n=== SUMMARY ===")
print(f"Average correction correlation: ρ = {np.mean([corr_gpt_llama, corr_gpt_deep, corr_llama_deep]):.3f}")
print(f"Average false insertion correlation: ρ = {np.mean([corr_gpt_llama_fi, corr_gpt_deep_fi, corr_llama_deep_fi]):.3f}")