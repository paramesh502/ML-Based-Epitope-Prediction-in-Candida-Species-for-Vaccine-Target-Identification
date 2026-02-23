import pandas as pd
import numpy as np

# Load final dataset (balanced dataset with positives and negatives)
df = pd.read_csv("training_dataset_final.csv")

# Standard 20 amino acids
amino_acids = list("ACDEFGHIKLMNPQRSTVWY")

# Function to compute Amino Acid Composition (AAC)
def compute_aac(sequence):
    sequence = str(sequence)
    length = len(sequence)
    aac = {}

    for aa in amino_acids:
        count = sequence.count(aa)
        aac[aa] = count / length if length > 0 else 0

    return pd.Series(aac)

# Apply AAC to each peptide
aac_features = df['Peptide'].apply(compute_aac)

# Combine AAC features with original dataset
df_final = pd.concat([df, aac_features], axis=1)

# Save new dataset
df_final.to_csv("dataset_with_aac.csv", index=False)

print("AAC feature extraction complete.")
print("Original dataset shape:", df.shape)
print("New dataset shape with AAC:", df_final.shape)
