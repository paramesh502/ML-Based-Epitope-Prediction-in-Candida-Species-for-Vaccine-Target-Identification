import pandas as pd

# Load dataset
df = pd.read_csv("dataset_with_physico.csv")

# Hydrophilicity scale (Hopp-Woods)
hydrophilicity_scale = {
    'A': -0.5, 'C': -1.0, 'D': 3.0, 'E': 3.0, 'F': -2.5,
    'G': 0.0, 'H': -0.5, 'I': -1.8, 'K': 3.0, 'L': -1.8,
    'M': -1.3, 'N': 0.2, 'P': 0.0, 'Q': 0.2, 'R': 3.0,
    'S': 0.3, 'T': -0.4, 'V': -1.5, 'W': -3.4, 'Y': -2.3
}

# Disorder propensity scale (approximation)
disorder_scale = {
    'A': 0.06, 'C': 0.02, 'D': 0.19, 'E': 0.19, 'F': 0.03,
    'G': 0.16, 'H': 0.14, 'I': 0.03, 'K': 0.19, 'L': 0.04,
    'M': 0.05, 'N': 0.16, 'P': 0.20, 'Q': 0.16, 'R': 0.18,
    'S': 0.15, 'T': 0.12, 'V': 0.04, 'W': 0.01, 'Y': 0.03
}

surface_scores = []
disorder_scores = []

for seq in df["Peptide"]:
    seq = str(seq)
    
    if len(seq) == 0:
        surface_scores.append(0)
        disorder_scores.append(0)
        continue
    
    # Surface accessibility proxy (average hydrophilicity)
    surface = sum(hydrophilicity_scale.get(aa, 0) for aa in seq) / len(seq)
    
    # Disorder proxy (average disorder propensity)
    disorder = sum(disorder_scale.get(aa, 0) for aa in seq) / len(seq)
    
    surface_scores.append(surface)
    disorder_scores.append(disorder)

# Add new features
df["Surface_Accessibility_Score"] = surface_scores
df["Disorder_Score"] = disorder_scores

# Save final dataset
df.to_csv("dataset_with_structural_features.csv", index=False)

print("✅ Surface accessibility and disorder features added.")
print("New dataset shape:", df.shape)
