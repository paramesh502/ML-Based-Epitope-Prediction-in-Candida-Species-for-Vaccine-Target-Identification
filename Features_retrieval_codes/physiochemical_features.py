import pandas as pd
import re
from Bio.SeqUtils.ProtParam import ProteinAnalysis

# ==============================
# 1️⃣ Load Dataset
# ==============================

# Change this to your actual file name if needed
df = pd.read_csv("dataset_with_aac.csv")

# ==============================
# 2️⃣ Clean Peptide Sequences
# ==============================

def clean_sequence(seq):
    seq = str(seq).upper()
    # Keep only standard amino acid letters
    seq = re.sub(r'[^ACDEFGHIKLMNPQRSTVWY]', '', seq)
    return seq

# Apply cleaning
df["Peptide"] = df["Peptide"].apply(clean_sequence)

# ==============================
# 3️⃣ Initialize Feature Lists
# ==============================

lengths = []
molecular_weights = []
pI_values = []
aromaticities = []
instability_indices = []
gravy_scores = []
net_charges = []

# ==============================
# 4️⃣ Compute Physicochemical Features
# ==============================

for seq in df["Peptide"]:

    if len(seq) == 0:
        lengths.append(0)
        molecular_weights.append(0)
        pI_values.append(0)
        aromaticities.append(0)
        instability_indices.append(0)
        gravy_scores.append(0)
        net_charges.append(0)
        continue

    analysis = ProteinAnalysis(seq)

    lengths.append(len(seq))
    molecular_weights.append(analysis.molecular_weight())
    pI_values.append(analysis.isoelectric_point())
    aromaticities.append(analysis.aromaticity())
    instability_indices.append(analysis.instability_index())
    gravy_scores.append(analysis.gravy())
    net_charges.append(analysis.charge_at_pH(7.0))

# ==============================
# 5️⃣ Add Features to DataFrame
# ==============================

df["Length"] = lengths
df["Molecular_Weight"] = molecular_weights
df["Isoelectric_Point"] = pI_values
df["Aromaticity"] = aromaticities
df["Instability_Index"] = instability_indices
df["Hydrophobicity_GRAVY"] = gravy_scores
df["Net_Charge_pH7"] = net_charges

# ==============================
# 6️⃣ Save Updated Dataset
# ==============================

df.to_csv("dataset_with_physico.csv", index=False)

print("✅ Physicochemical feature extraction complete.")
print("New dataset shape:", df.shape)
