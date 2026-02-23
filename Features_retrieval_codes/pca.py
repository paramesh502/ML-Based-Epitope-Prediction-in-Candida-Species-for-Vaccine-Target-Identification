import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ==============================
# 1️⃣ Load Dataset
# ==============================

# Make sure this file is in same folder as this script
file_name = "final_dataset_with_bert.csv"

df = pd.read_csv(file_name)

print("Dataset loaded successfully.")
print("Total samples:", df.shape[0])
print("Total columns:", df.shape[1])

# ==============================
# 2️⃣ Separate Features & Label
# ==============================

X = df.drop(columns=["Peptide", "Label"])
y = df["Label"]

print("Feature matrix shape:", X.shape)

# ==============================
# 3️⃣ Standardize Features
# ==============================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Feature scaling completed.")

# ==============================
# 4️⃣ Apply PCA (95% Variance)
# ==============================

pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

print("\n----- PCA Results -----")
print("Original feature count:", X.shape[1])
print("Reduced feature count:", X_pca.shape[1])
print("Variance retained:", sum(pca.explained_variance_ratio_))

# ==============================
# 5️⃣ Save Reduced Dataset
# ==============================

df_pca = pd.DataFrame(X_pca)
df_pca["Label"] = y.values

output_file = "dataset_pca_95.csv"
df_pca.to_csv(output_file, index=False)

print("\nPCA dataset saved as:", output_file)
