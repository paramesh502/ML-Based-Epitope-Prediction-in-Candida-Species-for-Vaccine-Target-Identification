import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

# ==============================
# 1️⃣ Load Dataset
# ==============================

df = pd.read_csv("dataset_with_structural_features.csv")

peptides = df["Peptide"].astype(str).tolist()

# ==============================
# 2️⃣ Set Device (GPU if available)
# ==============================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ==============================
# 3️⃣ Load ProtBERT
# ==============================

tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
model = BertModel.from_pretrained("Rostlab/prot_bert")
model = model.to(device)
model.eval()

# ==============================
# 4️⃣ Function to Process in Batches
# ==============================

def get_embeddings_batch(sequences, batch_size=16):
    embeddings = []
    
    for i in tqdm(range(0, len(sequences), batch_size)):
        batch_seqs = sequences[i:i+batch_size]
        
        # Add spaces between amino acids
        batch_seqs = [" ".join(list(seq)) for seq in batch_seqs]
        
        inputs = tokenizer(batch_seqs, return_tensors="pt", padding=True, truncation=True)
        inputs = {key: val.to(device) for key, val in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Mean pooling (better than CLS for proteins)
        batch_embeddings = outputs.last_hidden_state.mean(dim=1)
        
        embeddings.extend(batch_embeddings.cpu().numpy())
    
    return embeddings

# ==============================
# 5️⃣ Generate Embeddings
# ==============================

bert_features = get_embeddings_batch(peptides, batch_size=16)

# Convert to DataFrame
bert_df = pd.DataFrame(bert_features)

embedding_dim = bert_df.shape[1]
bert_df.columns = [f"BERT_{i}" for i in range(embedding_dim)]

# ==============================
# 6️⃣ Merge with Existing Features
# ==============================

final_df = pd.concat([df.reset_index(drop=True), bert_df], axis=1)

final_df.to_csv("final_dataset_with_bert.csv", index=False)

print("✅ BERT feature extraction complete.")
print("Final dataset shape:", final_df.shape)
