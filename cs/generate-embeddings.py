import os
import pandas as pd
import json
import glob
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

VECTOR_DB_PATH = "./vector_store/faiss_index"
CASE_DIR = "./case-summary/"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# Step 1: Load and merge CSVs
def load_and_dedupe_cases():
    all_files = glob.glob(os.path.join(CASE_DIR, "*.csv"))
    df_list = [pd.read_csv(f) for f in all_files]
    full_df = pd.concat(df_list, ignore_index=True)

    # Deduplicate based on COMPLAINT_CASE_ID, keep the most complete entry
    def longer_row(a, b):
        return a if a.astype(str).str.len().sum() >= b.astype(str).str.len().sum() else b

    deduped = (
        full_df.groupby("COMPLAINT_CASE_ID")
        .apply(lambda grp: grp.iloc[0] if len(grp) == 1 else grp.apply(lambda col: col.dropna().astype(str).sort_values(key=len).iloc[-1]))
        .reset_index(drop=True)
    )

    return deduped

# Step 2: Create embedding text block
def build_embedding_text(row):
    fields = [
        f"Complaint ID: {row.get('COMPLAINT_CASE_ID', '')}",
        f"Product: {row.get('PRODUCT_FAMILY_NAME', '')}",
        f"Main Issue: {row.get('COMPLAINT_CASE_CATEGORY_DRIVER', '')}",
        f"Sub Issue: {row.get('COMPLAINT_CASE_CATEGORY_SUBDRIVER', '')}",
        f"Customer Narrative: {row.get('COMPLAINT_NARRATIVE', '')}",
        f"Agent Summary: {row.get('COMPLAINT_COMMENT', '')}",
        f"Agent Resolution: {row.get('COMPLAINT_RESOLUTION_COMMENT', '')}",
        f"Resolution Type: {row.get('RESOLUTION_TYPE', '')}",
        f"Complaint Tags: {row.get('COMPLAINT_TAG_SUMMARY', '')}"
    ]
    return "\n".join(fields)

# Step 3: Build vector store
def build_vector_db(df):
    embeddings = []
    metadata = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        text = build_embedding_text(row)
        emb = model.encode(text, normalize_embeddings=True)
        embeddings.append(emb)
        metadata.append({
            "COMPLAINT_CASE_ID": row.get("COMPLAINT_CASE_ID"),
            "PRODUCT_FAMILY_NAME": row.get("PRODUCT_FAMILY_NAME"),
            "COMPLAINT_CASE_CATEGORY_DRIVER": row.get("COMPLAINT_CASE_CATEGORY_DRIVER"),
            "COMPLAINT_NARRATIVE": row.get("COMPLAINT_NARRATIVE")
        })

    embeddings_np = np.vstack(embeddings).astype('float32')
    index = faiss.IndexFlatL2(embeddings_np.shape[1])
    index.add(embeddings_np)

    faiss.write_index(index, VECTOR_DB_PATH)
    with open(VECTOR_DB_PATH + ".meta.json", "w") as f:
        json.dump(metadata, f)

if __name__ == "__main__":
    print("Loading and deduplicating complaints...")
    df = load_and_dedupe_cases()
    print(f"{len(df)} unique complaints loaded.")
    print("Generating embeddings and saving to vector DB...")
    build_vector_db(df)
    print("Done.")

