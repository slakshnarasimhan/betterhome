import os
import pandas as pd
import json
import glob
import requests
from tqdm import tqdm
from bs4 import BeautifulSoup
import faiss
import numpy as np

VECTOR_DB_PATH = "./vector_store/faiss_index"
CASE_DIR = "./case-summary/"
PROCESS_KB_CSV = "./knowledge_base/process_articles.csv"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"

# Step 1: Load and merge CSVs
def load_and_dedupe_cases():
    all_files = glob.glob(os.path.join(CASE_DIR, "*.csv"))
    df_list = [pd.read_csv(f, dtype=str, keep_default_na=False) for f in all_files]
    full_df = pd.concat(df_list, ignore_index=True)

    def merge_rows(group):
        merged = {}
        for col in group.columns:
            non_nulls = group[col].dropna().astype(str)
            merged[col] = non_nulls.iloc[non_nulls.str.len().argmax()] if not non_nulls.empty else None
        return pd.Series(merged)

    deduped = full_df.groupby("COMPLAINT_CASE_ID", as_index=False).apply(merge_rows).reset_index(drop=True)
    return deduped

# Step 2: Create embedding text block
def build_embedding_text(row):
    def parse_json_field(field):
        try:
            parsed = json.loads(field)
            if isinstance(parsed, dict):
                return "; ".join(f"{k}: {v}" for k, v in parsed.items())
            elif isinstance(parsed, list):
                return ", ".join(map(str, parsed))
            else:
                return str(parsed)
        except:
            return field

    agent_note = parse_json_field(row.get('ACTIVITY_NOTE', ''))
    activity_details = parse_json_field(row.get('ACTIVITY_DETAILS', ''))

    fields = [
        f"Complaint ID: {row.get('COMPLAINT_CASE_ID', '')}",
        f"Product: {row.get('PRODUCT_FAMILY_NAME', '')}",
        f"Main Issue: {row.get('COMPLAINT_CASE_CATEGORY_DRIVER', '')}",
        f"Sub Issue: {row.get('COMPLAINT_CASE_CATEGORY_SUBDRIVER', '')}",
        f"Customer Narrative: {row.get('COMPLAINT_NARRATIVE', '')}",
        f"Agent Summary: {row.get('COMPLAINT_COMMENT', '')}",
        f"Agent Resolution: {row.get('COMPLAINT_RESOLUTION_COMMENT', '')}",
        f"Resolution Type: {row.get('RESOLUTION_TYPE', '')}",
        f"Complaint Tags: {row.get('COMPLAINT_TAG_SUMMARY', '')}",
        f"Agent Notes: {agent_note}",
        f"Activity Details: {activity_details}"
    ]
    return "\n".join(fields)

# Step 3: Download and extract knowledge base articles
def load_knowledge_base():
    if not os.path.exists(PROCESS_KB_CSV):
        return []
    kb_df = pd.read_csv(PROCESS_KB_CSV)
    kb_texts = []
    for _, row in kb_df.iterrows():
        description, url = row.get("Process Description", ""), row.get("link", "")
        try:
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.text, "html.parser")
            content = soup.get_text(separator=" ", strip=True)
            kb_texts.append((f"Process: {description}\nContent: {content[:2000]}", url))
        except Exception as e:
            print(f"Failed to fetch {url}: {e}")
    return kb_texts

# Step 4: Use Ollama to generate embeddings
def get_ollama_embedding(text):
    response = requests.post(
        OLLAMA_EMBED_URL,
        json={"model": "nomic-embed-text", "prompt": text}
    )
    result = response.json()
    return np.array(result["embedding"], dtype=np.float32)

# Step 5: Build vector store
def build_vector_db(df, kb_data):
    embeddings = []
    metadata = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        text = build_embedding_text(row)
        emb = get_ollama_embedding(text)
        embeddings.append(emb)
        metadata.append({
            "type": "complaint",
            "COMPLAINT_CASE_ID": row.get("COMPLAINT_CASE_ID"),
            "PRODUCT_FAMILY_NAME": row.get("PRODUCT_FAMILY_NAME"),
            "COMPLAINT_CASE_CATEGORY_DRIVER": row.get("COMPLAINT_CASE_CATEGORY_DRIVER"),
            "COMPLAINT_NARRATIVE": row.get("COMPLAINT_NARRATIVE")
        })

    for kb_text, url in kb_data:
        emb = get_ollama_embedding(kb_text)
        embeddings.append(emb)
        metadata.append({
            "type": "knowledge_base",
            "source_url": url,
            "description": kb_text[:100] + "..."
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
    print("Loading external knowledge base articles...")
    kb_data = load_knowledge_base()
    print(f"{len(kb_data)} articles loaded.")
    print("Generating embeddings via Ollama and saving to vector DB...")
    build_vector_db(df, kb_data)
    print("Done.")

