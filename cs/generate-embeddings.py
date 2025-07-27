import os
import pandas as pd
import json
import glob
import requests
from tqdm import tqdm
from bs4 import BeautifulSoup
import faiss
import numpy as np

COMPLAINT_INDEX_PATH = "./vector_store/complaints_index"
KB_INDEX_PATH = "./vector_store/kb_index"
CASE_DIR = "./case-summary/"
PROCESS_KB_CSV = "./knowledge_base/process_articles.csv"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"

# Load and deduplicate complaints
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

# Build complaint text block
def build_complaint_text(row):
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

    return "\n".join([
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
    ])

# Load knowledge base articles
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
            kb_texts.append((f"Process: {description}\nContent: {content[:2000]}", description, url))
        except Exception as e:
            print(f"Failed to fetch {url}: {e}")
    return kb_texts

# Generate embeddings via Ollama
def get_ollama_embedding(text):
    response = requests.post(
        OLLAMA_EMBED_URL,
        json={"model": "nomic-embed-text", "prompt": text}
    )
    result = response.json()
    return np.array(result["embedding"], dtype=np.float32)

# Build FAISS index
def build_faiss_index(embedding_texts, output_path):
    embeddings = [get_ollama_embedding(text) for text, _ in embedding_texts]
    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(np.vstack(embeddings).astype('float32'))
    faiss.write_index(index, output_path + ".faiss")
    with open(output_path + ".meta.json", "w") as f:
        json.dump([meta for _, meta in embedding_texts], f)

if __name__ == "__main__":
    print("Processing complaint data...")
    complaint_df = load_and_dedupe_cases()
    complaint_data = [
        (build_complaint_text(row), {
            "COMPLAINT_CASE_ID": row.get("COMPLAINT_CASE_ID"),
            "DRIVER": row.get("COMPLAINT_CASE_CATEGORY_DRIVER"),
            "SUBDRIVER": row.get("COMPLAINT_CASE_CATEGORY_SUBDRIVER"),
            "NARRATIVE": row.get("COMPLAINT_NARRATIVE")
        })
        for _, row in complaint_df.iterrows()
    ]
    print(f"{len(complaint_data)} complaints processed.")

    print("Processing knowledge base articles...")
    kb_data_raw = load_knowledge_base()
    kb_data = [(text, {"description": desc, "source_url": url}) for text, desc, url in kb_data_raw]
    print(f"{len(kb_data)} KB articles processed.")

    print("Building complaint vector index...")
    build_faiss_index(complaint_data, COMPLAINT_INDEX_PATH)

    print("Building knowledge base vector index...")
    build_faiss_index(kb_data, KB_INDEX_PATH)

    print("✅ Index generation complete.")

