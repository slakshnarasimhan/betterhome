import faiss
import json
import numpy as np
import pandas as pd
import os
import requests
from tqdm import tqdm
from collections import defaultdict
import re

VECTOR_DB_PATH = "./vector_store/faiss_index"
META_PATH = VECTOR_DB_PATH + ".meta.json"
CSV_DIR = "./case-summary"

EMBED_URL = "http://localhost:11434/api/embeddings"

ESCALATION_KEYWORDS = [
    "fraud", "fraud team", "referred", "escalated", "escalation",
    "investigation", "compliance", "security", "risk", "alert", "dispute"
]

TOOL_KEYWORDS = [
    "TSYS", "OMEGA", "EPM", "CTL", "CTI", "CTR",
    "Document Scanner", "Case Management Software"
]

TEAM_KEYWORDS = [
    "CAO", "Customer Service", "Operations", "CTI", "CTR", "Research Team"
]

# ------------------- Helper: Embedding -------------------
def get_embedding(text):
    response = requests.post(
        EMBED_URL,
        json={"model": "nomic-embed-text", "prompt": text}
    )
    result = response.json()
    return np.array(result["embedding"], dtype=np.float32)

# ------------------- Helper: Flatten JSON -------------------
def flatten_activity_json(value):
    try:
        obj = json.loads(value)
        if isinstance(obj, list):
            return " | ".join(json.dumps(x) for x in obj)
        elif isinstance(obj, dict):
            return json.dumps(obj)
        return str(obj)
    except Exception:
        return str(value)

# ------------------- Helper: Infer Escalations/Tools -------------------
def infer_keywords(text, keywords):
    found = set()
    for keyword in keywords:
        if re.search(rf"\\b{re.escape(keyword)}\\b", text, re.IGNORECASE):
            found.add(keyword)
    return ", ".join(sorted(found))

# ------------------- Step 1: Load and clean all CSVs -------------------
def load_all_complaints():
    merged = defaultdict(dict)

    for file in os.listdir(CSV_DIR):
        if not file.endswith(".csv"): continue
        df = pd.read_csv(os.path.join(CSV_DIR, file))
        for _, row in df.iterrows():
            case_id = str(row.get("COMPLAINT_CASE_ID", "")).strip()
            if not case_id: continue

            if case_id not in merged:
                merged[case_id] = row.to_dict()
            else:
                for key, val in row.items():
                    if pd.notnull(val) and not pd.isnull(merged[case_id].get(key)):
                        merged[case_id][key] = val
    return list(merged.values())

# ------------------- Step 2: Build full context -------------------
def build_context(entry):
    note = flatten_activity_json(entry.get("ACTIVITY_NOTE", ""))
    detail = flatten_activity_json(entry.get("ACTIVITY_DETAILS", ""))
    all_text = f"{note} {detail}"
    escalations = infer_keywords(all_text, ESCALATION_KEYWORDS)
    tools = infer_keywords(all_text, TOOL_KEYWORDS)
    teams = infer_keywords(all_text, TEAM_KEYWORDS)

    return f"""
Complaint ID: {entry.get('COMPLAINT_CASE_ID', '')}
Category: {entry.get('COMPLAINT_CASE_CATEGORY_DRIVER', '')} > {entry.get('COMPLAINT_CASE_CATEGORY_SUBDRIVER', '')}
Narrative: {entry.get('COMPLAINT_NARRATIVE', '')}
Agent Notes: {note}
Activity Details: {detail}
Escalation Path: {escalations or 'None'}
Tools Used: {tools or 'Not specified'}
Teams Involved: {teams or 'Not captured'}
Agent Resolution Comment: {entry.get('COMPLAINT_RESOLUTION_COMMENT', '')}
Tags: {entry.get('COMPLAINT_TAG_SUMMARY', '')}
Resolution Type: {entry.get('RESOLUTION_TYPE', '')}
Product: {entry.get('PRODUCT_FAMILY_NAME', '')}
    """

# ------------------- Step 3: Embed and Save -------------------
def main():
    print("📥 Loading complaints...")
    complaints = load_all_complaints()

    print("🧠 Generating embeddings...")
    embeddings = []
    metadata = []
    for entry in tqdm(complaints):
        context = build_context(entry)
        try:
            emb = get_embedding(context)
            embeddings.append(emb)
            metadata.append(entry)
        except Exception as e:
            print(f"⚠️ Failed embedding for case {entry.get('COMPLAINT_CASE_ID')}: {e}")

    print("💾 Saving FAISS index and metadata...")
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))

    os.makedirs(os.path.dirname(VECTOR_DB_PATH), exist_ok=True)
    faiss.write_index(index, VECTOR_DB_PATH)

    with open(META_PATH, "w") as f:
        json.dump(metadata, f)

    print("✅ Embedding generation complete.")

if __name__ == "__main__":
    main()

