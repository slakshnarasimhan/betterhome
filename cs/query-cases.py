import faiss
import json
import numpy as np
import requests

COMPLAINT_INDEX_PATH = "./vector_store/complaints_index"
KB_INDEX_PATH = "./vector_store/kb_index"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
OLLAMA_GEN_URL = "http://localhost:11434/api/generate"

# Load indexes and metadata
complaint_index = faiss.read_index(COMPLAINT_INDEX_PATH + ".faiss")
kb_index = faiss.read_index(KB_INDEX_PATH + ".faiss")
with open(COMPLAINT_INDEX_PATH + ".meta.json", "r") as f:
    complaint_meta = json.load(f)
with open(KB_INDEX_PATH + ".meta.json", "r") as f:
    kb_meta = json.load(f)

# Embedding
def get_ollama_embedding(text):
    response = requests.post(
        OLLAMA_EMBED_URL,
        json={"model": "nomic-embed-text", "prompt": text}
    )
    return np.array(response.json()["embedding"], dtype=np.float32)

# Query FAISS index
def query_index(index, meta, query_text, top_k=5):
    emb = get_ollama_embedding(query_text).reshape(1, -1)
    D, I = index.search(emb.astype('float32'), top_k)
    return [meta[i] for i in I[0]], D[0]

# Build mixed prompt
def build_answer_prompt(kb_hits, complaint_hits, query):
    kb_part = "\n".join(f"Process KB: {hit.get('description')}\nSource: {hit.get('source_url')}" for hit in kb_hits)
    complaint_part = "\n".join(f"Complaint Narrative: {hit.get('NARRATIVE')}\nCategory: {hit.get('DRIVER')} > {hit.get('SUBDRIVER')}" for hit in complaint_hits)
    return f"""
You are a regulatory-compliant customer support expert for credit card complaints.
You are given two sets of information:
1. Official process documentation
2. Real complaint narratives with category context

Based on both, answer the question below with a clear and concise summary.

Question: {query}

---
Knowledge Articles:
{kb_part if kb_part else 'None'}

Complaint Examples:
{complaint_part if complaint_part else 'None'}

Answer:
"""

# Answer logic
def answer_query(user_input):
    user_input_clean = user_input.strip()
    if user_input_clean.isdigit():
        for item in complaint_meta:
            if item.get("COMPLAINT_CASE_ID") == user_input_clean:
                return f"Case ID: {item['COMPLAINT_CASE_ID']}\nNarrative: {item.get('NARRATIVE')}\nCategory: {item.get('DRIVER')} > {item.get('SUBDRIVER')}"
        return "Case ID not found."
    else:
        kb_hits, kb_scores = query_index(kb_index, kb_meta, user_input, top_k=3)
        if all(score > 1.5 for score in kb_scores):  # no good KB match
            fallback_driver = user_input.split(" ")[0]
            complaint_hits = [m for m in complaint_meta if fallback_driver.lower() in str(m.get("DRIVER", "")).lower()][:5]
        else:
            complaint_hits, _ = query_index(complaint_index, complaint_meta, user_input, top_k=5)

        prompt = build_answer_prompt(kb_hits, complaint_hits, user_input)
        response = requests.post(OLLAMA_GEN_URL, json={"model": "llama3.2", "prompt": prompt, "stream": False})
        return response.json().get("response", "No response from model.")

if __name__ == "__main__":
    while True:
        user_input = input("Enter case ID or question (q to quit): ")
        if user_input.lower() == 'q':
            break
        print("\n--- RESPONSE ---")
        print(answer_query(user_input))
        print("\n----------------\n")

