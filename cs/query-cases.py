import faiss
import json
import numpy as np
import requests

VECTOR_DB_PATH = "./vector_store/faiss_index"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
OLLAMA_GEN_URL = "http://localhost:11434/api/generate"

# Load vector index and metadata
index = faiss.read_index(VECTOR_DB_PATH)
with open(VECTOR_DB_PATH + ".meta.json", "r") as f:
    metadata = json.load(f)

# Use Ollama to embed user query
def get_ollama_embedding(text):
    response = requests.post(
        OLLAMA_EMBED_URL,
        json={"model": "nomic-embed-text", "prompt": text}
    )
    result = response.json()
    return np.array(result["embedding"], dtype=np.float32)

def query_vector_db(user_query, top_k=10):
    query_emb = get_ollama_embedding(user_query).reshape(1, -1)
    D, I = index.search(query_emb.astype('float32'), top_k)
    return [metadata[i] for i in I[0]]

def build_mixed_prompt(results, user_input):
    complaints = []
    kb_articles = []

    for item in results:
        if item.get("type") == "complaint":
            complaints.append(
                f"Complaint: {item.get('COMPLAINT_NARRATIVE', '')}\nTags: {item.get('COMPLAINT_CASE_CATEGORY_DRIVER', '')}\nAgent Notes: {item.get('ACTIVITY_NOTE', '')}\nActivity Details: {item.get('ACTIVITY_DETAILS', '')}"
            )
        elif item.get("type") == "knowledge_base":
            kb_articles.append(
                f"Source: {item.get('source_url', '')}\nSummary: {item.get('description', '')}"
            )

    prompt = f"""
You are a banking domain expert and complaint resolution assistant. You are well-versed in:
- The Credit Card Accountability Responsibility and Disclosure (CARD) Act of 2009
- The Truth in Lending Act (TILA)
- Federal regulations enforced by the Consumer Financial Protection Bureau (CFPB)

You are provided with two sources of information:
1. Historical complaint cases with agent actions.
2. Official process documentation from a knowledge base.

Answer the following question using evidence from both sources:
"{user_input}"

---
Relevant Complaints:
{chr(10).join(complaints[:5])}

---
Relevant Knowledge Articles:
{chr(10).join(kb_articles[:3])}

Provide a clear, evidence-based summary.
"""
    return prompt

def query_llama(prompt):
    response = requests.post(
        OLLAMA_GEN_URL,
        json={"model": "llama3.2", "prompt": prompt, "stream": False}
    )
    return response.json().get("response", "No response from model.")

def answer_query(user_input):
    user_input_lower = user_input.lower()
    if user_input.strip().isdigit():
        for item in metadata:
            if item.get("COMPLAINT_CASE_ID") == user_input.strip():
                return f"Case ID: {item['COMPLAINT_CASE_ID']}\nNarrative: {item['COMPLAINT_NARRATIVE']}\nTags: {item['COMPLAINT_CASE_CATEGORY_DRIVER']}\nAgent Notes: {item.get('ACTIVITY_NOTE', '')}\nActivity Details: {item.get('ACTIVITY_DETAILS', '')}"
        return "Case ID not found."
    else:
        results = query_vector_db(user_input)
        prompt = build_mixed_prompt(results, user_input)
        return query_llama(prompt)

if __name__ == "__main__":
    while True:
        user_input = input("Enter case ID or scenario-based complaint question (q to quit): ")
        if user_input.lower() == 'q':
            break
        print("\n--- RESPONSE ---")
        print(answer_query(user_input))
        print("\n----------------\n")

