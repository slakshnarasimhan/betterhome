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

def query_vector_db(user_query, top_k=8):
    query_emb = get_ollama_embedding(user_query).reshape(1, -1)
    D, I = index.search(query_emb.astype('float32'), top_k)
    return [metadata[i] for i in I[0]]

def build_resolution_prompt(similar_cases, user_input):
    cases_str = "\n---\n".join(
        f"Complaint: {case['COMPLAINT_NARRATIVE']}\nTags: {case['COMPLAINT_CASE_CATEGORY_DRIVER']}\nAgent Notes: {case.get('ACTIVITY_NOTE', '')}\nActivity Details: {case.get('ACTIVITY_DETAILS', '')}\nResolution (if known): Unknown"
        for case in similar_cases
    )
    return f"""
You are a customer complaint resolution expert specializing in credit card disputes. You are well-versed in:
- The Credit Card Accountability Responsibility and Disclosure (CARD) Act of 2009
- The Truth in Lending Act (TILA)
- Federal regulations enforced by the Consumer Financial Protection Bureau (CFPB)

Based on the following historical complaint cases and agent actions, summarize the standard course of action agents typically take when handling a scenario like:
"{user_input}"

Only refer to what is evident in the historical cases. Structure your answer clearly.

{cases_str}

Summary:
"""

def build_fraud_routing_prompt(similar_cases, user_input):
    cases_str = "\n---\n".join(
        f"Complaint: {case['COMPLAINT_NARRATIVE']}\nAgent Notes: {case.get('ACTIVITY_NOTE', '')}\nActivity Details: {case.get('ACTIVITY_DETAILS', '')}"
        for case in similar_cases
    )
    return f"""
You are analyzing complaint cases to understand under what scenarios agents escalate issues to the Fraud Team.
Based only on the following examples, summarize the types of complaints or patterns that typically lead to a fraud referral:

{cases_str}

Summarize the types of complaints and triggers that caused fraud escalation:
"""

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
            if item["COMPLAINT_CASE_ID"] == user_input.strip():
                return f"Case ID: {item['COMPLAINT_CASE_ID']}\nNarrative: {item['COMPLAINT_NARRATIVE']}\nTags: {item['COMPLAINT_CASE_CATEGORY_DRIVER']}\nAgent Notes: {item.get('ACTIVITY_NOTE', '')}\nActivity Details: {item.get('ACTIVITY_DETAILS', '')}"
        return "Case ID not found."
    elif "fraud" in user_input_lower and ("team" in user_input_lower or "escalate" in user_input_lower or "referred" in user_input_lower):
        similar = query_vector_db("cases escalated to fraud team")
        prompt = build_fraud_routing_prompt(similar, user_input)
        return query_llama(prompt)
    else:
        similar = query_vector_db(user_input)
        prompt = build_resolution_prompt(similar, user_input)
        return query_llama(prompt)

if __name__ == "__main__":
    while True:
        user_input = input("Enter case ID or scenario-based complaint question (q to quit): ")
        if user_input.lower() == 'q':
            break
        print("\n--- RESPONSE ---")
        print(answer_query(user_input))
        print("\n----------------\n")

