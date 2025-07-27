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

def query_vector_db(user_query, top_k=5):
    query_emb = get_ollama_embedding(user_query).reshape(1, -1)
    D, I = index.search(query_emb.astype('float32'), top_k)
    return [metadata[i] for i in I[0]]

def build_prompt(similar_cases, user_input):
    cases_str = "\n---\n".join(
        f"Complaint: {case['COMPLAINT_NARRATIVE']}\nTags: {case['COMPLAINT_CASE_CATEGORY_DRIVER']}\nResolution (if known): Unknown"
        for case in similar_cases
    )
    return f"""
You are a banking complaint resolution assistant.
Given the following similar past complaints:
{cases_str}

Now answer this:
Customer said: "{user_input}"

What is the most likely resolution we should offer?
"""

def build_tool_team_extraction_prompt(similar_cases):
    cases_str = "\n---\n".join(
        f"""
Complaint Narrative: {case['COMPLAINT_NARRATIVE']}
Case Tags: {case['COMPLAINT_CASE_CATEGORY_DRIVER']}
Agent Notes: {case.get('COMPLAINT_COMMENT', '')}
Resolution: {case.get('COMPLAINT_RESOLUTION_COMMENT', '')}"""
        for case in similar_cases
    )

    return f"""
You are an expert analyst reviewing customer complaint cases in the banking industry.
Based strictly on the following past complaints, generate a comprehensive list of:
1. Tools or platforms mentioned (e.g., document scanner, TSYS)
2. Teams involved in resolution (e.g., CAO, Customer Service)

Do not assume or add any tools/teams not found in the provided cases.
Group the tools and teams in separate bullet-point lists. If none are found, say so.

Here are the cases:
{cases_str}

Your output should look like:

**Tools Used:**
- Tool A
- Tool B

**Teams Involved:**
- Team X
- Team Y
"""

def query_llama(prompt):
    response = requests.post(
        OLLAMA_GEN_URL,
        json={"model": "llama3.2", "prompt": prompt, "stream": False}
    )
    return response.json().get("response", "No response from model.")

def answer_query(user_input):
    if user_input.strip().isdigit():
        for item in metadata:
            if item["COMPLAINT_CASE_ID"] == user_input.strip():
                return f"Case ID: {item['COMPLAINT_CASE_ID']}\nNarrative: {item['COMPLAINT_NARRATIVE']}\nTags: {item['COMPLAINT_CASE_CATEGORY_DRIVER']}"
        return "Case ID not found."
    elif "tools" in user_input.lower() or "teams" in user_input.lower():
        similar = query_vector_db("tool and team references in complaint cases")
        prompt = build_tool_team_extraction_prompt(similar)
        return query_llama(prompt)
    else:
        similar = query_vector_db(user_input)
        prompt = build_prompt(similar, user_input)
        return query_llama(prompt)

if __name__ == "__main__":
    while True:
        user_input = input("Enter case ID, hypothetical complaint, or ask about tools/teams (q to quit): ")
        if user_input.lower() == 'q':
            break
        print("\n--- RESPONSE ---")
        print(answer_query(user_input))
        print("\n----------------\n")

