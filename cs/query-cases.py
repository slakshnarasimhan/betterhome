import streamlit as st
import faiss
import json
import numpy as np
import requests

VECTOR_DB_PATH = "./vector_store/faiss_index"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
OLLAMA_GEN_URL = "http://localhost:11434/api/generate"

# Load vector index and metadata
@st.cache_resource
def load_index_and_metadata():
    index = faiss.read_index(VECTOR_DB_PATH)
    with open(VECTOR_DB_PATH + ".meta.json", "r") as f:
        metadata = json.load(f)
    return index, metadata

# Use Ollama to embed user query
def get_ollama_embedding(text):
    response = requests.post(
        OLLAMA_EMBED_URL,
        json={"model": "nomic-embed-text", "prompt": text}
    )
    result = response.json()
    return np.array(result["embedding"], dtype=np.float32)

def query_vector_db(index, metadata, user_query, top_k=5):
    query_emb = get_ollama_embedding(user_query).reshape(1, -1)
    D, I = index.search(query_emb.astype('float32'), top_k)
    return [metadata[i] for i in I[0]]

def build_prompt(similar_cases, user_input):
    cases_str = "\n---\n".join(
        f"Complaint: {case['COMPLAINT_NARRATIVE']}\nTags: {case['COMPLAINT_CASE_CATEGORY_DRIVER']}\nAgent Notes: {case.get('ACTIVITY_NOTE', '')}\nActivity Details: {case.get('ACTIVITY_DETAILS', '')}\nResolution (if known): Unknown"
        for case in similar_cases
    )
    return f"""
You are a customer complaint resolution expert specializing in credit card disputes. You are well-versed in:
- The Credit Card Accountability Responsibility and Disclosure (CARD) Act of 2009
- The Truth in Lending Act (TILA)
- Federal regulations enforced by the Consumer Financial Protection Bureau (CFPB)

Using the following real-world complaint cases, determine the most appropriate resolution for the user complaint provided at the end.
Do not hallucinate or invent any facts. Use only what is present in the historical cases.

{cases_str}

Now answer this:
Customer said: "{user_input}"

What is the most likely resolution we should offer?
"""

def query_llama(prompt):
    response = requests.post(
        OLLAMA_GEN_URL,
        json={"model": "llama3.2", "prompt": prompt, "stream": False}
    )
    return response.json().get("response", "No response from model.")

# Streamlit App
st.set_page_config(page_title="Credit Card Complaint Resolution Assistant")
st.title("🔍 Credit Card Complaint Resolution Assistant")

user_input = st.text_area("Enter a case ID or a hypothetical complaint:", height=150)

if st.button("Analyze Complaint"):
    if not user_input.strip():
        st.warning("Please enter a valid input.")
    else:
        index, metadata = load_index_and_metadata()

        if user_input.strip().isdigit():
            found = False
            for item in metadata:
                if item["COMPLAINT_CASE_ID"] == user_input.strip():
                    st.subheader("Case Summary")
                    st.write(f"**Case ID:** {item['COMPLAINT_CASE_ID']}")
                    st.write(f"**Narrative:** {item['COMPLAINT_NARRATIVE']}")
                    st.write(f"**Tags:** {item['COMPLAINT_CASE_CATEGORY_DRIVER']}")
                    st.write(f"**Agent Notes:** {item.get('ACTIVITY_NOTE', '')}")
                    st.write(f"**Activity Details:** {item.get('ACTIVITY_DETAILS', '')}")
                    found = True
                    break
            if not found:
                st.error("Case ID not found.")
        else:
            similar = query_vector_db(index, metadata, user_input)
            prompt = build_prompt(similar, user_input)
            st.subheader("Resolution Recommendation")
            with st.spinner("Consulting LLaMA 3.2..."):
                result = query_llama(prompt)
            st.success("Resolution Generated")
            st.markdown(result)

