import faiss
import numpy as np
import json
from openai import OpenAI
OPENAI_API_KEY="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"
client = OpenAI(api_key=OPENAI_API_KEY)
import pandas as pd
import streamlit as st


# ==========================
# Configuration (Hardcoded Values)
# ==========================

OPENAI_API_KEY = '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'  # Replace with your actual OpenAI API Key
CSV_FILE_PATH = 'products-clean.csv'  # Path to your product catalog CSV file
EMBEDDINGS_FILE_PATH = 'embeddings.json'  # Path to your embeddings file



# ==========================
# Step 1: Load Embeddings & Product Entries
# ==========================

def load_embeddings(file_path):
    with open(file_path, 'r') as f:
        embeddings = json.load(f)
    return np.array(embeddings)


def load_product_catalog(file_path):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    return df


# ==========================
# Step 2: Build FAISS Index
# ==========================

def build_faiss_index(embeddings):
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index


# ==========================
# Step 3: Query the Index
# ==========================

def query_index(index, query_embedding, top_k=5):
    distances, indices = index.search(np.array([query_embedding]), top_k)
    return indices[0], distances[0]


# ==========================
# Step 4: Generate Query Embedding (Using OpenAI)
# ==========================

def get_openai_embedding(text):
    response = client.embeddings.create(model="text-embedding-ada-002",
    input=text)
    return np.array(response.data[0].embedding)


# ==========================
# Step 5: Retrieve & Generate Response
# ==========================

def retrieve_and_generate(query, index, entries):
    query_embedding = get_openai_embedding(query)
    indices, distances = query_index(index, query_embedding)

    # Retrieve relevant entries
    retrieved_texts = [entries[i] for i in indices]
    context = "\n".join(retrieved_texts)

    # Generate answer using OpenAI
    response = client.chat.completions.create(model="gpt-4",
    messages=[
        {"role": "system", "content": "You are an expert assistant helping with a product catalog."},
        {"role": "user", "content": f"Based on the following context, answer the question:\n{context}\nQuestion: {query}"}
    ])
    answer = response.choices[0].message.content
    return answer


# ==========================
# Streamlit Interface
# ==========================

def main():
    st.title('Product Catalog Q&A System')

    # Load data
    df = load_product_catalog(CSV_FILE_PATH)
    embeddings = load_embeddings(EMBEDDINGS_FILE_PATH)

    # Prepare richer entries with more details
    entries = [
        f"Title: {row['title']}. Price: {row['price']}. Description: {row['description']}. Features: {row['features']}. "
        f"Brand: {row['brand']}. Type: {row['type']}. Tags: {row['tags']}. Warranty: {row['warranty']}"
        for index, row in df.iterrows()
    ]

    # Build FAISS index
    index = build_faiss_index(embeddings)

    # Query box
    query = st.text_input("Ask a question about your catalog:")

    if query:
        answer = retrieve_and_generate(query, index, entries)
        st.write(f"### Answer: {answer}")


if __name__ == "__main__":
    main()
