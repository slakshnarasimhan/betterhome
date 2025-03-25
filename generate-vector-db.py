import faiss
import numpy as np
import json
import os
from openai import OpenAI
import streamlit as st

openai_key = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=openai_key)

import pandas as pd

# ==========================
# Step 1: Load Embeddings & Product Entries
# ==========================

def load_embeddings(file_name):
    with open(file_name, 'r') as f:
        embeddings = json.load(f)
    return np.array(embeddings)

def load_product_catalog(file_path):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    return df

# ==========================
# Step 2: Build & Persist FAISS Index
# ==========================

def build_faiss_index(embeddings):
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    faiss.write_index(index, 'faiss_index.index')  # Persisting the index to disk
    return index

def load_or_build_index(embeddings):
    if os.path.exists('faiss_index.index'):
        print("Loading existing FAISS index from disk.")
        return faiss.read_index('faiss_index.index')
    else:
        print("Building new FAISS index and saving to disk.")
        return build_faiss_index(embeddings)

# ==========================
# Step 3: Query the Index
# ==========================

def query_index(index, query_embedding, top_k=5):
    distances, indices = index.search(np.array([query_embedding]), top_k)
    return indices[0], distances[0]

# ==========================
# Step 4: Generate Query Embedding (Using OpenAI)
# ==========================

def get_openai_embedding(text, openai_key):
    response = client.embeddings.create(model="text-embedding-ada-002", input=text)
    return np.array(response.data[0].embedding)

# ==========================
# Step 5: Retrieve & Generate Response
# ==========================

def retrieve_and_generate(query, index, entries, openai_key):
    query_embedding = get_openai_embedding(query, openai_key)
    indices, distances = query_index(index, query_embedding)

    # Retrieve relevant entries
    retrieved_texts = [entries[i] for i in indices]
    context = "\n".join(retrieved_texts)

    # Generate answer using OpenAI
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an expert assistant helping with Better Home's product catalog."},
            {"role": "user", "content": f"Based on the following context, answer the question:\n{context}\nQuestion: {query}"}
        ]
    )
    answer = response.choices[0].message.content
    return answer


if __name__ == "__main__":
    # Load Embeddings & Product Catalog
    embeddings = load_embeddings('embeddings.json')
    df = load_product_catalog('cleaned-products.csv')

    # Create a list of entries
    entries = [
        f"Title: {row['title']}. Better Home Price: {row['price']} INR. Retail Price: {row['compare_at_price']} INR. URL: https://betterhomeapp.com/products/{row['handle']}."
        for index, row in df.iterrows()
    ]

    # Load or Build FAISS Index
    index = load_or_build_index(embeddings)

    # Test the System
    user_query = "which fan brands are the most expensive"
    answer = retrieve_and_generate(user_query, index, entries, openai_key)

    print(f"Answer: {answer}")

