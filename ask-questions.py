import faiss
import numpy as np
import json
import os
from openai import OpenAI
import pandas as pd
import streamlit as st


# ==========================
# Configuration (Hardcoded Values)
# ==========================
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=OPENAI_API_KEY)
CSV_FILE_PATH = 'cleaned-products.csv'  # Updated to use the new cleaned products file
EMBEDDINGS_FILE_PATH = 'embeddings.json'  # Path to your embeddings file
INDEX_FILE_PATH = 'faiss_index.index'  # Persisted index file


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
# Step 2: Load or Build FAISS Index
# ==========================

def load_or_build_index(embeddings):
    if os.path.exists(INDEX_FILE_PATH):
        print("Loading existing FAISS index from disk.")
        return faiss.read_index(INDEX_FILE_PATH)
    else:
        print("Building new FAISS index and saving to disk.")
        dimension = len(embeddings[0])
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        faiss.write_index(index, INDEX_FILE_PATH)
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
    response = client.embeddings.create(model="text-embedding-ada-002", input=text)
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
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an expert assistant helping with Better Home's product catalog."},
            {"role": "user", "content": f"Based on the following context, answer the question:\n{context}\nQuestion: {query}"}
        ]
    )
    answer = response.choices[0].message.content
    return answer


# ==========================
# Streamlit Interface
# ==========================

def main():
    st.title('Better Home Product Q&A System')

    # Load data
    df = load_product_catalog(CSV_FILE_PATH)
    embeddings = load_embeddings(EMBEDDINGS_FILE_PATH)

    # Prepare richer entries with more details
    entries = [
        f"Title: {row['title']}. Better Home Price: {row['price']} INR. Retail Price: {row['compare_at_price']} INR. URL: https://betterhomeapp.com/products/{row['handle']}"
        for index, row in df.iterrows()
    ]

    # Load or Build FAISS index
    index = load_or_build_index(embeddings)

    # Query box
    query = st.text_input("Ask a question about your catalog:")

    if query:
        answer = retrieve_and_generate(query, index, entries)
        st.write(f"### Answer: {answer}")


if __name__ == "__main__":
    main()

