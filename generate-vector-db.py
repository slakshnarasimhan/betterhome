import faiss
import numpy as np
import json
from openai import OpenAI
openai_key=st.secrets["OPENAI_API_KEY"]
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

def get_openai_embedding(text, openai_api_key):
    response = client.embeddings.create(model="text-embedding-ada-002",
    input=text)
    return np.array(response.data[0].embedding)


# ==========================
# Step 5: Retrieve & Generate Response
# ==========================

def retrieve_and_generate(query, index, entries, openai_api_key):
    query_embedding = get_openai_embedding(query, openai_api_key)
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


if __name__ == "__main__":
    # Load Embeddings & Product Catalog
    embeddings = load_embeddings('embeddings.json')
    df = load_product_catalog('products-clean.csv')

    # Create a list of entries
    entries = df['title'].tolist()  # Using 'title' column as product entries

    # Build FAISS index
    index = build_faiss_index(embeddings)

    # Test the System
    openai_api_key = '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'  # Replace with your OpenAI API Key
    user_query = "What is the price of Daikin AC 1.0 Ton - Non Inverter - 3 Star - Split AC - FTL35UV16W1 and RL35UV16W1 - Copper Condenser"
    answer = retrieve_and_generate(user_query, index, entries, openai_api_key)

    print(f"Answer: {answer}")

