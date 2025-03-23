import faiss
import numpy as np
import json
from openai import OpenAI

client = OpenAI(api_key=openai_api_key)
import pandas as pd
import streamlit as st


# ==========================
# Step 1: Load Embeddings & Product Entries
# ==========================

def load_embeddings(uploaded_file):
    # Read uploaded file
    content = uploaded_file.read()
    embeddings = json.loads(content.decode('utf-8'))
    return np.array(embeddings)


def load_product_catalog(uploaded_file):
    # Read CSV file from the uploaded file object
    df = pd.read_csv(uploaded_file)
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


# ==========================
# Streamlit Interface
# ==========================

def main():
    st.title('Product Catalog Q&A System')
    openai_api_key = st.text_input("Enter your OpenAI API Key:", type="password")

    if openai_api_key:
        uploaded_file = st.file_uploader("Upload your product catalog CSV file", type="csv")
        embeddings_file = st.file_uploader("Upload your embeddings.json file", type="json")

        if uploaded_file and embeddings_file:
            df = load_product_catalog(uploaded_file)
            entries = df['title'].tolist()  # Using 'title' column as product entries

            embeddings = load_embeddings(embeddings_file)
            index = build_faiss_index(embeddings)

            query = st.text_input("Ask a question about your catalog:")

            if query:
                answer = retrieve_and_generate(query, index, entries, openai_api_key)
                st.write(f"### Answer: {answer}")


if __name__ == "__main__":
    main()

