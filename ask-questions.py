import faiss
import numpy as np
import json
import openai
import pandas as pd
import streamlit as st
openai.api_key=st.secrets["OPENAI_API_KEY"]


# ==========================
# Configuration
# ==========================
CSV_FILE_PATH = 'products-clean.csv'  # Your product catalog CSV file
EMBEDDINGS_FILE_PATH = 'embeddings.json'  # Your embeddings file


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
    df['price'] = df['price'].astype(str) + ' INR'  # Convert price to string and add INR
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
# Step 3: Retrieve & Generate Response with Memory
# ==========================
conversation_memory = []


def retrieve_and_generate(query, index, entries, top_k=5):
    # Generate query embedding
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=query
    )
    query_embedding = np.array(response['data'][0]['embedding'])

    # Perform similarity search
    indices, distances = index.search(np.array([query_embedding]), top_k)

    # Retrieve relevant entries
    retrieved_texts = [entries[int(i)] for i in indices[0].tolist()]
    context = "\n".join(retrieved_texts)

    # Add query to memory
    conversation_memory.append({"role": "user", "content": query})

    # Generate response from GPT-4
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an expert assistant helping with a product catalog."}
        ] + conversation_memory + [
            {"role": "user", "content": f"Based on the following context, answer the question:\n{context}\nQuestion: {query}"}
        ]
    )

    answer = response['choices'][0]['message']['content']
    conversation_memory.append({"role": "assistant", "content": answer})

    return answer


# ==========================
# Streamlit Interface
# ==========================

def main():
    st.set_page_config(page_title="Enhanced Product Q&A", layout="wide")
    st.title('📌 Enhanced Product Catalog Q&A System')

    # Load Data
    df = load_product_catalog(CSV_FILE_PATH)
    entries = [
        f"Title: {row['title']}. Price: {row['price']}. Description: {row['description']}. Features: {row['features']}. "
        f"Brand: {row['brand']}. Type: {row['type']}. Tags: {row['tags']}. Warranty: {row['warranty']}"
        for _, row in df.iterrows()
    ]

    embeddings = load_embeddings(EMBEDDINGS_FILE_PATH)
    index = build_faiss_index(embeddings)

    # User Interaction
    with st.sidebar:
        st.subheader("Settings")
        top_k = st.slider("Number of Results to Retrieve", 1, 10, 5)

        if st.button('Clear Memory'):
            conversation_memory.clear()

    tab1, tab2 = st.tabs(["Q&A", "Conversation History"])

    with tab1:
        query = st.text_input("Ask a question about your catalog:")

        if query:
            answer = retrieve_and_generate(query, index, entries, top_k)
            st.write(f"### 💬 Answer: {answer}")

    with tab2:
        st.write("### Conversation History")
        for message in conversation_memory:
            if message['role'] == 'user':
                st.write(f"**You:** {message['content']}")
            else:
                st.write(f"**Assistant:** {message['content']}")


if __name__ == "__main__":
    main()

