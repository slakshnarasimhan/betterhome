import faiss
import numpy as np
import json
import os
import pandas as pd
import streamlit as st
import requests  # For making HTTP requests to the remote model


# ==========================
# Configuration (Hardcoded Values)
# ==========================
CSV_FILE_PATH = 'cleaned_products.csv'  # Updated to use the new cleaned products file
EMBEDDINGS_FILE_PATH = 'embeddings.json'  # Path to your embeddings file
INDEX_FILE_PATH = 'faiss_index.index'  # Persisted index file

REMOTE_MODEL_URL = "https://c8b3-34-125-31-103.ngrok-free.app/api/generate"  # Remote LLaMA model URL


# Session State for Memory
if 'conversation_history' not in st.session_state:
    st.session_state['conversation_history'] = []


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
# Step 4: Generate Query Embedding (Using Local Model)
# ==========================

def get_local_embedding(text):
    # Mock embedding generation for now
    return np.random.rand(512)


# ==========================
# Step 5: Retrieve & Generate Response
# ==========================

def retrieve_and_generate(query, index, entries):
    query_embedding = get_local_embedding(query)
    indices, distances = query_index(index, query_embedding)

    # Retrieve relevant entries
    retrieved_texts = [entries[i] for i in indices]
    context = "\n".join(retrieved_texts)

    # Adding memory context
    if st.session_state['conversation_history']:
        memory_context = "\n".join(st.session_state['conversation_history'])
        context = f"Previous Conversation:\n{memory_context}\n\n{context}"

    # Instruct model to keep URLs and formatting intact
    prompt = f"You are an assistant helping to provide product information. Ensure all product details, URLs, and price comparisons are preserved.\n{context}\nQuestion: {query}\nAnswer with product details, URLs, and variants if applicable."

    payload = {
        "model": "llama3.2",
        "prompt": prompt,
        "temperature": 0.7,
        "max_tokens": 512
    }

    try:
        response = requests.post(REMOTE_MODEL_URL, json=payload, stream=True)
        
        response_text = ""

        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line.decode('utf-8'))
                    if 'response' in data:
                        response_text += data['response']
                    if data.get('done', False):
                        break
                except json.JSONDecodeError:
                    continue
        
        st.session_state['conversation_history'].append(f"User: {query}\nAssistant: {response_text}")
        return response_text
    except Exception as e:
        return f"Error: {e}"


# ==========================
# Streamlit Interface
# ==========================

def main():
    st.title('Better Home Product Q&A System (Local Model)')

    # Load data
    df = load_product_catalog(CSV_FILE_PATH)
    embeddings = load_embeddings(EMBEDDINGS_FILE_PATH)

    # Prepare richer entries with more details
    entries = []
    for index, row in df.iterrows():
        bh_price = row.get('better_home_price', row.get('variant_price', None))
        retail_price = row.get('retail_price', row.get('variant_compare_at_price', None))
        handle = row['handle']
        product_url = f"https://betterhomeapp.com/products/{handle}"

        if pd.notna(bh_price) and pd.notna(retail_price) and retail_price > 0:
            discount_percentage = ((retail_price - bh_price) / retail_price) * 100
            discount_text = f"Better Home Price: {bh_price} INR (Retail Price: {retail_price} INR). Better Home is {discount_percentage:.2f}% cheaper."
        else:
            discount_text = f"Better Home Price: {bh_price} INR. No discount available."

        variants = ', '.join([str(row.get(x, '')) for x in ['color', 'finish', 'material', 'style'] if pd.notna(row.get(x, ''))])
        variant_text = f"Available Variants: {variants}" if variants else ""

        entries.append(
            f"Title: {row['title']}. {discount_text} URL: {product_url} {variant_text}"
        )

    # Load or Build FAISS index
    index = load_or_build_index(embeddings)

    query = st.text_input("Ask a question about your catalog:")

    if query:
        answer = retrieve_and_generate(query, index, entries)
        st.write(f"### Answer: {answer}")

    if st.button('Show Conversation History'):
        st.write("### Conversation History:")
        for item in st.session_state['conversation_history'][-5:]:
            st.write(item)


if __name__ == "__main__":
    main()

