
import faiss
import numpy as np
import json
import os
import openai
import pandas as pd
import streamlit as st
from ollama import Client as Ollama
import requests


# ==========================
# Configuration
# ==========================
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
openai.api_key = OPENAI_API_KEY
client = Ollama()

CSV_FILE_PATH = 'cleaned_products.csv'
EMBEDDINGS_FILE_PATH = 'embeddings.json'
INDEX_FILE_PATH = 'faiss_index.index'
PRODUCT_TERMS_FILE = 'product_terms.json'  # New file for product terms

# Session State for Memory
if 'conversation_history' not in st.session_state:
    st.session_state['conversation_history'] = []


# ==========================
# Step 1: Load Product Terms Dictionary
# ==========================

def load_product_terms(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


def find_product_type(query, product_terms):
    query_lower = query.lower()
    for product_type, info in product_terms.items():
        if product_type.lower() in query_lower:
            return product_type
        for alternative in info['alternatives']:
            if alternative.lower() in query_lower:
                return product_type
        for category in info['categories']:
            if category.lower() in query_lower:
                return product_type
    return None


# ==========================
# Step 2: Load Embeddings & Product Entries
# ==========================

def load_embeddings(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return {
        'product_embeddings': np.array(data['product_embeddings']),
        'product_type_embeddings': np.array(data['product_type_embeddings']),
        'brand_embeddings': np.array(data['brand_embeddings']),
        'metadata': data['metadata']
    }


def load_product_catalog(file_path):
    df = pd.read_csv(file_path)
    return df


# ==========================
# Step 3: Generate Query Embedding
# ==========================

def get_openai_embedding(text, model='llama3.2'):
    if model == 'llama3.2':
        try:
            response = client.create(model="llama3.2", messages=[{"role": "user", "content": text}])
            embedding = response['data'][0]['embedding']
            embedding = np.array(embedding) / np.linalg.norm(embedding)
            return embedding
        except Exception as e:
            print(f"Error generating embedding with Ollama: {e}")
            return np.random.rand(3072)
    else:
        try:
            response = openai.Embedding.create(
                model="text-embedding-ada-002",
                input=text
            )
            embedding = response['data'][0]['embedding']
            return np.array(embedding)
        except Exception as e:
            print(f"Error generating embedding with OpenAI: {e}")
            return np.random.rand(1536)


# ==========================
# Step 4: Generate Response with OpenAI
# ==========================

def retrieve_and_generate_openai(query, context, model='gpt-4'):
    system_prompt = (
        "You are an expert assistant helping with Better Home's product catalog. "
        "When discussing prices, always mention both the Better Home Price and how much cheaper it is compared to the Retail Price. "
        "Be specific about the savings in INR and percentage terms."
    )

    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context: {context}\nQuestion: {query}\nAnswer:"}
            ],
            max_tokens=300,
            temperature=0.7
        )
        answer = response.choices[0].message.content.strip()
        return answer
    except Exception as e:
        print(f"Error generating response with OpenAI: {e}")
        return "I'm sorry, I couldn't generate a response at this time."


# ==========================
# Step 5: Streamlit Interface
# ==========================

def main():
    st.title('Better Home Product Q&A System')

    # Load data
    df = load_product_catalog(CSV_FILE_PATH)
    embeddings_dict = load_embeddings(EMBEDDINGS_FILE_PATH)

    # Display metadata
    st.sidebar.write("### Catalog Statistics")
    st.sidebar.write(f"Total Products: {embeddings_dict['metadata']['total_products']}")
    st.sidebar.write(f"Unique Product Types: {embeddings_dict['metadata']['unique_product_types']}")
    st.sidebar.write(f"Unique Brands: {embeddings_dict['metadata']['unique_brands']}")

    # Query box
    query = st.text_input("Ask a question about Better Home products:")

    if query:
        # Generate answer
        context = "Your context preparation logic here..."
        answer = retrieve_and_generate_openai(query, context)
        st.write(f"### Answer: {answer}")

    # Toggle button for conversation history
    if st.button('Show Conversation History'):
        st.write("### Conversation History:")
        for item in st.session_state['conversation_history'][-5:]:
            st.write(item)


if __name__ == "__main__":
    main()
