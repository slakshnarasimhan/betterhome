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
def get_openai_embedding(text, model='text-embedding-ada-002'):
    try:
        response = openai.Embedding.create(
            model=model,
            input=text
        )
        embedding = response['data'][0]['embedding']
        return np.array(embedding)
    except Exception as e:
        print(f"Error generating embedding with OpenAI: {e}")
        return np.random.rand(1536)

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
# Step 4: FAISS Index Operations
# ==========================

def build_or_load_faiss_index(embeddings, dimension):
    if os.path.exists(INDEX_FILE_PATH):
        index = faiss.read_index(INDEX_FILE_PATH)
    else:
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        faiss.write_index(index, INDEX_FILE_PATH)
    return index


def search_products(query, df, embeddings_dict, k=5):
    # Generate query embedding
    query_embedding = get_openai_embedding(query)
    query_embedding = query_embedding.reshape(1, -1).astype('float32')
    
    # Build or load FAISS index
    index = build_or_load_faiss_index(embeddings_dict['product_embeddings'], len(query_embedding[0]))
    
    # Search
    D, I = index.search(query_embedding, k)
    return I[0]  # Return indices of top k matches


# ==========================
# Step 5: Price-based Queries
# ==========================

def handle_price_query(query, df):
    query_lower = query.lower()
    product_type = None
    
    # Check for geyser/water heater in query
    if 'geyser' in query_lower or 'water heater' in query_lower:
        product_type = 'Water Heater'
    
    # Filter by product type if specified
    if product_type:
        df = df[df['Product Type'] == product_type]
    
    # Sort based on query type
    if 'most expensive' in query_lower or 'highest price' in query_lower:
        sorted_df = df.sort_values('Better Home Price', ascending=False)
        return sorted_df.head(5)
    elif 'least expensive' in query_lower or 'cheapest' in query_lower or 'lowest price' in query_lower:
        sorted_df = df.sort_values('Better Home Price', ascending=True)
        return sorted_df.head(5)
    
    return None


# ==========================
# Step 6: Generate Response
# ==========================

def format_product_response(products_df):
    response = "Results:\n"
    for _, product in products_df.iterrows():
        title = product['title']
        brand = product['Brand']
        price = product['Better Home Price']
        url = product.get('url', '#')
        
        response += f"{title} by {brand}: ₹{price:,.2f}\n"
        response += f"Buy now: {url}\n\n"
    
    return response


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
        # Check if it's a price-based query
        price_results = handle_price_query(query, df)
        
        if price_results is not None:
            # Handle price-based query
            answer = format_product_response(price_results)
        else:
            # Handle regular query
            indices = search_products(query, df, embeddings_dict)
            context = df.iloc[indices].to_string()
            answer = retrieve_and_generate_openai(query, context)
        
        st.write("### Answer:")
        st.write(answer)
        
        # Update conversation history
        st.session_state['conversation_history'].append({
            'query': query,
            'answer': answer
        })

    # Toggle button for conversation history
    if st.button('Show Conversation History'):
        st.write("### Conversation History:")
        for item in st.session_state['conversation_history'][-5:]:
            st.write(f"Q: {item['query']}")
            st.write(f"A: {item['answer']}\n")


if __name__ == "__main__":
    main()
