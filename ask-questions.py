import faiss
import numpy as np
import json
import os
#from openai import OpenAI
from openai import OpenAI

client = OpenAI()
import pandas as pd
import streamlit as st
#from openai import OpenAI
from ollama import Client as Ollama
import requests
#import openai


# ==========================
# Configuration
# ==========================
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
#client = OpenAI(api_key=OPENAI_API_KEY)
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
    """
    Find the most relevant product type for a given query using the product terms dictionary.
    """
    query_lower = query.lower()
    best_match = None
    best_score = 0

    for product_type, info in product_terms.items():
        # Check standard name
        if product_type.lower() in query_lower:
            return product_type

        # Check alternatives
        for alternative in info['alternatives']:
            if alternative.lower() in query_lower:
                return product_type

        # Check categories
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
# Step 3: Load or Build FAISS Index
# ==========================

def load_or_build_index(embeddings_dict):
    if os.path.exists(INDEX_FILE_PATH):
        print("Loading existing FAISS index from disk.")
        return {
            'product_index': faiss.read_index(f"{INDEX_FILE_PATH}_product"),
            'product_type_index': faiss.read_index(f"{INDEX_FILE_PATH}_type"),
            'brand_index': faiss.read_index(f"{INDEX_FILE_PATH}_brand")
        }
    else:
        print("Building new FAISS indices and saving to disk.")
        indices = {}

        # Create and save product index
        dimension = len(embeddings_dict['product_embeddings'][0])
        indices['product_index'] = faiss.IndexFlatL2(dimension)
        indices['product_index'].add(embeddings_dict['product_embeddings'])
        faiss.write_index(indices['product_index'], f"{INDEX_FILE_PATH}_product")

        # Create and save product type index
        dimension = len(embeddings_dict['product_type_embeddings'][0])
        indices['product_type_index'] = faiss.IndexFlatL2(dimension)
        indices['product_type_index'].add(embeddings_dict['product_type_embeddings'])
        faiss.write_index(indices['product_type_index'], f"{INDEX_FILE_PATH}_type")

        # Create and save brand index
        dimension = len(embeddings_dict['brand_embeddings'][0])
        indices['brand_index'] = faiss.IndexFlatL2(dimension)
        indices['brand_index'].add(embeddings_dict['brand_embeddings'])
        faiss.write_index(indices['brand_index'], f"{INDEX_FILE_PATH}_brand")

        return indices


# ==========================
# Step 4: Query the Index
# ==========================

def query_index(index, query_embedding, top_k=5):
    distances, indices = index.search(np.array([query_embedding]), top_k)
    return indices[0], distances[0]


# ==========================
# Step 5: Generate Query Embedding
# ==========================

def get_openai_embedding(text):
    # Use the local Ollama client to generate embeddings
    try:
        response = client.embed(model="llama3.2", input=text)
        if hasattr(response, 'embeddings') and response.embeddings:
            embedding = np.array(response.embeddings[0])
            # Normalize the embedding (L2 normalization)
            embedding = embedding / np.linalg.norm(embedding)
            return embedding
        else:
            print("Warning: No embedding received from model. Using fallback embedding.")
            return np.random.rand(3072)  # Fallback to random embedding with correct dimension
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return np.random.rand(3072)  # Fallback to random embedding with correct dimension


# ==========================
# Step 6: Determine Query Type
# ==========================

def determine_query_type(query, product_terms):
    """
    Determine the type of query and find relevant product type if applicable.
    """
    query_lower = query.lower()

    # Price-related keywords
    price_keywords = ['price', 'cost', 'expensive', 'cheap', 'cheaper', 'discount', 'savings']

    # Product type keywords
    type_keywords = ['type', 'category', 'kind', 'variety']

    # Brand keywords
    brand_keywords = ['brand', 'make', 'manufacturer', 'company']

    # First check if we can identify a specific product type
    product_type = find_product_type(query, product_terms)
    if product_type:
        return 'product_type', product_type

    # If no specific product type found, check other query types
    if any(keyword in query_lower for keyword in price_keywords):
        return 'price', None
    elif any(keyword in query_lower for keyword in type_keywords):
        return 'product_type', None
    elif any(keyword in query_lower for keyword in brand_keywords):
        return 'brand', None
    else:
        return 'general', None


# ==========================
# Step 7: Retrieve & Generate Response
# ==========================

def retrieve_and_generate(query, indices, df, query_type, product_type=None):
    query_embedding = get_openai_embedding(query)

    # Select appropriate index based on query type
    if query_type == 'price':
        index = indices['product_index']
        top_k = 5
    elif query_type == 'product_type':
        index = indices['product_type_index']
        top_k = 3
    elif query_type == 'brand':
        index = indices['brand_index']
        top_k = 3
    else:
        index = indices['product_index']
        top_k = 5

    indices, distances = query_index(index, query_embedding, top_k)

    # Prepare context based on query type
    context_parts = []
    for idx in indices:
        row = df.iloc[idx]
        if query_type == 'price':
            context_parts.append(
                f"Product: {row['title']}. "
                f"Better Home Price: {row['Better Home Price']} INR. "
                f"Retail Price: {row['Retail Price']} INR. "
                f"Discount: {((row['Retail Price'] - row['Better Home Price']) / row['Retail Price'] * 100):.2f}% off."
            )
        else:
            context_parts.append(
                f"Product: {row['title']}. "
                f"Type: {row['Product Type']}. "
                f"Brand: {row['Brand']}. "
                f"Better Home Price: {row['Better Home Price']} INR."
            )

    context = "\n".join(context_parts)

    # Add conversation history
    if st.session_state['conversation_history']:
        memory_context = "\n".join(st.session_state['conversation_history'])
        context = f"Previous Conversation:\n{memory_context}\n\n{context}"

    # Prepare the prompt for the model
    system_prompt = """You are an expert assistant helping with Better Home's product catalog. 
    When discussing prices, always mention both the Better Home Price and how much cheaper it is compared to the Retail Price.
    Be specific about the savings in INR and percentage terms."""

    prompt = f"{system_prompt}\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"

    # Ensure the correct parameters are passed to the Ollama client
    try:
        response = client.create(model="llama3.2", messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\nQuestion: {query}\nAnswer:"}
        ])
        answer = response.choices[0].message.content
    except Exception as e:
        print(f"Error generating response: {e}")
        answer = "I'm sorry, I couldn't generate a response at this time."

    # Save to memory
    st.session_state['conversation_history'].append(f"User: {query}\nAssistant: {answer}")

    return answer


def retrieve_and_generate_openai(query, indices, df, embeddings_dict, query_type, product_type=None):
    query_embedding = get_openai_embedding(query)

    # Initialize df_filtered
    df_filtered = pd.DataFrame()

    # Select appropriate index based on query type
    if query_type == 'product_type' and product_type:
        # Filter the dataframe to only include the specified product type
        df_filtered = df[df['Product Type'].str.lower() == product_type.lower()]

        # Sort by price if the query indicates a price-related request
        if 'cheapest' in query.lower() or 'expensive' in query.lower():
            df_filtered = df_filtered.sort_values(by='Better Home Price', ascending='cheapest' in query.lower())

        embeddings_filtered = embeddings_dict['product_type_embeddings'][df_filtered.index]
        index = faiss.IndexFlatL2(len(embeddings_filtered[0]))
        index.add(embeddings_filtered)
        top_k = len(df_filtered)  # Adjust top_k to include all items
    else:
        index = indices['product_index']
        top_k = 5

    indices, distances = query_index(index, query_embedding, top_k)

    # Prepare context based on query type
    context_parts = []
    for idx in indices:
        row = df.iloc[idx]
        product_url = row.get('url', '#')  # Use 'url' as the column name
        context_parts.append(
            f"Product: [{row['title']}]({product_url}). "  # Hyperlink the product URL
            f"Type: {row['Product Type']}. "
            f"Brand: {row['Brand']}. "
            f"Better Home Price: {row['Better Home Price']} INR."
        )

    # Add total number of items in the catalog for the product type
    total_items = len(df_filtered)
    context_parts.append(f"Total number of {product_type} in catalog: {total_items}")

    context = "\n".join(context_parts)

    # Add conversation history
    if st.session_state['conversation_history']:
        memory_context = "\n".join(st.session_state['conversation_history'])
        context = f"Previous Conversation:\n{memory_context}\n\n{context}"

    # Prepare the prompt for the model
    system_prompt = """You are an expert assistant helping with Better Home's product catalog. 
    When discussing prices, always mention both the Better Home Price and how much cheaper it is compared to the Retail Price.
    Be specific about the savings in INR and percentage terms."""

    # Use OpenAI API to generate response with chat model
    try:
        response = client.chat.completions.create(model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\nQuestion: {query}\nAnswer:"}
        ],
        max_tokens=300,
        temperature=0.7)
        answer = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating response with OpenAI: {e}")
        answer = "I'm sorry, I couldn't generate a response at this time."

    # Save to memory
    st.session_state['conversation_history'].append(f"User: {query}\nAssistant: {answer}")

    return answer


# ==========================
# Streamlit Interface
# ==========================

def main():
    st.title('Better Home Product Q&A System')

    # Load data
    df = load_product_catalog(CSV_FILE_PATH)
    embeddings_dict = load_embeddings(EMBEDDINGS_FILE_PATH)
    product_terms = load_product_terms(PRODUCT_TERMS_FILE)

    # Display metadata
    st.sidebar.write("### Catalog Statistics")
    st.sidebar.write(f"Total Products: {embeddings_dict['metadata']['total_products']}")
    st.sidebar.write(f"Unique Product Types: {embeddings_dict['metadata']['unique_product_types']}")
    st.sidebar.write(f"Unique Brands: {embeddings_dict['metadata']['unique_brands']}")

    # Load or Build FAISS indices
    indices = load_or_build_index(embeddings_dict)

    # Query box
    query = st.text_input("Ask a question about Better Home products:")

    if query:
        # Determine query type and product type
        query_type, product_type = determine_query_type(query, product_terms)

        # Generate answer
        answer = retrieve_and_generate_openai(query, indices, df, embeddings_dict, query_type, product_type)
        st.write(f"### Answer: {answer}")

    # Toggle button for conversation history
    if st.button('Show Conversation History'):
        st.write("### Conversation History:")
        for item in st.session_state['conversation_history'][-5:]:
            st.write(item)


if __name__ == "__main__":
    main()

