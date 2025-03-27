import faiss
import numpy as np
import json
import os
from openai import OpenAI
import pandas as pd
import streamlit as st


# ==========================
# Configuration
# ==========================
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=OPENAI_API_KEY)
CSV_FILE_PATH = 'cleaned_products.csv'
EMBEDDINGS_FILE_PATH = 'embeddings.json'
INDEX_FILE_PATH = 'faiss_index.index'

# Session State for Memory
if 'conversation_history' not in st.session_state:
    st.session_state['conversation_history'] = []


# ==========================
# Step 1: Load Embeddings & Product Entries
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
# Step 2: Load or Build FAISS Index
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
# Step 3: Query the Index
# ==========================

def query_index(index, query_embedding, top_k=5):
    distances, indices = index.search(np.array([query_embedding]), top_k)
    return indices[0], distances[0]


# ==========================
# Step 4: Generate Query Embedding
# ==========================

def get_openai_embedding(text):
    response = client.embeddings.create(model="text-embedding-ada-002", input=text)
    return np.array(response.data[0].embedding)


# ==========================
# Step 5: Determine Query Type
# ==========================

def determine_query_type(query):
    query_lower = query.lower()
    
    # Price-related keywords
    price_keywords = ['price', 'cost', 'expensive', 'cheap', 'cheaper', 'discount', 'savings']
    
    # Product type keywords
    type_keywords = ['type', 'category', 'kind', 'variety']
    
    # Brand keywords
    brand_keywords = ['brand', 'make', 'manufacturer', 'company']
    
    if any(keyword in query_lower for keyword in price_keywords):
        return 'price'
    elif any(keyword in query_lower for keyword in type_keywords):
        return 'product_type'
    elif any(keyword in query_lower for keyword in brand_keywords):
        return 'brand'
    else:
        return 'general'


# ==========================
# Step 6: Retrieve & Generate Response
# ==========================

def retrieve_and_generate(query, indices, df, query_type='general'):
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
    
    # Generate answer using OpenAI
    system_prompt = """You are an expert assistant helping with Better Home's product catalog. 
    When discussing prices, always mention both the Better Home Price and how much cheaper it is compared to the Retail Price.
    Be specific about the savings in INR and percentage terms."""
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Based on the following context, answer the question:\n{context}\nQuestion: {query}"}
        ]
    )
    answer = response.choices[0].message.content
    
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
        # Determine query type
        query_type = determine_query_type(query)
        
        # Generate answer
        answer = retrieve_and_generate(query, indices, df, query_type)
        st.write(f"### Answer: {answer}")
    
    # Toggle button for conversation history
    if st.button('Show Conversation History'):
        st.write("### Conversation History:")
        for item in st.session_state['conversation_history'][-5:]:
            st.write(item)


if __name__ == "__main__":
    main()

