import faiss
import numpy as np
import json
import os
import pandas as pd
import streamlit as st
import requests  # For making HTTP requests to the local Ollama endpoint


# ==========================
# Configuration
# ==========================
CSV_FILE_PATH = 'cleaned_products.csv'
EMBEDDINGS_FILE_PATH = 'embeddings.json'
INDEX_FILE_PATH = 'faiss_index.index'
OLLAMA_URL = "http://localhost:11434/api/generate"  # Local Ollama endpoint

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

def get_local_embedding(text):
    """
    Generate embeddings using the local Llama model's last layer output.
    This approach uses the model's final hidden states as embeddings.
    """
    # Prepare the prompt for embedding generation
    prompt = f"Represent the following text for retrieval: {text}"
    
    # Call the local Llama model with specific parameters for embedding generation
    payload = {
        "model": "llama2",  # or your specific model name
        "prompt": prompt,
        "stream": False,  # We don't need streaming for embeddings
        "temperature": 0.0,  # Lower temperature for more consistent embeddings
        "max_tokens": 1,  # We only need the last token's embedding
        "embedding": True  # Request embeddings from the model
    }

    try:
        # Make the request to the local Ollama endpoint
        response = requests.post(OLLAMA_URL, json=payload)
        response_data = response.json()
        
        if 'embedding' in response_data:
            # Convert the embedding to numpy array
            embedding = np.array(response_data['embedding'])
            
            # Normalize the embedding (L2 normalization)
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding
        else:
            print("Warning: No embedding received from model. Using fallback embedding.")
            return np.random.rand(512)  # Fallback to random embedding
            
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return np.random.rand(512)  # Fallback to random embedding


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
    query_embedding = get_local_embedding(query)
    
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
    
    # Prepare the prompt for the local model
    system_prompt = """You are an expert assistant helping with Better Home's product catalog. 
    When discussing prices, always mention both the Better Home Price and how much cheaper it is compared to the Retail Price.
    Be specific about the savings in INR and percentage terms."""
    
    prompt = f"{system_prompt}\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    
    payload = {
        "model": "llama2",  # or your specific model name
        "prompt": prompt,
        "stream": True,
        "temperature": 0.7,
        "max_tokens": 512
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, stream=True)
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
        
        # Save to memory
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

