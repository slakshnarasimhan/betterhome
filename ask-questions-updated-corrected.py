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
BLOG_EMBEDDINGS_FILE_PATH = 'blog_embeddings.json'
INDEX_FILE_PATH = 'faiss_index.index'
BLOG_INDEX_FILE_PATH = 'blog_faiss_index.index'
PRODUCT_INDEX_FILE_PATH = 'faiss_index.index_product'
TYPE_INDEX_FILE_PATH = 'faiss_index.index_type'
BRAND_INDEX_FILE_PATH = 'faiss_index.index_brand'
PRODUCT_TERMS_FILE = 'product_terms.json'

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
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return {
            'product_embeddings': np.array(data.get('product_embeddings', [])),
            'product_type_embeddings': np.array(data.get('product_type_embeddings', [])),
            'brand_embeddings': np.array(data.get('brand_embeddings', [])),
            'metadata': data.get('metadata', {
                'total_products': 0,
                'unique_product_types': 0,
                'unique_brands': 0
            })
        }
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        return {
            'product_embeddings': np.array([]),
            'product_type_embeddings': np.array([]),
            'brand_embeddings': np.array([]),
            'metadata': {
                'total_products': 0,
                'unique_product_types': 0,
                'unique_brands': 0
            }
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

def retrieve_and_generate_openai(query, context, blog_suggestions=None, model='gpt-4'):
    system_prompt = (
        "You are an expert assistant helping with Better Home's product catalog. "
        "When discussing prices, always mention both the Better Home Price and how much cheaper it is compared to the Retail Price. "
        "Be specific about the savings in INR and percentage terms. "
        "Always include the product URL as a markdown link in your response using the format [Click here to buy](url). "
        "When mentioning blog articles, include them as clickable links using the format [Title](url)."
    )
    
    # Add blog suggestions to context if available
    if blog_suggestions:
        blog_context = "\n\nRelevant blog articles:\n"
        for blog in blog_suggestions:
            blog_context += f"- [{blog['title']}]({blog['url']}) (Published: {blog['date']})\n"
        context += blog_context

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
        
        # Add blog suggestions to the response with clickable links
        if blog_suggestions:
            answer += "\n\n### Related Articles:\n"
            for blog in blog_suggestions:
                answer += f"- [{blog['title']}]({blog['url']}) (Published: {blog['date']})\n"
        
        return answer
    except Exception as e:
        print(f"Error generating response with OpenAI: {e}")
        return "I'm sorry, I couldn't generate a response at this time."

# ==========================
# Step 4: FAISS Index Operations
# ==========================

def build_or_load_faiss_index(embeddings, dimension, index_path):
    if os.path.exists(index_path):
        index = faiss.read_index(index_path)
    else:
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        faiss.write_index(index, index_path)
    return index


def search_products(query, df, embeddings_dict, k=5):
    # Generate query embedding
    query_embedding = get_openai_embedding(query)
    query_embedding = query_embedding.reshape(1, -1).astype('float32')
    
    # Build or load FAISS index
    index = build_or_load_faiss_index(
        embeddings_dict['product_embeddings'], 
        len(query_embedding[0]),
        PRODUCT_INDEX_FILE_PATH
    )
    
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
    response = "### Results:\n\n"
    for _, product in products_df.iterrows():
        title = product['title']
        brand = product['Brand']
        price = product['Better Home Price']
        retail_price = product.get('Retail Price', 0)
        url = product.get('url', '#')
        
        # Calculate discount percentage if retail price is available
        if retail_price > 0:
            discount = ((retail_price - price) / retail_price) * 100
            discount_text = f"({discount:.1f}% off retail price ₹{retail_price:,.2f})"
        else:
            discount_text = ""
        
        response += f"#### {title}\n"
        response += f"**Brand:** {brand}\n"
        response += f"**Price:** ₹{price:,.2f} {discount_text}\n"
        response += f"**[Click here to buy]({url})**\n\n"
        response += "---\n\n"
    
    return response


# ==========================
# Load Blog Embeddings
# ==========================
def load_blog_embeddings(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return {
            'blog_embeddings': np.array(data['blog_embeddings']),
            'metadata': data['metadata']
        }
    except FileNotFoundError:
        return None


# ==========================
# Search Blogs
# ==========================
def search_relevant_blogs(query, blog_embeddings_dict, k=2):
    if blog_embeddings_dict is None:
        return []
        
    query_embedding = get_openai_embedding(query)
    query_embedding = query_embedding.reshape(1, -1).astype('float32')
    
    # Build or load FAISS index for blogs
    if os.path.exists(BLOG_INDEX_FILE_PATH):
        blog_index = faiss.read_index(BLOG_INDEX_FILE_PATH)
    else:
        blog_index = faiss.IndexFlatL2(len(query_embedding[0]))
        blog_index.add(blog_embeddings_dict['blog_embeddings'])
        faiss.write_index(blog_index, BLOG_INDEX_FILE_PATH)
    
    D, I = blog_index.search(query_embedding, k)
    return [blog_embeddings_dict['metadata'][i] for i in I[0]]


def main():
    st.title('Better Home Product Q&A System')

    # Load data
    df = load_product_catalog(CSV_FILE_PATH)
    
    # Load blog embeddings
    blog_embeddings_dict = load_blog_embeddings(BLOG_EMBEDDINGS_FILE_PATH)

    # Display metadata
    st.sidebar.write("### Catalog Statistics")
    st.sidebar.write(f"Total Products: {len(df)}")
    st.sidebar.write(f"Unique Product Types: {df['Product Type'].nunique()}")
    st.sidebar.write(f"Unique Brands: {df['Brand'].nunique()}")
    if blog_embeddings_dict:
        st.sidebar.write(f"Blog Articles: {len(blog_embeddings_dict['metadata'])}")

    # Query box
    query = st.text_input("Ask a question about Better Home products:")

    if query:
        # Check if it's a price-based query
        price_results = handle_price_query(query, df)
        
        # Find relevant blog articles
        blog_suggestions = search_relevant_blogs(query, blog_embeddings_dict)
        
        if price_results is not None:
            # Handle price-based query
            answer = format_product_response(price_results)
            if blog_suggestions:
                answer += "\n\n### Related Articles:\n"
                for blog in blog_suggestions:
                    answer += f"- [{blog['title']}]({blog['url']}) (Published: {blog['date']})\n"
            st.markdown(answer, unsafe_allow_html=True)
        else:
            # Handle regular query
            query_embedding = get_openai_embedding(query)
            query_embedding = query_embedding.reshape(1, -1).astype('float32')
            
            # Load product index
            product_index = faiss.read_index(PRODUCT_INDEX_FILE_PATH)
            D, I = product_index.search(query_embedding, 5)
            
            # Get relevant products
            context = df.iloc[I[0]].to_string()
            answer = retrieve_and_generate_openai(query, context, blog_suggestions)
            st.markdown("### Answer:", unsafe_allow_html=True)
            st.markdown(answer, unsafe_allow_html=True)
        
        # Update conversation history
        st.session_state['conversation_history'].append({
            'query': query,
            'answer': answer
        })

    # Toggle button for conversation history
    if st.button('Show Conversation History'):
        st.write("### Conversation History:")
        for item in st.session_state['conversation_history'][-5:]:
            st.markdown(f"**Q:** {item['query']}")
            st.markdown(f"**A:** {item['answer']}", unsafe_allow_html=True)
            st.markdown("---")


if __name__ == "__main__":
    main()
