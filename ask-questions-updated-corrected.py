import faiss
import numpy as np
import json
import os
import openai
import pandas as pd
import streamlit as st
from ollama import Client as Ollama
import requests
import traceback
import re


# ==========================
# Configuration
# ==========================
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    openai.api_key = OPENAI_API_KEY
except Exception as e:
    st.error(f"Error loading OpenAI API key: {str(e)}")
    OPENAI_API_KEY = "not-set"  # Placeholder to allow app to continue loading
    openai.api_key = OPENAI_API_KEY

client = Ollama()

CSV_FILE_PATH = 'cleaned_products.csv'
EMBEDDINGS_FILE_PATH = 'embeddings.json'
# BLOG_EMBEDDINGS_FILE_PATH = 'blog_embeddings.json'  # Commented out
BLOG_EMBEDDINGS_FILE_PATH = 'blog_embeddings.json'
INDEX_FILE_PATH = 'faiss_index.index'
# BLOG_INDEX_FILE_PATH = 'blog_faiss_index.index'    # Commented out
BLOG_INDEX_FILE_PATH = 'faiss_index.index_blog'
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
    
    # Special case for HDHMR/plywood
    if 'hdhmr' in query_lower or 'plywood' in query_lower or 'ply' in query_lower:
        return 'Plywood'
    
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
    # Convert 'Product Type' column to string to avoid type errors
    if 'Product Type' in df.columns:
        df['Product Type'] = df['Product Type'].astype(str)
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

def retrieve_and_generate_openai(query, context):
    # Check if it's a "best" query
    is_best_query = "best" in query.lower() and any(product_type in query.lower() for product_type in ["fan", "water heater", "geyser", "refrigerator", "washing machine", "air conditioner", "ac", "chimney", "hob", "plywood", "hdhmr"])
    
    # Debug logging
    print(f"Retrieve and generate - Query: {query}")
    print(f"Retrieve and generate - Is best query: {is_best_query}")
    
    # Create system prompt based on query type
    if is_best_query:
        system_prompt = """You are a helpful assistant for Better Home, a home improvement store. 
        When answering questions about the "best" product, remember that this is a subjective question.
        Provide a structured response that includes:
        1. Acknowledge that "best" is subjective and depends on specific needs
        2. List 2-3 top recommendations with clear reasoning for each
        3. Include price ranges and key features
        4. Mention any special considerations (e.g., energy efficiency for fans)
        5. End with a suggestion to consider specific needs and budget
        
        Format your response in markdown with clear sections and bullet points."""
        print("Using best query system prompt")
    else:
        system_prompt = """You are a helpful assistant for Better Home, a home improvement store. 
        Answer questions about products based on the provided context. 
        Be concise and informative, focusing on key features and benefits.
        Format your response in markdown."""
        print("Using regular system prompt")
    
    try:
        # Create messages for the API call
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]
        
        # Make the API call
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7,
            max_tokens=1000
        )
        
        # Extract and return the answer
        answer = response.choices[0].message.content
        print(f"Generated answer length: {len(answer)}")
        return answer
        
    except Exception as e:
        print(f"Error in retrieve_and_generate_openai: {str(e)}")
        raise e

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
# Step 5: Price & Brand-based Queries
# ==========================

def handle_price_query(query, df, product_terms=None):
    query_lower = query.lower()
    product_type = None
    is_bldc = False
    
    # Check if the query matches a product type in the product_terms dictionary
    if product_terms:
        product_type = find_product_type(query, product_terms)
        if product_type:
            print(f"Found product type in dictionary: {product_type}")
    
    # If no match in the dictionary, use manual checks
    if not product_type:
        # Check for HDHMR/plywood specific queries
        if 'hdhmr' in query_lower or 'plywood' in query_lower or 'ply' in query_lower:
            product_type = 'Plywood'
        # Check for BLDC specific queries
        elif 'bldc' in query_lower or 'brushless' in query_lower or 'energy efficient fan' in query_lower:
            product_type = 'Ceiling Fan'
            is_bldc = True
        # Check for geyser/water heater in query
        elif 'geyser' in query_lower or 'water heater' in query_lower:
            product_type = 'Water Heater'
        # Check for fan related queries
        elif 'fan' in query_lower or 'ceiling fan' in query_lower:
            product_type = 'Ceiling Fan'
        # Add more product type checks as needed
    else:
        # Check if the matched product type is a fan and needs BLDC filtering
        if 'fan' in product_type.lower() and ('bldc' in query_lower or 'brushless' in query_lower or 'energy efficient' in query_lower):
            is_bldc = True
    
    # Filter by product type if specified
    if product_type:
        df = df[df['Product Type'].astype(str) == product_type]
        
        # Further filter for BLDC fans if specified
        if is_bldc:
            df = df[df['title'].str.contains('BLDC|Brushless', case=False, na=False)]
    
    # Sort based on query type
    if 'most expensive' in query_lower or 'highest price' in query_lower:
        sorted_df = df.sort_values('Better Home Price', ascending=False)
        return sorted_df.head(5)
    elif 'least expensive' in query_lower or 'cheapest' in query_lower or 'lowest price' in query_lower:
        sorted_df = df.sort_values('Better Home Price', ascending=True)
        return sorted_df.head(5)
    
    return None

def format_brand_response(products_df, product_type, is_warranty_query=False):
    # Create a title that reflects the query focus
    if is_warranty_query:
        response = f"### Available Brands for {product_type} with Warranty Information:\n\n"
    else:
        response = f"### Available Brands for {product_type}:\n\n"
    
    # Get total count
    brand_count = len(products_df)
    response += f"We have {brand_count} brand{'s' if brand_count > 1 else ''} of {product_type.lower()} available:\n\n"
    
    for _, product in products_df.iterrows():
        title = product['title']
        brand = product['Brand']
        price = product['Better Home Price']
        retail_price = product.get('Retail Price', 0)
        url = product.get('url', '#')
        
        # Calculate discount percentage if retail price is available
        if retail_price > 0:
            discount = ((retail_price - price) / retail_price) * 100
            discount_text = f"(up to {discount:.1f}% off retail prices)"
        else:
            discount_text = ""
        
        # Highlight warranty information if present in the title
        warranty_text = ""
        if is_warranty_query:
            # Extract warranty info from title using regex
            warranty_match = re.search(r'with\s+(\d+)\s+years?\s+warranty', title, re.IGNORECASE)
            if warranty_match:
                warranty_years = warranty_match.group(1)
                warranty_text = f"**⭐ {warranty_years} Year Warranty**\n"
            elif 'warranty' in title.lower():
                warranty_text = "**⭐ Includes Warranty**\n"
            elif 'year' in title.lower() and re.search(r'(\d+)\s+years?', title, re.IGNORECASE):
                warranty_years = re.search(r'(\d+)\s+years?', title, re.IGNORECASE).group(1)
                warranty_text = f"**⭐ {warranty_years} Year Guarantee**\n"
        
        response += f"#### {brand} {discount_text}\n"
        response += f"**Example product:** {title}\n"
        if warranty_text:
            response += warranty_text
        response += f"**Starting at:** ₹{price:,.2f}\n"
        response += f"**[Browse {brand} products]({url})**\n\n"
        response += "---\n\n"
    
    return response

def handle_brand_query(query, df, product_terms=None):
    query_lower = query.lower()
    product_type = None
    
    # Check if it contains warranty-related terms
    is_warranty_query = any(term in query_lower for term in [
        'warranty', 'guarantee', 'guaranty', 'quality', 'life', 'lifetime', 'longevity',
        'years', 'replacement', 'reliable', 'reliability'
    ])
    
    # Check if it's a brand-related query - more comprehensive patterns
    is_brand_query = any(term in query_lower for term in [
        'brand', 'brands', 'manufacturer', 'manufacturers', 'make', 'makes', 'company', 'companies', 'list'
    ])
    
    # Also detect patterns like "what brands of X do you have" or "list brands of X"
    if not is_brand_query:
        # Pattern: "what brands of X" or "which brands of X"
        if re.search(r'what|which|list|show|tell|different|available', query_lower) and re.search(r'of|for', query_lower):
            is_brand_query = True
    
    if not is_brand_query and not is_warranty_query:
        return None
    
    # First try to match product type using the dictionary
    if product_terms:
        product_type = find_product_type(query, product_terms)
        if product_type:
            print(f"Found product type in dictionary for brand query: {product_type}")
    
    # If no match in the dictionary, use manual checks
    if not product_type:
        # Check for HDHMR/plywood specific queries
        if 'hdhmr' in query_lower or 'plywood' in query_lower or 'ply' in query_lower:
            product_type = 'Plywood'
        # Check for product type mentions
        elif 'geyser' in query_lower or 'water heater' in query_lower or 'heater' in query_lower:
            product_type = 'Water Heater'
        elif 'bldc' in query_lower or 'brushless' in query_lower or 'energy efficient fan' in query_lower:
            product_type = 'Ceiling Fan'
            # Specifically filter for BLDC fans
            df = df[df['title'].str.contains('BLDC|Brushless', case=False, na=False)]
        elif 'fan' in query_lower or 'ceiling fan' in query_lower:
            product_type = 'Ceiling Fan'
        elif 'washing machine' in query_lower or 'washer' in query_lower:
            product_type = 'Washing Machine'
        elif 'refrigerator' in query_lower or 'fridge' in query_lower:
            product_type = 'Refrigerator'
        elif 'air conditioner' in query_lower or 'ac' in query_lower:
            product_type = 'Air Conditioner'
        # Add more product type checks as needed
    else:
        # If it's a fan query from the dictionary and mentions BLDC, filter for that
        if 'fan' in product_type.lower() and ('bldc' in query_lower or 'brushless' in query_lower or 'energy efficient' in query_lower):
            df = df[df['title'].str.contains('BLDC|Brushless', case=False, na=False)]
    
    # If specific product type not found but query contains product keywords,
    # look for possible product types in the catalog
    if not product_type:
        # Print all unique product types for debugging
        unique_types = df['Product Type'].unique()
        print(f"Available product types: {unique_types}")
        
        # Get all unique product types from the catalog
        for pt in df['Product Type'].unique():
            pt_lower = str(pt).lower()  # Ensure conversion to string
            # Check if any word in the query is in the product type
            for word in query_lower.split():
                if word in pt_lower and len(word) > 2:  # Minimum word length to avoid false matches
                    product_type = pt
                    break
            if product_type:
                break
    
    if product_type:
        # Filter by product type - ensure both are strings for comparison
        product_type_str = str(product_type)
        filtered_df = df[df['Product Type'].astype(str) == product_type_str]
        
        # If no results, try a partial match
        if len(filtered_df) == 0:
            print(f"No exact matches for '{product_type}', trying partial matches")
            for pt in df['Product Type'].unique():
                pt_str = str(pt).lower()
                product_type_lower = product_type_str.lower()
                if product_type_lower in pt_str or pt_str in product_type_lower:
                    print(f"Found partial match: {pt}")
                    filtered_df = pd.concat([filtered_df, df[df['Product Type'].astype(str) == str(pt)]])
        
        # Log how many products found for debugging
        print(f"Found {len(filtered_df)} products for product type: {product_type}")
        
        # Get unique brands for this product type
        unique_brands = filtered_df['Brand'].unique()
        print(f"Found {len(unique_brands)} unique brands: {unique_brands}")
        
        # For each brand, get a representative product
        brand_products = []
        for brand in unique_brands:
            brand_df = filtered_df[filtered_df['Brand'] == brand]
            if len(brand_df) > 0:  # Ensure there are products for this brand
                # For warranty queries, prioritize products that mention warranty in the title
                if is_warranty_query:
                    warranty_products = brand_df[brand_df['title'].str.contains('warranty|year', case=False, na=False)]
                    if len(warranty_products) > 0:
                        brand_products.append(warranty_products.iloc[0])  # Use product that mentions warranty
                    else:
                        brand_products.append(brand_df.iloc[0])  # Fallback to first product if none mention warranty
                else:
                    brand_products.append(brand_df.iloc[0])  # Get the first product for each brand
        
        # Create brand products dataframe
        if brand_products:
            result = {
                'dataframe': pd.DataFrame(brand_products),
                'product_type': product_type_str,
                'is_warranty_query': is_warranty_query
            }
            return result
    else:
        print(f"Could not determine product type from query: {query}")
    
    return None


# ==========================
# Step 6: Generate Response
# ==========================

def format_product_response(products_df):
    response = "### Results:\n\n"
    
    # Add a special message for BLDC fans if they're in the results
    has_bldc = any(products_df['title'].str.contains('BLDC|Brushless', case=False, na=False))
    if has_bldc:
        response += "💚 **Energy Efficient BLDC Fan Featured** - These fans use up to 70% less electricity!\n\n"
    
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
        
        # Add a special highlight for BLDC fans
        is_bldc = 'BLDC' in title or 'Brushless' in title
        energy_label = "💚 ENERGY EFFICIENT: " if is_bldc else ""
        
        response += f"#### {energy_label}{title}\n"
        response += f"**Brand:** {brand}\n"
        response += f"**Price:** ₹{price:,.2f} {discount_text}\n"
        
        # Add energy saving information for BLDC fans
        if is_bldc:
            response += f"**Energy Savings:** Up to 70% less electricity compared to regular fans\n"
            
        response += f"**[Click here to buy]({url})**\n\n"
        response += "---\n\n"
    
    return response


# ==========================
# Load Blog Embeddings
# ==========================
def load_blog_embeddings(file_path):
    try:
        print(f"Loading blog embeddings from {file_path}")
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Validate the format of the loaded data
        if 'blog_embeddings' not in data or 'metadata' not in data:
            print(f"Invalid blog embeddings file format. Expected 'blog_embeddings' and 'metadata' keys. Found: {list(data.keys())}")
            return None
        
        # Check if there are any blog embeddings
        if len(data['blog_embeddings']) == 0:
            print("Blog embeddings array is empty")
            return None
            
        # Check if there is metadata for each embedding
        if len(data['blog_embeddings']) != len(data['metadata']):
            print(f"Mismatch between embeddings ({len(data['blog_embeddings'])}) and metadata ({len(data['metadata'])})")
        
        # Check if metadata has the necessary fields (title, url, etc.)
        if data['metadata'] and len(data['metadata']) > 0:
            sample_metadata = data['metadata'][0]
            print(f"Sample metadata fields: {list(sample_metadata.keys())}")
            
            # Check for URL field
            url_field_present = any('url' in item for item in data['metadata'])
            if not url_field_present:
                print("WARNING: No 'url' field found in blog metadata. URLs may not display correctly.")
            
            # Check for title field
            title_field_present = any('title' in item for item in data['metadata'])
            if not title_field_present:
                print("WARNING: No 'title' field found in blog metadata. Titles may not display correctly.")
        
        print(f"Successfully loaded {len(data['blog_embeddings'])} blog embeddings with {len(data['metadata'])} metadata entries")
        return {
            'blog_embeddings': np.array(data['blog_embeddings']),
            'metadata': data['metadata']
        }
    except FileNotFoundError:
        print(f"Blog embeddings file not found: {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error decoding JSON from blog embeddings file: {file_path}")
        return None
    except Exception as e:
        print(f"Error loading blog embeddings: {str(e)}")
        traceback.print_exc()
        return None


# ==========================
# Search Blogs
# ==========================
def search_relevant_blogs(query, blog_embeddings_dict, k=3, similarity_threshold=0.5, product_filter=None):
    print("Starting blog search...")
    print(f"Query: {query}")
    print(f"Similarity threshold: {similarity_threshold}")
    print(f"Product filter: {product_filter}")

    # Use OpenAI embeddings instead of Ollama for consistent results
    query_embedding = get_openai_embedding(query)
    print(f"Generated query embedding using OpenAI: {len(query_embedding)} dimensions")
    
    # Reshape for processing
    query_embedding = query_embedding.reshape(1, -1).astype('float32')
    
    # Convert blog embeddings to numpy array if it's a list
    blog_embeddings = np.array(blog_embeddings_dict['blog_embeddings']).astype('float32')
    
    # Build or load FAISS index for blogs
    if os.path.exists(BLOG_INDEX_FILE_PATH):
        blog_index = faiss.read_index(BLOG_INDEX_FILE_PATH)
    else:
        blog_index = faiss.IndexFlatL2(blog_embeddings.shape[1])
        if len(blog_embeddings) > 0:
            blog_index.add(blog_embeddings)
        faiss.write_index(blog_index, BLOG_INDEX_FILE_PATH)
    
    # Log the dimensions of the query embedding and the FAISS index
    print(f"Query embedding dimensions: {query_embedding.shape}")
    print(f"Blog index dimensions: {blog_index.d}")

    # Check for dimension mismatch and handle it
    if query_embedding.shape[1] != blog_index.d:
        print(f"Dimension mismatch: query {query_embedding.shape[1]} vs index {blog_index.d}")
        # Handle dimension mismatch by truncating or padding the query embedding
        if query_embedding.shape[1] > blog_index.d:
            query_embedding = query_embedding[:, :blog_index.d]
            print(f"Truncated query embedding to {blog_index.d} dimensions")
        else:
            padding = np.zeros((1, blog_index.d - query_embedding.shape[1]), dtype=np.float32)
            query_embedding = np.hstack((query_embedding, padding))
            print(f"Padded query embedding to {blog_index.d} dimensions")
    
    # Only search if we have blog embeddings
    if len(blog_embeddings) > 0:
        # Search for more blog posts than needed so we can filter
        search_k = min(k * 5, len(blog_embeddings_dict['metadata']))
        D, I = blog_index.search(query_embedding, search_k)
        print(f"Found {len(I[0])} initial matching blog articles with distances: {D[0]}")
        
        # Get the metadata for found articles
        results = []
        for idx, (distance, i) in enumerate(zip(D[0], I[0])):
            if i < len(blog_embeddings_dict['metadata']):
                metadata = blog_embeddings_dict['metadata'][i]
                
                # Ensure title and URL are present
                if 'title' not in metadata or not metadata['title']:
                    metadata['title'] = f"Blog Article {i}"
                
                if 'url' not in metadata or not metadata['url']:
                    if 'slug' in metadata and metadata['slug']:
                        metadata['url'] = f"https://betterhomeapp.com/blogs/articles/{metadata['slug']}"
                    else:
                        metadata['url'] = "https://betterhomeapp.com/blogs/articles"
                
                # Calculate similarity score
                similarity_score = 1.0 / (1.0 + distance)  # Convert distance to similarity (0-1 scale)
                print(f"Blog {i}: Similarity score = {similarity_score:.3f}")
                
                # Check if query matches related words or categories
                query_lower = query.lower()
                matches_related_words = any(word.lower() in query_lower for word in metadata.get('related_words', []))
                matches_categories = any(tag.lower() in query_lower for tag in metadata.get('categories', []))
                print(f"Blog {i}: Matches related words = {matches_related_words}, Matches categories = {matches_categories}")
                
                # Only include if similarity score is above threshold or matches related words/categories
                if similarity_score > similarity_threshold or matches_related_words or matches_categories:
                    metadata['_similarity_score'] = similarity_score
                    results.append(metadata)
                else:
                    print(f"Skipping blog {i} due to low similarity score: {similarity_score:.3f}")
            else:
                print(f"Index {i} is out of bounds for metadata array of length {len(blog_embeddings_dict['metadata'])}")
        
        # Sort by similarity score and limit to k results
        if results:
            results.sort(key=lambda x: x.get('_similarity_score', 0), reverse=True)
            results = results[:k]
            print(f"Returning {len(results)} most relevant blog articles")
        else:
            print("No sufficiently relevant blog articles found")
        
        return results
    else:
        print("No blog embeddings to search")
        return []


def format_blog_response(blog_results, query=None):
    """
    Format blog search results for display
    
    Parameters:
    - blog_results: List of blog metadata dictionaries
    - query: The original user query (for context)
    
    Returns:
    - Formatted markdown text or None if no results
    """
    if not blog_results or len(blog_results) == 0:
        return None
    
    # Extract query topic for header if available
    topic_header = ""
    if query:
        # Try to extract the subject of the query
        query_words = query.lower().split()
        
        # Simplified logic to extract likely topic words
        stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'is', 'are', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 
                      'about', 'like', 'of', 'do', 'does', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 
                      'those', 'list', 'show', 'tell', 'me', 'get', 'can', 'could', 'would', 'should', 'how'}
        
        # Get potential topic words (non-stop words)
        topic_words = [word for word in query_words if word not in stop_words and len(word) > 3]
        
        if topic_words:
            topic_header = f" About {' '.join(topic_words[:3]).title()}"
    
    response = f"### 📚 Related Blog Articles{topic_header}\n\n"
    
    # Only include this introduction if we have actual content in the blogs
    content_exists = any(blog.get('content') for blog in blog_results)
    if content_exists:
        response += "I found these relevant articles that may provide additional information:\n\n"
    
    for blog in blog_results:
        title = blog.get('title', 'Untitled Article')
        url = blog.get('url', '#')
        
        # Make sure we're not displaying generic titles like "Article X"
        if not title or title.startswith('Article '):
            title = "Blog Article (Click to read)"
        
        # Ensure we have a valid URL
        if not url or url == '#':
            # If no URL is available, try to construct one from other metadata
            if 'slug' in blog:
                url = f"https://betterhomeapp.com/blogs/articles/{blog['slug']}"
            else:
                url = "https://betterhomeapp.com/blogs/articles"
        
        date = blog.get('date', '')
        
        # Get relevance scores if available
        similarity_score = blog.get('_similarity_score', 0)
        
        # Add a relevance indicator based on scores
        relevance_indicator = ""
        if similarity_score > 0.8:
            relevance_indicator = "🔍 "  # High relevance
        
        # Create a brief excerpt if content is available
        content = blog.get('content', '')
        
        # Improved excerpt that focuses on relevant parts if possible
        excerpt = ""
        if content and query:
            # Try to find most relevant sentence containing query words
            query_words = set(query.lower().split()) - {'a', 'an', 'the', 'and', 'or', 'but', 'is', 'are', 'in', 'on', 'at'}
            sentences = content.replace('\n', ' ').split('. ')
            
            # Score sentences by how many query words they contain
            sentence_scores = [(s, sum(1 for w in query_words if w in s.lower())) for s in sentences if len(s) > 20]
            
            if sentence_scores:
                # Get the most relevant sentence (highest score)
                best_sentence = max(sentence_scores, key=lambda x: x[1])[0]
                excerpt = best_sentence.strip()
                
                # Truncate if too long
                if len(excerpt) > 200:
                    excerpt = excerpt[:200] + "..."
            else:
                # No good match, use beginning of content
                excerpt = content[:200] + "..." if len(content) > 200 else content
        elif content:
            # No query provided, just use the beginning
            excerpt = content[:200] + "..." if len(content) > 200 else content
        
        # Format the article entry with enhanced readability
        response += f"#### {relevance_indicator}📝 [{title}]({url})\n"
        
        # Only show date if it exists and isn't empty
        if date and date.strip():
            response += f"**Date:** {date}\n"
        
        # Add excerpt if available
        if excerpt:
            response += f"**Preview:** {excerpt}\n"
        
        # Add a cleaner "Read more" link
        response += f"[**Read Full Article** ↗]({url})\n\n"
        response += "---\n\n"
    
    return response


def main():
    st.title('Better Home Product Q&A System')

    # Add debugging information
    #st.sidebar.markdown("### Debug Info")
    #st.sidebar.text(f"OpenAI API Key: {'Set' if OPENAI_API_KEY != 'not-set' else 'Not Set'}")
    #st.sidebar.text(f"CSV File exists: {os.path.exists(CSV_FILE_PATH)}")
    #st.sidebar.text(f"Embeddings File exists: {os.path.exists(EMBEDDINGS_FILE_PATH)}")
    #st.sidebar.text(f"Product Index exists: {os.path.exists(PRODUCT_INDEX_FILE_PATH)}")
    #st.sidebar.text(f"Blog Embeddings File exists: {os.path.exists(BLOG_EMBEDDINGS_FILE_PATH)}")
    #st.sidebar.text(f"Blog Index exists: {os.path.exists(BLOG_INDEX_FILE_PATH)}")
    
    # Load data with error handling
    try:
        df = load_product_catalog(CSV_FILE_PATH)
        st.sidebar.text(f"Product catalog loaded: {len(df)} items")
        
        # Load product terms dictionary
        product_terms = {}
        if os.path.exists(PRODUCT_TERMS_FILE):
            try:
                product_terms = load_product_terms(PRODUCT_TERMS_FILE)
                st.sidebar.text(f"Product terms loaded: {len(product_terms)} product types")
            except Exception as e:
                st.sidebar.text(f"Error loading product terms: {str(e)}")
        else:
            st.sidebar.text("Product terms file not found")
        
        # Load blog embeddings if available
        blog_embeddings = None
        if os.path.exists(BLOG_EMBEDDINGS_FILE_PATH):
            blog_embeddings = load_blog_embeddings(BLOG_EMBEDDINGS_FILE_PATH)
            if blog_embeddings:
                st.sidebar.text(f"Blog embeddings loaded: {len(blog_embeddings['metadata'])} articles")
            else:
                st.sidebar.text("Failed to load blog embeddings")
        else:
            st.sidebar.text("Blog embeddings file not found")
        
        # Display available product types for debugging
        #st.sidebar.markdown("### Available Product Types")
        unique_types = df['Product Type'].unique()
        # Convert all product types to strings before joining
        unique_types_str = [str(pt) for pt in unique_types]
        #st.sidebar.text(f"{', '.join(unique_types_str)}")
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        traceback.print_exc()  # Print detailed error information
        st.stop()
        return

    # Display metadata
    #st.sidebar.write("### Catalog Statistics")
    #st.sidebar.write(f"Total Products: {len(df)}")
    #st.sidebar.write(f"Unique Product Types: {df['Product Type'].nunique()}")
    #st.sidebar.write(f"Unique Brands: {df['Brand'].nunique()}")

    # Query box
    query = st.text_input("Ask a question about Better Home products:")

    if query:
        query_lower = query.lower()  # Define query_lower here
        try:
            # Check if it's a "best" query first - this needs to be checked before any other processing
            is_best_query = "best" in query_lower and any(product_type in query_lower for product_type in ["fan", "water heater", "geyser", "refrigerator", "washing machine", "air conditioner", "ac", "chimney", "hob", "plywood", "hdhmr"])
            
            # Debug logging
            st.sidebar.text(f"Query: {query}")
            st.sidebar.text(f"Is best query: {is_best_query}")
            st.sidebar.text(f"Query lower: {query_lower}")
            
            # If it's a "best" query, use the special handling for all product types
            if is_best_query:
                # Determine the product type from the query
                product_type = None
                
                # More comprehensive product type detection
                if 'fan' in query_lower or 'ceiling fan' in query_lower:
                    product_type = 'Ceiling Fan'
                    st.sidebar.text("Detected fan in best query")
                elif 'water heater' in query_lower or 'geyser' in query_lower or 'heater' in query_lower:
                    product_type = 'Water Heater'
                elif 'refrigerator' in query_lower or 'fridge' in query_lower:
                    product_type = 'Refrigerator'
                elif 'washing machine' in query_lower or 'washer' in query_lower:
                    product_type = 'Washing Machine'
                elif 'air conditioner' in query_lower or 'ac' in query_lower:
                    product_type = 'Air Conditioner'
                elif 'chimney' in query_lower:
                    product_type = 'Chimney'
                elif 'hob' in query_lower or 'hob top' in query_lower:
                    product_type = 'Hob Top'
                elif 'plywood' in query_lower or 'hdhmr' in query_lower or 'ply' in query_lower:
                    product_type = 'Plywood'
                
                # If product type not found by direct matching, try using product_terms
                if not product_type and product_terms:
                    product_type = find_product_type(query, product_terms)
                    st.sidebar.text(f"Found product type from dictionary: {product_type}")
                
                # Debug logging
                st.sidebar.text(f"Product type for best query: {product_type}")
                
                # If we identified a product type, filter the dataframe
                if product_type:
                    filtered_df = df[df['Product Type'].astype(str) == product_type]
                    
                    # Debug logging
                    st.sidebar.text(f"Found {len(filtered_df)} products of type {product_type}")
                    
                    # Create a context with all products of this type
                    context = f"Here are all the {product_type} products available:\n\n"
                    for i, (idx, product) in enumerate(filtered_df.iterrows()):
                        context += f"Product {i+1}:\n"
                        context += f"Title: {product.get('title', 'N/A')}\n"
                        context += f"Brand: {product.get('Brand', 'N/A')}\n"
                        context += f"Product Type: {product.get('Product Type', 'N/A')}\n"
                        context += f"Better Home Price: ₹{product.get('Better Home Price', 0):,.2f}\n"
                        context += f"Retail Price: ₹{product.get('Retail Price', 0):,.2f}\n"
                        context += f"URL: {product.get('url', '#')}\n\n"
                    
                    # Get relevant blog articles if available
                    blog_results = []
                    if blog_embeddings:
                        try:
                            blog_results = search_relevant_blogs(query, blog_embeddings, product_filter=product_type)
                            if blog_results and len(blog_results) > 0:
                                context += "\n\nHere are some relevant blog articles that may contain useful information:\n\n"
                                for i, blog in enumerate(blog_results):
                                    blog_title = blog.get('title', 'Untitled')
                                    blog_content = blog.get('content', '')
                                    if len(blog_content) > 1000:
                                        blog_content = blog_content[:1000] + "..."
                                    
                                    context += f"Blog Article {i+1}:\n"
                                    context += f"Title: {blog_title}\n"
                                    context += f"Content: {blog_content}\n\n"
                        except Exception as e:
                            print(f"Error searching blog embeddings: {str(e)}")
                            st.sidebar.text(f"Error searching blog embeddings: {str(e)}")
                    
                    # Debug logging
                    st.sidebar.text(f"Context length: {len(context)} characters")
                    
                    # Generate the answer using the updated system prompt
                    try:
                        # Add a timeout to the OpenAI API call
                        st.sidebar.text("Calling OpenAI API for best query response...")
                        answer = retrieve_and_generate_openai(query, context)
                        st.sidebar.text("Received response from OpenAI API")
                        st.markdown("### Answer:", unsafe_allow_html=True)
                        st.markdown(answer, unsafe_allow_html=True)
                        
                        # Display related blog articles if found
                        if blog_results and len(blog_results) > 0:
                            blog_answer = format_blog_response(blog_results, query)
                            if blog_answer:
                                st.markdown(blog_answer, unsafe_allow_html=True)
                                st.markdown("---")
                        
                        # Update conversation history and return
                        st.session_state['conversation_history'].append({
                            'query': query,
                            'answer': answer
                        })
                        return
                    except Exception as e:
                        st.sidebar.text(f"Error generating answer: {str(e)}")
                        st.error(f"Error generating answer: {str(e)}")
                        
                        # Fallback to a simpler response if OpenAI fails
                        st.markdown("### Answer:")
                        st.markdown(f"**It is subjective.** Here are some {product_type} options based on different criteria:")
                        
                        # Create a simple structured response
                        if len(filtered_df) > 0:
                            # Sort by price for price recommendation
                            price_sorted = filtered_df.sort_values('Better Home Price')
                            
                            # Get the most expensive for performance recommendation
                            performance_sorted = filtered_df.sort_values('Better Home Price', ascending=False)
                            
                            # Create a simple response
                            response = f"**If you consider performance:** {performance_sorted.iloc[0]['title']} offers the best features.\n\n"
                            response += f"**If you consider price:** {price_sorted.iloc[0]['title']} is the most affordable option starting from ₹{price_sorted.iloc[0]['Better Home Price']:,.2f}.\n\n"
                            
                            # Add more criteria if we have enough products
                            if len(filtered_df) >= 3:
                                response += f"**If you consider brand reputation:** {filtered_df.iloc[0]['Brand']} is a well-known and reliable brand.\n\n"
                            
                            st.markdown(response)
                            
                            # Update conversation history
                            st.session_state['conversation_history'].append({
                                'query': query,
                                'answer': response
                            })
                        return
            
            # Check if query is just a product type directly - special handling for direct product queries
            direct_product_query = False
            product_type = None
            
            # Check for HDHMR/plywood in direct query
            if 'hdhmr' in query_lower or 'plywood' in query_lower or 'ply' in query_lower:
                product_type = 'Plywood'
                direct_product_query = True
                st.sidebar.text(f"Direct query for product type: {product_type}")
            
            # Check other direct product type queries
            if not direct_product_query and product_terms:
                product_type = find_product_type(query, product_terms)
                if product_type:
                    direct_product_query = True
                    st.sidebar.text(f"Found direct product type in dictionary: {product_type}")
            
            # Handle direct product type queries
            if direct_product_query and product_type:
                # Filter by product type
                filtered_df = df[df['Product Type'].astype(str) == str(product_type)]
                
                # Special handling for BLDC fans if that's the query
                if product_type == 'Ceiling Fan' and ('bldc' in query_lower or 'brushless' in query_lower):
                    filtered_df = filtered_df[filtered_df['title'].str.contains('BLDC|Brushless', case=False, na=False)]
                
                if len(filtered_df) > 0:
                    st.markdown(f"### {product_type} Products")
                    answer = format_product_response(filtered_df.head(5))
                    st.markdown(answer, unsafe_allow_html=True)
                    
                    # Search for related blog articles
                    if blog_embeddings:
                        st.sidebar.text(f"Searching for blog articles related to {product_type}...")
                        blog_results = search_relevant_blogs(query, blog_embeddings, product_filter=product_type)
                        if blog_results and len(blog_results) > 0:
                            blog_answer = format_blog_response(blog_results, query)
                            st.markdown(blog_answer, unsafe_allow_html=True)
                    
                    # Update conversation history
                    st.session_state['conversation_history'].append({
                        'query': query,
                        'answer': answer
                    })
                    return
                else:
                    st.sidebar.text(f"No products found for type: {product_type}")
            
            # Continue with other query types if not a direct product query or no results found
            # First, check if it's a product type availability query (e.g., "Do you have ceiling fans?")
            is_availability_query = any(term in query_lower for term in [
                'do you have', 'do you sell', 'do you carry', 'available', 'in stock', 'sell', 'offer', 'show me'
            ])
            
            if is_availability_query:
                # Print debugging info
                st.sidebar.text(f"Detected availability query: {query}")
                
                # Check if query matches any product type using product_terms dictionary
                matched_product_type = None
                if product_terms:
                    matched_product_type = find_product_type(query, product_terms)
                    if matched_product_type:
                        st.sidebar.text(f"Matched product type from dictionary: {matched_product_type}")
                
                # Special check for geysers/water heaters if not already matched
                if not matched_product_type and any(term in query_lower for term in [
                    'geyser', 'geysers', 'water heater', 'water heaters', 'heater', 'heaters'
                ]):
                    st.sidebar.text("Detected geyser/water heater query")
                    # Get all water heaters
                    water_heaters = df[df['Product Type'].str.contains('Water Heater|Geyser', case=False, na=False)]
                    
                    if len(water_heaters) > 0:
                        st.markdown("### Yes, we have Water Heaters/Geysers!")
                        answer = format_product_response(water_heaters.head(5))
                        st.markdown(answer, unsafe_allow_html=True)
                        
                        # Update conversation history
                        st.session_state['conversation_history'].append({
                            'query': query,
                            'answer': answer
                        })
                        return
                
                # Use the matched product type from dictionary if found
                if matched_product_type:
                    product_df = df[df['Product Type'].str.contains(matched_product_type, case=False, na=False)]
                    
                    if len(product_df) > 0:
                        st.markdown(f"### Yes, we have {matched_product_type} products!")
                        
                        # Special case for fans - prioritize BLDC fans 
                        if 'fan' in matched_product_type.lower():
                            bldc_products = product_df[product_df['title'].str.contains('BLDC|Brushless', case=False, na=False)]
                            regular_products = product_df[~product_df['title'].str.contains('BLDC|Brushless', case=False, na=False)]
                            
                            # Combine with BLDC fans first
                            sample_products = pd.concat([bldc_products.head(3), regular_products.head(2)])
                            if len(bldc_products) > 0:
                                st.markdown("💚 **Featured: Energy-efficient BLDC fans**")
                        else:
                            sample_products = product_df.head(5)
                        
                        answer = format_product_response(sample_products)
                        st.markdown(answer, unsafe_allow_html=True)
                        
                        # Update conversation history
                        st.session_state['conversation_history'].append({
                            'query': query,
                            'answer': answer
                        })
                        return
                
                # Special handling for BLDC fan queries
                is_bldc_query = 'bldc' in query_lower or 'brushless' in query_lower or 'energy efficient' in query_lower
                
                if is_bldc_query and ('fan' in query_lower or 'ceiling fan' in query_lower):
                    # Handle BLDC fan availability query
                    fan_df = df[df['Product Type'].astype(str) == 'Ceiling Fan']
                    bldc_fans = fan_df[fan_df['title'].str.contains('BLDC|Brushless', case=False, na=False)]
                    
                    if len(bldc_fans) > 0:
                        st.markdown("### Yes, we have BLDC Ceiling Fans!")
                        st.markdown("💚 These energy-efficient fans use up to 70% less electricity than conventional fans.")
                        answer = format_product_response(bldc_fans.head(5))
                        st.markdown(answer, unsafe_allow_html=True)
                        
                        # Update conversation history
                        st.session_state['conversation_history'].append({
                            'query': query,
                            'answer': answer
                        })
                        return
                
                # Regular product availability check (fallback method)
                product_found = False
                for pt in df['Product Type'].unique():
                    pt_lower = str(pt).lower()  # Ensure conversion to string
                    
                    # Debug print for each product type being checked
                    st.sidebar.text(f"Checking product type: {pt}")
                    
                    # More relaxed matching for product types
                    if any(word in pt_lower for word in query_lower.split() if len(word) > 2) or \
                       any(word in query_lower for word in pt_lower.split() if len(word) > 2):
                        product_found = True
                        st.markdown(f"### Yes, we have {pt} products!")
                        
                        # Special case for fans - prioritize BLDC fans
                        if pt.lower() == 'ceiling fan' or 'fan' in pt.lower():
                            fan_df = df[df['Product Type'].astype(str) == pt]
                            bldc_fans = fan_df[fan_df['title'].str.contains('BLDC|Brushless', case=False, na=False)]
                            regular_fans = fan_df[~fan_df['title'].str.contains('BLDC|Brushless', case=False, na=False)]
                            
                            # Combine with BLDC fans first
                            sample_products = pd.concat([bldc_fans.head(3), regular_fans.head(2)])
                            st.markdown("💚 **Featured: Energy-efficient BLDC fans**")
                        else:
                            # For other product types, take the first 5
                            sample_products = df[df['Product Type'].astype(str) == str(pt)].head(5)
                            
                        answer = format_product_response(sample_products)
                        st.markdown(answer, unsafe_allow_html=True)
                        break
                
                if not product_found:
                    st.markdown("### Product Type Not Found")
                    st.markdown(f"I'm sorry, but I couldn't find that product type in our catalog. Here are the product types we currently offer:")
                    unique_types_str = [str(pt) for pt in df['Product Type'].unique()]
                    st.markdown(", ".join(unique_types_str))
                    
                    # Store conversation history
                    st.session_state['conversation_history'].append({
                        'query': query,
                        'answer': "Product type not found in our catalog."
                    })
                    return
            
            # Check if it's a price-based query
            price_results = handle_price_query(query, df, product_terms)
            
            # Check if it's a brand-related query
            brand_results = handle_brand_query(query, df, product_terms)
            
            if brand_results is not None:
                # Handle brand-specific query
                results_df = brand_results['dataframe']
                product_type = brand_results['product_type']
                st.sidebar.text(f"Handling brand query - found {len(results_df)} brands for {product_type}")
                answer = format_brand_response(results_df, product_type, brand_results['is_warranty_query'])
                st.markdown(answer, unsafe_allow_html=True)
                
                # Add blog results for brand queries if available
                if blog_embeddings:
                    # Create a more specific query for blog search
                    is_warranty_query = brand_results.get('is_warranty_query', False)
                    
                    if is_warranty_query:
                        # For warranty queries, specifically search for warranty content
                        brand_blog_query = f"{product_type} brands warranty guarantee quality"
                    else:
                        brand_blog_query = f"{product_type} brands information"
                    
                    # Add the product_filter parameter to ensure we get relevant blogs
                    blog_results = search_relevant_blogs(
                        brand_blog_query, 
                        blog_embeddings, 
                        product_filter=product_type
                    )
                    
                    if blog_results and len(blog_results) > 0:
                        blog_answer = format_blog_response(blog_results, query)
                        st.markdown(blog_answer, unsafe_allow_html=True)
            
            elif price_results is not None:
                # Handle price-based query
                answer = format_product_response(price_results)
                st.markdown(answer, unsafe_allow_html=True)
            else:
                # Handle regular query
                try:
                    # Special handling for ceiling fan and BLDC fan queries
                    if ('fan' in query_lower or 'ceiling fan' in query_lower) and not price_results and not is_best_query:
                        # Debug logging
                        st.sidebar.text("Special fan handling triggered")
                        st.sidebar.text(f"is_best_query: {is_best_query}")
                        
                        # Check if it's specifically about BLDC fans
                        is_bldc_query = 'bldc' in query_lower or 'brushless' in query_lower or 'energy efficient' in query_lower
                        
                        # Filter for ceiling fans
                        fan_df = df[df['Product Type'].astype(str) == 'Ceiling Fan']
                        
                        if is_bldc_query:
                            # If specifically asking about BLDC fans, filter for those
                            bldc_fans = fan_df[fan_df['title'].str.contains('BLDC|Brushless', case=False, na=False)]
                            if len(bldc_fans) > 0:
                                answer = format_product_response(bldc_fans.head(5))
                                st.markdown("### BLDC Ceiling Fans (Energy Efficient):", unsafe_allow_html=True)
                                st.markdown(answer, unsafe_allow_html=True)
                                
                                # Get relevant blogs for BLDC fans
                                if blog_embeddings:
                                    blog_results = search_relevant_blogs("BLDC ceiling fans energy efficient", blog_embeddings, product_filter='Ceiling Fan')
                                    if blog_results and len(blog_results) > 0:
                                        blog_answer = format_blog_response(blog_results, query)
                                        st.markdown(blog_answer, unsafe_allow_html=True)
                                
                                # Update conversation history and return
                                st.session_state['conversation_history'].append({
                                    'query': query,
                                    'answer': answer
                                })
                                return
                        else:
                            # For general fan queries, prioritize BLDC fans first, then show regular fans
                            bldc_fans = fan_df[fan_df['title'].str.contains('BLDC|Brushless', case=False, na=False)]
                            regular_fans = fan_df[~fan_df['title'].str.contains('BLDC|Brushless', case=False, na=False)]
                            
                            # Combine with BLDC fans first
                            prioritized_fans = pd.concat([bldc_fans.head(3), regular_fans.head(2)])
                            
                            if len(prioritized_fans) > 0:
                                answer = format_product_response(prioritized_fans.head(5))
                                st.markdown("### Ceiling Fans (Energy Efficient BLDC Fans Highlighted):", unsafe_allow_html=True)
                                st.markdown(answer, unsafe_allow_html=True)
                                
                                # Get relevant blogs for fans
                                if blog_embeddings:
                                    blog_results = search_relevant_blogs("ceiling fans", blog_embeddings, product_filter='Ceiling Fan')
                                    if blog_results and len(blog_results) > 0:
                                        blog_answer = format_blog_response(blog_results, query)
                                        st.markdown(blog_answer, unsafe_allow_html=True)
                                
                                # Update conversation history and return
                                st.session_state['conversation_history'].append({
                                    'query': query,
                                    'answer': answer
                                })
                                return
                    
                    # Continue with the regular embedding-based query if no special handling applied
                    query_embedding = get_openai_embedding(query)
                    st.sidebar.text(f"Query embedding generated: {len(query_embedding)} dimensions")
                    query_embedding = query_embedding.reshape(1, -1).astype('float32')
                    
                    # Load product index
                    try:
                        product_index = faiss.read_index(PRODUCT_INDEX_FILE_PATH)
                        index_dim = product_index.d
                        st.sidebar.text(f"Product index loaded: {product_index.ntotal} vectors, {index_dim} dimensions")
                        
                        # Check dimension mismatch and adapt if necessary
                        query_dim = query_embedding.shape[1]
                        if query_dim != index_dim:
                            st.sidebar.text(f"Dimension mismatch: query {query_dim} vs index {index_dim}")
                            if query_dim > index_dim:
                                # Truncate the query embedding
                                query_embedding = query_embedding[:, :index_dim]
                                st.sidebar.text(f"Truncated query embedding to {index_dim} dimensions")
                            else:
                                # Pad the query embedding
                                padding = np.zeros((1, index_dim - query_dim), dtype=np.float32)
                                query_embedding = np.hstack((query_embedding, padding))
                                st.sidebar.text(f"Padded query embedding to {index_dim} dimensions")
                        
                        # Search for more products initially to account for possible duplicates
                        k_search = 10  # Search for more products initially
                        D, I = product_index.search(query_embedding, min(k_search, product_index.ntotal))
                        
                        # Get relevant products with better formatting
                        all_products = df.iloc[I[0]]
                        
                        # Remove duplicate products based on title
                        unique_products = all_products.drop_duplicates(subset=['title'])
                        
                        # Take only the top 5 unique products
                        products = unique_products.head(5)
                        st.sidebar.text(f"Found {len(products)} unique relevant products")
                        
                        # Create a more structured context
                        context = "Here are the relevant products:\n\n"
                        for i, (idx, product) in enumerate(products.iterrows()):
                            context += f"Product {i+1}:\n"
                            context += f"Title: {product.get('title', 'N/A')}\n"
                            context += f"Brand: {product.get('Brand', 'N/A')}\n"
                            context += f"Product Type: {product.get('Product Type', 'N/A')}\n"
                            context += f"Better Home Price: ₹{product.get('Better Home Price', 0):,.2f}\n"
                            context += f"Retail Price: ₹{product.get('Retail Price', 0):,.2f}\n"
                            context += f"URL: {product.get('url', '#')}\n\n"
                        
                        # Search for relevant blog articles if available
                        blog_results = []
                        if blog_embeddings:
                            try:
                                st.sidebar.text("Searching for relevant blog articles...")
                                # Determine appropriate product filter based on query
                                product_filter = None
                                query_lower = query.lower()
                                
                                # For BLDC or ceiling fan queries, set product filter
                                if 'bldc' in query_lower or 'fan' in query_lower or 'ceiling' in query_lower:
                                    product_filter = 'Ceiling Fan'
                                elif 'plywood' in query_lower or 'hdhmr' in query_lower:
                                    product_filter = 'Plywood'
                                elif 'water heater' in query_lower or 'geyser' in query_lower:
                                    product_filter = 'Water Heater'
                                
                                # For general queries about products, don't use a strict product filter
                                if query_lower.startswith('tell me about') or query_lower.startswith('what is') or query_lower.startswith('how'):
                                    # Use lower similarity threshold for informational queries
                                    blog_results = search_relevant_blogs(query, blog_embeddings, similarity_threshold=0.5)
                                else:
                                    # Use product filter if available, otherwise do a general search
                                    blog_results = search_relevant_blogs(query, blog_embeddings, product_filter=product_filter)
                                
                                if blog_results and len(blog_results) > 0:
                                    st.sidebar.text(f"Found {len(blog_results)} relevant blog articles")
                                    
                                    # Add blog content to context
                                    context += "\n\nHere are some relevant blog articles that may contain useful information:\n\n"
                                    for i, blog in enumerate(blog_results):
                                        blog_title = blog.get('title', 'Untitled')
                                        blog_content = blog.get('content', '')
                                        # Trim content if too long
                                        if len(blog_content) > 1000:
                                            blog_content = blog_content[:1000] + "..."
                                        
                                        context += f"Blog Article {i+1}:\n"
                                        context += f"Title: {blog_title}\n"
                                        context += f"Content: {blog_content}\n\n"
                            except Exception as e:
                                st.sidebar.text(f"Error searching blog embeddings: {str(e)}")
                                traceback.print_exc()
                        else:
                            st.sidebar.text("Blog embeddings not available")
                        answer = retrieve_and_generate_openai(query, context)
                        st.markdown("### Answer:", unsafe_allow_html=True)
                        st.markdown(answer, unsafe_allow_html=True)
                        
                        # Display related blog articles if found
                        if blog_results and len(blog_results) > 0:
                            blog_answer = format_blog_response(blog_results, query)
                            if blog_answer:
                                st.markdown(blog_answer, unsafe_allow_html=True)
                                # Add a separator to make it distinct
                                st.markdown("---")
                        else:
                            st.sidebar.text("No blog content to display")
                    except Exception as e:
                        st.error(f"Error searching product index: {str(e)}")
                        st.sidebar.text(f"Product index error: {traceback.format_exc()}")
                except Exception as e:
                    st.error(f"Error generating query embedding: {str(e)}")
                    st.sidebar.text(f"Embedding error: {traceback.format_exc()}")
            
            # Update conversation history
            st.session_state['conversation_history'].append({
                'query': query,
                'answer': answer if 'answer' in locals() else "No results found."
            })
        except Exception as e:
            st.error(f"An error occurred while processing your query: {str(e)}")
            st.sidebar.text(f"General error: {traceback.format_exc()}")
            print(f"Error details: {traceback.format_exc()}")

    # Toggle button for conversation history
    if st.button('Show Conversation History'):
        st.write("### Conversation History:")
        for item in st.session_state['conversation_history'][-5:]:
            st.markdown(f"**Q:** {item['query']}")
            st.markdown(f"**A:** {item['answer']}", unsafe_allow_html=True)
            st.markdown("---")


if __name__ == "__main__":
    main()
