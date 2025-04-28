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
import yaml  # Add yaml import
import sys  # Add sys import for path debugging
from typing import Dict, Any, List, Optional, Tuple  # Add missing imports
import networkx as nx
from collections import defaultdict

# Print Python path for debugging
print("Python path:")
for path in sys.path:
    print(f"  - {path}")

# Import common modules
try:
    from betterhome.common.config import (
        CSV_FILE_PATH,
        EMBEDDINGS_FILE_PATH,
        PRODUCT_INDEX_FILE_PATH,
        PRODUCT_TERMS_FILE,
        HOME_CONFIG_FILE,
        BLOG_EMBEDDINGS_FILE_PATH,
        BLOG_INDEX_FILE_PATH,
        load_product_terms,
        load_home_config
    )
    print("Successfully imported config module")
except Exception as e:
    print(f"Error importing config module: {str(e)}")
    traceback.print_exc()

try:
    from betterhome.common.embeddings import (
        get_query_embedding,
        load_embeddings,
        build_or_load_faiss_index,
        search_products
    )
    print("Successfully imported embeddings module")
except Exception as e:
    print(f"Error importing embeddings module: {str(e)}")
    traceback.print_exc()

try:
    from betterhome.common.product_utils import (
        find_product_type,
        handle_price_query,
        handle_brand_query,
        format_brand_response,
        format_product_response,
        search_catalog,
        format_answer
    )
    print("Successfully imported product_utils module")
except Exception as e:
    print(f"Error importing product_utils module: {str(e)}")
    traceback.print_exc()

try:
    from betterhome.common.blog_utils import is_how_to_query
    print("Successfully imported blog_utils module")
except Exception as e:
    print(f"Error importing blog_utils module: {str(e)}")
    traceback.print_exc()

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

# Load data
df = pd.read_csv(CSV_FILE_PATH)
embedding_data = load_embeddings(EMBEDDINGS_FILE_PATH)
product_terms = load_product_terms(PRODUCT_TERMS_FILE)
home_config = load_home_config(HOME_CONFIG_FILE)

# Load the product graph
try:
    # Handle different NetworkX versions 
    if hasattr(nx, 'read_gpickle'):
        # For older versions of NetworkX
        product_graph = nx.read_gpickle('product_graph.gpickle')
    else:
        # For newer versions of NetworkX, try the current API
        import pickle
        with open('product_graph.gpickle', 'rb') as f:
            product_graph = pickle.load(f)
    print("Successfully loaded product graph")
except Exception as e:
    print(f"Error loading product graph: {str(e)}")
    product_graph = None

# Load the knowledge graph if available
try:
    # Handle different NetworkX versions
    if hasattr(nx, 'read_gpickle'):
        # For older versions of NetworkX
        knowledge_graph = nx.read_gpickle('product_graph.gpickle')
    else:
        # For newer versions of NetworkX, try the current API
        import pickle
        with open('product_graph.gpickle', 'rb') as f:
            knowledge_graph = pickle.load(f)
    print("Successfully loaded product knowledge graph")
except Exception as e:
    print(f"Error loading product knowledge graph: {str(e)}")
    knowledge_graph = None

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
    
    # First check exact matches in product terms dictionary
    for product_type, info in product_terms.items():
        if product_type.lower() in query_lower:
            return product_type
            
        # Check categories if they exist
        if 'categories' in info:
            for category in info['categories']:
                if category.lower() in query_lower:
                    return product_type
                
    # If no exact match found, try partial matches
    for product_type, info in product_terms.items():
        # Check if any word in the product type is in the query
        product_type_words = product_type.lower().split()
        if any(word in query_lower for word in product_type_words):
            return product_type
            
        # Check categories if they exist
        if 'categories' in info:
            for category in info['categories']:
                category_words = category.lower().split()
                if any(word in query_lower for word in category_words):
                    return product_type
    
    # Special cases for common terms not in categories
    if 'geyser' in query_lower or 'water heater' in query_lower:
        return 'Water Heater'
    elif 'fan' in query_lower or 'ceiling fan' in query_lower:
        return 'Ceiling Fan'
    elif 'ac' in query_lower or 'air conditioner' in query_lower:
        return 'Air Conditioner'
    elif 'fridge' in query_lower or 'refrigerator' in query_lower:
        return 'Refrigerator'
    elif 'washing machine' in query_lower or 'washer' in query_lower:
        return 'Washing Machine'
    
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
    is_best_query = any(term in query.lower() for term in ['best', 'recommend', 'suggest', 'top', 'ideal', 'perfect'])
    
    # Check if we have home configuration for personalization
    has_home_config = home_config is not None and 'home' in home_config
    
    # Create system prompt based on query type and home configuration
    if is_best_query:
        if has_home_config:
            system_prompt = """You are a helpful assistant for Better Home, a home improvement store. 
            When answering questions about the "best" product, provide a concise response suitable for WhatsApp:
            1. Keep your response under 300 words total
            2. Start with a brief acknowledgment that "best" depends on specific needs
            3. Make 1-2 personalized recommendations based on the user's home configuration
            4. Consider the user's location, property type, and floor level when relevant
            5. Include only essential details: key features, price range, and 1-2 main benefits
            6. Use bullet points for readability
            7. End with a very brief suggestion to consider specific needs
            8. For fan queries, explicitly mention considerations like room size and power consumption
            
            Format your response in simple markdown with minimal sections."""
            print("Using personalized best query system prompt with home configuration")
        else:
            system_prompt = """You are a helpful assistant for Better Home, a home improvement store. 
            When answering questions about the "best" product, provide a concise response suitable for WhatsApp:
            1. Keep your response under 300 words total
            2. Start with a brief acknowledgment that "best" depends on specific needs
            3. Make 1-2 recommendations based on general criteria
            4. Include only essential details: key features, price range, and 1-2 main benefits
            5. Use bullet points for readability
            6. End with a very brief suggestion to consider specific needs
            
            Format your response in simple markdown with minimal sections."""
            print("Using concise best query system prompt")
    else:
        if has_home_config:
            system_prompt = """You are a helpful assistant for Better Home, a home improvement store. 
            Answer questions about products based on the provided context and the user's home configuration. 
            Be extremely concise (under 150 words) and direct, focusing only on the most important information.
            Consider the user's location, property type, and floor level when relevant.
            Format your response in simple markdown with minimal formatting.
            Avoid unnecessary introductions or explanations."""
            print("Using personalized regular system prompt with home configuration")
        else:
            system_prompt = """You are a helpful assistant for Better Home, a home improvement store. 
            Answer questions about products based on the provided context. 
            Be extremely concise (under 150 words) and direct, focusing only on the most important information.
            Format your response in simple markdown with minimal formatting.
            Avoid unnecessary introductions or explanations."""
            print("Using concise regular system prompt")
    
    try:
        print(f"Processing query: {query}")

        # Create messages for the API call
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]
        
        # Make the API call with reduced max_tokens for more concise responses
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7,
            max_tokens=500  # Reduced from 1000 to encourage brevity
        )
        
        # Extract and return the answer
        answer = response.choices[0].message.content
        print(f"Generated answer: {answer}")
        
        # Filter the answer to include only products from the catalog
        catalog_titles = df['title'].str.lower().tolist()
        print(f"Catalog titles: {catalog_titles}")  # Debug: Print catalog titles
        filtered_answer = '\n'.join([line for line in answer.split('\n') if any(title in line.lower() for title in catalog_titles)])
        print(f"Filtered answer: {filtered_answer}")  # Debug: Print filtered answer
        
        return filtered_answer
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
    
    # Search with more results to account for potential invalid indices
    D, I = index.search(query_embedding, min(k*3, index.ntotal))
    
    # Filter out indices that would be out of bounds
    valid_indices = [idx for idx in I[0] if 0 <= idx < len(df)]
    
    # Return top k valid indices or all valid ones if fewer than k
    return valid_indices[:k]


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
    # Limit to 3 brands maximum for brevity
    products_df = products_df.head(3)
    
    # Create a concise title
    if is_warranty_query:
        response = f"### {product_type} Brands with Warranty:\n\n"
    else:
        response = f"### {product_type} Brands:\n\n"
    
    # Get total count
    brand_count = len(products_df)
    response += f"{brand_count} brand{'s' if brand_count > 1 else ''} available:\n\n"
    
    for _, product in products_df.iterrows():
        title = product['title']
        brand = product['Brand']
        price = product['Better Home Price']
        retail_price = product.get('Retail Price', 0)
        url = product.get('url', '#')
        
        # Calculate discount percentage if retail price is available
        if retail_price > 0:
            discount = ((retail_price - price) / retail_price) * 100
            discount_text = f"({discount:.1f}% off)"
        else:
            discount_text = ""
        
        # Highlight warranty information if present in the title
        warranty_text = ""
        if is_warranty_query:
            # Extract warranty info from title using regex
            warranty_match = re.search(r'with\s+(\d+)\s+years?\s+warranty', title, re.IGNORECASE)
            if warranty_match:
                warranty_years = warranty_match.group(1)
                warranty_text = f"⭐ {warranty_years} Year Warranty\n"
            elif 'warranty' in title.lower():
                warranty_text = "⭐ Includes Warranty\n"
            elif 'year' in title.lower() and re.search(r'(\d+)\s+years?', title, re.IGNORECASE):
                warranty_years = re.search(r'(\d+)\s+years?', title, re.IGNORECASE).group(1)
                warranty_text = f"⭐ {warranty_years} Year Guarantee\n"
        
        # More concise brand listing
        response += f"**{brand}** {discount_text}\n"
        response += f"₹{price:,.2f}\n"
        if warranty_text:
            response += warranty_text
        # Make the buy link more prominent
        response += f"🛒 [Buy Now]({url})\n\n"
    
    # Add a note about clicking the links
    response += "*Click on 'Buy Now' to purchase the product.*\n"
    
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
    # Limit to 3 products maximum for brevity
    products_df = products_df.head(3)
    
    response = "### Products:\n\n"
    
    # Add a special message for BLDC fans if they're in the results
    has_bldc = any(products_df['title'].str.contains('BLDC|Brushless', case=False, na=False))
    if has_bldc:
        response += "💚 **BLDC Fans** - 70% less electricity!\n\n"
    
    for _, product in products_df.iterrows():
        title = product['title']
        price = product['Better Home Price']
        retail_price = product.get('Retail Price', 0)
        url = product.get('url', '#')
        
        # Calculate discount percentage if retail price is available
        if retail_price > 0:
            discount = ((retail_price - price) / retail_price) * 100
            discount_text = f"({discount:.1f}% off)"
        else:
            discount_text = ""
        
        # Add a special highlight for BLDC fans
        is_bldc = 'BLDC' in title or 'Brushless' in title
        energy_label = "💚 " if is_bldc else ""
        
        # More concise product listing
        response += f"**{energy_label}{title}**\n"
        response += f"₹{price:,.2f} {discount_text}\n"
        
        # Add energy saving information for BLDC fans
        if is_bldc:
            response += f"70% energy savings\n"
        
        # Make the buy link more prominent
        response += f"🛒 [Buy Now]({url})\n\n"
    
    # Add a note about clicking the links
    response += "*Click on 'Buy Now' to purchase the product.*\n"
    
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
def search_relevant_blogs(query, blog_embeddings_dict, k=3, similarity_threshold=0.1):
    # Debug: Check if blog_embeddings_dict is valid
    if not blog_embeddings_dict:
        return []
    
    blog_embeddings = blog_embeddings_dict['blog_embeddings']
    query_lower = query.lower()
    
    # Extract product type from query
    product_type = None
    for pt in ['ceiling fan', 'fan', 'washing machine', 'chimney', 'refrigerator', 'air conditioner', 'microwave']:
        if pt in query_lower:
            product_type = pt
            break
    
    print(f"Detected product type in query: {product_type if product_type else 'None'}")
    
    # Preprocess blog contents to filter out metadata
    for metadata in blog_embeddings_dict['metadata']:
        if 'content' in metadata:
            # Filter out date patterns
            content = metadata['content']
            content = re.sub(r'\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}', '', content)
            # Filter out blog metadata terms
            content = re.sub(r'\b(?:recent post|post|blog|comment|author|published|updated)\b', '', content, flags=re.IGNORECASE)
            metadata['filtered_content'] = content
    
    # First, try to find exact matches for specific appliance types
    if any(appliance in query_lower for appliance in ['washing machine', 'chimney', 'refrigerator', 'air conditioner', 'microwave', 'ceiling fan', 'fan']):
        exact_matches = []
        search_appliance = product_type  # Use the detected product type
        
        # If no product type was detected, try again with more specific matching
        if not search_appliance:
            if 'washing machine' in query_lower:
                search_appliance = 'washing machine'
            elif 'chimney' in query_lower:
                search_appliance = 'chimney'
            elif 'refrigerator' in query_lower:
                search_appliance = 'refrigerator'
            elif 'air conditioner' in query_lower:
                search_appliance = 'air conditioner'
            elif 'microwave' in query_lower:
                search_appliance = 'microwave'
            elif 'ceiling fan' in query_lower:
                search_appliance = 'ceiling fan'
            elif 'fan' in query_lower:
                search_appliance = 'fan'
        
        for metadata in blog_embeddings_dict['metadata']:
            title = metadata.get('title', '').lower()
            content = metadata.get('filtered_content', '').lower()
            
            # Skip irrelevant articles
            if search_appliance and search_appliance not in title and search_appliance not in content:
                continue
                
            # Skip articles that don't match the appliance type but match other appliances
            if any(other_appliance in title 
                   for other_appliance in ['washing machine', 'chimney', 'refrigerator', 'air conditioner', 'microwave', 'ceiling fan', 'fan', 'roti', 'toaster', 'cooker', 'mixer', 'dishwasher'] 
                   if other_appliance != search_appliance):
                continue
                
            # Check for appliance type in title with relevant keywords
            if (search_appliance in title and 
                any(word in title + ' ' + content for word in ['choose', 'guide', 'buying', 'types', 'best'])):
                metadata['_similarity_score'] = 1.0  # Highest score for title matches
                exact_matches.append(metadata)
            # If no title match but appliance type is in content with relevant keywords
            elif (search_appliance in content and 
                  any(word in title + ' ' + content for word in ['choose', 'guide', 'buying', 'types', 'best'])):
                metadata['_similarity_score'] = 0.8  # Lower score for content matches
                exact_matches.append(metadata)
        
        # If we found exact matches, return them first
        if exact_matches:
            # Sort by similarity score
            exact_matches.sort(key=lambda x: x.get('_similarity_score', 0), reverse=True)
            return exact_matches[:k]
    
    # If no exact matches or not a specific appliance query, proceed with regular search
    try:
        # Enhance query with product type for more accurate embedding
        enhanced_query = query
        if product_type:
            # Emphasize product type in query for embedding
            enhanced_query = f"{product_type} {query} {product_type}"
            print(f"Enhanced query for embedding: {enhanced_query}")
            query_embedding = get_openai_embedding(enhanced_query)
        else:
            query_embedding = get_openai_embedding(query)
    except Exception as e:
        traceback.print_exc()
        return []
    
    # Debug: Check query embedding dimensions
    if query_embedding.shape[0] != blog_embeddings.shape[1]:
        return []
    
    # Build or load FAISS index for blog search
    try:
        blog_index = build_or_load_faiss_index(
            blog_embeddings,
            query_embedding.shape[0],
            BLOG_INDEX_FILE_PATH
        )
    except Exception as e:
        traceback.print_exc()
        return []
    
    # Only search if we have blog embeddings
    if len(blog_embeddings) > 0:
        # Search for more blog posts than needed so we can filter
        search_k = min(k * 2, len(blog_embeddings_dict['metadata']))
        try:
            query_embedding = query_embedding.reshape(1, -1).astype('float32')
            D, I = blog_index.search(query_embedding, search_k)
        except Exception as e:
            traceback.print_exc()
            return []
        
        # Get the metadata for found articles
        results = []
        for idx, (distance, i) in enumerate(zip(D[0], I[0])):
            if i < len(blog_embeddings_dict['metadata']):
                metadata = blog_embeddings_dict['metadata'][i]
                title = metadata.get('title', '').lower()
                content = metadata.get('filtered_content', '').lower()
                
                # Filter by product type if specified
                if product_type and not (product_type in title or product_type in content):
                    # For fan queries, also check for ceiling fan
                    if not (product_type == 'fan' and ('ceiling fan' in title or 'ceiling fan' in content)):
                        continue
                
                # Skip dishwasher articles when searching for washing machines
                if 'washing machine' in query_lower and 'dishwasher' in title:
                    continue
                
                # Skip kitchen appliance articles unless specifically asked for
                if not any(term in query_lower for term in ['kitchen', 'chimney', 'roti', 'toaster', 'cooker', 'mixer']):
                    if any(term in title.lower() for term in ['roti', 'toaster', 'cooker', 'mixer', 'kitchen']):
                        continue
                    
                similarity_score = 1.0 / (1.0 + distance)
                
                # Boost score if title matches query intent
                if any(word in title for word in query_lower.split()):
                    similarity_score *= 2.0
                
                # Boost score if product type is in title
                if product_type and product_type in title:
                    similarity_score *= 3.0
                
                # Only include if similarity score is above threshold
                if similarity_score > similarity_threshold:
                    metadata['_similarity_score'] = similarity_score
                    results.append(metadata)
        
            # Sort by similarity score
        if results:
            results.sort(key=lambda x: x.get('_similarity_score', 0), reverse=True)
            return results[:k]
    
        return []


def format_blog_response(blog_results, query=None):
    """
    Format blog search results for display in a concise, WhatsApp-friendly format
    """
    if not blog_results or len(blog_results) == 0:
        return None
    
    # Extract product type from query
    product_type = None
    query_lower = query.lower() if query else ""
    
    # Look for product types in query
    for pt in ['ceiling fan', 'fan', 'washing machine', 'chimney', 'refrigerator', 'air conditioner', 'microwave']:
        if pt in query_lower:
            product_type = pt
            break
    
    # Filter blog results to only include relevant product types
    filtered_results = []
    if product_type:
        print(f"Filtering blog results for product type: {product_type}")
        for blog in blog_results:
            title = blog.get('title', '').lower()
            # Use filtered_content if available, otherwise fallback to content
            content = blog.get('filtered_content', blog.get('content', '')).lower()
            
            # Include if product type is mentioned in title or content
            if product_type in title or product_type in content:
                filtered_results.append(blog)
            # For "fan" also match "ceiling fan"
            elif product_type == 'fan' and ('ceiling fan' in title or 'ceiling fan' in content):
                filtered_results.append(blog)
        
        # If we have filtered results, use them
        if filtered_results:
            blog_results = filtered_results
            print(f"Found {len(blog_results)} relevant articles for {product_type}")
        else:
            print(f"No relevant articles found for {product_type}")
    
    # Extract query topic for header
    topic_header = ""
    if product_type:
        topic_header = f" About {product_type.title()}"
    elif query:
        query_words = query_lower.split()
        stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'is', 'are', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 
                      'about', 'like', 'of', 'do', 'does', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 
                      'those', 'list', 'show', 'tell', 'me', 'get', 'can', 'could', 'would', 'should', 'how'}
        topic_words = [word for word in query_words if word not in stop_words and len(word) > 3]
        if topic_words:
            topic_header = f" About {' '.join(topic_words[:3]).title()}"
    
    response = f"### 📚 Articles{topic_header}\n\n"
    
    # Limit to 2 blog articles maximum
    blog_count = 0
    for blog in blog_results:
        if blog_count >= 2:
            break
            
        title = blog.get('title', '')
        url = blog.get('url', '#')
        # Use filtered_content if available, otherwise fallback to content
        content = blog.get('filtered_content', blog.get('content', ''))
        
        # Skip if no title or content
        if not title or not content:
            continue
        
        # Format the article entry
        response += f"**{title}**\n\n"
        
        # Extract relevant sections based on query
        key_points = extract_key_points(content, query)
        
        # If we have key points, show them
        if key_points:
            response += "**Key Points:**\n"
            for i, point in enumerate(key_points):
                response += f"• {point}\n"
            response += "\n"
        # Otherwise try to extract a summary
        else:
            summary = extract_summary(content, query)
            if summary:
                response += f"{summary[:250]}...\n\n"
            # Complete fallback for ceiling fan/fan queries when no points or summary could be extracted
            elif product_type in ['ceiling fan', 'fan'] and ('how to choose' in query_lower or 'how to buy' in query_lower):
                response += "**Key Factors for Choosing a Ceiling Fan:**\n"
                response += "• **Warranty**: Look for fans with at least 2-year warranty for peace of mind and reliability.\n"
                response += "• **Energy Efficiency**: BLDC motors can save up to 70% on electricity compared to regular fans.\n"
                response += "• **Air Delivery**: Higher air delivery (measured in CMM) means better cooling performance.\n"
                response += "• **Noise Level**: Lower RPM fans with balanced blades operate more quietly.\n"
                response += "• **Value**: Consider long-term benefits like energy savings rather than just upfront cost.\n\n"
        
        # Add link to full article
        response += f"📖 [Read Full Article]({url})\n\n"
        
        blog_count += 1
    
    # If nothing was found or extracted, provide a generic response for the product type
    if blog_count == 0 and product_type:
        response += f"I couldn't find specific articles about {product_type}s, but here are some general buying tips:\n\n"
        
        if product_type in ['ceiling fan', 'fan']:
            response += "**Key Factors for Choosing a Ceiling Fan:**\n"
            response += "• **Warranty**: Look for fans with at least 2-year warranty for peace of mind and reliability.\n"
            response += "• **Energy Efficiency**: BLDC motors can save up to 70% on electricity compared to regular fans.\n"
            response += "• **Air Delivery**: Higher air delivery (measured in CMM) means better cooling performance.\n"
            response += "• **Noise Level**: Lower RPM fans with balanced blades operate more quietly.\n"
            response += "• **Value**: Consider long-term benefits like energy savings rather than just upfront cost.\n\n"
        elif product_type == 'washing machine':
            response += "**Key Factors for Choosing a Washing Machine:**\n"
            response += "• **Capacity**: 6-7kg for a family of 3-4 people, 8-9kg for larger families.\n"
            response += "• **Type**: Front load (more efficient, gentler) vs top load (more convenient, less expensive).\n"
            response += "• **Energy Efficiency**: Look for star ratings to save on utility bills.\n"
            response += "• **Features**: Consider wash programs, quick wash, spin speed based on your needs.\n\n"
    
    # Add personalized product recommendations based on query type
    query_intent = ""
    if 'how to choose' in query_lower or 'how to buy' in query_lower or 'buying guide' in query_lower:
        query_intent = "buying"
    elif 'price' in query_lower or 'cost' in query_lower or 'budget' in query_lower:
        query_intent = "price"
    elif 'best' in query_lower or 'top' in query_lower or 'recommend' in query_lower:
        query_intent = "best"
    
    # Different prompts based on product type and query intent
    if product_type:
        if query_intent == "buying":
            response += f"Based on these buying tips, I can recommend some {product_type}s that match these criteria.\n"
        elif query_intent == "price":
            response += f"I can show you {product_type}s at different price points to help you find one within your budget.\n"
        elif query_intent == "best":
            response += f"I can show you some of our top-rated {product_type}s based on customer feedback.\n"
        else:
            response += f"Would you like to see some recommended {product_type}s available for purchase?\n"
    else:
        response += f"Would you like to see some recommended products based on this information?\n"
    
    # Common call-to-action
    response += "Just say 'yes' or let me know if you're looking for specific features."
    
    return response

def extract_key_points(content, query):
    """
    Extract key points from blog content based on the user's query.
    
    Parameters:
    - content: The full content of the blog article
    - query: The user's original query
    
    Returns:
    - List of key points extracted from the content
    """
    if not content or not query:
        return []
    
    # Clean up the content
    content = content.replace('\n', ' ').replace('\r', ' ')
    
    # Identify the type of query to extract relevant information
    query_lower = query.lower()
    
    # Identify product type in query
    product_type = None
    for pt in ['ceiling fan', 'fan', 'washing machine', 'chimney', 'refrigerator', 'air conditioner', 'microwave']:
        if pt in query_lower:
            product_type = pt
            break
    
    # Extract key points based on query type
    key_points = []
    
    # For "how to choose" or "buying guide" queries
    is_buying_guide = any(term in query_lower for term in [
        'how to choose', 'how to select', 'how to buy', 'buying guide', 
        'purchase guide', 'buying tips', 'selection guide', 'best', 'recommend'
    ])
    
    # Key criteria that are relevant for product selection (especially for ceiling fans)
    key_criteria = [
        'warranty', 'guarantee', 'energy', 'efficiency', 'power consumption', 'electricity',
        'value', 'price', 'cost', 'budget', 'quality', 'durability', 'life', 'lifespan',
        'features', 'performance', 'noise', 'silent', 'quiet', 'airflow', 'air delivery',
        'speed', 'rpm', 'bldc', 'brushless', 'star rating', 'expert', 'endorsed', 'recommended',
        'brand', 'size', 'design'
    ]
    
    # If it's a buying guide question
    if is_buying_guide and product_type:
        # Look for sentences containing key criteria related to the product
        for criterion in key_criteria:
            pattern = f"[^.!?]*{criterion}[^.!?]*[.!?]"
            matches = re.findall(pattern, content, re.IGNORECASE)
            
            for match in matches:
                # Filter out sentences that are likely to be metadata or dates
                if re.search(r'\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}', match):
                    continue
                    
                # Filter out sentences that contain "post" or "blog" which are likely metadata
                if re.search(r'\b(?:post|blog|comment|author|published|updated)\b', match.lower()):
                    continue
                
                # Skip extremely short sentences
                if len(match.strip()) < 30:
                    continue
                    
                # Skip sentences that don't mention the product type or aren't about selection criteria
                if product_type not in match.lower() and not any(word in match.lower() for word in ['choose', 'select', 'buy', 'consider', 'important', 'look for']):
                    continue
                
                # Add clean sentence to key points if it's not already included
                clean_match = match.strip()
                if clean_match not in key_points and len(clean_match) > 30 and len(clean_match) < 200:
                    key_points.append(clean_match)
        
        # Look for sections with headings like "Factors to consider", "What to look for", etc.
        buying_patterns = [
            r'(?:factors|things|points|what|features) to (?:consider|look for|check|know)',
            r'(?:key|important) (?:factors|considerations|features|aspects|points)',
            r'(?:buying|purchase) (?:guide|tips|considerations|advice)',
            r'how to (?:choose|select|buy|pick)',
            r'before (?:buying|purchasing)',
            r'(?:choosing|selecting) the (?:right|best|perfect)'
        ]
        
        # Look for these patterns in the content
        for pattern in buying_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                # Extract the paragraph or section containing this match
                start_pos = max(0, match.start() - 100)
                end_pos = min(len(content), match.end() + 500)
                section = content[start_pos:end_pos]
                
                # Extract bullet points or numbered lists
                points = re.findall(r'(?:•|\d+\.|\*)\s*(.*?)(?=(?:•|\d+\.|\*|$))', section)
                if points:
                    for p in points:
                        clean_point = p.strip()
                        # Filter out points that are likely to be metadata
                        if re.search(r'\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}', clean_point):
                            continue
                            
                        # Filter out points about posts or blogs
                        if re.search(r'\b(?:post|blog|comment|author|published|updated)\b', clean_point.lower()):
                            continue
                            
                        if len(clean_point) > 30 and len(clean_point) < 200 and clean_point not in key_points:
                            key_points.append(clean_point)
                else:
                    # If no bullet points, extract sentences with important keywords
                    sentences = re.split(r'(?<=[.!?])\s+', section)
                    important_words = ['consider', 'important', 'factor', 'key', 'check', 'ensure', 'look for', 'choose']
                    for sentence in sentences:
                        # Skip sentences with metadata
                        if re.search(r'\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}', sentence):
                            continue
                            
                        if re.search(r'\b(?:post|blog|comment|author|published|updated)\b', sentence.lower()):
                            continue
                            
                        if any(word in sentence.lower() for word in important_words):
                            clean_sentence = sentence.strip()
                            if len(clean_sentence) > 30 and len(clean_sentence) < 200 and clean_sentence not in key_points:
                                key_points.append(clean_sentence)
        
        # For ceiling fans, specifically look for sentences about the key criteria
        if product_type in ['ceiling fan', 'fan']:
            fan_factors = [
                'warranty', 'energy efficiency', 'electricity bill', 'power consumption',
                'bldc', 'brushless', 'noise', 'silent', 'quiet', 'air delivery', 'airflow',
                'rpm', 'speed', 'sweep size', 'blade size', 'remote', 'value', 'price'
            ]
            for factor in fan_factors:
                pattern = f"[^.!?]*{factor}[^.!?]*[.!?]"
                matches = re.findall(pattern, content, re.IGNORECASE)
                for match in matches:
                    # Filter out metadata and short sentences
                    if re.search(r'\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}', match):
                        continue
                        
                    if re.search(r'\b(?:post|blog|comment|author|published|updated)\b', match.lower()):
                        continue
                        
                    clean_match = match.strip()
                    if len(clean_match) > 30 and len(clean_match) < 200 and clean_match not in key_points:
                        key_points.append(clean_match)
    
    # If we still don't have enough key points, try extracting structured information
    if len(key_points) < 3 and product_type:
        # Look for sections with structured information (headings followed by content)
        heading_patterns = [
            r'(?:Warranty|Guarantee).*?[:.]\s*(.*?)(?=\n|$)',
            r'(?:Energy Efficiency|Power Consumption|Electricity).*?[:.]\s*(.*?)(?=\n|$)',
            r'(?:Value|Price|Cost|Budget).*?[:.]\s*(.*?)(?=\n|$)',
            r'(?:Expert|Professional|Industry).*?[:.]\s*(.*?)(?=\n|$)',
            r'(?:Features|Benefits|Advantages).*?[:.]\s*(.*?)(?=\n|$)'
        ]
        
        for pattern in heading_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):  # Handle groups in regex
                    match = match[0]
                clean_match = match.strip()
                if len(clean_match) > 30 and len(clean_match) < 200 and clean_match not in key_points:
                    key_points.append(clean_match)
    
    # Clean up and format key points
    cleaned_points = []
    for point in key_points:
        # Remove extra whitespace
        point = re.sub(r'\s+', ' ', point).strip()
        
        # Skip points that are too short or too long
        if len(point) < 30 or len(point) > 200:
            continue
        
        # Skip points that are just numbers or symbols
        if re.match(r'^[\d\s\.\-\*]+$', point):
            continue
            
        # Skip points that are likely dates or metadata
        if re.search(r'^\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}', point):
            continue
            
        # Skip points that mention "post", "blog", etc.
        if re.search(r'\b(?:post|blog|comment|author|published|updated)\b', point.lower()):
            continue
        
        # Skip duplicate points
        if point not in cleaned_points:
            cleaned_points.append(point)
    
    # If we still don't have enough key points, generate generic advice for the product type
    if len(cleaned_points) < 2 and product_type:
        if product_type == 'ceiling fan' or product_type == 'fan':
            cleaned_points = [
                "Look for energy-efficient ceiling fans with BLDC motors that can save up to 70% on electricity bills.",
                "Consider the warranty period - quality ceiling fans typically come with 2-5 year warranties.",
                "Check the air delivery (measured in CMM) which indicates how effectively the fan circulates air.",
                "Consider the sweep size (blade diameter) - larger rooms need fans with 1200mm or more sweep.",
                "Look for fans with low noise levels, especially for bedrooms and study areas."
            ]
        elif product_type == 'washing machine':
            cleaned_points = [
                "Consider the capacity based on your family size - typically 6-7kg for a family of four.",
                "Choose between front load (more efficient, gentler) and top load (more convenient, less expensive).",
                "Check for energy and water efficiency ratings to save on utility bills.",
                "Look for key features like multiple wash programs, quick wash option, and spin speed."
            ]
    
    # Limit to 5 key points maximum
    return cleaned_points[:5]

def extract_summary(content, query):
    """
    Extract a summary from the blog content when key points cannot be extracted.
    
    Parameters:
    - content: The full content of the blog article
    - query: The user's original query
    
    Returns:
    - A summary of the content or None if no summary can be extracted
    """
    if not content or not query:
        return None
    
    # Clean up the content
    content = content.replace('\n', ' ').replace('\r', ' ')
    
    # Identify product type in query
    product_type = None
    query_lower = query.lower()
    for pt in ['ceiling fan', 'fan', 'washing machine', 'chimney', 'refrigerator', 'air conditioner', 'microwave']:
        if pt in query_lower:
            product_type = pt
            break
    
    # Try to find paragraphs containing query keywords
    if product_type:
        # First look for paragraphs with both the product type and query intent
        intent_words = ['choose', 'select', 'buy', 'types', 'best', 'guide', 'how', 'what']
        intent_matches = []
        
        for word in intent_words:
            if word in query_lower:
                # Try to find paragraphs containing both product type and this intent word
                pattern = f"[^.!?]*{product_type}[^.!?]*{word}[^.!?]*[.!?]|[^.!?]*{word}[^.!?]*{product_type}[^.!?]*[.!?]"
                matches = re.findall(pattern, content, re.IGNORECASE)
                
                if matches:
                    for match in matches:
                        if len(match.strip()) > 50 and len(match.strip()) < 500:
                            intent_matches.append(match.strip())
        
        if intent_matches:
            # Return the first match or concatenate multiple short ones
            if len(intent_matches[0]) > 150:
                return intent_matches[0]
            else:
                combined = " ".join(intent_matches[:3])
                return combined if len(combined) < 500 else combined[:500]
        
        # If no intent matches, just look for paragraphs with the product type
        product_paragraphs = []
        paragraphs = re.split(r'\n\n|\r\n\r\n', content)
        
        for paragraph in paragraphs:
            if product_type in paragraph.lower():
                clean_paragraph = paragraph.strip()
                if len(clean_paragraph) > 50 and len(clean_paragraph) < 500:
                    product_paragraphs.append(clean_paragraph)
        
        if product_paragraphs:
            # Return the first good paragraph
            return product_paragraphs[0]
    
    # Try to find a summary section
    summary_match = re.search(r'(?:Summary|Conclusion|Overview).*?(?=\n\n|\Z)', content, re.IGNORECASE | re.DOTALL)
    if summary_match:
        summary = summary_match.group(0)
        # Clean up the summary
        summary = re.sub(r'(?:Summary|Conclusion|Overview):\s*', '', summary, flags=re.IGNORECASE)
        summary = summary.strip()
        if len(summary) > 50 and len(summary) < 500:
            return summary
    
    # If no summary section found, try to extract the introduction or first paragraph
    intro_match = re.search(r'(?:Introduction|Overview).*?(?=\n\n|\Z)', content, re.IGNORECASE | re.DOTALL)
    if intro_match:
        intro = intro_match.group(0)
        intro = re.sub(r'(?:Introduction|Overview):\s*', '', intro, flags=re.IGNORECASE)
        intro = intro.strip()
        if len(intro) > 50 and len(intro) < 500:
            return intro
            
    # If still nothing, extract the first substantive paragraph (at least 50 characters)
    paragraphs = re.split(r'\n\n', content)
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if len(paragraph) > 50 and len(paragraph) < 500:
            return paragraph
    
    return None

def get_fan_recommendations_by_user_profile(query: str, df: pd.DataFrame, home_config: Dict[str, Any]) -> str:
    """Get personalized fan recommendations based on user profile information."""
    if not home_config or 'home' not in home_config or 'rooms' not in home_config['home']:
        return None
        
    # Filter for ceiling fans
    fans_df = df[df['Product Type'] == 'Ceiling Fan'].copy()
    if fans_df.empty:
        return None
        
    # Check if query is about BLDC fans
    is_bldc_query = 'bldc' in query.lower() or 'brushless' in query.lower() or 'energy efficient' in query.lower()
    
    # Initialize recommendations with a concise header
    recommendations = []
    recommendations.append("## Fan Recommendations\n")
    
    # Get home information for context
    home_info = home_config.get('home', {})
    city = home_info.get('city', '')
    property_type = home_info.get('property_type', '')
    floor_level = home_info.get('floor_level', 0)
    total_floors = home_info.get('total_floors', 0)
    
    # Add home context if available
    if city:
        recommendations.append(f"Based on your location in {city} and your home setup:")
    
    # Process each room - limit to 3 rooms maximum for brevity
    room_count = 0
    for room in home_config['home']['rooms']:
        if room_count >= 3:  # Limit to 3 rooms maximum
            break
            
        room_name = room.get('name', 'Unknown Room')
        room_color = room.get('room_color', '')
        used_by = room.get('used_by', '')
        
        recommendations.append(f"### {room_name}")
        
        # Add room context
        if room_color:
            recommendations.append(f"Room color: {room_color}")
        if used_by:
            recommendations.append(f"Used by: {used_by}")
        
        # Add recommendations based on user type
        if 'elderly' in used_by.lower():
            recommendations.append("- Quiet operation with remote control")
        if 'child' in used_by.lower():
            recommendations.append("- Child-safe features with secure mounting")
        if 'adult' in used_by.lower():
            recommendations.append("- Good airflow and energy efficiency")
            
        # BLDC recommendation if applicable
        if is_bldc_query:
            recommendations.append("- BLDC fans: 70% energy savings, quieter operation")
        
        room_count += 1
    
    # Add very brief summary
    recommendations.append("\n### Summary")
    recommendations.append("- Consider room size and user needs")
    recommendations.append("- BLDC fans offer energy savings and quiet operation")
    
    # Add floor level considerations if applicable
    if floor_level and total_floors:
        recommendations.append(f"- Being on floor {floor_level} of {total_floors}: Consider noise levels and ventilation needs")
        
    return "\n".join(recommendations)


def match_user_profile_with_products(user_profile, product_embeddings, product_metadata, k=3):
    """
    Match a user profile with products for personalized recommendations.
    
    Args:
        user_profile: Dictionary containing user profile information
        product_embeddings: List of product embeddings
        product_metadata: List of product metadata dictionaries
        k: Number of top matches to return
        
    Returns:
        DataFrame containing the top k matching products
    """
    # Create a user profile embedding
    profile_entry = (
        f"Age Group: {user_profile.get('age_group', 'Not Available')}. "
        f"Room Type: {user_profile.get('room_type', 'Not Available')}. "
        f"Preferences: {', '.join(user_profile.get('preferences', []))}. "
        f"Budget: {user_profile.get('budget', 'Not Available')}."
    )
    
    # Generate embedding for the user profile
    client = Ollama()
    try:
        response = client.embed(model=MODEL_NAME, input=profile_entry)
        if isinstance(response, dict) and 'embeddings' in response:
            profile_embedding = response['embeddings'][0]
        else:
            print("Failed to generate user profile embedding")
            return None
    except Exception as e:
        print(f"Error generating user profile embedding: {str(e)}")
        return None
    
    # Calculate similarity scores
    similarity_scores = []
    for i, product_embedding in enumerate(product_embeddings):
        # Calculate cosine similarity
        similarity = np.dot(profile_embedding, product_embedding) / (
            np.linalg.norm(profile_embedding) * np.linalg.norm(product_embedding)
        )
        similarity_scores.append((i, similarity))
    
    # Sort by similarity score (descending)
    similarity_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Get top k matches
    top_k_indices = [idx for idx, _ in similarity_scores[:k]]
    
    # Create a DataFrame with the top k products
    top_products = []
    for idx in top_k_indices:
        if idx < len(product_metadata):
            product = product_metadata[idx]
            product['similarity_score'] = similarity_scores[top_k_indices.index(idx)][1]
            top_products.append(product)
    
    return pd.DataFrame(top_products)

def get_personalized_recommendations(query, df, user_profile=None):
    """
    Get personalized product recommendations based on user profile.
    
    Args:
        query: User query
        df: Product DataFrame
        user_profile: Dictionary containing user profile information
        
    Returns:
        DataFrame containing personalized recommendations
    """
    # If no user profile is provided, return regular search results
    if not user_profile:
        return None
    
    # Load product embeddings
    try:
        with open(EMBEDDINGS_FILE_PATH, 'r') as f:
            embeddings_dict = json.load(f)
        
        product_embeddings = embeddings_dict.get('product_embeddings', [])
        product_metadata = []
        
        # Create product metadata
        for _, row in df.iterrows():
            metadata = {
                'title': row.get('title', 'Not Available'),
                'Product Type': row.get('Product Type', 'Not Available'),
                'Brand': row.get('Brand', 'Not Available'),
                'Better Home Price': row.get('Better Home Price', 'Not Available'),
                'url': row.get('url', '#'),
                'Description': row.get('Description', 'Not Available')
            }
            product_metadata.append(metadata)
        
        # Match user profile with products
        recommendations = match_user_profile_with_products(
            user_profile, 
            product_embeddings, 
            product_metadata
        )
        
        if recommendations is not None and not recommendations.empty:
            print(f"Found {len(recommendations)} personalized recommendations")
            return recommendations
        else:
            print("No personalized recommendations found")
            return None
            
    except Exception as e:
        print(f"Error getting personalized recommendations: {str(e)}")
        return None

# Function to find related products using the graph
def find_related_products(product_title, graph, max_related=5):
    if not graph:
        return []
    
    # Find the node in the graph
    product_node = None
    for node, data in graph.nodes(data=True):
        if data.get('title', '').lower() == product_title.lower():
            product_node = node
            break
    
    if not product_node:
        return []
    
    # Get related products
    related_products = list(graph.neighbors(product_node))[:max_related]
    return related_products

# Example usage in a recommendation function
def recommend_products_based_on_graph(query, df, graph):
    # Find related products using the graph
    related_products = find_related_products(query, graph)
    
    # Filter the DataFrame for these products
    recommended_df = df[df['title'].isin(related_products)]
    
    # Format the response
    response = format_product_response(recommended_df)
    return response

# Function to get recommendations using the knowledge graph
def get_recommendations_from_graph(product_title, knowledge_graph, df, max_recommendations=5):
    """
    Get product recommendations based on the knowledge graph
    """
    if knowledge_graph is None:
        return None
    
    # Find the product node in the graph based on title
    product_node = None
    for node, data in knowledge_graph.nodes(data=True):
        if data.get('type') == 'product' and data.get('title', '').lower() == product_title.lower():
            product_node = node
            break
    
    if not product_node:
        return None
    
    product_id = product_node.replace('product_', '')
    
    # Get product category
    category_nodes = [n for u, n in knowledge_graph.out_edges(product_node) 
                    if knowledge_graph.nodes[n].get('type') == 'category']
    
    # Get product features
    feature_nodes = [n for u, n in knowledge_graph.out_edges(product_node) 
                    if knowledge_graph.nodes[n].get('type') == 'feature']
    
    # Get product brand
    brand_nodes = [n for u, n in knowledge_graph.out_edges(product_node) 
                  if knowledge_graph.nodes[n].get('type') == 'brand']
    
    # Get price range
    price_nodes = [n for u, n in knowledge_graph.out_edges(product_node) 
                  if knowledge_graph.nodes[n].get('type') == 'price_range']
    
    # Find similar products based on shared category, features, brand, and price range
    similarity_scores = defaultdict(float)
    
    # Get all product nodes
    product_nodes = [n for n in knowledge_graph.nodes() if knowledge_graph.nodes[n].get('type') == 'product' and n != product_node]
    
    for other_product in product_nodes:
        # Check if same category
        other_categories = [n for u, n in knowledge_graph.out_edges(other_product) 
                          if knowledge_graph.nodes[n].get('type') == 'category']
        
        category_match = len(set(category_nodes).intersection(set(other_categories)))
        if category_match > 0:
            similarity_scores[other_product] += 1.0  # Base similarity for same category
            
            # Check feature overlap
            other_features = [n for u, n in knowledge_graph.out_edges(other_product) 
                            if knowledge_graph.nodes[n].get('type') == 'feature']
            
            feature_overlap = len(set(feature_nodes).intersection(set(other_features)))
            similarity_scores[other_product] += 0.2 * feature_overlap  # Add score for each shared feature
            
            # Check if same brand
            other_brands = [n for u, n in knowledge_graph.out_edges(other_product) 
                          if knowledge_graph.nodes[n].get('type') == 'brand']
            
            brand_match = len(set(brand_nodes).intersection(set(other_brands)))
            similarity_scores[other_product] += 0.5 * brand_match  # Add score for brand match
            
            # Check if same price range
            other_price = [n for u, n in knowledge_graph.out_edges(other_product) 
                          if knowledge_graph.nodes[n].get('type') == 'price_range']
            
            price_match = len(set(price_nodes).intersection(set(other_price)))
            similarity_scores[other_product] += 0.3 * price_match  # Add score for price range match
    
    # Sort products by similarity score
    sorted_products = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Get top recommendations
    recommendations = []
    for product_node, score in sorted_products[:max_recommendations]:
        product_title = knowledge_graph.nodes[product_node].get('title', 'Unknown Product')
        product_id = product_node.replace('product_', '')
        
        # Find this product in the dataframe
        product_row = None
        for idx, row in df.iterrows():
            if row.get('title', '') == product_title:
                product_row = row
                break
        
        if product_row is not None:
            recommendations.append(product_row)
    
    if recommendations:
        return pd.DataFrame(recommendations)
    
    return None

# Integrate the knowledge graph into the search_products function
def search_products_with_graph(query, df, embeddings_dict, knowledge_graph=None, k=5):
    """
    Search for products using embeddings and enhance with knowledge graph
    """
    # First search products using existing embedding-based method
    product_indices = search_products(query, df, embeddings_dict, k)
    
    # If no knowledge graph, return the embedding-based results
    if knowledge_graph is None:
        return product_indices
    
    # Get the first product from embedding-based search
    if len(product_indices) > 0:
        first_product_idx = product_indices[0]
        first_product = df.iloc[first_product_idx]
        first_product_title = first_product.get('title', '')
        
        # Get recommendations from knowledge graph
        graph_recommendations = get_recommendations_from_graph(
            first_product_title, 
            knowledge_graph, 
            df, 
            max_recommendations=k
        )
        
        if graph_recommendations is not None:
            # Convert DataFrame back to list of indices
            graph_indices = []
            for idx, row in graph_recommendations.iterrows():
                original_idx = df[df['title'] == row['title']].index
                if len(original_idx) > 0:
                    graph_indices.append(original_idx[0])
            
            # Combine embedding-based and graph-based recommendations
            combined_indices = []
            # Start with first product from embedding search
            combined_indices.append(product_indices[0])
            
            # Add graph-based recommendations not already included
            for idx in graph_indices:
                if idx not in combined_indices:
                    combined_indices.append(idx)
            
            # Add remaining embedding-based recommendations not already included
            for idx in product_indices[1:]:
                if idx not in combined_indices:
                    combined_indices.append(idx)
            
            # Limit to k recommendations
            return combined_indices[:k]
    
    # If no recommendations from graph, return original embedding-based results
    return product_indices

# ==========================
# Main Function
# ==========================
def main():
    st.title("Better Home Assistant")
    
    # Initialize session state
    if 'conversation_history' not in st.session_state:
        st.session_state['conversation_history'] = []
    
    # Add a session state variable to track the last query product type
    if 'last_product_type' not in st.session_state:
        st.session_state['last_product_type'] = None
    
    # Add a session state variable to track if we're awaiting a response to view products
    if 'awaiting_product_view' not in st.session_state:
        st.session_state['awaiting_product_view'] = False
    
    # Build or load FAISS index for product search
    try:
        # Rebuild FAISS index to match current DataFrame
        new_index = faiss.IndexFlatL2(embedding_data['product_embeddings'].shape[1])
        new_index.add(embedding_data['product_embeddings'][:len(df)])
        faiss.write_index(new_index, PRODUCT_INDEX_FILE_PATH)
    except Exception as e:
        st.error(f"Error rebuilding product index: {str(e)}")
        index = None
    
    # User input
    user_input = st.text_input("Ask me anything about home products:", key="user_input")
    
    if user_input:
        query = user_input.lower().strip()
        
        try:
            # Check if this is a response to view products
            if st.session_state['awaiting_product_view']:
                if any(term in query for term in ['yes', 'show', 'sure', 'products', 'view', 'see', 'okay', 'ok']):
                    # User wants to see product recommendations based on the last query
                    product_type = st.session_state['last_product_type']
                    
                    if product_type:
                        # Filter dataframe for the specific product type
                        filtered_df = df[df['Product Type'].astype(str) == product_type]
                        
                        if len(filtered_df) > 0:
                            # Get top products
                            top_products = filtered_df.sort_values('Better Home Price', ascending=True).head(5)
                            response = format_product_response(top_products)
                            st.write(response)
                            
                            # Reset the awaiting_product_view flag
                            st.session_state['awaiting_product_view'] = False
                            return
                    
                    # If we couldn't find products or product_type is None
                    # Fall back to regular search
                    st.session_state['awaiting_product_view'] = False
                else:
                    # User asked something else, reset the flag
                    st.session_state['awaiting_product_view'] = False
            
            # Check if it's a how-to question
            if is_how_to_query(query):
                # Load blog embeddings
                try:
                    print(f"Loading blog embeddings from {BLOG_EMBEDDINGS_FILE_PATH}")
                    blog_embeddings_dict = load_blog_embeddings(BLOG_EMBEDDINGS_FILE_PATH)
                    if blog_embeddings_dict and blog_embeddings_dict['blog_embeddings'].shape[0] > 0:
                        print(f"Successfully loaded {blog_embeddings_dict['blog_embeddings'].shape[0]} blog embeddings")
                        # Search for relevant blogs
                        blog_results = search_relevant_blogs(query, blog_embeddings_dict, k=3)
                        if blog_results:
                            print(f"Found {len(blog_results)} relevant blog articles")
                            response = format_blog_response(blog_results, query)
                            st.write(response)
                            
                            # Save the product type for follow-up
                            for pt in ['ceiling fan', 'fan', 'washing machine', 'chimney', 'refrigerator', 'air conditioner', 'microwave']:
                                if pt in query.lower():
                                    st.session_state['last_product_type'] = pt
                                    break
                            
                            # Set flag to indicate we're awaiting a response to view products
                            st.session_state['awaiting_product_view'] = True
                            return
                        else:
                            print("No relevant blog articles found")
                            response = "I couldn't find any articles that directly answer your how-to question. Please try rephrasing your question or ask about a specific product."
                            st.write(response)
                            return
                    else:
                        print("Failed to load blog embeddings or no embeddings found")
                        response = "I'm having trouble accessing our knowledge base right now. Please try again later or ask about a specific product."
                        st.write(response)
                        return
                except Exception as e:
                    print(f"Error searching blogs: {str(e)}")
                    traceback.print_exc()
                    response = "I encountered an error while searching for information. Please try again later or ask about a specific product."
                    st.write(response)
                    return

            # Handle bestseller queries
            bestseller_results = handle_bestseller_query(query, df)
            if bestseller_results is not None:
                response = format_product_response(bestseller_results)
                st.write(response)
                # Append to conversation history without displaying
                st.session_state['conversation_history'].append(("user", user_input))
                st.session_state['conversation_history'].append(("assistant", response))
                return

            # Handle price queries
            if any(word in query for word in ['price', 'cost', 'expensive', 'cheap', 'budget']):
                products = handle_price_query(query, df, product_terms)
                if products is not None:
                    response = format_product_response(products)
                    st.write(response)
                    return

            # Handle brand queries
            if any(word in query for word in ['brand', 'warranty', 'company', 'manufacturer']):
                brand_result = handle_brand_query(query, df, product_terms)
                if brand_result is not None:
                    response = format_brand_response(brand_result['dataframe'], brand_result['product_type'], brand_result['is_warranty_query'])
                    st.write(response)
                    return

            # Handle general "best" product queries
            if any(term in query for term in ['best', 'recommend', 'suggest', 'top', 'ideal', 'perfect']):
                response = retrieve_and_generate_openai(query, "")
                st.write(response)
                return
                
            # Handle similar product queries with knowledge graph
            if any(term in query for term in ['similar', 'related', 'like']) and knowledge_graph is not None:
                # Try to extract a product name from the query
                product_words = query.replace('similar to', '').replace('related to', '')
                product_words = product_words.replace('like', '').strip()
                
                # Try to find a match in our catalog
                potential_matches = []
                for _, row in df.iterrows():
                    title = str(row.get('title', '')).lower()
                    match_score = 0
                    for word in product_words.split():
                        if len(word) > 3 and word in title:  # Only consider words longer than 3 chars
                            match_score += 1
                    
                    if match_score > 0:
                        potential_matches.append((row, match_score))
                
                # Sort by match score
                potential_matches.sort(key=lambda x: x[1], reverse=True)
                
                if potential_matches:
                    # Get the best matching product
                    best_match = potential_matches[0][0]
                    product_title = best_match.get('title', '')
                    
                    print(f"Found potential product match: {product_title}")
                    
                    # Get recommendations from knowledge graph
                    graph_recommendations = get_recommendations_from_graph(
                        product_title, 
                        knowledge_graph, 
                        df, 
                        max_recommendations=5
                    )
                    
                    if graph_recommendations is not None:
                        print(f"Found {len(graph_recommendations)} recommendations from knowledge graph")
                        response = format_product_response(graph_recommendations)
                        st.write(response)
                        
                        # Append to conversation history without displaying
                        st.session_state['conversation_history'].append(("user", user_input))
                        st.session_state['conversation_history'].append(("assistant", response))
                        return

            # Handle general product search
            if index is not None:
                # First try to use the knowledge graph if appropriate
                if knowledge_graph is not None and any(term in query for term in ['recommend', 'similar', 'related', 'like']):
                    indices = search_products_with_graph(query, df, embedding_data, knowledge_graph)
                    products = [df.iloc[idx] for idx in indices]
                else:
                    products = search_catalog(query, df, index)
                
                response = format_answer(products, query)
                st.write(response)
                if products:  # Check if the list is not empty
                    # Convert the list of dictionaries to a DataFrame for display
                    products_df = pd.DataFrame(products)
                    st.dataframe(products_df)
                # Append to conversation history without displaying
                st.session_state['conversation_history'].append(("user", user_input))
                st.session_state['conversation_history'].append(("assistant", response))
        except Exception as e:
            st.error(f"I apologize, but I encountered an error: {str(e)}")
            # Append error to conversation history without displaying
            st.session_state['conversation_history'].append(("user", user_input))
            st.session_state['conversation_history'].append(("assistant", f"I apologize, but I encountered an error: {str(e)}"))

def handle_bestseller_query(query, df):
    """
    Handle bestseller queries and return bestselling products.
    
    Args:
        query: User query
        df: DataFrame containing product data
        
    Returns:
        DataFrame containing bestselling products or None if not a bestseller query
    """
    # Check if it's a bestseller-related query
    query_lower = query.lower()
    
    is_bestseller_query = any(term in query_lower for term in [
        'bestseller', 'best seller', 'best selling', 'most popular', 'popular', 'top selling',
        'trending', 'most sold', 'most purchased', 'best performing'
    ])
    
    if not is_bestseller_query:
        return None
    
    # Try to find if there's a specific product type in the query
    product_type = None
    
    # Common product type keywords 
    product_types = {
        'fan': 'Ceiling Fan',
        'water heater': 'Water Heater',
        'geyser': 'Water Heater',
        'ac': 'Air Conditioner',
        'air conditioner': 'Air Conditioner',
        'refrigerator': 'Refrigerator',
        'fridge': 'Refrigerator',
        'washing machine': 'Washing Machine',
        'washer': 'Washing Machine',
        'chimney': 'Chimney'
    }
    
    # Check if query contains any specific product type
    for keyword, pt in product_types.items():
        if keyword in query_lower:
            product_type = pt
            break
    
    try:
        # Check if dataframe has is_bestseller column
        if 'is_bestseller' not in df.columns:
            print("No is_bestseller column found in dataframe")
            return None
            
        # Filter for bestseller products
        bestsellers = df[df['is_bestseller'] == True]
        
        if len(bestsellers) == 0:
            # Try case-insensitive boolean matching
            bestsellers = df[df['is_bestseller'].astype(str).str.lower() == 'true']
            
        # If no bestsellers found, try matching on string "true"
        if len(bestsellers) == 0:
            bestsellers = df[df['is_bestseller'] == "true"]
        
        print(f"Found {len(bestsellers)} bestseller products")
        
        # Further filter by product type if specified
        if product_type:
            filtered_bestsellers = bestsellers[bestsellers['Product Type'].astype(str) == product_type]
            if len(filtered_bestsellers) > 0:
                bestsellers = filtered_bestsellers
                print(f"Filtered to {len(bestsellers)} {product_type} bestsellers")
        
        # If still empty, return None
        if len(bestsellers) == 0:
            return None
            
        # Return the top bestselling products
        return bestsellers.head(5)
    
    except Exception as e:
        print(f"Error in handle_bestseller_query: {str(e)}")
        return None

if __name__ == "__main__":
    main()
