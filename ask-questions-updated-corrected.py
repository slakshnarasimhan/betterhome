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
            
        # Check categories
        for category in info['categories']:
            if category.lower() in query_lower:
                return product_type
                
    # If no exact match found, try partial matches
    for product_type, info in product_terms.items():
        # Check if any word in the product type is in the query
        product_type_words = product_type.lower().split()
        if any(word in query_lower for word in product_type_words):
            return product_type
            
        # Check categories
        for category in info['categories']:
            category_words = category.lower().split()
            if any(word in query_lower for word in category_words):
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
        print(f"Generated answer length: {len(answer)}")
        
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
def search_relevant_blogs(query, blog_embeddings_dict, k=3, similarity_threshold=0.2):
    # Debug: Check if blog_embeddings_dict is valid
    if not blog_embeddings_dict:
        return []
    
    blog_embeddings = blog_embeddings_dict['blog_embeddings']
    
    # Generate query embedding using OpenAI for consistency
    try:
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
            query_embedding.shape[0],  # Use the correct dimension of the query embedding
            BLOG_INDEX_FILE_PATH
        )
    except Exception as e:
        traceback.print_exc()
        return []
    
    # Only search if we have blog embeddings
    if len(blog_embeddings) > 0:
        # Search for more blog posts than needed so we can filter
        search_k = min(k * 15, len(blog_embeddings_dict['metadata']))  # Increased from k * 10 to k * 15
        # Add detailed exception logging around the search process
        try:
            query_embedding = query_embedding.reshape(1, -1).astype('float32')
            D, I = blog_index.search(query_embedding, search_k)
        except Exception as e:
            traceback.print_exc()
            return []
        
        # Get the metadata for found articles
        results = []
        query_lower = query.lower()
        is_chimney_query = "chimney" in query_lower
        
        # First pass: collect all potential matches with their scores
        potential_matches = []
        for idx, (distance, i) in enumerate(zip(D[0], I[0])):
            if i < len(blog_embeddings_dict['metadata']):
                metadata = blog_embeddings_dict['metadata'][i]
                
                # Check if the article is chimney-related
                is_chimney_related = 'chimney' in metadata.get('title', '').lower() or 'chimney' in metadata.get('content', '').lower()
                
                # Calculate base similarity score
                similarity_score = 1.0 / (1.0 + distance)  # Convert distance to similarity (0-1 scale)
                
                # Check if query matches related words or categories
                matches_related_words = any(word.lower() in query_lower for word in metadata.get('related_words', []))
                matches_categories = any(tag.lower() in query_lower for tag in metadata.get('categories', []))
                
                # For chimney queries, check if the title or content contains "chimney"
                matches_chimney = False
                if is_chimney_query:
                    matches_chimney = is_chimney_related
                    
                    # Boost score for articles with "chimney" in title or content
                    if matches_chimney:
                        if "chimney" in metadata.get('title', '').lower():
                            similarity_score *= 2.0  # Double the score for title matches
                        else:
                            similarity_score *= 1.5  # 1.5x score for content matches
                
                # Only include if similarity score is above threshold or matches criteria
                if similarity_score > similarity_threshold or matches_related_words or matches_categories or matches_chimney:
                    metadata['_similarity_score'] = similarity_score
                    potential_matches.append(metadata)
        except Exception as e:
            traceback.print_exc()
            return []
        
        # Sort by similarity score and limit to k results
        if potential_matches:
            # Sort by similarity score
            potential_matches.sort(key=lambda x: x.get('_similarity_score', 0), reverse=True)
            
            # For chimney queries, ensure chimney-related articles appear first
            if is_chimney_query:
                chimney_articles = [article for article in potential_matches 
                                  if "chimney" in article.get('title', '').lower() 
                                  or "chimney" in article.get('content', '').lower()]
                other_articles = [article for article in potential_matches 
                                if article not in chimney_articles]
                results = chimney_articles + other_articles
            else:
                results = potential_matches
            
            # Limit to k results
            results = results[:k]
            return results
        else:
            return []
    else:
        return []


def format_blog_response(blog_results, query=None):
    """
    Format blog search results for display in a concise, WhatsApp-friendly format
    
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
    
    # Concise header
    response = f"### 📚 Articles{topic_header}\n\n"
    
    # Limit to 2 blog articles maximum for brevity
    blog_count = 0
    for blog in blog_results:
        if blog_count >= 2:
            break
            
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
        
        # Create a very brief excerpt if content is available
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
                
                # Truncate if too long - make it shorter for WhatsApp
                if len(excerpt) > 100:
                    excerpt = excerpt[:100] + "..."
            else:
                # No good match, use beginning of content
                excerpt = content[:100] + "..." if len(content) > 100 else content
        elif content:
            # No query provided, just use the beginning
            excerpt = content[:100] + "..." if len(content) > 100 else content
        
        # Format the article entry with minimal formatting
        response += f"**{title}**\n"
        
        # Add very brief excerpt if available
        if excerpt:
            response += f"{excerpt}\n"
        
        # Add a more prominent link
        response += f"📖 [Read Article]({url})\n\n"
        
        blog_count += 1
    
    # Add a note about clicking the links
    response += "*Click on 'Read Article' to view the full article.*\n"
    
    return response


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

# ==========================
# Main Function
# ==========================
def main():
    st.title("Better Home Assistant")
    
    # Initialize session state
    if 'conversation_history' not in st.session_state:
        st.session_state['conversation_history'] = []
    
    # Build or load FAISS index for product search
    try:
        index = build_or_load_faiss_index(
            embedding_data['product_embeddings'],
            embedding_data['product_embeddings'].shape[1] if len(embedding_data['product_embeddings']) > 0 else 1536,
            PRODUCT_INDEX_FILE_PATH
        )
    except Exception as e:
        st.error(f"Error building product index: {str(e)}")
        index = None
    
    # User input
    user_input = st.text_input("Ask me anything about home products:", key="user_input")
    
    if user_input:
        query = user_input.lower().strip()
        
        try:
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
                            st.session_state['conversation_history'].append(("user", user_input))
                            st.session_state['conversation_history'].append(("assistant", response))
                            return
                        else:
                            print("No relevant blog articles found")
                            response = "I couldn't find any articles that directly answer your how-to question. Please try rephrasing your question or ask about a specific product."
                            st.write(response)
                            st.session_state['conversation_history'].append(("user", user_input))
                            st.session_state['conversation_history'].append(("assistant", response))
                            return
                    else:
                        print("Failed to load blog embeddings or no embeddings found")
                        response = "I'm having trouble accessing our knowledge base right now. Please try again later or ask about a specific product."
                        st.write(response)
                        st.session_state['conversation_history'].append(("user", user_input))
                        st.session_state['conversation_history'].append(("assistant", response))
                        return
                except Exception as e:
                    print(f"Error searching blogs: {str(e)}")
                    traceback.print_exc()
                    response = "I encountered an error while searching for information. Please try again later or ask about a specific product."
                    st.write(response)
                    st.session_state['conversation_history'].append(("user", user_input))
                    st.session_state['conversation_history'].append(("assistant", response))
                    return

            # Temporarily disable product search
            # Handle price queries
            if any(word in query for word in ['price', 'cost', 'expensive', 'cheap', 'budget']):
                products = handle_price_query(query, df, product_terms)
                if products is not None:
                    response = format_product_response(products)
                    st.write(response)
                    st.session_state['conversation_history'].append(("user", user_input))
                    st.session_state['conversation_history'].append(("assistant", response))
                    return

            # Handle brand queries
            if any(word in query for word in ['brand', 'warranty', 'company', 'manufacturer']):
                brand_result = handle_brand_query(query, df, product_terms)
                if brand_result is not None:
                    response = format_brand_response(brand_result['dataframe'], brand_result['product_type'], brand_result['is_warranty_query'])
                    st.write(response)
                    st.session_state['conversation_history'].append(("user", user_input))
                    st.session_state['conversation_history'].append(("assistant", response))
                    return

            # Handle general "best" product queries
            if any(term in query for term in ['best', 'recommend', 'suggest', 'top', 'ideal', 'perfect']):
                response = retrieve_and_generate_openai(query, "")
                st.write(response)
                st.session_state['conversation_history'].append(("user", user_input))
                st.session_state['conversation_history'].append(("assistant", response))
                return

            # Handle general product search
            if index is not None:
                products = search_catalog(query, df, index)
                response = format_answer(products, query)
                st.write(response)
                if products:  # Check if the list is not empty
                    # Convert the list of dictionaries to a DataFrame for display
                    products_df = pd.DataFrame(products)
                    st.dataframe(products_df)
                st.session_state['conversation_history'].append(("user", user_input))
                st.session_state['conversation_history'].append(("assistant", response))
            
        except Exception as e:
            st.error(f"I apologize, but I encountered an error: {str(e)}")
            st.session_state['conversation_history'].append(("user", user_input))
            st.session_state['conversation_history'].append(("assistant", f"I apologize, but I encountered an error: {str(e)}"))
    
    # Display conversation history
    if st.session_state['conversation_history']:
        st.subheader("Conversation History")
        for role, message in st.session_state['conversation_history']:
            if role == "user":
                st.write(f"You: {message}")
            else:
                st.write(f"Assistant: {message}")

if __name__ == "__main__":
    main()
