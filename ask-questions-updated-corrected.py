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
import time
import random
import threading

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
    product_graph = nx.read_gpickle('product_graph.gpickle')
    print("Successfully loaded product graph")
except Exception as e:
    print(f"Error loading product graph: {str(e)}")
    product_graph = None

# Load the knowledge graph if available
try:
    knowledge_graph = nx.read_gpickle('product_knowledge_graph.gpickle')
    print("Successfully loaded product knowledge graph")
except Exception as e:
    print(f"Error loading product knowledge graph: {str(e)}")
    knowledge_graph = None

# ==========================
# Animation Utilities
# ==========================
def animate_typing(text, container, speed=0.03, variance=0.01):
    """
    Display text with a typing animation effect
    
    Args:
        text: The text to display
        container: The Streamlit container to update
        speed: Base delay between characters in seconds
        variance: Random variance to add to the typing speed
    """
    displayed_text = ""
    # Split by markdown formatting so we don't break it
    segments = []
    current_segment = ""
    in_formatting = False
    
    for char in text:
        if char == '*' or char == '#' or char == '[' or char == '`' or char == '_':
            in_formatting = True
            current_segment += char
        elif in_formatting and (char == ' ' or char == '\n' or char == ']'):
            in_formatting = False
            current_segment += char
            segments.append(current_segment)
            current_segment = ""
        else:
            current_segment += char
            
    if current_segment:
        segments.append(current_segment)
    
    # Display text with typing animation effect
    for segment in segments:
        # Add segment all at once if it's a formatting element
        if segment.startswith('*') or segment.startswith('#') or segment.startswith('[') or segment.startswith('`') or segment.startswith('_'):
            displayed_text += segment
            container.markdown(displayed_text)
            time.sleep(speed)
        else:
            # Type out regular text character by character
            for char in segment:
                displayed_text += char
                container.markdown(displayed_text)
                
                # Add natural variation to typing speed
                delay = speed + random.uniform(-variance, variance)
                if char in '.!?':  # Pause a bit longer after end of sentences
                    delay *= 3
                elif char == ',':  # Slight pause after commas
                    delay *= 2
                elif char == '\n':  # Pause after line breaks
                    delay *= 1.5
                    
                time.sleep(max(0.01, delay))  # Ensure delay is at least 10ms

def show_thinking_animation(container, duration=0.5):
    """
    Display a thinking animation while processing
    
    Args:
        container: The Streamlit container to update
        duration: Time between animation frames in seconds
    """
    thinking_dots = ["â ‹", "â ™", "â ¹", "â ¸", "â ¼", "â ´", "â ¦", "â §", "â ‡", "â �"]
    thinking_messages = [
        "Searching for relevant products",
        "Analyzing your question",
        "Retrieving information",
        "Finding the best matches",
        "Checking product details",
        "Preparing your response"
    ]
    
    # Pick a random message
    message = random.choice(thinking_messages)
    
    # Show animation until stopped externally
    i = 0
    while True:
        container.markdown(f"**{message}** {thinking_dots[i % len(thinking_dots)]}")
        time.sleep(duration)
        i += 1
        
        # Check if we should stop
        if not getattr(st.session_state, 'thinking', True):
            break

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
    
    if is_warranty_query:
        response = f"### Warranty Information for {product_type.title()} Brands:\n\n"
    else:
        response = f"### Top {product_type.title()} Brands:\n\n"
    
    for brand_name, group in products_df.groupby('Brand'):
        response += f"**{brand_name}**\n"
        
        # Warranty info
        warranty_info = group.iloc[0].get('Warranty', 'Not specified')
        if warranty_info and warranty_info != 'Not specified' and is_warranty_query:
            response += f"- Warranty: {warranty_info}\n"
        
        # Sample products with price range
        min_price = group['Better Home Price'].min()
        max_price = group['Better Home Price'].max()
        
        if min_price == max_price:
            response += f"- Price: â‚¹{min_price:,.2f}\n"
        else:
            response += f"- Price Range: â‚¹{min_price:,.2f} - â‚¹{max_price:,.2f}\n"
        
        # Key features or benefits (if available)
        features = group.iloc[0].get('Key Features', '').split(', ')
        if features and features[0]:  # Check if there are actual features
            response += "- Key Features: "
            response += ", ".join(features[:3])  # Limit to 3 features
            response += "\n"
        
        # Sample product
        sample_product = group.iloc[0]
        sample_url = sample_product.get('url', '#')
        response += f"- [Browse {brand_name} Products]({sample_url})\n\n"
    
    # Add follow-up prompt
    response += "\n**Would you like to:**\n"
    response += f"- Learn more about a specific {product_type} brand?\n"
    response += f"- Compare prices across {product_type} brands?\n"
    response += f"- See the top-rated {product_type} products?\n"
    
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
        response += "ğŸ’š **BLDC Fans** - 70% less electricity!\n\n"
    
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
        energy_label = "ğŸ’š " if is_bldc else ""
        
        # More concise product listing
        response += f"**{energy_label}{title}**\n"
        response += f"â‚¹{price:,.2f} {discount_text}\n"
        
        # Add energy saving information for BLDC fans
        if is_bldc:
            response += f"70% energy savings\n"
        
        # Make the buy link more prominent
        response += f"ğŸ›’ [Buy Now]({url})\n\n"
    
    # Add a note about clicking the links
    response += "*Click on 'Buy Now' to purchase the product.*\n"
    
    # Add follow-up prompt
    response += "\n**Would you like to:**\n"
    response += "- See more details about these products?\n"
    response += "- Compare features across these products?\n"
    response += "- Find similar alternatives?\n"
    
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
    
    # First, try to find exact matches for specific appliance types
    if any(appliance in query_lower for appliance in ['washing machine', 'chimney', 'refrigerator', 'air conditioner', 'microwave']):
        exact_matches = []
        search_appliance = None
        
        # Determine which appliance we're searching for
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
        
        for metadata in blog_embeddings_dict['metadata']:
            title = metadata.get('title', '').lower()
            content = metadata.get('content', '').lower()
            
            # Skip irrelevant articles
            if any(irrelevant in title for irrelevant in ['roti', 'toaster', 'cooker', 'mixer', 'dishwasher']) and search_appliance != 'dishwasher':
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
    
    # If no exact matches or not a washing machine query, proceed with regular search
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
                
                # Skip dishwasher articles when searching for washing machines
                if 'washing machine' in query_lower and 'dishwasher' in title:
                    continue
                    
                similarity_score = 1.0 / (1.0 + distance)
                
                # Boost score if title matches query intent
                if any(word in title for word in query_lower.split()):
                    similarity_score *= 2.0
                
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
    response = "### Article Recommendations:\n\n"
    
    for result in blog_results:
        # Extract metadata
        title = result.get('title', 'No Title')
        url = result.get('url', '#')
        content = result.get('content', '')
        
        # Add article title and link
        response += f"#### [{title}]({url})\n\n"
        
        # Extract important information from content based on query
        if query and content:
            important_info = extract_key_points(content, query)
            response += f"{important_info}\n\n"
    
    # Add follow-up prompt
    response += "\n**Would you like to:**\n"
    response += "- Get more details on these articles?\n"
    response += "- Ask a more specific question about this topic?\n"
    response += "- See related products instead?\n"
    
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
    
    # Extract key points based on query type
    key_points = []
    
    # For "how to choose" queries
    if 'how to choose' in query_lower or 'how to select' in query_lower or 'how to buy' in query_lower:
        # Look for sections with headings like "Factors to consider", "What to look for", etc.
        factor_sections = re.findall(r'(?:Factors to consider|What to look for|Key factors|Important considerations|Tips for choosing|How to choose|Buying guide|Selection criteria|Summary).*?(?=\n\n|\Z)', content, re.IGNORECASE | re.DOTALL)
        
        if factor_sections:
            for section in factor_sections:
                # Extract bullet points or numbered lists
                points = re.findall(r'(?:â€¢|\d+\.)\s*(.*?)(?=\n|$)', section)
                if points:
                    key_points.extend(points)
            else:
                    # If no bullet points, extract sentences
                    sentences = re.split(r'(?<=[.!?])\s+', section)
                    key_points.extend([s.strip() for s in sentences if len(s.strip()) > 20 and len(s.strip()) < 200])
        
        # For washing machine specific queries, look for comparison sections
        if 'washing machine' in query_lower:
            comparison_sections = re.findall(r'(?:Fully automatic vs Semi-automatic|Top load vs front load|Comparison).*?(?=\n\n|\Z)', content, re.IGNORECASE | re.DOTALL)
            
            if comparison_sections:
                for section in comparison_sections:
                    # Extract comparison points
                    points = re.findall(r'(?:â€¢|\d+\.)\s*(.*?)(?=\n|$)', section)
                    if points:
                        key_points.extend(points)
                    else:
                        # If no bullet points, extract sentences with comparison keywords
                        sentences = re.split(r'(?<=[.!?])\s+', section)
                        comparison_sentences = [s.strip() for s in sentences if any(word in s.lower() for word in ['better', 'more', 'less', 'versus', 'vs', 'compared', 'difference'])]
                        key_points.extend(comparison_sentences)
    
    # For "types of" queries
    elif 'types of' in query_lower or 'kinds of' in query_lower or 'varieties of' in query_lower:
        # Look for sections about types or categories
        type_sections = re.findall(r'(?:Types of|Categories of|Different types|Varieties of|Fully automatic vs Semi-automatic|Top load vs front load).*?(?=\n\n|\Z)', content, re.IGNORECASE | re.DOTALL)
        
        if type_sections:
            for section in type_sections:
                # Extract bullet points or numbered lists
                points = re.findall(r'(?:â€¢|\d+\.)\s*(.*?)(?=\n|$)', section)
                if points:
                    key_points.extend(points)
                else:
                    # If no bullet points, extract sentences
                    sentences = re.split(r'(?<=[.!?])\s+', section)
                    key_points.extend([s.strip() for s in sentences if len(s.strip()) > 20 and len(s.strip()) < 200])
    
    # For "best" queries
    elif 'best' in query_lower or 'recommend' in query_lower or 'top' in query_lower:
        # Look for recommendations or conclusions
        recommendation_sections = re.findall(r'(?:Recommendation|Conclusion|Summary|Best choice|Top pick).*?(?=\n\n|\Z)', content, re.IGNORECASE | re.DOTALL)
        
        if recommendation_sections:
            for section in recommendation_sections:
                # Extract bullet points or numbered lists
                points = re.findall(r'(?:â€¢|\d+\.)\s*(.*?)(?=\n|$)', section)
                if points:
                    key_points.extend(points)
                else:
                    # If no bullet points, extract sentences
                    sentences = re.split(r'(?<=[.!?])\s+', section)
                    key_points.extend([s.strip() for s in sentences if len(s.strip()) > 20 and len(s.strip()) < 200])
        
        # For washing machine specific queries, look for pros and cons
        if 'washing machine' in query_lower:
            pros_cons_sections = re.findall(r'(?:Pros and cons|Advantages and disadvantages|Benefits and drawbacks).*?(?=\n\n|\Z)', content, re.IGNORECASE | re.DOTALL)
            
            if pros_cons_sections:
                for section in pros_cons_sections:
                    # Extract pros and cons points
                    points = re.findall(r'(?:â€¢|\d+\.)\s*(.*?)(?=\n|$)', section)
                    if points:
                        key_points.extend(points)
    
    # For general queries, extract the introduction and any summary sections
    else:
        # Extract introduction
        intro_match = re.search(r'(?:Introduction|Overview).*?(?=\n\n|\Z)', content, re.IGNORECASE | re.DOTALL)
        if intro_match:
            intro = intro_match.group(0)
            sentences = re.split(r'(?<=[.!?])\s+', intro)
            key_points.extend([s.strip() for s in sentences if len(s.strip()) > 20 and len(s.strip()) < 200])
        
        # Extract summary
        summary_match = re.search(r'(?:Summary|Conclusion).*?(?=\n\n|\Z)', content, re.IGNORECASE | re.DOTALL)
        if summary_match:
            summary = summary_match.group(0)
            sentences = re.split(r'(?<=[.!?])\s+', summary)
            key_points.extend([s.strip() for s in sentences if len(s.strip()) > 20 and len(s.strip()) < 200])
    
    # If we still don't have enough key points, extract general information
    if len(key_points) < 3:
        # Look for bullet points or numbered lists throughout the content
        points = re.findall(r'(?:â€¢|\d+\.)\s*(.*?)(?=\n|$)', content)
        if points:
            key_points.extend(points)
        
        # If still not enough, extract sentences with important keywords
        if len(key_points) < 3:
            # Extract product type from query
            product_type = None
            if product_terms:
                product_type = find_product_type(query, product_terms)
            
            if product_type:
                # Look for sentences containing the product type
                product_sentences = re.findall(f'[^.!?]*{product_type}[^.!?]*[.!?]', content, re.IGNORECASE)
                key_points.extend([s.strip() for s in product_sentences if len(s.strip()) > 20 and len(s.strip()) < 200])
    
    # Clean up and format key points
    cleaned_points = []
    for point in key_points:
        # Remove extra whitespace
        point = re.sub(r'\s+', ' ', point).strip()
        
        # Skip points that are too short or too long
        if len(point) < 20 or len(point) > 200:
            continue
        
        # Skip points that are just numbers or symbols
        if re.match(r'^[\d\s\.\-\*]+$', point):
            continue
        
        # Skip duplicate points
        if point not in cleaned_points:
            cleaned_points.append(point)
    
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
    
    # Try to find a summary section
    summary_match = re.search(r'(?:Summary|Conclusion|Overview).*?(?=\n\n|\Z)', content, re.IGNORECASE | re.DOTALL)
    if summary_match:
        summary = summary_match.group(0)
        # Clean up the summary
        summary = re.sub(r'(?:Summary|Conclusion|Overview):\s*', '', summary, flags=re.IGNORECASE)
        summary = summary.strip()
        if len(summary) > 50 and len(summary) < 500:
            return summary
    
    # If no summary section found, try to extract the first paragraph
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

def generate_concise_description(row):
    """
    Generate a concise 2-3 line description for a product.
    If description is NaN, create a meaningful description from available data.
    """
    description = row.get('Description', '')
    
    # If description is NaN or empty, create one from available data
    if pd.isna(description) or not description.strip():
        product_type = row.get('Product Type', '')
        brand = row.get('Brand', '')
        features = row.get('Features', '')
        
        # Create a meaningful description from available data
        description_parts = []
        
        # Add product type and brand
        if product_type and brand:
            description_parts.append(f"{brand} {product_type}")
        
        # Add key features if available
        if features and isinstance(features, str):
            # Take first 2-3 key features
            features_list = [f.strip() for f in features.split(',') if f.strip()]
            if features_list:
                description_parts.append(f"Key features: {', '.join(features_list[:3])}")
        
        # Add capacity if available
        capacity = row.get('Capacity', '')
        if capacity:
            description_parts.append(f"Capacity: {capacity}")
        
        description = '. '.join(description_parts)
    
    # If we still have a description, make it concise (2-3 sentences)
    if description:
        # Split into sentences
        sentences = [s.strip() for s in description.split('.') if s.strip()]
        if len(sentences) > 3:
            # Take first 3 sentences and join them
            description = '. '.join(sentences[:3]) + '.'
    
    return description

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
                'Description': generate_concise_description(row)
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
    # Set page title and configuration
    st.title("Better Home Assistant")
    
    # Initialize session state variables if they don't exist
    if 'conversation_history' not in st.session_state:
        st.session_state['conversation_history'] = []
    if 'follow_up_state' not in st.session_state:
        st.session_state['follow_up_state'] = False
    if 'current_context' not in st.session_state:
        st.session_state['current_context'] = None
    if 'last_products' not in st.session_state:
        st.session_state['last_products'] = None
    if 'product_type' not in st.session_state:
        st.session_state['product_type'] = None
    if 'thinking' not in st.session_state:
        st.session_state['thinking'] = False
    
    # Create containers for dynamic content
    if 'response_container' not in st.session_state:
        st.session_state.response_container = st.container()
    if 'follow_up_container' not in st.session_state:
        st.session_state.follow_up_container = st.container()
    if 'thinking_container' not in st.session_state:
        st.session_state.thinking_container = st.container()
    
    # Reset button in sidebar
    with st.sidebar:
        st.title("Options")
        if st.button("Clear History"):
            # Reset all conversation state
            st.session_state['conversation_history'] = []
            st.session_state['follow_up_state'] = False
            st.session_state['last_products'] = None
            st.session_state['current_context'] = None
            st.session_state['product_type'] = None
            st.session_state['thinking'] = False
            st.session_state.response_container.empty()
            st.session_state.follow_up_container.empty() 
            st.session_state.thinking_container.empty()
            st.success("Conversation history cleared!")
            time.sleep(1)  # Show success message briefly
            st.rerun()
    
    # Load product terms for identifying product types
    product_terms = load_product_terms(PRODUCT_TERMS_FILE_PATH)
    
    # Show conversation history
    for speaker, text in st.session_state['conversation_history']:
        if speaker == "user":
            st.chat_message("user").write(text)
        else:
            st.chat_message("assistant").write(text)
    
    # Load embeddings and create index
    try:
        embedding_data = load_embeddings(EMBEDDINGS_FILE_PATH)
        df = load_product_catalog(PRODUCT_CATALOG_FILE_PATH)
        
        # Build or load the FAISS index
        if embedding_data['product_embeddings'].shape[0] > 0:
            dimension = embedding_data['product_embeddings'].shape[1]
            index = build_or_load_faiss_index(embedding_data['product_embeddings'], dimension, INDEX_PATH)
            print(f"Successfully loaded FAISS index with {embedding_data['metadata'].get('total_products', 0)} products")
        else:
            print("No embeddings found, cannot build index")
            index = None
    except Exception as e:
        print(f"Error loading embeddings or building index: {str(e)}")
        traceback.print_exc()
        embedding_data = None
        df = None
        index = None
    
    # Handle follow-up queries
    if st.session_state['follow_up_state']:
        product_type = st.session_state.get('product_type', '')
        
        st.markdown(f"### I'm currently showing information about {product_type or 'products'}. What would you like to know?")
        
        # Display follow-up options based on context
        if st.session_state['current_context'] == 'product':
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("More details"):
                    process_follow_up("Tell me more about these products", df, index, "product_details")
            with col2:
                if st.button("Compare products"):
                    process_follow_up("Compare these products", df, index, "product_comparison")
            with col3:
                if st.button("Similar products"):
                    process_follow_up("Show me similar products", df, index, "product_alternatives")
        
        elif st.session_state['current_context'] == 'brand':
            product_type = st.session_state.get('product_type', '')
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Compare features"):
                    process_follow_up(f"Compare features across {product_type} brands", df, index, "brand_features")
            with col2:
                if st.button("Compare prices"):
                    process_follow_up(f"Compare prices across {product_type} brands", df, index, "brand_price_comparison")
            with col3:
                if st.button("Top-rated products"):
                    process_follow_up(f"Show me top-rated {product_type} products", df, index, "top_rated_products")
        
        # User input - only show if not in follow-up mode or explicitly reset
        if not st.session_state['follow_up_state']:
            user_input = st.text_input("Ask me anything about home products:", key="user_input")
            
            if user_input:
                process_query(user_input, df, index)

def process_follow_up(follow_up_query, df, index, follow_up_type):
    """Handle follow-up queries based on previous context"""
    # Get the last products context
    last_products = st.session_state['last_products']
    context_type = st.session_state['current_context']
    
    # Safety check - if last_products is None, respond with a message
    if last_products is None:
        # Add follow-up to conversation history
        st.session_state['conversation_history'].append(("user", follow_up_query))
        
        response = "I'm sorry, but I don't have any product information to follow up on. Please ask a new question."
        st.session_state['conversation_history'].append(("assistant", response))
        
        # Show response with typing animation
        animate_typing(response, st.session_state.follow_up_container)
        
        # Reset follow-up state to allow for new input
        st.session_state['follow_up_state'] = False
        
        st.rerun()
        return
    
    # Add follow-up to conversation history
    st.session_state['conversation_history'].append(("user", follow_up_query))
    
    # Start thinking animation
    st.session_state.thinking = True
    thinking_thread = threading.Thread(target=show_thinking_animation, args=(st.session_state.thinking_container, 0.3))
    thinking_thread.daemon = True
    thinking_thread.start()
    
    response = ""
    
    # Handle different follow-up types
    if follow_up_type == "product_details":
        # Show more detailed information about the last products
        if isinstance(last_products, pd.DataFrame) and not last_products.empty:
            response = "### Detailed Product Information:\n\n"
            for _, product in last_products.iterrows():
                title = product['title']
                specs = product.get('Specifications', 'No detailed specifications available')
                features = product.get('Key Features', 'No features listed')
                
                response += f"**{title}**\n"
                response += f"- **Key Features:** {features}\n"
                response += f"- **Specifications:** {specs}\n\n"
        else:
            response = "I don't have detailed product information available. Please try searching for products first."
    
    elif follow_up_type == "product_comparison":
        # Compare features across the last shown products
        if isinstance(last_products, pd.DataFrame) and not last_products.empty and len(last_products) >= 2:
            response = "### Product Comparison:\n\n"
            comparison_table = "| Feature | " + " | ".join([p['title'] for _, p in last_products.iterrows()]) + " |\n"
            comparison_table += "| --- | " + " | ".join(["---" for _ in range(len(last_products))]) + " |\n"
            
            # Compare key features
            comparison_table += "| Price | " + " | ".join([f"â‚¹{p['Better Home Price']:,.2f}" for _, p in last_products.iterrows()]) + " |\n"
            
            # Add other common features if available
            for feature in ['Energy Rating', 'Warranty', 'Brand']:
                if any(feature in p for _, p in last_products.iterrows()):
                    comparison_table += f"| {feature} | " + " | ".join([str(p.get(feature, 'N/A')) for _, p in last_products.iterrows()]) + " |\n"
            
            response += comparison_table
        else:
            response = "I need at least two products to compare. Please search for more products first."
    
    elif follow_up_type == "product_alternatives":
        # Find similar products based on the last shown products
        if isinstance(last_products, pd.DataFrame) and not last_products.empty:
            try:
                # Take the first product title and find alternatives
                first_product = last_products.iloc[0]
                product_title = first_product['title']
                
                # Get recommendations from knowledge graph if available
                if knowledge_graph is not None:
                    alternatives = get_recommendations_from_graph(product_title, knowledge_graph, df, max_recommendations=3)
                    if alternatives is not None and len(alternatives) > 0:
                        response = "### Similar Products You Might Like:\n\n"
                        response += format_product_response(alternatives)
                        # Update the last products to these alternatives
                        st.session_state['last_products'] = alternatives
                        st.session_state['follow_up_state'] = True
                        st.session_state['current_context'] = 'product'
                        
                        # Update conversation history
                        st.session_state['conversation_history'].append(("assistant", response))
                        
                        # Stop thinking animation
                        st.session_state.thinking = False
                        st.session_state.thinking_container.empty()
                        
                        # Show response with typing animation
                        animate_typing(response, st.session_state.follow_up_container)
                        
                        st.rerun()
                        return
                    else:
                        response = "I couldn't find similar products. Would you like to see other recommended products instead?"
                else:
                    response = "I'm sorry, but I don't have the knowledge graph available to find similar products."
            except Exception as e:
                print(f"Error finding alternatives: {str(e)}")
                response = "I had trouble finding alternatives for this product. Please try a different query."
        else:
            response = "I don't have any products to find alternatives for. Please search for products first."
    
    # Format response based on follow-up type
    if not response:
        response = "I'm sorry, I couldn't find the information you requested. Please try asking a different question."
    
    # Add response to conversation history
    st.session_state['conversation_history'].append(("assistant", response))
    
    # Stop thinking animation
    st.session_state.thinking = False
    st.session_state.thinking_container.empty()
    
    # Show response with typing animation
    animate_typing(response, st.session_state.follow_up_container)
    
    # Keep follow-up state active
    st.session_state['follow_up_state'] = True
    
    # Rerun to update the UI
    st.rerun()


def process_query(user_input, df, index):
    """Process a new user query"""
    query = user_input.lower().strip()
    
    # Reset follow-up state for new queries
    st.session_state['follow_up_state'] = False
    
    # Add user input to conversation history
    st.session_state['conversation_history'].append(("user", user_input))
    
    # Start thinking animation
    st.session_state.thinking = True
    thinking_thread = threading.Thread(target=show_thinking_animation, args=(st.session_state.thinking_container, 0.3))
    thinking_thread.daemon = True
    thinking_thread.start()
    
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
                        
                        # Save context and enable follow-up state
                        st.session_state['last_products'] = blog_results
                        st.session_state['current_context'] = 'blog'
                        st.session_state['follow_up_state'] = True
                        
                        # Update conversation history
                        st.session_state['conversation_history'].append(("assistant", response))
                        
                        # Stop thinking animation
                        st.session_state.thinking = False
                        st.session_state.thinking_container.empty()
                        
                        # Show response with typing animation
                        animate_typing(response, st.session_state.response_container)
                        
                        st.rerun()
                        return
                    else:
                        print("No relevant blog articles found")
                        response = "I couldn't find any articles that directly answer your how-to question. Please try rephrasing your question or ask about a specific product."
                        st.session_state['conversation_history'].append(("assistant", response))
                        
                        # Stop thinking animation
                        st.session_state.thinking = False
                        st.session_state.thinking_container.empty()
                        
                        # Show response with typing animation
                        animate_typing(response, st.session_state.response_container)
                        
                        st.rerun()
                        return
                else:
                    print("Failed to load blog embeddings or no embeddings found")
                    response = "I'm having trouble accessing our knowledge base right now. Please try again later or ask about a specific product."
                    st.session_state['conversation_history'].append(("assistant", response))
                    
                    # Stop thinking animation
                    st.session_state.thinking = False
                    st.session_state.thinking_container.empty()
                    
                    # Show response with typing animation
                    animate_typing(response, st.session_state.response_container)
                    
                    st.rerun()
                    return
            except Exception as e:
                print(f"Error searching blogs: {str(e)}")
                traceback.print_exc()
                response = "I encountered an error while searching for information. Please try again later or ask about a specific product."
                st.session_state['conversation_history'].append(("assistant", response))
                
                # Stop thinking animation
                st.session_state.thinking = False
                st.session_state.thinking_container.empty()
                
                # Show response with typing animation
                animate_typing(response, st.session_state.response_container)
                
                st.rerun()
                return

        # Handle price queries
        if any(word in query for word in ['price', 'cost', 'expensive', 'cheap', 'budget']):
            products = handle_price_query(query, df, product_terms)
            if products is not None:
                response = format_product_response(products)
                
                # Save context and enable follow-up state
                st.session_state['last_products'] = products
                st.session_state['current_context'] = 'product'
                st.session_state['follow_up_state'] = True
                
                # Update conversation history
                st.session_state['conversation_history'].append(("assistant", response))
                
                # Stop thinking animation
                st.session_state.thinking = False
                st.session_state.thinking_container.empty()
                
                # Show response with typing animation
                animate_typing(response, st.session_state.response_container)
                
                st.rerun()
                return

        # Handle brand queries
        if any(word in query for word in ['brand', 'warranty', 'company', 'manufacturer']):
            brand_result = handle_brand_query(query, df, product_terms)
            if brand_result is not None:
                response = format_brand_response(brand_result['dataframe'], brand_result['product_type'], brand_result['is_warranty_query'])
                
                # Save context and enable follow-up state
                st.session_state['last_products'] = brand_result['dataframe']
                st.session_state['current_context'] = 'brand'
                st.session_state['product_type'] = brand_result['product_type']
                st.session_state['follow_up_state'] = True
                
                # Update conversation history
                st.session_state['conversation_history'].append(("assistant", response))
                
                # Stop thinking animation
                st.session_state.thinking = False
                st.session_state.thinking_container.empty()
                
                # Show response with typing animation
                animate_typing(response, st.session_state.response_container)
                
                st.rerun()
                return

        # Handle general "best" product queries
        if any(term in query for term in ['best', 'recommend', 'suggest', 'top', 'ideal', 'perfect']):
            # If we have a valid index
            if index is not None:
                # Try to find the product type
                product_type = find_product_type(query, product_terms)
                
                if knowledge_graph is not None:
                    # Get recommendations from knowledge graph
                    recommendations = recommend_products_based_on_graph(query, df, knowledge_graph)
                    if recommendations is not None and not recommendations.empty:
                        response = "### Based on what others are buying:\n\n"
                        response += format_product_response(recommendations)
                        
                        # Save context
                        st.session_state['last_products'] = recommendations
                        st.session_state['current_context'] = 'product'
                        st.session_state['follow_up_state'] = True
                        
                        # Update conversation history
                        st.session_state['conversation_history'].append(("assistant", response))
                        
                        # Stop thinking animation
                        st.session_state.thinking = False
                        st.session_state.thinking_container.empty()
                        
                        # Show response with typing animation
                        animate_typing(response, st.session_state.response_container)
                        
                        st.rerun()
                        return
                
                # If graph recommendations failed, do regular search
                result = search_catalog(query, df, embedding_data, index, k=5, product_type=product_type)
                if result and not result.empty:
                    products_df = result.head(3)  # Limit to top 3
                    response = "### Top Recommended Products:\n\n"
                    response += format_product_response(products_df)
                    
                    # Save context
                    st.session_state['last_products'] = products_df
                    st.session_state['current_context'] = 'product'
                    st.session_state['follow_up_state'] = True
                    
                    # Update conversation history
                    st.session_state['conversation_history'].append(("assistant", response))
                    
                    # Stop thinking animation
                    st.session_state.thinking = False
                    st.session_state.thinking_container.empty()
                    
                    # Show response with typing animation
                    animate_typing(response, st.session_state.response_container)
                    
                    st.rerun()
                    return
                else:
                    # Fallback response
                    response = "I couldn't find specific recommendations for that query. Could you please be more specific about what type of product you're looking for?"
                    
                    # Update conversation history
                    st.session_state['conversation_history'].append(("assistant", response))
                    
                    # Stop thinking animation
                    st.session_state.thinking = False
                    st.session_state.thinking_container.empty()
                    
                    # Show response with typing animation
                    animate_typing(response, st.session_state.response_container)
                    
                    st.rerun()
                    return
        
        # Otherwise, do a general search
        try:
            # First see if we can identify the product type
            product_type = find_product_type(query, product_terms)
            
            # If index is valid, search the catalog
            if index is not None:
                result = search_catalog(query, df, embedding_data, index, k=5, product_type=product_type)
                if result is not None and not result.empty:
                    products_df = result.head(3)  # Limit to top 3
                    response = "### Here's what I found:\n\n"
                    response += format_product_response(products_df)
                    
                    # Set context for follow up
                    st.session_state['last_products'] = products_df
                    st.session_state['current_context'] = 'product'
                    st.session_state['follow_up_state'] = True
                    
                    # Update conversation history
                    st.session_state['conversation_history'].append(("assistant", response))
                    
                    # Stop thinking animation
                    st.session_state.thinking = False
                    st.session_state.thinking_container.empty()
                    
                    # Show response with typing animation
                    animate_typing(response, st.session_state.response_container)
                    
                    st.rerun()
                    return
            
            # If we get here, we couldn't provide a helpful response
            response = "I don't have specific information about that, but I can help you find ceiling fans, air conditioners, water heaters, and other home appliances. Could you please clarify what you're looking for?"
            
            # Update conversation history
            st.session_state['conversation_history'].append(("assistant", response))
            
            # Stop thinking animation
            st.session_state.thinking = False
            st.session_state.thinking_container.empty()
            
            # Show response with typing animation
            animate_typing(response, st.session_state.response_container)
            
            st.rerun()
            
        except Exception as e:
            # Handle errors
            error_msg = str(e)
            print(f"Error in query processing: {error_msg}")
            traceback.print_exc()
            
            response = f"I apologize, but I encountered an error: {error_msg}"
            
            # Update conversation history
            st.session_state['conversation_history'].append(("assistant", response))
            
            # Stop thinking animation
            st.session_state.thinking = False
            st.session_state.thinking_container.empty()
            
            # Show response with typing animation
            animate_typing(response, st.session_state.response_container)
            
            st.rerun()
            
    except Exception as e:
        # Handle errors at the outermost level
        error_msg = str(e)
        print(f"Outer error in query processing: {error_msg}")
        traceback.print_exc()
        
        response = f"I apologize, but I encountered an error: {error_msg}"
        
        # Update conversation history
        st.session_state['conversation_history'].append(("assistant", response))
        
        # Stop thinking animation
        st.session_state.thinking = False
        st.session_state.thinking_container.empty()
        
        # Show response with typing animation
        animate_typing(response, st.session_state.response_container)
        
        st.rerun()

def is_comparison_query(query):
    """
    Detect if a query is asking to compare products.
    
    Parameters:
    - query: The user's query
    
    Returns:
    - Boolean indicating if this is a comparison query
    """
    query_lower = query.lower()
    
    # Check for direct comparison phrases
    comparison_phrases = [
        "vs", "versus", "compare", "better than", "difference between",
        "which is better", "compare", "or", "over", "against"
    ]
    
    # Check if any of the comparison phrases are in the query
    for phrase in comparison_phrases:
        if phrase in query_lower:
            return True
    
    return False

def extract_products_to_compare(query, df):
    """
    Extract product names from a comparison query.
    
    Parameters:
    - query: The user's query
    - df: Product dataframe
    
    Returns:
    - A tuple of (product_a, product_b) names to compare
    """
    query_lower = query.lower()
    
    # Try to extract product names based on comparison keywords
    for keyword in ["vs", "versus", "or", "compare", "better than", "difference between"]:
        if keyword in query_lower:
            parts = query_lower.split(keyword, 1)
            if len(parts) == 2:
                product_a_text = parts[0].strip()
                product_b_text = parts[1].strip()
                
                # Clean up the extracted product names
                for term in ["which is better", "what is better", "?", "the"]:
                    product_a_text = product_a_text.replace(term, "").strip()
                    product_b_text = product_b_text.replace(term, "").strip()
                
                # Search for these products in the catalog
                product_a_matches = search_products(product_a_text, df, embeddings_dict, k=1)
                product_b_matches = search_products(product_b_text, df, embeddings_dict, k=1)
                
                if not product_a_matches.empty and not product_b_matches.empty:
                    return (product_a_matches.iloc[0], product_b_matches.iloc[0])
    
    return None
    
def generate_product_comparison(product_a, product_b):
    """
    Generate a detailed comparison between two products.
    
    Parameters:
    - product_a: First product dictionary/Series
    - product_b: Second product dictionary/Series
    
    Returns:
    - Formatted string with the comparison
    """
    if product_a is None or product_b is None:
        return "I couldn't find enough information about these products to compare them."
    
    # Get product titles
    title_a = product_a.get('title', 'Product A')
    title_b = product_b.get('title', 'Product B')
    
    # Start building the comparison
    comparison = f"## Comparing {title_a} vs {title_b}\n\n"
    
    # Add a comparison table
    comparison += "| Feature | " + title_a + " | " + title_b + " | Winner |\n"
    comparison += "| --- | --- | --- | --- |\n"
    
    # Compare prices
    price_a = product_a.get('Better Home Price', 0)
    price_b = product_b.get('Better Home Price', 0)
    price_winner = "ğŸ�† " + title_a if price_a < price_b else "ğŸ�† " + title_b if price_b < price_a else "Tie"
    comparison += f"| Price | â‚¹{price_a:,.2f} | â‚¹{price_b:,.2f} | {price_winner} |\n"
    
    # Compare ratings if available
    if 'rating' in product_a and 'rating' in product_b:
        rating_a = product_a.get('rating', 0)
        rating_b = product_b.get('rating', 0)
        rating_winner = "ğŸ�† " + title_a if rating_a > rating_b else "ğŸ�† " + title_b if rating_b > rating_a else "Tie"
        comparison += f"| Rating | {rating_a} | {rating_b} | {rating_winner} |\n"
    
    # Compare energy ratings if available
    if 'Energy Rating' in product_a or 'Energy Rating' in product_b:
        energy_a = product_a.get('Energy Rating', 'N/A')
        energy_b = product_b.get('Energy Rating', 'N/A')
        
        # Determine winner based on star rating
        # Higher star rating is better
        energy_winner = "Tie"
        if energy_a != 'N/A' and energy_b != 'N/A':
            # Extract star numbers if possible
            try:
                stars_a = int(energy_a.split()[0])
                stars_b = int(energy_b.split()[0])
                energy_winner = "ğŸ�† " + title_a if stars_a > stars_b else "ğŸ�† " + title_b if stars_b > stars_a else "Tie"
            except:
                # If we can't parse the stars, just use string comparison
                energy_winner = "ğŸ�† " + title_a if energy_a > energy_b else "ğŸ�† " + title_b if energy_b > energy_a else "Tie"
        
        comparison += f"| Energy Efficiency | {energy_a} | {energy_b} | {energy_winner} |\n"
    
    # Compare warranty
    warranty_a = product_a.get('Warranty', 'N/A')
    warranty_b = product_b.get('Warranty', 'N/A')
    
    # Try to determine which warranty is better
    warranty_winner = "Tie"
    if warranty_a != 'N/A' and warranty_b != 'N/A':
        # Extract years if possible 
        try:
            years_a = int(re.search(r'(\d+)', warranty_a).group(1))
            years_b = int(re.search(r'(\d+)', warranty_b).group(1))
            warranty_winner = "ğŸ�† " + title_a if years_a > years_b else "ğŸ�† " + title_b if years_b > years_a else "Tie"
        except:
            # If we can't extract years, just compare strings
            warranty_winner = "Tie"
    
    comparison += f"| Warranty | {warranty_a} | {warranty_b} | {warranty_winner} |\n"
    
    # Compare brand reputation if available
    brand_a = product_a.get('Brand', 'N/A')
    brand_b = product_b.get('Brand', 'N/A')
    comparison += f"| Brand | {brand_a} | {brand_b} | - |\n"
    
    # Extract and compare key features
    features_a = extract_features_from_title(title_a)
    features_b = extract_features_from_title(title_b)
    
    # Add a detailed feature analysis section
    comparison += "\n### ğŸ”� Key Features Analysis\n\n"
    
    # Product A features
    comparison += f"**{title_a} highlights:**\n"
    for feature in features_a:
        comparison += f"- {feature}\n"
    
    # Product B features
    comparison += f"\n**{title_b} highlights:**\n"
    for feature in features_b:
        comparison += f"- {feature}\n"
    
    # Add recommendation section
    comparison += "\n### ğŸ’¡ Recommendation\n\n"
    
    # Count the winners
    a_wins = comparison.count("ğŸ�† " + title_a)
    b_wins = comparison.count("ğŸ�† " + title_b)
    
    if a_wins > b_wins:
        comparison += f"**{title_a}** appears to be the better choice overall. "
    elif b_wins > a_wins:
        comparison += f"**{title_b}** appears to be the better choice overall. "
    else:
        comparison += f"Both products have their strengths. "
    
    # Add budget vs premium recommendation
    if price_a < price_b * 0.8:  # Product A is at least 20% cheaper
        comparison += f"If you're on a budget, **{title_a}** offers better value for money. "
    elif price_b < price_a * 0.8:  # Product B is at least 20% cheaper
        comparison += f"If you're on a budget, **{title_b}** offers better value for money. "
    
    if price_a > price_b * 1.2:  # Product A is at least 20% more expensive
        comparison += f"If you're looking for premium features, **{title_a}** might be worth the extra investment."
    elif price_b > price_a * 1.2:  # Product B is at least 20% more expensive
        comparison += f"If you're looking for premium features, **{title_b}** might be worth the extra investment."
    
    # Add call to action
    comparison += "\n\n### ğŸ”¥ Limited Time Offers\n\n"
    comparison += f"**Buy {title_a}:** [View Deal](https://betterhome.co.in/product/{product_a.get('id', '')})\n\n"
    comparison += f"**Buy {title_b}:** [View Deal](https://betterhome.co.in/product/{product_b.get('id', '')})\n\n"
    comparison += "ğŸ�� Use code **COMPARE10** for an extra 10% off your purchase!"
    
    return comparison

def extract_features_from_title(title):
    """
    Extract key features from a product title.
    
    Parameters:
    - title: Product title string
    
    Returns:
    - List of feature strings
    """
    features = []
    
    # Look for specific patterns in the title
    # Capacity (for appliances)
    capacity_match = re.search(r'(\d+\.?\d*)\s*(L|liters?|litres?|kg|tons?|cu\.?\s*ft)', title, re.IGNORECASE)
    if capacity_match:
        features.append(f"Capacity: {capacity_match.group(1)} {capacity_match.group(2)}")
    
    # Power consumption
    power_match = re.search(r'(\d+)\s*(W|watts?|kW|kilowatts?)', title, re.IGNORECASE)
    if power_match:
        features.append(f"Power: {power_match.group(1)} {power_match.group(2)}")
    
    # Dimensions
    dimension_match = re.search(r'(\d+\.?\d*)\s*x\s*(\d+\.?\d*)\s*x\s*(\d+\.?\d*)', title)
    if dimension_match:
        features.append(f"Dimensions: {dimension_match.group(0)}")
    
    # Color
    color_match = re.search(r'(White|Black|Silver|Blue|Red|Grey|Gray|Brown|Green|Gold|Rose Gold)', title, re.IGNORECASE)
    if color_match:
        features.append(f"Color: {color_match.group(1)}")
    
    # Smart features
    if re.search(r'smart|wifi|alexa|google assistant|voice control|app', title, re.IGNORECASE):
        features.append("Smart features with app/voice control")
    
    # Energy efficiency
    if re.search(r'energy\s+efficient|energy\s+saving|inverter|BLDC', title, re.IGNORECASE):
        features.append("Energy efficient design")
    
    # Advanced technology
    if re.search(r'AI|ML|digital|inverter|advanced|smart', title, re.IGNORECASE):
        features.append("Advanced technology features")
    
    # If we couldn't extract specific features, add generic ones based on the product type
    if not features:
        if re.search(r'refrigerator|fridge', title, re.IGNORECASE):
            features.append("Cooling technology")
            features.append("Storage capacity")
        elif re.search(r'washing machine|washer', title, re.IGNORECASE):
            features.append("Washing capacity")
            features.append("Multiple wash programs")
        elif re.search(r'ac|air condition', title, re.IGNORECASE):
            features.append("Cooling capacity")
            features.append("Temperature control")
        elif re.search(r'fan|ceiling fan', title, re.IGNORECASE):
            features.append("Air circulation")
            features.append("Speed control")
        else:
            features.append("Premium build quality")
            features.append("Reliable performance")
    
    return features

if __name__ == "__main__":
    main()
