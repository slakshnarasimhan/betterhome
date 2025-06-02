import faiss
import numpy as np
import json
import os
import pandas as pd
import streamlit as st
from ollama import Client as Ollama
import requests
import traceback
import re
import yaml
import sys
from typing import Dict, Any, List, Optional, Tuple
import networkx as nx
from collections import defaultdict
import time
import random
import threading

# Constants for Ollama models
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text:latest"
OLLAMA_COMPLETION_MODEL = "gemma3:latest"

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

# Initialize Ollama client
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
# Step 1: Load Product Terms Dictionary
# ==========================

def load_product_terms(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


def find_product_type(query, product_terms):
    """
    Find the most relevant product type for a given query using the product terms dictionary.
    """
    if not product_terms:
        return None
        
    query_lower = query.lower()
    
    # First check exact matches in product terms dictionary
    for product_type, info in product_terms.items():
        if product_type.lower() in query_lower:
            return product_type
            
        # Check categories if they exist
        if isinstance(info, dict) and 'categories' in info:
            try:
                for category in info.get('categories', []):
                    if isinstance(category, str) and category.lower() in query_lower:
                        return product_type
            except Exception as e:
                print(f"Error checking categories for {product_type}: {str(e)}")
                continue
                
    # If no exact match found, try partial matches
    for product_type, info in product_terms.items():
        # Check if any word in the product type is in the query
        try:
            product_type_words = product_type.lower().split()
            if any(word in query_lower for word in product_type_words):
                return product_type
        except Exception as e:
            print(f"Error checking product type words for {product_type}: {str(e)}")
            continue
            
        # Check categories if they exist
        if isinstance(info, dict) and 'categories' in info:
            try:
                for category in info.get('categories', []):
                    if isinstance(category, str):
                        category_words = category.lower().split()
                        if any(word in query_lower for word in category_words):
                            return product_type
            except Exception as e:
                print(f"Error checking category words for {product_type}: {str(e)}")
                continue
    
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
# Step 3: Generate Query Embedding with Ollama
# ==========================
def get_ollama_embedding(text, model=OLLAMA_EMBEDDING_MODEL):
    """Generate embedding using Ollama"""
    if not text:
        print("Warning: Empty text provided for embedding")
        return np.random.rand(768)  # Return random embedding for empty text
        
    try:
        response = client.embeddings(model=model, prompt=text)
        
        # Validate response format
        if not isinstance(response, dict):
            print(f"Unexpected response format from Ollama: {type(response)}")
            return np.random.rand(768)
            
        embedding = response.get('embedding', [])
        
        # Check if embedding is empty
        if not embedding:
            print("Empty embedding returned from Ollama")
            return np.random.rand(768)
            
        return np.array(embedding)
    except Exception as e:
        print(f"Error generating embedding with Ollama: {e}")
        traceback.print_exc()
        # Return a random embedding with appropriate dimension as fallback
        # Nomic Embed typically returns 768 dimensions
        return np.random.rand(768)

def retrieve_and_generate_ollama(query, context):
    """Generate responses using Ollama instead of OpenAI"""
    if not query:
        return "Please provide a query."
        
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

        # Create full prompt for Ollama
        full_prompt = f"{system_prompt}\n\nContext:\n{context}\n\nQuestion: {query}"
        
        # Make the Ollama API call
        response = client.chat(
            model=OLLAMA_COMPLETION_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
            ],
            temperature=0.7
        )
        
        # Validate response format
        if not isinstance(response, dict):
            print(f"Unexpected response format from Ollama: {type(response)}")
            return "I'm sorry, I encountered an error processing your request. Please try again."
        
        # Extract and return the answer
        message = response.get('message', {})
        if not isinstance(message, dict):
            print(f"Unexpected message format from Ollama: {type(message)}")
            return "I'm sorry, I encountered an error processing your request. Please try again."
            
        answer = message.get('content', '')
        if not answer:
            print("Empty answer returned from Ollama")
            return "I'm sorry, I couldn't generate a response. Please try rephrasing your question."
            
        print(f"Generated answer: {answer}")
        
        # Filter the answer to include only products from the catalog
        try:
            catalog_titles = df['title'].str.lower().tolist()
            filtered_answer = '\n'.join([line for line in answer.split('\n') if any(title in line.lower() for title in catalog_titles)])
            print(f"Filtered answer: {filtered_answer}")  # Debug: Print filtered answer
            
            # If filtering removed everything, just return the original answer
            if not filtered_answer.strip():
                return answer
            
            return filtered_answer
        except Exception as e:
            print(f"Error filtering answer: {str(e)}")
            return answer  # Return the original answer if filtering fails
    except Exception as e:
        print(f"Error in retrieve_and_generate_ollama: {str(e)}")
        traceback.print_exc()
        return f"I'm sorry, I encountered an error while processing your request: {str(e)}"

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
    # Generate query embedding using Ollama
    query_embedding = get_ollama_embedding(query)
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

# ==========================
# Search Blogs with Ollama embeddings
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
        query_embedding = get_ollama_embedding(query)
    except Exception as e:
        traceback.print_exc()
        return []
    
    # Debug: Check query embedding dimensions
    if query_embedding.shape[0] != blog_embeddings.shape[1]:
        print(f"Dimension mismatch: Query embedding shape: {query_embedding.shape}, Blog embedding shape: {blog_embeddings.shape[1]}")
        # Try to reshape if possible, otherwise return empty
        try:
            # Option 1: Truncate or pad
            if query_embedding.shape[0] > blog_embeddings.shape[1]:
                query_embedding = query_embedding[:blog_embeddings.shape[1]]
            else:
                padding = np.zeros(blog_embeddings.shape[1] - query_embedding.shape[0])
                query_embedding = np.concatenate([query_embedding, padding])
        except:
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

# ==========================
# Personalized Recommendations with Ollama
# ==========================
def match_user_profile_with_products(user_profile, product_embeddings, product_metadata, k=3):
    """
    Match a user profile with products for personalized recommendations using Ollama.
    """
    # Create a user profile embedding
    profile_entry = (
        f"Age Group: {user_profile.get('age_group', 'Not Available')}. "
        f"Room Type: {user_profile.get('room_type', 'Not Available')}. "
        f"Preferences: {', '.join(user_profile.get('preferences', []))}. "
        f"Budget: {user_profile.get('budget', 'Not Available')}."
    )
    
    # Generate embedding for the user profile
    try:
        profile_embedding = get_ollama_embedding(profile_entry)
        
        # Calculate similarity scores
        similarity_scores = []
        for i, product_embedding in enumerate(product_embeddings):
            # Handle dimension mismatch if necessary
            if len(profile_embedding) != len(product_embedding):
                # If dimensions don't match, use the smaller dimension
                min_dim = min(len(profile_embedding), len(product_embedding))
                p_embed = profile_embedding[:min_dim]
                prod_embed = product_embedding[:min_dim]
                
                # Calculate cosine similarity
                similarity = np.dot(p_embed, prod_embed) / (
                    np.linalg.norm(p_embed) * np.linalg.norm(prod_embed)
                )
            else:
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
    except Exception as e:
        print(f"Error matching user profile with products: {str(e)}")
        traceback.print_exc()
        return None

# Custom conversational response formatter for products
def format_product_response_conversational(products_df):
    """
    Format product results in a conversational format
    """
    # Limit to 3 products maximum for brevity
    products_df = products_df.head(3)
    
    # Count products
    product_count = len(products_df)
    
    # Initialize response with conversational intro
    response = "### Product Recommendations\n\n"
    response += f"I found {product_count} product{'s' if product_count > 1 else ''} that might interest you:\n\n"
    
    # Add a special message for BLDC fans if they're in the results
    has_bldc = any(products_df['title'].str.contains('BLDC|Brushless', case=False, na=False))
    if has_bldc:
        response += "ğŸ’š **BLDC Fans** - 70% less electricity! These energy-efficient options will help reduce your power bills.\n\n"
    
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
    response += "*Click on 'Buy Now' to purchase the product.*\n\n"
    
    # Add a conversational follow-up question
    response += "**Would you like to know more about any specific product or compare their features?**"
    
    return response

def format_brand_response_conversational(products_df, product_type, is_warranty_query=False):
    """
    Format brand query results in a conversational format
    """
    # Limit to 5 brands maximum for brevity
    products_df = products_df.head(5)
    
    # Group by brand
    brand_groups = products_df.groupby('Brand')
    brand_count = len(brand_groups)
    
    # Create a conversational introduction
    if is_warranty_query:
        response = f"### Warranty Information for {product_type.title()} Brands\n\n"
        response += f"I found warranty details for {brand_count} brands of {product_type.title()}:\n\n"
    else:
        response = f"### Top {product_type.title()} Brands\n\n"
        response += f"Here are {brand_count} great brands of {product_type.title()} for you to consider:\n\n"
    
    # Add each brand with details
    for brand_name, group in brand_groups:
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
        features = []
        if 'Key Features' in group.iloc[0]:
            if isinstance(group.iloc[0]['Key Features'], str):
                features = group.iloc[0]['Key Features'].split(', ')
                if features and features[0]:  # Check if there are actual features
                    response += "- Key Features: "
                    response += ", ".join(features[:3])  # Limit to 3 features
                    response += "\n"
        
        # Add a sample product link
        sample_product = group.iloc[0]
        sample_url = sample_product.get('url', '#')
        response += f"- [Browse {brand_name} Products]({sample_url})\n\n"
    
    # Add conversational follow-up
    response += f"\n**Would you like me to help you choose a {product_type.lower()}? Or would you like to know more about any specific brand?**"
    
    return response

# ==========================
# Main Function
# ==========================
def main():
    st.title("Better Home Assistant (Ollama Version)")
    
    # Initialize session state variables if they don't exist
    if 'conversation_history' not in st.session_state:
        st.session_state['conversation_history'] = []
    if 'follow_up_state' not in st.session_state:
        st.session_state['follow_up_state'] = None
    if 'last_products' not in st.session_state:
        st.session_state['last_products'] = None
    if 'last_product_type' not in st.session_state:
        st.session_state['last_product_type'] = None
    
    # Create all containers at the beginning
    # This container holds the chat history (past messages)
    chat_history_container = st.container()
    
    # This spacer helps push new content to the bottom
    st.markdown("<br>" * 5, unsafe_allow_html=True)
    
    # Create a fixed container at the bottom for input and new messages
    bottom_container = st.container()
    
    # Create containers for animations and responses
    if 'thinking_container' not in st.session_state:
        st.session_state.thinking_container = bottom_container.empty()
    if 'response_container' not in st.session_state:
        st.session_state.response_container = bottom_container.empty()
    if 'follow_up_container' not in st.session_state:
        st.session_state.follow_up_container = bottom_container.empty()
    
    # Display session state for debugging (can be removed in production)
    with st.sidebar.expander("Debug Session State"):
        st.write(f"Follow-up state: {st.session_state['follow_up_state']}")
        st.write(f"Last product type: {st.session_state['last_product_type']}")
        st.write(f"Has products: {st.session_state['last_products'] is not None}")
        
        # Add a reset button for debugging
        if st.button("Reset Session State"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    # Display model information
    st.sidebar.title("LLM Configuration")
    st.sidebar.info(f"Using Ollama with models:\n- Embeddings: {OLLAMA_EMBEDDING_MODEL}\n- Completions: {OLLAMA_COMPLETION_MODEL}")
    
    # Test Ollama availability
    try:
        client.list()
        st.sidebar.success("âœ… Ollama connection successful")
    except Exception as e:
        st.sidebar.error(f"â�Œ Ollama connection failed: {str(e)}")
        st.error("Could not connect to Ollama. Please make sure Ollama is running.")
        return
    
    # Check if required models are available
    try:
        models = client.list()
        available_models = [model.get('name', '') for model in models.get('models', [])]
        
        embedding_model_name = OLLAMA_EMBEDDING_MODEL.split(':')[0] if ':' in OLLAMA_EMBEDDING_MODEL else OLLAMA_EMBEDDING_MODEL
        completion_model_name = OLLAMA_COMPLETION_MODEL.split(':')[0] if ':' in OLLAMA_COMPLETION_MODEL else OLLAMA_COMPLETION_MODEL
        
        embedding_model_found = any(model.startswith(embedding_model_name) for model in available_models)
        completion_model_found = any(model.startswith(completion_model_name) for model in available_models)
        
        if not embedding_model_found:
            st.sidebar.warning(f"âš ï¸� Embedding model {embedding_model_name} not found. Run: ollama pull {embedding_model_name}")
            
        if not completion_model_found:
            st.sidebar.warning(f"âš ï¸� Completion model {completion_model_name} not found. Run: ollama pull {completion_model_name}")
            
        if embedding_model_found and completion_model_found:
            st.sidebar.success("âœ… All required models are available")
    except Exception as e:
        st.sidebar.warning(f"âš ï¸� Could not check available models: {str(e)}")
    
    # Build or load FAISS index for product search
    try:
        index = build_or_load_faiss_index(
            embedding_data['product_embeddings'],
            embedding_data['product_embeddings'].shape[1] if len(embedding_data['product_embeddings']) > 0 else 768,
            PRODUCT_INDEX_FILE_PATH
        )
    except Exception as e:
        st.error(f"Error building product index: {str(e)}")
        index = None
    
    # Display conversation history in the history container
    with chat_history_container:
        for i, (role, message) in enumerate(st.session_state['conversation_history']):
            if role == "user":
                st.markdown(f"**You:** {message}")
            else:
                st.markdown(f"**Assistant:** {message}")
    
    # Bottom area for input and new responses
    with bottom_container:
        # Check if we need to handle follow-up interactions
        if st.session_state['follow_up_state'] is not None:
            print(f"Handling follow-up state: {st.session_state['follow_up_state']}")
            handle_follow_up(st.session_state['follow_up_state'])
            return
        
        # User input
        user_input = st.text_input("Ask me anything about home products:", key="user_input")
        
        if user_input:
            # Add user input to conversation history
            st.session_state['conversation_history'].append(("user", user_input))
            
            query = user_input.lower().strip()
            
            # Set thinking flag to True
            st.session_state.thinking = True
            
            # Start thinking animation in a separate thread
            thinking_thread = threading.Thread(target=show_thinking_animation, args=(st.session_state.thinking_container, 0.3))
            thinking_thread.daemon = True
            thinking_thread.start()
            
            # Process user query
            try:
                # Rest of the query processing code remains the same
                # Check if it's a how-to question
                if is_how_to_query(query):
                    # Load blog embeddings
                    try:
                        print(f"Loading blog embeddings from {BLOG_EMBEDDINGS_FILE_PATH}")
                        blog_embeddings_dict = load_blog_embeddings(BLOG_EMBEDDINGS_FILE_PATH)
                        if blog_embeddings_dict and blog_embeddings_dict['blog_embeddings'].shape[0] > 0:
                            print(f"Successfully loaded {blog_embeddings_dict['blog_embeddings'].shape[0]} blog embeddings")
                            # Search for relevant blogs using Ollama embeddings
                            blog_results = search_relevant_blogs(query, blog_embeddings_dict, k=3)
                            if blog_results:
                                print(f"Found {len(blog_results)} relevant blog articles")
                                response = format_blog_response(blog_results, query)
                                st.empty()  # Clear the "Processing" message
                                animate_typing(response, st.session_state.response_container)
                                
                                # Add follow-up options
                                st.session_state['follow_up_state'] = 'blog_followup'
                                col1, col2 = st.columns(2)
                                with col1:
                                    if st.button("Find related products"):
                                        st.session_state['follow_up_state'] = 'related_to_blogs'
                                        st.rerun()
                                with col2:
                                    if st.button("Ask another question"):
                                        st.session_state['follow_up_state'] = None
                                        st.rerun()
                                return
                            else:
                                print("No relevant blog articles found")
                                response = "I couldn't find any articles that directly answer your how-to question. Please try rephrasing your question or ask about a specific product."
                                st.empty()  # Clear the "Processing" message
                                animate_typing(response, st.session_state.response_container)
                                return
                        else:
                            print("Failed to load blog embeddings or no embeddings found")
                            response = "I'm having trouble accessing our knowledge base right now. Please try again later or ask about a specific product."
                            st.empty()  # Clear the "Processing" message
                            animate_typing(response, st.session_state.response_container)
                            return
                    except Exception as e:
                        print(f"Error searching blogs: {str(e)}")
                        traceback.print_exc()
                        response = "I encountered an error while searching for information. Please try again later or ask about a specific product."
                        st.empty()  # Clear the "Processing" message
                        animate_typing(response, st.session_state.response_container)
                        return

                # Handle price queries
                try:
                    if any(word in query for word in ['price', 'cost', 'expensive', 'cheap', 'budget']):
                        products = handle_price_query(query, df, product_terms)
                        if products is not None:
                            # Store the products for follow-up
                            st.session_state['last_products'] = products
                            
                            # Get product type
                            if 'Product Type' in products.columns:
                                st.session_state['last_product_type'] = products.iloc[0]['Product Type']
                            
                            response = format_product_response_conversational(products)  # Use our conversational formatter
                            st.empty()  # Clear the "Processing" message
                            animate_typing(response, st.session_state.response_container)
                            
                            # Add follow-up buttons
                            st.session_state['follow_up_state'] = 'price_results'
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                if st.button("Compare features"):
                                    st.session_state['follow_up_state'] = 'compare_features'
                                    st.rerun()
                            with col2:
                                if st.button("Find alternatives"):
                                    st.session_state['follow_up_state'] = 'find_alternatives'
                                    st.rerun()
                            with col3:
                                if st.button("Ask another question"):
                                    st.session_state['follow_up_state'] = None
                                    st.rerun()
                            return
                except Exception as e:
                    print(f"Error handling price query: {str(e)}")
                    traceback.print_exc()

                # Handle brand queries
                try:
                    if any(word in query for word in ['brand', 'warranty', 'company', 'manufacturer']):
                        brand_result = handle_brand_query(query, df, product_terms)
                        if brand_result is not None:
                            # Store the products and product type for follow-up
                            st.session_state['last_products'] = brand_result['dataframe']
                            st.session_state['last_product_type'] = brand_result['product_type']
                            
                            response = format_brand_response_conversational(brand_result['dataframe'], brand_result['product_type'], brand_result['is_warranty_query'])
                            st.empty()  # Clear the "Processing" message
                            animate_typing(response, st.session_state.response_container)
                            
                            # Add follow-up buttons
                            st.session_state['follow_up_state'] = 'brand_results'
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                if st.button("Help me choose one"):
                                    st.session_state['follow_up_state'] = 'help_choose'
                                    st.rerun()
                            with col2:
                                if st.button("Show features"):
                                    st.session_state['follow_up_state'] = 'show_features'
                                    st.rerun()
                            with col3:
                                if st.button("Ask another question"):
                                    st.session_state['follow_up_state'] = None
                                    st.rerun()
                            return
                except Exception as e:
                    print(f"Error handling brand query: {str(e)}")
                    traceback.print_exc()

                # Handle general "best" product queries
                try:
                    if any(term in query for term in ['best', 'recommend', 'suggest', 'top', 'ideal', 'perfect']):
                        response = retrieve_and_generate_ollama(query, "")
                        st.empty()  # Clear the "Processing" message
                        animate_typing(response, st.session_state.response_container)
                        
                        # Add follow-up buttons
                        st.session_state['follow_up_state'] = 'best_results'
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("Show me more options"):
                                st.session_state['follow_up_state'] = 'more_options'
                                st.rerun()
                        with col2:
                            if st.button("Ask another question"):
                                st.session_state['follow_up_state'] = None
                                st.rerun()
                        return
                except Exception as e:
                    print(f"Error handling best product query: {str(e)}")
                    traceback.print_exc()
                
                # Handle similar product queries with knowledge graph
                if any(term in query for term in ['similar', 'related', 'like']) and knowledge_graph is not None:
                    try:
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
                                # Store products for follow-up
                                st.session_state['last_products'] = graph_recommendations
                                
                                response = format_product_response_conversational(graph_recommendations)  # Use conversational format
                                st.empty()  # Clear the "Processing" message
                                animate_typing(response, st.session_state.response_container)
                                
                                # Add follow-up buttons
                                st.session_state['follow_up_state'] = 'similar_results'
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    if st.button("Compare products"):
                                        st.session_state['follow_up_state'] = 'compare_products'
                                        st.rerun()
                                with col2:
                                    if st.button("Help me decide"):
                                        st.session_state['follow_up_state'] = 'help_decide'
                                        st.rerun()
                                with col3:
                                    if st.button("Ask another question"):
                                        st.session_state['follow_up_state'] = None
                                        st.rerun()
                                
                                # Append to conversation history without displaying
                                st.session_state['conversation_history'].append(("user", user_input))
                                st.session_state['conversation_history'].append(("assistant", response))
                                return
                    except Exception as e:
                        print(f"Error in knowledge graph search: {str(e)}")
                        # Continue to other search methods
                
                # Handle general product search
                try:
                    if index is not None:
                        # First try to use the knowledge graph if appropriate
                        if knowledge_graph is not None and any(term in query for term in ['recommend', 'similar', 'related', 'like']):
                            indices = search_products_with_graph(query, df, embedding_data, knowledge_graph)
                            products = [df.iloc[idx] for idx in indices]
                        else:
                            products = search_catalog(query, df, index)
                        
                        if products and len(products) > 0:
                            # Store products for follow-up
                            try:
                                products_df = pd.DataFrame(products)
                                st.session_state['last_products'] = products_df
                                
                                # Try to determine product type
                                if 'Product Type' in products_df.columns:
                                    st.session_state['last_product_type'] = products_df.iloc[0]['Product Type']
                                
                                response = format_product_response_conversational(products_df)  # Use conversational format
                            except Exception as e:
                                print(f"Error formatting products: {str(e)}")
                                response = format_answer(products, query)  # Fallback to standard format
                        else:
                            response = "I couldn't find any products matching your query. Could you try a different search term?"
                            
                        st.empty()  # Clear the "Processing" message
                        animate_typing(response, st.session_state.response_container)
                        
                        # Show products in a table (optional)
                        if products and len(products) > 0:
                            with st.expander("Show detailed product information"):
                                st.dataframe(pd.DataFrame(products))
                        
                        # Add follow-up buttons
                        if products and len(products) > 0:
                            st.session_state['follow_up_state'] = 'search_results'
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                if st.button("Which one should I buy?"):
                                    st.session_state['follow_up_state'] = 'recommendation'
                                    st.rerun()
                            with col2:
                                if st.button("Compare top 3"):
                                    st.session_state['follow_up_state'] = 'compare_top'
                                    st.rerun()
                            with col3:
                                if st.button("Ask another question"):
                                    st.session_state['follow_up_state'] = None
                                    st.rerun()
                                
                        # Append to conversation history without displaying
                        st.session_state['conversation_history'].append(("user", user_input))
                        st.session_state['conversation_history'].append(("assistant", response))
                        return
                except Exception as e:
                    print(f"Error in product search: {str(e)}")
                    traceback.print_exc()
                    
                # If all specialized handlers failed, try general query with Ollama
                try:
                    response = retrieve_and_generate_ollama(query, "I couldn't find specific product information for your query. Here's a general response:")
                    st.empty()  # Clear the "Processing" message
                    animate_typing(response, st.session_state.response_container)
                    
                    # Add button to ask another question
                    if st.button("Ask another question"):
                        st.session_state['follow_up_state'] = None
                        st.rerun()
                    return
                except Exception as e:
                    print(f"Error in fallback response: {str(e)}")
                    traceback.print_exc()
                    st.error("I apologize, but I encountered multiple errors trying to process your request. Please try a different query.")
                    
            except Exception as e:
                error_msg = str(e)
                print(f"Top-level error: {error_msg}")
                traceback.print_exc()
                st.error(f"I apologize, but I encountered an error: {error_msg}")
                # Append error to conversation history without displaying
                st.session_state['conversation_history'].append(("user", user_input))
                st.session_state['conversation_history'].append(("assistant", f"I apologize, but I encountered an error: {error_msg}"))

            # Stop thinking animation
            st.session_state.thinking = False
            time.sleep(0.5)  # Give time for the animation to stop
            st.session_state.thinking_container.empty()  # Clear the thinking message
            
            # Display the animated response
            animate_typing(response, st.session_state.response_container)
            
            # ... rest of the existing code ...

def handle_follow_up(state):
    """Handle follow-up interactions based on state"""
    # Create a response container for follow-ups if it doesn't exist
    if 'follow_up_container' not in st.session_state:
        st.session_state.follow_up_container = st.empty()
    
    # Display the current conversation history
    for i, (role, message) in enumerate(st.session_state['conversation_history']):
        if role == "user":
            st.markdown(f"**You:** {message}")
        else:
            st.markdown(f"**Assistant:** {message}")

    # Print debug info
    print(f"Handling follow-up state: {state}")
    print(f"Last product type: {st.session_state.get('last_product_type')}")
    print(f"Has products: {st.session_state.get('last_products') is not None}")
    
    # Map state values to handlers
    state_map = {
        'help_choose': 'help_choose',
        'show_features': 'show_features',
        'compare_features': 'compare_products',
        'compare_products': 'compare_products',
        'compare_top': 'compare_products',
        'find_alternatives': 'product_alternatives',
        'installation_tips': 'installation_tips',
        'maintenance_tips': 'maintenance_tips',
        'energy_tips': 'energy_tips',
        'price_results': 'handle_price_results',
        'brand_results': 'brand_results',
        'search_results': 'search_results',
        'similar_results': 'similar_results',
        'best_results': 'best_results',
    }
    
    # Count conversation turns to determine if we should push for purchase
    conversation_turns = len([msg for msg in st.session_state['conversation_history'] if msg[0] == "assistant"])
    should_encourage_purchase = conversation_turns >= 2
    
    # Get handler name or default to "default"
    handler = state_map.get(state, "default")
    
    # Add user query to conversation history for the follow-up if needed
    if len(st.session_state['conversation_history']) > 0 and st.session_state['conversation_history'][-1][0] != "user":
        follow_up_query = f"I'm interested in more information about these products."
        st.session_state['conversation_history'].append(("user", follow_up_query))
    
    # Helper function to add purchase links
    def add_purchase_links(response, products):
        """Add purchase links to the response"""
        if not isinstance(products, pd.DataFrame) or len(products) == 0:
            return response
            
        response += "\n\n### Ready to Buy?\n\n"
        for _, product in products.head(3).iterrows():
            title = product['title']
            price = product['Better Home Price']
            url = product.get('url', '#')
            response += f"â€¢ **{title}** - â‚¹{price:,.2f} - [ğŸ›’ Buy Now]({url})\n"
        
        return response
    
    # Dispatch to specific handlers
    if handler == 'help_choose':
        products = st.session_state['last_products']
        product_type = st.session_state.get('last_product_type', 'product')
        
        # Safety check for None products
        if products is None or not isinstance(products, pd.DataFrame) or products.empty:
            response = f"I don't have any {product_type} information available to help you choose. Would you like to search for some {product_type}s?"
            st.session_state['conversation_history'].append(("assistant", response))
            animate_typing(response, st.session_state.follow_up_container)
            
            if st.button("Start a new search"):
                st.session_state['follow_up_state'] = None
                st.rerun()
            return
        
        response = f"## Helping You Choose a {product_type}\n\nBased on the products we discussed, here's my recommendation:"
        
        # Get a recommendation based on price and features
        try:
            # Sort by price (mid-range usually best value)
            sorted_products = products.sort_values('Better Home Price')
            mid_index = len(sorted_products) // 2
            recommended = sorted_products.iloc[mid_index]
            
            response += f"\n\n### Recommended: {recommended['title']}\n**Price**: â‚¹{recommended['Better Home Price']:,.2f}\n\n"
            
            # Explain why this is recommended
            response += "**Why I recommend this:**\n"
            response += "- Good balance of price and features\n"
            response += "- Well-regarded brand with reliable warranty\n"
            response += "- Suitable for most homes and family needs\n\n"
            
            # Add buy link for recommended product
            url = recommended.get('url', '#')
            response += f"**[ğŸ›’ Buy This {product_type} Now]({url})**\n\n"
            
            # Provide alternative options
            response += "### Other Options:\n"
            response += "**If you're on a budget:**\n"
            if len(sorted_products) > 0:
                budget_option = sorted_products.iloc[0]
                budget_url = budget_option.get('url', '#')
                response += f"â€¢ {budget_option['title']} (â‚¹{budget_option['Better Home Price']:,.2f}) - [Buy Now]({budget_url})\n\n"
            
            response += "**If you want premium features:**\n"
            if len(sorted_products) > 2:
                premium_option = sorted_products.iloc[-1]
                premium_url = premium_option.get('url', '#')
                response += f"â€¢ {premium_option['title']} (â‚¹{premium_option['Better Home Price']:,.2f}) - [Buy Now]({premium_url})"
        except Exception as e:
            response += "\n\nI couldn't generate a specific recommendation based on the products, but here's some general advice:\n"
            response += "- Choose a product with a good warranty (2+ years)\n"
            response += "- Look for energy-efficient options if available\n"
            response += "- Consider well-known brands for reliability"
            print(f"Error in recommendation: {str(e)}")
            
            # Still try to add purchase links
            response = add_purchase_links(response, products)
        
        # Add decisive call to action if we've had a few exchanges
        if should_encourage_purchase:
            response += "\n\n### Make Your Purchase Today!\n\n"
            response += "These products are in stock and ready to ship. Complete your purchase now for:"
            response += "\n- Fast delivery options"
            response += "\n- Easy installation support"
            response += "\n- Full manufacturer warranty"
            
        # Add response to conversation history
        st.session_state['conversation_history'].append(("assistant", response))
        
        # Display the response
        animate_typing(response, st.session_state.follow_up_container)
        
        # Continue the conversation with purchase-focused options
        st.write("### What would you like to do next?")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Installation tips"):
                st.session_state['follow_up_state'] = 'installation_tips'
                st.rerun()
        with col2:
            if st.button("See payment options"):
                payment_response = "### Payment & Delivery Options\n\n"
                payment_response += "- **Cash on Delivery**: Available for orders under â‚¹10,000\n"
                payment_response += "- **Credit/Debit Card**: All major cards accepted\n"
                payment_response += "- **EMI Options**: Available for purchases above â‚¹5,000\n"
                payment_response += "- **Bank Transfer**: Get 2% discount on direct transfers\n\n"
                payment_response += "**Delivery**: 2-5 business days for most locations"
                
                st.session_state['conversation_history'].append(("user", "What are the payment and delivery options?"))
                st.session_state['conversation_history'].append(("assistant", payment_response))
                st.session_state['follow_up_state'] = None
                st.rerun()
        with col3:
            if st.button("Ask another question"):
                st.session_state['follow_up_state'] = None
                st.rerun()
                
    elif handler == 'show_features':
        products = st.session_state['last_products']
        product_type = st.session_state.get('last_product_type', 'product')
        
        # Safety check for None products
        if products is None or not isinstance(products, pd.DataFrame) or products.empty:
            response = f"I don't have any {product_type} information available to show features. Would you like to search for some {product_type}s?"
            st.session_state['conversation_history'].append(("assistant", response))
            animate_typing(response, st.session_state.follow_up_container)
            
            if st.button("Start a new search"):
                st.session_state['follow_up_state'] = None
                st.rerun()
            return
        
        response = f"## Features Comparison for {product_type}\n\n"
        
        # Create a comparison table
        comparison_data = []
        for _, product in products.iterrows():
            features = []
            title = product['title']
            # Extract potential features from title
            if 'BLDC' in title or 'Brushless' in title:
                features.append("Energy-efficient BLDC motor")
            if 'Smart' in title or 'IoT' in title or 'Remote' in title:
                features.append("Smart controls")
            if 'LED' in title:
                features.append("LED lighting")
            if 'Year' in title or 'Warranty' in title:
                # Try to extract warranty period
                warranty_match = re.search(r'(\d+)\s*Year', title)
                if warranty_match:
                    features.append(f"{warranty_match.group(1)} year warranty")
                else:
                    features.append("Warranty included")
                    
            comparison_data.append({
                'Brand': product['Brand'],
                'Product': title,
                'Price': f"â‚¹{product['Better Home Price']:,.2f}",
                'Features': ", ".join(features)
            })
        
        # Create markdown table
        comparison_table = "| Brand | Product | Price | Features |\n"
        comparison_table += "| ----- | ------- | ----- | -------- |\n"
        
        for data in comparison_data:
            comparison_table += f"| {data['Brand']} | {data['Product']} | {data['Price']} | {data['Features']} |\n"
        
        response += comparison_table + "\n\n"
        
        # Add product-specific advice
        if product_type == 'Ceiling Fan':
            response += "### Ceiling Fan Features to Consider:\n"
            response += "- **Air Delivery**: Higher values (>210 CMM) mean better air circulation\n"
            response += "- **BLDC Motors**: Reduce electricity consumption by up to 70%\n"
            response += "- **Remote Control**: Convenient for adjusting settings without getting up"
        elif product_type == 'Water Heater':
            response += "### Water Heater Features to Consider:\n"
            response += "- **Capacity**: Choose based on your family size (10-15L for small families)\n"
            response += "- **Star Rating**: Higher ratings mean better energy efficiency\n"
            response += "- **Safety Features**: Look for anti-scalding and auto-shutoff"
            
        # Add response to conversation history
        st.session_state['conversation_history'].append(("assistant", response))
        
        # Display the response
        animate_typing(response, st.session_state.follow_up_container)
        
        # Continue the conversation
        st.write("### What would you like to do next?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Help me decide"):
                st.session_state['follow_up_state'] = 'help_choose'
                st.rerun()
        with col2:
            if st.button("Ask another question"):
                st.session_state['follow_up_state'] = None
                st.rerun()
    
    elif handler == 'compare_products':
        # Reset user query but keep follow-up state
        user_query = "Let me see a comparison of these products"
        st.session_state['conversation_history'].append(("user", user_query))
        
        # Similar to show_features but with more detailed comparison
        products = st.session_state['last_products']
        
        # Safety check for None products
        if products is None or not isinstance(products, pd.DataFrame) or products.empty:
            product_type = st.session_state.get('last_product_type', 'products')
            response = f"I don't have any {product_type} information to compare. Would you like to search for some {product_type}?"
            st.session_state['conversation_history'].append(("assistant", response))
            animate_typing(response, st.session_state.follow_up_container)
            
            if st.button("Start a new search"):
                st.session_state['follow_up_state'] = None
                st.rerun()
            return
        
        if 'Product Type' in products.columns:
            product_type = products.iloc[0]['Product Type']
        else:
            product_type = st.session_state.get('last_product_type', 'Products')
            
        response = f"## Detailed Comparison of {product_type}\n\n"
        
        # Create a comparison table with more details
        comparison_data = []
        for _, product in products.iterrows():
            entry = {
                'Brand': product['Brand'],
                'Product': product['title'][:50] + ('...' if len(product['title']) > 50 else ''),
                'Price': f"â‚¹{product['Better Home Price']:,.2f}"
            }
            
            # Extract other details if available
            for col in ['rating', 'energy_rating', 'power', 'capacity']:
                if col in product:
                    entry[col.capitalize()] = product[col]
            
            comparison_data.append(entry)
        
        # Create markdown table
        comparison_table = "| Brand | Product | Price |\n"
        comparison_table += "| ----- | ------- | ----- |\n"
        
        for data in comparison_data:
            comparison_table += f"| {data['Brand']} | {data['Product']} | {data['Price']} |\n"
        
        response += comparison_table + "\n\n"
        
        # Add recommendation
        response += "### Recommendation Based on Comparison:\n"
        response += "After comparing these products, here's what stands out:\n\n"
        
        # Sort by price to identify budget and premium options
        sorted_products = products.sort_values('Better Home Price')
        
        if len(sorted_products) >= 3:
            budget = sorted_products.iloc[0]
            mid = sorted_products.iloc[len(sorted_products)//2]
            premium = sorted_products.iloc[-1]
            
            response += f"**Best Value**: {mid['title']}\n"
            response += f"**Budget Option**: {budget['title']}\n"
            response += f"**Premium Choice**: {premium['title']}"
        elif len(sorted_products) > 0:
            response += f"**Recommended**: {sorted_products.iloc[0]['title']}"
            
        # Add purchase links
        response = add_purchase_links(response, products)
            
        # Add decisive call to action if we've had a few exchanges
        if should_encourage_purchase:
            response += "\n\n### Limited Time Offer!\n\n"
            response += "Make your purchase today and receive complimentary installation worth â‚¹1,500!"
            
        # Add response to conversation history
        st.session_state['conversation_history'].append(("assistant", response))
        
        # Display the response
        animate_typing(response, st.session_state.follow_up_container)
        
        # Continue the conversation
        st.write("### What would you like to do next?")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Help me decide"):
                st.session_state['follow_up_state'] = 'help_choose'
                st.rerun()
        with col2:
            if st.button("Installation advice"):
                st.session_state['follow_up_state'] = 'installation_tips'
                st.rerun()
        with col3:
            if st.button("Ask another question"):
                st.session_state['follow_up_state'] = None
                st.rerun()
    
    elif handler == 'product_alternatives':
        # Show alternative products
        products = st.session_state['last_products']
        
        # Safety check for None products
        if products is None or not isinstance(products, pd.DataFrame) or products.empty:
            product_type = st.session_state.get('last_product_type', 'products')
            response = f"I don't have any {product_type} information to find alternatives for. Would you like to search for some {product_type}?"
            st.session_state['conversation_history'].append(("assistant", response))
            animate_typing(response, st.session_state.follow_up_container)
            
            if st.button("Start a new search"):
                st.session_state['follow_up_state'] = None
                st.rerun()
            return
        
        if 'Product Type' in products.columns:
            product_type = products.iloc[0]['Product Type']
        else:
            product_type = st.session_state.get('last_product_type', 'Products')
            
        response = f"## Alternative {product_type} Options\n\n"
        response += "Here are some alternative products you might want to consider:\n\n"
        
        # Try to find alternatives
        try:
            # First try the knowledge graph if available
            if knowledge_graph is not None and len(products) > 0:
                first_product = products.iloc[0]
                product_title = first_product['title']
                
                alternatives = get_recommendations_from_graph(product_title, knowledge_graph, df, max_recommendations=3)
                
                if alternatives is not None and len(alternatives) > 0:
                    # Create markdown product list
                    for _, product in alternatives.iterrows():
                        title = product['title']
                        price = product['Better Home Price']
                        brand = product['Brand']
                        url = product.get('url', '#')
                        
                        response += f"**{title}**\n"
                        response += f"- Brand: {brand}\n"
                        response += f"- Price: â‚¹{price:,.2f}\n"
                        
                        # Add special notes for certain product types
                        if product_type == 'Ceiling Fan' and ('BLDC' in title or 'Brushless' in title):
                            response += "- Energy-efficient BLDC motor (saves up to 70% electricity)\n"
                        
                        # Add buy link
                        response += f"- [ğŸ›’ Buy Now]({url})\n\n"
                    
                    # Update the last products to these alternatives
                    st.session_state['last_products'] = alternatives
            else:
                # Fallback to similar products by brand
                response += "Based on your interests, here are some similar products from different brands:\n\n"
                
                # Get unique brands
                all_brands = df[df['Product Type'] == product_type]['Brand'].unique()
                
                # Take up to 3 different brands
                brands_shown = 0
                found_products = []
                
                for brand in all_brands:
                    if brands_shown >= 3:
                        break
                        
                    brand_products = df[(df['Product Type'] == product_type) & (df['Brand'] == brand)]
                    
                    if len(brand_products) > 0:
                        # Show the first product from this brand
                        product = brand_products.iloc[0]
                        
                        response += f"**{product['title']}**\n"
                        response += f"- Brand: {brand}\n"
                        response += f"- Price: â‚¹{product['Better Home Price']:,.2f}\n"
                        
                        # Add buy link
                        url = product.get('url', '#')
                        response += f"- [ğŸ›’ Buy Now]({url})\n\n"
                        
                        found_products.append(product)
                        brands_shown += 1
                
                # If we found alternative products, update the last_products state
                if found_products:
                    st.session_state['last_products'] = pd.DataFrame(found_products)
        except Exception as e:
            print(f"Error finding alternatives: {str(e)}")
            response += "I couldn't find specific alternatives, but I recommend looking at different brands or price points for more options.\n"
        
        # Add purchase encouragement based on conversation length
        conversation_turns = len([msg for msg in st.session_state['conversation_history'] if msg[0] == "assistant"])
        if conversation_turns >= 2:
            response += "## ğŸŒŸ Special Offer ğŸŒŸ\n\n"
            response += "Purchase any of these products today and receive:\n"
            response += "- **Free express shipping** (save â‚¹500)\n"
            response += "- **10% off** your next purchase\n"
            response += "- **Priority customer support**\n\n"
            response += "These offers are valid for a limited time only. Click any Buy Now link above to take advantage of this special pricing!\n\n"
        
        # Add response to conversation history
        st.session_state['conversation_history'].append(("assistant", response))
        
        # Display the response
        animate_typing(response, st.session_state.follow_up_container)
        
        # Continue the conversation
        st.write("### Would you like more information?")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Help me choose the best one"):
                st.session_state['follow_up_state'] = 'help_choose'
                st.rerun()
        with col2:
            if st.button("Compare all options"):
                st.session_state['follow_up_state'] = 'compare_products'
                st.rerun()
        with col3:
            if st.button("Ask another question"):
                st.session_state['follow_up_state'] = None
                st.rerun()
    
    elif handler == 'installation_tips':
        product_type = st.session_state.get('last_product_type', '')
        
        response = f"## Installation Tips for {product_type}\n\n"
        
        if 'Ceiling Fan' in product_type:
            response += "### Ceiling Fan Installation Guidelines:\n"
            response += "1. **Safety First**: Turn off power at the circuit breaker\n"
            response += "2. **Height**: Install at least 7-8 feet from the floor and 10-12 inches from the ceiling\n"
            response += "3. **Support**: Ensure the ceiling box can support the fan's weight (should be rated for ceiling fans)\n"
            response += "4. **Balancing**: If the fan wobbles after installation, use the balancing kit provided\n"
            response += "5. **Professional Help**: For safety, consider professional installation if you're not experienced"
        
        elif 'Water Heater' in product_type:
            response += "### Water Heater Installation Guidelines:\n"
            response += "1. **Location**: Install in a well-ventilated area with easy access for maintenance\n"
            response += "2. **Mounting**: Secure firmly to the wall with proper brackets\n"
            response += "3. **Plumbing**: Use high-quality pipes rated for hot water\n"
            response += "4. **Electrical**: Ensure proper earthing and use an MCB for protection\n"
            response += "5. **Professional Installation**: Strongly recommended for safety and warranty validity"
        
        elif 'Air Conditioner' in product_type:
            response += "### Air Conditioner Installation Guidelines:\n"
            response += "1. **Location**: Install indoor unit high on the wall, outdoor unit in a well-ventilated space\n"
            response += "2. **Distance**: Keep at least 15cm clearance around the units\n"
            response += "3. **Drainage**: Ensure proper slope for condensate drainage\n"
            response += "4. **Electrical**: Use a dedicated circuit with proper capacity\n"
            response += "5. **Professional Installation**: Required for warranty validity and optimal performance"
        
        else:
            response += "### General Installation Tips:\n"
            response += "1. **Read the Manual**: Always read the manufacturer's instructions carefully\n"
            response += "2. **Tools**: Gather all necessary tools before starting installation\n"
            response += "3. **Safety**: Turn off electricity at the circuit breaker before installation\n"
            response += "4. **Warranty**: Use authorized installers if required by warranty terms\n"
            response += "5. **Testing**: Test the product thoroughly after installation"
            
        # Add response to conversation history
        st.session_state['conversation_history'].append(("assistant", response))
        
        # Display the response
        animate_typing(response, st.session_state.follow_up_container)
        
        # Continue the conversation
        st.write("### Any other help needed?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Maintenance tips"):
                st.session_state['follow_up_state'] = 'maintenance_tips'
                st.rerun()
        with col2:
            if st.button("Ask another question"):
                st.session_state['follow_up_state'] = None
                st.rerun()
    
    elif handler == 'maintenance_tips':
        product_type = st.session_state.get('last_product_type', '')
        
        response = f"## Maintenance Tips for {product_type}\n\n"
        
        if 'Ceiling Fan' in product_type:
            response += "### Ceiling Fan Maintenance:\n"
            response += "1. **Regular Cleaning**: Dust the blades monthly\n"
            response += "2. **Tighten Screws**: Check and tighten all screws every 6 months\n"
            response += "3. **Lubrication**: Oil the motor once a year (if required by manufacturer)\n"
            response += "4. **Wobbling**: If the fan wobbles, rebalance the blades\n"
            response += "5. **Noise**: If you hear unusual noises, check for loose parts"
        
        elif 'Water Heater' in product_type:
            response += "### Water Heater Maintenance:\n"
            response += "1. **Descaling**: Clean the heating element every 6 months in hard water areas\n"
            response += "2. **Anode Rod**: Check the anode rod annually and replace if needed\n"
            response += "3. **Temperature Setting**: Keep at 50-55Â°C to prevent scaling\n"
            response += "4. **Pressure Relief**: Test the pressure relief valve every 6 months\n"
            response += "5. **Draining**: Drain the tank every 6 months to remove sediment"
        
        else:
            response += "### General Maintenance Tips:\n"
            response += "1. **Regular Cleaning**: Keep the product clean and dust-free\n"
            response += "2. **Check Connections**: Periodically check electrical and plumbing connections\n"
            response += "3. **Follow Manual**: Refer to the user manual for specific maintenance\n"
            response += "4. **Professional Service**: Schedule annual professional maintenance\n"
            response += "5. **Replacement Parts**: Use only manufacturer-approved replacement parts"
            
        # Add response to conversation history
        st.session_state['conversation_history'].append(("assistant", response))
        
        # Display the response
        animate_typing(response, st.session_state.follow_up_container)
        
        # Continue the conversation
        st.write("### Would you like to know anything else?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Energy saving tips"):
                st.session_state['follow_up_state'] = 'energy_tips'
                st.rerun()
        with col2:
            if st.button("Ask another question"):
                st.session_state['follow_up_state'] = None
                st.rerun()
    
    elif handler == 'energy_tips':
        product_type = st.session_state.get('last_product_type', '')
        
        response = f"## Energy Saving Tips for {product_type}\n\n"
        
        if 'Ceiling Fan' in product_type:
            response += "### How to Save Energy with Ceiling Fans:\n"
            response += "1. **Use BLDC Fans**: They consume up to 70% less electricity than regular fans\n"
            response += "2. **Optimal Speed**: Use lower speeds when possible\n"
            response += "3. **Turn Off When Not in Room**: Fans cool people, not rooms\n"
            response += "4. **Regular Maintenance**: Clean dust regularly for optimal performance\n"
            response += "5. **Pair with AC**: Raise AC temperature by 3-4Â°C and use fan to feel cooler"
        
        elif 'Water Heater' in product_type:
            response += "### How to Save Energy with Water Heaters:\n"
            response += "1. **Optimal Temperature**: Set to 50-55Â°C instead of maximum\n"
            response += "2. **Timer Usage**: Use built-in timers to heat water only when needed\n"
            response += "3. **Insulate Pipes**: Prevent heat loss through pipes\n"
            response += "4. **Remove Sediment**: Regularly clean to maintain efficiency\n"
            response += "5. **Upgrade to Higher Star Rating**: Each star increases efficiency by 8-10%"
        
        elif 'Air Conditioner' in product_type:
            response += "### How to Save Energy with Air Conditioners:\n"
            response += "1. **Optimal Temperature**: Set to 24-26Â°C for balance of comfort and efficiency\n"
            response += "2. **Regular Cleaning**: Clean filters monthly for optimal performance\n"
            response += "3. **Fan Mode**: Use fan mode when full cooling isn't necessary\n"
            response += "4. **Seal Leaks**: Ensure no cool air escapes through windows or doors\n"
            response += "5. **Night Mode**: Use sleep/night mode to gradually increase temperature"
        
        else:
            response += "### General Energy Saving Tips:\n"
            response += "1. **Star Rating**: Always choose higher star-rated appliances\n"
            response += "2. **Right Sizing**: Select appliances appropriate for your needs\n"
            response += "3. **Regular Maintenance**: Keep appliances clean and well-maintained\n"
            response += "4. **Smart Usage**: Use appliances during off-peak hours when possible\n"
            response += "5. **Power Strips**: Use smart power strips to eliminate phantom power consumption"
            
        # Add response to conversation history
        st.session_state['conversation_history'].append(("assistant", response))
        
        # Display the response
        animate_typing(response, st.session_state.follow_up_container)
        
        # Continue the conversation
        st.write("### Is there anything else you'd like to know?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("View products again"):
                st.session_state['follow_up_state'] = 'show_features'
                st.rerun()
        with col2:
            if st.button("Ask another question"):
                st.session_state['follow_up_state'] = None
                st.rerun()
    
    # Handle price results follow-up (common handler for price-related queries)
    elif handler == 'handle_price_results':
        # Set the user question to follow up on price results
        products = st.session_state['last_products']
        product_type = st.session_state.get('last_product_type', 'product')
        user_query = f"Tell me more about these {product_type} options"
        st.session_state['conversation_history'].append(("user", user_query))
        
        # Call the detailed compare function
        # Create a detailed comparison
        response = f"## Detailed Information About These {product_type} Options\n\n"
        
        # Add detailed information for each product
        for _, product in products.iterrows():
            title = product['title']
            price = product['Better Home Price']
            brand = product['Brand']
            url = product.get('url', '#')
            
            response += f"### {title}\n\n"
            response += f"**Brand**: {brand}\n"
            response += f"**Price**: â‚¹{price:,.2f}\n"
            
            # Add any additional information available
            for field in ['power', 'energy_rating', 'capacity', 'weight', 'color', 'warranty']:
                if field in product and str(product[field]) != 'nan':
                    field_title = field.replace('_', ' ').title()
                    response += f"**{field_title}**: {product[field]}\n"
            
            # Add BLDC specific information for fans
            if product_type == 'Ceiling Fan' and ('BLDC' in title or 'Brushless' in title):
                response += "**Energy Savings**: Up to 70% less electricity consumption\n"
                response += "**Special Feature**: Brushless DC motor for efficient operation\n"
            
            # Add inverter specific information for ACs
            if product_type == 'Air Conditioner' and 'Inverter' in title:
                response += "**Special Feature**: Inverter technology for efficient operation\n"
                response += "**Energy Savings**: Variable speed compressor adjusts to cooling needs\n"
            
            # Add buy link
            response += f"**[ğŸ›’ Buy Now]({url})**\n\n"
        
        # Add recommendation based on price
        response += "## Recommendation Based on Price\n\n"
        sorted_by_price = products.sort_values('Better Home Price')
        
        if len(sorted_by_price) >= 3:
            budget = sorted_by_price.iloc[0]
            mid = sorted_by_price.iloc[len(sorted_by_price)//2]
            premium = sorted_by_price.iloc[-1]
            
            budget_url = budget.get('url', '#')
            mid_url = mid.get('url', '#') 
            premium_url = premium.get('url', '#')
            
            response += f"**Most Economical**: {budget['title']} at â‚¹{budget['Better Home Price']:,.2f} - [Buy Now]({budget_url})\n"
            response += f"**Best Value**: {mid['title']} at â‚¹{mid['Better Home Price']:,.2f} - [Buy Now]({mid_url})\n" 
            response += f"**Premium Option**: {premium['title']} at â‚¹{premium['Better Home Price']:,.2f} - [Buy Now]({premium_url})\n\n"
        elif len(sorted_by_price) > 0:
            product = sorted_by_price.iloc[0]
            url = product.get('url', '#')
            response += f"**Recommended Option**: {product['title']} at â‚¹{product['Better Home Price']:,.2f} - [Buy Now]({url})\n\n"
        
        # Add general advice based on product type
        if product_type == 'Ceiling Fan':
            response += "**Tips for Choosing a Fan**:\n"
            response += "- Look for BLDC motors for energy savings\n"
            response += "- Consider air delivery (higher = better cooling)\n"
            response += "- Check warranty period (longer is better)\n"
        elif product_type == 'Air Conditioner':
            response += "**Tips for Choosing an AC**:\n"
            response += "- Select appropriate ton capacity based on room size\n"
            response += "- Higher star rating means more energy efficiency\n"
            response += "- Inverter ACs cost more initially but save electricity long-term\n"
        
        # Add decisive call to action if we've had a few exchanges
        if should_encourage_purchase:
            response += "\n\n### Ready to Purchase?\n\n"
            response += "Our customers who purchased these products also enjoyed:\n"
            response += "- Free extended warranty registration\n"
            response += "- Professional installation service\n"
            response += "- 24/7 customer support\n\n"
            response += "Click the Buy Now links above to complete your purchase!"
        
        # Add response to conversation history
        st.session_state['conversation_history'].append(("assistant", response))
        
        # Display the response
        animate_typing(response, st.session_state.follow_up_container)
        
        # Continue the conversation
        st.write("### What would you like to do next?")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Compare features"):
                st.session_state['follow_up_state'] = 'compare_features'
                st.rerun()
        with col2:
            if st.button("Help me decide"):
                st.session_state['follow_up_state'] = 'help_choose'
                st.rerun()
        with col3:
            if st.button("Ask another question"):
                st.session_state['follow_up_state'] = None
                st.rerun()
    
    # Handle other result types
    elif handler == 'brand_results' or handler == 'search_results' or handler == 'similar_results' or handler == 'best_results':
        # Generic handler for other result types
        products = st.session_state['last_products']
        product_type = st.session_state.get('last_product_type', 'product')
        
        # Safety check for None products
        if products is None or not isinstance(products, pd.DataFrame) or products.empty:
            response = f"I don't have any {product_type} information to show you. Would you like to search for some {product_type}s?"
            st.session_state['conversation_history'].append(("assistant", response))
            animate_typing(response, st.session_state.follow_up_container)
            
            # Reset follow-up state to allow new queries
            if st.button("Start a new search"):
                st.session_state['follow_up_state'] = None
                st.rerun()
            return
        
        user_query = f"I'd like to compare the {product_type} options"
        st.session_state['conversation_history'].append(("user", user_query))
        
        # Create a simple comparison
        response = f"## Comparison of {product_type} Options\n\n"
        
        # Create a comparison table
        comparison_table = "| Brand | Product | Price | Features | Action |\n"
        comparison_table += "| ----- | ------- | ----- | -------- | ------ |\n"
        
        for _, product in products.iterrows():
            title = product['title'][:50] + ('...' if len(product['title']) > 50 else '')
            price = f"â‚¹{product['Better Home Price']:,.2f}"
            brand = product['Brand']
            url = product.get('url', '#')
            
            # Extract features
            features = []
            if 'BLDC' in product['title'] or 'Brushless' in product['title']:
                features.append("Energy-efficient")
            if 'Inverter' in product['title']:
                features.append("Inverter technology")
            if 'Smart' in product['title'] or 'IoT' in product['title']:
                features.append("Smart controls")
            if 'Year' in product['title'] or 'Warranty' in product['title']:
                warranty_match = re.search(r'(\d+)\s*Year', product['title'])
                if warranty_match:
                    features.append(f"{warranty_match.group(1)}yr warranty")
                else:
                    features.append("Warranty")
            
            # Fallback if no features extracted
            if not features:
                features = ["Standard features"]
            
            buy_link = f"[Buy Now]({url})"
            comparison_table += f"| {brand} | {title} | {price} | {', '.join(features)} | {buy_link} |\n"
        
        response += comparison_table + "\n\n"
        
        # Add a simple recommendation
        response += "### Quick Recommendation\n\n"
        sorted_by_price = products.sort_values('Better Home Price')
        lowest_price = sorted_by_price.iloc[0]
        lowest_url = lowest_price.get('url', '#')
        
        response += f"**Best Budget Option**: {lowest_price['title']} at â‚¹{lowest_price['Better Home Price']:,.2f} - [ğŸ›’ Buy Now]({lowest_url})\n\n"
        
        if len(sorted_by_price) > 1:
            highest_price = sorted_by_price.iloc[-1]
            highest_url = highest_price.get('url', '#')
            response += f"**Premium Option**: {highest_price['title']} at â‚¹{highest_price['Better Home Price']:,.2f} - [ğŸ›’ Buy Now]({highest_url})\n\n"
        
        # If we've had several exchanges, add a strong purchase call-to-action
        conversation_turns = len([msg for msg in st.session_state['conversation_history'] if msg[0] == "assistant"])
        if conversation_turns >= 2:
            response += "### ğŸ”¥ Limited Time Offer! ğŸ”¥\n\n"
            if product_type == 'Ceiling Fan':
                response += "- **FREE installation** with any ceiling fan purchase today\n"
                response += "- **Extended warranty** when you register within 7 days\n"
                response += "- **10% off** on your next purchase of any home appliance\n\n"
            elif product_type == 'Air Conditioner':
                response += "- **FREE standard installation** with any AC purchase\n"
                response += "- **Additional 1-year warranty** on compressor\n"
                response += "- **No-cost EMI** available on select cards\n\n"
            else:
                response += "- **FREE delivery** on orders above â‚¹1,000\n"
                response += "- **Easy returns** within 7 days\n"
                response += "- **Installation assistance** available\n\n"
            
            response += "**Don't miss out! Click any Buy Now link to purchase.**"
        
        # Add response to conversation history
        st.session_state['conversation_history'].append(("assistant", response))
        
        # Display the response
        animate_typing(response, st.session_state.follow_up_container)
        
        # Continue the conversation
        st.write("### Would you like to know more?")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Detailed comparison"):
                st.session_state['follow_up_state'] = 'compare_features'
                st.rerun()
        with col2:
            if st.button("Help me choose"):
                st.session_state['follow_up_state'] = 'help_choose'
                st.rerun()
        with col3:
            if st.button("Ask another question"):
                st.session_state['follow_up_state'] = None
                st.rerun()
    
    else:
        # Default handler for other states or unrecognized states
        st.write("Let me know if you have any other questions about home products!")
        
        # Add a generic response to conversation history
        if st.session_state['conversation_history'] and st.session_state['conversation_history'][-1][0] == "user":
            st.session_state['conversation_history'].append(("assistant", "I'm not sure how to help with that specific follow-up. Please ask another question."))
        
        if st.button("Start a new conversation"):
            st.session_state['follow_up_state'] = None
            st.rerun()

def load_blog_embeddings(file_path):
    """
    Load blog embeddings from a JSON file
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        return {
            'blog_embeddings': np.array(data.get('blog_embeddings', [])),
            'metadata': data.get('metadata', [])
        }
    except Exception as e:
        print(f"Error loading blog embeddings: {str(e)}")
        return {
            'blog_embeddings': np.array([]),
            'metadata': []
        }

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

if __name__ == "__main__":
    main() 