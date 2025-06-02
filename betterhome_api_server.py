from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
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
from typing import List, Dict, Any
from datetime import datetime

# Custom JSON encoder to handle non-serializable values
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if pd.isna(obj) or obj is None:
            return None
        if isinstance(obj, (np.integer, np.floating)):
            if not np.isfinite(obj):
                return None
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        if isinstance(obj, pd.Series):
            return obj.to_dict()
        return super().default(obj)

# Import common modules
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
from betterhome.common.embeddings import (
    get_query_embedding,
    load_embeddings,
    build_or_load_faiss_index,
    search_products
)
from betterhome.common.product_utils import (
    find_product_type,
    handle_price_query,
    handle_brand_query,
    format_brand_response,
    format_product_response,
    search_catalog,
    format_answer
)
from betterhome.common.blog_utils import is_how_to_query

app = FastAPI()

# Configure FastAPI to use our custom JSON encoder
app.json_encoder = CustomJSONEncoder

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize OpenAI client
openai.api_key = os.environ.get("OPENAI_API_KEY")
if not openai.api_key:
    print("Warning: OPENAI_API_KEY environment variable not set")

# ====== Load Data and Models =======
df = pd.read_csv(CSV_FILE_PATH)
embedding_data = load_embeddings(EMBEDDINGS_FILE_PATH)
product_terms = load_product_terms(PRODUCT_TERMS_FILE)
home_config = load_home_config(HOME_CONFIG_FILE)

# Ensure embeddings and DataFrame have the same length
product_embeddings = np.array(embedding_data['product_embeddings']).astype('float32')
if len(product_embeddings) != len(df):
    print(f"Warning: Mismatch between embeddings ({len(product_embeddings)}) and DataFrame ({len(df)})")
    # Truncate to the smaller size
    min_size = min(len(product_embeddings), len(df))
    product_embeddings = product_embeddings[:min_size]
    df = df.iloc[:min_size].reset_index(drop=True)
    print(f"Truncated to {min_size} entries")

# Create FAISS index
index = faiss.IndexFlatL2(product_embeddings.shape[1])
index.add(product_embeddings)
print(f"Created FAISS index with {index.ntotal} vectors of dimension {index.d}")

# ====== Helper: Create embedding for query =======
def get_query_embedding(text):
    try:
        response = openai.Embedding.create(
            model="text-embedding-3-small",
            input=text
        )
        if 'data' in response and len(response['data']) > 0:
            return np.array(response['data'][0]['embedding'], dtype='float32')
        else:
            print(f"[Embedding Error] No data in response")
            return np.random.rand(index.d).astype('float32')  # fallback
    except Exception as e:
        print(f"[Embedding Error] {e}")
        return np.random.rand(index.d).astype('float32')  # fallback

# ====== Helper: Find product type =======
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

# ====== Helper: Handle price queries =======
def handle_price_query(query, df):
    try:
        if not query or not isinstance(query, str):
            print("[Price Query Error] Invalid query input")
            return []
            
        if df is None or df.empty:
            print("[Price Query Error] Invalid DataFrame input")
            return []
            
        price_range = extract_price_range(query)
        if not price_range:
            return []
            
        min_price, max_price = price_range
        mask = (df['Better Home Price'].notna() & 
                df['Better Home Price'].apply(lambda x: np.isfinite(float(x)) if isinstance(x, (int, float)) else False) &
                (df['Better Home Price'] >= min_price) & 
                (df['Better Home Price'] <= max_price))
        
        products = df[mask]
        if len(products) == 0:
            return []
            
        results = []
        for _, row in products.iterrows():
            product = {}
            for col in df.columns:
                val = row[col]
                if pd.isna(val):
                    product[col] = None
                elif isinstance(val, (np.integer, np.floating)):
                    product[col] = float(val) if np.isfinite(val) else None
                else:
                    product[col] = str(val)
            results.append(product)
            
        return results
    except Exception as e:
        print(f"[Price Query Error] {str(e)}")
        return []

# ====== Helper: Handle brand queries =======
def handle_brand_query(query, df):
    try:
        if not query or not isinstance(query, str):
            print("[Brand Query Error] Invalid query input")
            return {}
            
        if df is None or df.empty:
            print("[Brand Query Error] Invalid DataFrame input")
            return {}
            
        brand_name = extract_brand_name(query)
        if not brand_name:
            return {}
            
        mask = df['Brand'].str.lower() == brand_name.lower()
        products = df[mask]
        if len(products) == 0:
            return {}
            
        try:
            warranty_info = get_warranty_info(brand_name)
        except Exception as e:
            print(f"[Brand Query Error] Failed to get warranty info: {str(e)}")
            warranty_info = None
            
        product_list = []
        for _, row in products.iterrows():
            product = {}
            for col in df.columns:
                val = row[col]
                if pd.isna(val):
                    product[col] = None
                elif isinstance(val, (np.integer, np.floating)):
                    product[col] = float(val) if np.isfinite(val) else None
                else:
                    product[col] = str(val)
            product_list.append(product)
            
        return {
            'brand': brand_name,
            'warranty_info': warranty_info,
            'products': product_list
        }
    except Exception as e:
        print(f"[Brand Query Error] {str(e)}")
        return {}

# ====== Helper: Clean float values for JSON =======
def clean_float_for_json(value):
    """
    Clean float values to ensure they are JSON compliant.
    Returns None for non-finite values (inf, -inf, nan).
    """
    if value is None:
        return None
        
    if isinstance(value, float):
        if not np.isfinite(value):
            return None
        return value
        
    if isinstance(value, str):
        try:
            float_val = float(value)
            if not np.isfinite(float_val):
                return None
            return float_val
        except (ValueError, TypeError):
            return None
            
    return value

# ====== Helper: Format brand response =======
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
        price = clean_float_for_json(product['Better Home Price'])
        retail_price = clean_float_for_json(product.get('Retail Price', 0))
        url = product.get('url', '#')
        
        # Calculate discount percentage if retail price is available
        if retail_price is not None and retail_price > 0 and price is not None:
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
                warranty_text = f"â­� {warranty_years} Year Warranty\n"
            elif 'warranty' in title.lower():
                warranty_text = "â­� Includes Warranty\n"
            elif 'year' in title.lower() and re.search(r'(\d+)\s+years?', title, re.IGNORECASE):
                warranty_years = re.search(r'(\d+)\s+years?', title, re.IGNORECASE).group(1)
                warranty_text = f"â­� {warranty_years} Year Guarantee\n"
        
        # More concise brand listing
        response += f"**{brand}** {discount_text}\n"
        if price is not None:
            response += f"â‚¹{price:,.2f}\n"
        if warranty_text:
            response += warranty_text
        # Make the buy link more prominent
        response += f"ğŸ›’ [Buy Now]({url})\n\n"
    
    # Add a note about clicking the links
    response += "*Click on 'Buy Now' to purchase the product.*\n"
    
    return response

# ====== Helper: Format product response =======
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
        price = clean_float_for_json(product['Better Home Price'])
        retail_price = clean_float_for_json(product.get('Retail Price', 0))
        url = product.get('url', '#')
        
        # Calculate discount percentage if retail price is available
        if retail_price is not None and retail_price > 0 and price is not None:
            discount = ((retail_price - price) / retail_price) * 100
            discount_text = f"({discount:.1f}% off)"
        else:
            discount_text = ""
        
        # Add a special highlight for BLDC fans
        is_bldc = 'BLDC' in title or 'Brushless' in title
        energy_label = "ğŸ’š " if is_bldc else ""
        
        # More concise product listing
        response += f"**{energy_label}{title}**\n"
        if price is not None:
            response += f"â‚¹{price:,.2f} {discount_text}\n"
        
        # Add energy saving information for BLDC fans
        if is_bldc:
            response += f"70% energy savings\n"
        
        # Make the buy link more prominent
        response += f"ğŸ›’ [Buy Now]({url})\n\n"
    
    # Add a note about clicking the links
    response += "*Click on 'Buy Now' to purchase the product.*\n"
    
    return response

# ====== Helper: Retrieve top product entries =======
def search_catalog(query, top_k=5):
    """
    Search the product catalog using FAISS index.
    Returns a list of product dictionaries matching the query.
    """
    try:
        # Validate inputs
        if not query or not isinstance(query, str):
            print("[Search Error] Invalid query input")
            return []
            
        if not isinstance(top_k, int) or top_k < 1:
            print("[Search Error] Invalid top_k value")
            return []
            
        # Check if index and DataFrame are initialized
        if not hasattr(search_catalog, 'index') or search_catalog.index is None:
            print("[Search Error] FAISS index not initialized")
            return []
            
        if not hasattr(search_catalog, 'df') or search_catalog.df is None:
            print("[Search Error] DataFrame not initialized")
            return []
            
        # Get query embedding
        try:
            q_emb = get_query_embedding(query).reshape(1, -1)
        except Exception as e:
            print(f"[Search Error] Failed to get query embedding: {str(e)}")
            return []
        
        # Get the dimension of the FAISS index
        index_dim = search_catalog.index.d
        
        # Ensure query embedding matches index dimension
        if q_emb.shape[1] != index_dim:
            print(f"[Dimension Mismatch] Query dim: {q_emb.shape[1]}, Index dim: {index_dim}")
            try:
                # Pad or truncate the query embedding to match index dimension
                if q_emb.shape[1] > index_dim:
                    q_emb = q_emb[:, :index_dim]
                else:
                    padding = np.zeros((1, index_dim - q_emb.shape[1]), dtype='float32')
                    q_emb = np.hstack((q_emb, padding))
            except Exception as e:
                print(f"[Search Error] Failed to adjust embedding dimensions: {str(e)}")
                return []
        
        # Perform the search with a safe top_k value
        safe_top_k = min(top_k, search_catalog.index.ntotal)
        if safe_top_k == 0:
            print("[Search Error] No vectors in the index")
            return []
            
        try:
            D, I = search_catalog.index.search(q_emb, safe_top_k)
        except Exception as e:
            print(f"[Search Error] FAISS search failed: {str(e)}")
            return []
        
        # Handle case where no results are found
        if len(I[0]) == 0:
            print("[Search Error] No results found")
            return []
            
        # Get the results, ensuring indices are within bounds
        valid_indices = []
        for i in I[0]:
            if 0 <= i < len(search_catalog.df):
                valid_indices.append(i)
            else:
                print(f"[Search Warning] Index {i} is out of bounds for DataFrame of length {len(search_catalog.df)}")
        
        if not valid_indices:
            print("[Search Error] No valid indices found")
            return []
            
        # Create a list to store valid results
        results = []
        for idx in valid_indices:
            try:
                product = search_catalog.df.iloc[idx].to_dict()
                
                # Clean all values in the product dictionary
                cleaned_product = {}
                for key, value in product.items():
                    if pd.isna(value):
                        cleaned_product[key] = None
                    elif isinstance(value, (np.integer, np.floating)):
                        if not np.isfinite(value):
                            cleaned_product[key] = None
                        else:
                            cleaned_product[key] = float(value)
                    else:
                        cleaned_product[key] = value
                
                # Validate required fields
                if all(k in cleaned_product for k in ['title', 'Brand']):
                    results.append(cleaned_product)
                else:
                    print(f"[Search Warning] Product at index {idx} missing required fields")
            except Exception as e:
                print(f"[Search Error] Error accessing product at index {idx}: {str(e)}")
                continue
        
        return results
        
    except Exception as e:
        print(f"[Search Error] Unexpected error in search_catalog: {str(e)}")
        traceback.print_exc()
        return []

# ====== Helper: Format response =======
def format_answer(products, query):
    if not products:
        return f"I couldn't find any products matching your query: **{query}**. Please try rephrasing your question or check the product name."
    
    try:
        response = f"Found {len(products)} products matching your query: **{query}**\n\n"
        for p in products:
            try:
                title = p.get('title', 'N/A')
                price = clean_float_for_json(p.get('Better Home Price', 'N/A'))
                retail = clean_float_for_json(p.get('Retail Price', 'N/A'))
                brand = p.get('Brand', 'N/A')
                product_type = p.get('Product Type', 'N/A')
                category = p.get('Category', 'N/A')
                description = p.get('Description', 'N/A')
                sku = p.get('SKU', 'N/A')
                url = p.get('url', '#')
                
                response += f"### {title}\n"
                response += f"- SKU: {sku}\n"
                response += f"- Brand: {brand}\n"
                response += f"- Product Type: {product_type}\n"
                response += f"- Category: {category}\n"
                if price is not None:
                    response += f"- Better Home Price: â‚¹{price}\n"
                if retail is not None:
                    response += f"- Retail Price: â‚¹{retail}\n"
                response += f"- Description: {description}\n"
                response += f"[Click here to buy]({url})\n\n"
            except Exception as e:
                print(f"[Format Error] Error formatting product: {str(e)}")
                continue
        return response
    except Exception as e:
        print(f"[Format Error] Error in format_answer: {str(e)}")
        return f"I found some products matching your query: **{query}**, but encountered an error while formatting the response. Please try again."

# ====== API Model =======
class QueryInput(BaseModel):
    query: str

# ====== API Endpoint =======
@app.post("/api/ask")
async def ask_question(payload: QueryInput):
    try:
        # Validate input
        if not payload.query or not isinstance(payload.query, str):
            return {
                "answer": "Please provide a valid query string.",
                "products": []
            }
            
        query = payload.query.lower().strip()
        
        # Check if it's a how-to question
        if is_how_to_query(query):
            return {
                "answer": "I understand you're asking about how to do something. This feature is coming soon!",
                "products": []
            }
        
        # Handle price queries
        if any(word in query for word in ['price', 'cost', 'expensive', 'cheap', 'budget']):
            try:
                products = handle_price_query(query, df)
                if products:
                    formatted_response = format_product_response(pd.DataFrame(products))
                    if formatted_response:
                        return {
                            "answer": formatted_response,
                            "products": products
                        }
            except Exception as e:
                print(f"[Price Query Error] {str(e)}")
                traceback.print_exc()
            
            return {
                "answer": f"I couldn't find any price information for: **{query}**. Please try rephrasing your question.",
                "products": []
            }
        
        # Handle brand queries
        if any(word in query for word in ['brand', 'warranty', 'company', 'manufacturer']):
            try:
                brand_result = handle_brand_query(query, df)
                if brand_result and 'products' in brand_result:
                    products = brand_result['products']
                    if products:
                        formatted_response = format_brand_response(pd.DataFrame(products), brand_result['brand'], brand_result.get('warranty_info', False))
                        if formatted_response:
                            return {
                                "answer": formatted_response,
                                "products": products
                            }
            except Exception as e:
                print(f"[Brand Query Error] {str(e)}")
                traceback.print_exc()
            
            return {
                "answer": f"I couldn't find any brand information for: **{query}**. Please try rephrasing your question.",
                "products": []
            }
        
        # Handle general product search
        try:
            products = search_catalog(query)
            if products:
                formatted_response = format_answer(products, query)
                if formatted_response:
                    return {
                        "answer": formatted_response,
                        "products": products
                    }
        except Exception as e:
            print(f"[Search Error] {str(e)}")
            traceback.print_exc()
        
        return {
            "answer": f"I couldn't find any products matching your query: **{query}**. Please try rephrasing your question.",
            "products": []
        }
        
    except Exception as e:
        print(f"[API Error] Unexpected error in ask_question: {str(e)}")
        traceback.print_exc()
        return {
            "answer": "I apologize, but I encountered an error while processing your request. Please try again.",
            "products": []
        }

# Initialize search_catalog with index and DataFrame
search_catalog.index = index
search_catalog.df = df

