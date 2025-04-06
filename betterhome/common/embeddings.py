"""
Embeddings module for Better Home application.

This module contains functions for generating embeddings and performing
FAISS operations.
"""

import numpy as np
import faiss
import json
import openai
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Union
import os

from betterhome.common.config import OPENAI_API_KEY, EMBEDDINGS_FILE_PATH

# Set OpenAI API key
openai.api_key = OPENAI_API_KEY

def get_query_embedding(text: str, model: str = "text-embedding-3-small") -> np.ndarray:
    """
    Generate embedding for a query using OpenAI API.
    
    Args:
        text: Text to generate embedding for
        model: OpenAI model to use
        
    Returns:
        NumPy array containing the embedding
    """
    try:
        response = openai.Embedding.create(
            model=model,
            input=text
        )
        if 'data' in response and len(response['data']) > 0:
            return np.array(response['data'][0]['embedding'], dtype='float32')
        else:
            print(f"[Embedding Error] No data in response")
            return np.random.rand(1536).astype('float32')  # fallback
    except Exception as e:
        print(f"[Embedding Error] {e}")
        return np.random.rand(1536).astype('float32')  # fallback

def load_embeddings(file_path: str = EMBEDDINGS_FILE_PATH) -> Dict[str, Any]:
    """
    Load embeddings from JSON file.
    
    Args:
        file_path: Path to the embeddings JSON file
        
    Returns:
        Dictionary containing embeddings
    """
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

def build_or_load_faiss_index(embeddings: np.ndarray, dimension: int, index_path: str) -> faiss.Index:
    """
    Build or load FAISS index.
    
    Args:
        embeddings: NumPy array containing embeddings
        dimension: Dimension of the embeddings
        index_path: Path to save/load the FAISS index
        
    Returns:
        FAISS index
    """
    if os.path.exists(index_path):
        index = faiss.read_index(index_path)
    else:
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        faiss.write_index(index, index_path)
    return index

def search_products(query: str, df: pd.DataFrame, embeddings_dict: Dict[str, Any], k: int = 5) -> List[int]:
    """
    Search for products using FAISS.
    
    Args:
        query: Query text
        df: DataFrame containing product data
        embeddings_dict: Dictionary containing embeddings
        k: Number of results to return
        
    Returns:
        List of indices of top k matches
    """
    # Generate query embedding
    query_embedding = get_query_embedding(query)
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