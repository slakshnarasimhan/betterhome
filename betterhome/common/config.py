"""
Configuration module for Better Home application.

This module contains shared configuration variables used by both the API server
and the Streamlit application.
"""

import os
import json
import yaml
from typing import Dict, Any, Optional

# File paths
CSV_FILE_PATH = 'cleaned_products.csv'
EMBEDDINGS_FILE_PATH = 'embeddings.json'
PRODUCT_INDEX_FILE_PATH = 'faiss_index.index_product'
PRODUCT_TERMS_FILE = 'product_terms.json'
HOME_CONFIG_FILE = 'home_config.yaml'
BLOG_EMBEDDINGS_FILE_PATH = 'blog_embeddings.json'
BLOG_INDEX_FILE_PATH = 'faiss_index.index_blog'

# OpenAI configuration
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("Warning: OPENAI_API_KEY environment variable not set")

# Load product terms dictionary
def load_product_terms(file_path: str = PRODUCT_TERMS_FILE) -> Dict[str, Any]:
    """
    Load product terms from JSON file.
    
    Args:
        file_path: Path to the product terms JSON file
        
    Returns:
        Dictionary containing product terms
    """
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading product terms: {str(e)}")
        return {}

# Load home configuration
def load_home_config(file_path: str = HOME_CONFIG_FILE) -> Optional[Dict[str, Any]]:
    """
    Load home configuration from YAML file.
    
    Args:
        file_path: Path to the home configuration YAML file
        
    Returns:
        Dictionary containing home configuration or None if loading fails
    """
    try:
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading home configuration: {str(e)}")
        return None 