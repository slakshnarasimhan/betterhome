"""
Shared data utilities for the BetterHome project.
Contains common functions for loading and processing data files, embeddings, and configurations.
"""

import pandas as pd
import numpy as np
import json
import yaml
import faiss
import os
from typing import Dict, Any, List, Optional, Union


def load_json_file(file_path: str, default: Any = None) -> Any:
    """
    Load a JSON file with error handling.
    
    Args:
        file_path: Path to the JSON file
        default: Default value to return if file not found or invalid
    
    Returns:
        Loaded JSON data or default value
    """
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return default
    except json.JSONDecodeError as e:
        print(f"Invalid JSON in {file_path}: {e}")
        return default
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return default


def save_json_file(data: Any, file_path: str) -> bool:
    """
    Save data to a JSON file with error handling.
    
    Args:
        data: Data to save
        file_path: Path to save the JSON file
    
    Returns:
        True if successful, False otherwise
    """
    try:
        with open(file_path, 'w') as f:
            json.dump(data, f)
        print(f"Successfully saved data to {file_path}")
        return True
    except Exception as e:
        print(f"Error saving to {file_path}: {e}")
        return False


def load_yaml_file(file_path: str, default: Any = None) -> Any:
    """
    Load a YAML file with error handling.
    
    Args:
        file_path: Path to the YAML file
        default: Default value to return if file not found or invalid
    
    Returns:
        Loaded YAML data or default value
    """
    try:
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Configuration file not found: {file_path}")
        return default
    except yaml.YAMLError as e:
        print(f"Invalid YAML in {file_path}: {e}")
        return default
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return default


def load_product_catalog(file_path: str = 'cleaned_products.csv') -> Optional[pd.DataFrame]:
    """
    Load and preprocess the product catalog.
    
    Args:
        file_path: Path to the CSV file
    
    Returns:
        DataFrame with product catalog or None if error
    """
    try:
        df = pd.read_csv(file_path)
        
        # Convert text columns to string type
        text_columns = ['title', 'Product Type', 'Category', 'tags', 'SKU', 'Description', 
                       'Brand', 'Material', 'Warranty', 'Features', 'url']
        
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].astype(str)
        
        # Convert numeric columns to float
        numeric_columns = ['Better Home Price', 'Retail Price', 'Weight']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        print(f"Successfully loaded product catalog with {len(df)} entries.")
        return df
    except FileNotFoundError:
        print(f"Product catalog file not found: {file_path}")
        return None
    except Exception as e:
        print(f"Error loading product catalog: {str(e)}")
        return None


def load_product_catalog_json(file_path: str = 'product_catalog.json') -> Dict[str, Any]:
    """
    Load product catalog from JSON file.
    
    Args:
        file_path: Path to the JSON file
    
    Returns:
        Product catalog dictionary
    """
    return load_json_file(file_path, {})


def load_embeddings(file_path: str) -> Dict[str, Any]:
    """
    Load embeddings from JSON file.
    
    Args:
        file_path: Path to the embeddings JSON file
    
    Returns:
        Dictionary containing embeddings and metadata
    """
    data = load_json_file(file_path, {})
    
    if not data:
        return {}
    
    try:
        return {
            'product_embeddings': np.array(data.get('product_embeddings', [])),
            'product_type_embeddings': np.array(data.get('product_type_embeddings', [])),
            'brand_embeddings': np.array(data.get('brand_embeddings', [])),
            'metadata': data.get('metadata', {})
        }
    except Exception as e:
        print(f"Error processing embeddings: {e}")
        return {}


def load_product_terms(file_path: str = 'product_terms.json') -> Dict[str, Any]:
    """
    Load product terms dictionary.
    
    Args:
        file_path: Path to product terms JSON file
    
    Returns:
        Product terms dictionary
    """
    return load_json_file(file_path, {})


def load_faiss_index(index_path: str) -> Optional[faiss.Index]:
    """
    Load a FAISS index from disk.
    
    Args:
        index_path: Path to the index file
    
    Returns:
        FAISS index or None if error
    """
    try:
        if os.path.exists(index_path):
            return faiss.read_index(index_path)
        else:
            print(f"FAISS index file not found: {index_path}")
            return None
    except Exception as e:
        print(f"Error loading FAISS index from {index_path}: {e}")
        return None


def save_faiss_index(index: faiss.Index, index_path: str) -> bool:
    """
    Save a FAISS index to disk.
    
    Args:
        index: FAISS index to save
        index_path: Path to save the index
    
    Returns:
        True if successful, False otherwise
    """
    try:
        faiss.write_index(index, index_path)
        print(f"FAISS index saved successfully at {index_path}")
        return True
    except Exception as e:
        print(f"Error saving FAISS index: {e}")
        return False


def build_faiss_index(embeddings: np.ndarray, index_path: str = None) -> Optional[faiss.Index]:
    """
    Build and optionally save a FAISS index from embeddings.
    
    Args:
        embeddings: Numpy array of embeddings
        index_path: Optional path to save the index
    
    Returns:
        Built FAISS index or None if error
    """
    if len(embeddings) == 0:
        print("Error: No embeddings to build the index.")
        return None
    
    try:
        dimension = embeddings.shape[1] if len(embeddings.shape) > 1 else len(embeddings[0])
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings.astype('float32'))
        
        if index_path:
            save_faiss_index(index, index_path)
        
        print(f"FAISS index built with {len(embeddings)} embeddings, dimension: {dimension}")
        return index
    except Exception as e:
        print(f"Error building FAISS index: {e}")
        return None


def safe_int(val: Any, default: int = 0) -> int:
    """
    Safely convert value to integer.
    
    Args:
        val: Value to convert
        default: Default value if conversion fails
    
    Returns:
        Integer value or default
    """
    try:
        if pd.isna(val):
            return default
        return int(val)
    except (ValueError, TypeError):
        return default


def safe_float(val: Any, default: float = 0.0) -> float:
    """
    Safely convert value to float.
    
    Args:
        val: Value to convert
        default: Default value if conversion fails
    
    Returns:
        Float value or default
    """
    try:
        if pd.isna(val):
            return default
        return float(val)
    except (ValueError, TypeError):
        return default


def safe_str(val: Any, default: str = "") -> str:
    """
    Safely convert value to string.
    
    Args:
        val: Value to convert
        default: Default value if conversion fails
    
    Returns:
        String value or default
    """
    try:
        if pd.isna(val):
            return default
        return str(val).strip()
    except (ValueError, TypeError):
        return default


def format_currency(amount: Union[int, float], currency_symbol: str = "₹") -> str:
    """
    Format amount as currency.
    
    Args:
        amount: Amount to format
        currency_symbol: Currency symbol to use
    
    Returns:
        Formatted currency string
    """
    try:
        return f"{currency_symbol}{float(amount):,.2f}"
    except (ValueError, TypeError):
        return f"{currency_symbol}0.00"