"""
Shared embedding utilities for the BetterHome project.
Contains functions for generating embeddings using different providers and managing embedding operations.
"""

import numpy as np
import openai
import requests
import os
from typing import List, Optional, Dict, Any
from tqdm import tqdm
import time


class EmbeddingProvider:
    """Base class for embedding providers."""
    
    def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding for a single text."""
        raise NotImplementedError
    
    def get_batch_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Get embeddings for a batch of texts."""
        return [self.get_embedding(text) for text in texts]


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embedding provider."""
    
    def __init__(self, api_key: str = None, model: str = "text-embedding-3-small"):
        """
        Initialize OpenAI embedding provider.
        
        Args:
            api_key: OpenAI API key (if None, will use environment variable)
            model: OpenAI model to use for embeddings
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        openai.api_key = self.api_key
    
    def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Get embedding for a single text using OpenAI API.
        
        Args:
            text: Text to embed
        
        Returns:
            Embedding vector or None if error
        """
        try:
            response = openai.Embedding.create(
                model=self.model,
                input=text
            )
            if 'data' in response and len(response['data']) > 0:
                return np.array(response['data'][0]['embedding'])
            return None
        except Exception as e:
            print(f"Error generating OpenAI embedding: {e}")
            return None
    
    def get_batch_embeddings(self, texts: List[str], batch_size: int = 10, max_retries: int = 3) -> List[Optional[np.ndarray]]:
        """
        Get embeddings for a batch of texts with retry logic.
        
        Args:
            texts: List of texts to embed
            batch_size: Size of each batch
            max_retries: Maximum number of retries for failed batches
        
        Returns:
            List of embedding vectors (None for failed embeddings)
        """
        embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            for retry in range(max_retries):
                try:
                    print(f"Processing batch {i//batch_size + 1}/{len(texts)//batch_size + 1}, attempt {retry + 1}")
                    
                    batch_embeddings = []
                    for text in batch:
                        embedding = self.get_embedding(text)
                        if embedding is not None:
                            batch_embeddings.append(embedding)
                        else:
                            batch_embeddings.append(None)
                    
                    embeddings.extend(batch_embeddings)
                    break
                    
                except Exception as e:
                    print(f"Error in batch processing (attempt {retry + 1}): {e}")
                    if retry < max_retries - 1:
                        time.sleep(2 ** retry)  # Exponential backoff
                    else:
                        # Add None for failed batch
                        embeddings.extend([None] * len(batch))
        
        return embeddings


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Ollama local embedding provider."""
    
    def __init__(self, url: str = "http://localhost:11434/api/embeddings", model: str = "llama2"):
        """
        Initialize Ollama embedding provider.
        
        Args:
            url: Ollama API endpoint URL
            model: Model name to use
        """
        self.url = url
        self.model = model
    
    def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Get embedding for a single text using Ollama API.
        
        Args:
            text: Text to embed
        
        Returns:
            Embedding vector or None if error
        """
        try:
            response = requests.post(
                self.url,
                json={
                    "model": self.model,
                    "prompt": text
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                embedding = data.get('embedding', [])
                if embedding:
                    return np.array(embedding)
            
            print(f"Ollama API error: {response.status_code}")
            return None
            
        except Exception as e:
            print(f"Error generating Ollama embedding: {e}")
            return None


def create_embedding_provider(provider_type: str = "openai", **kwargs) -> EmbeddingProvider:
    """
    Factory function to create embedding providers.
    
    Args:
        provider_type: Type of provider ("openai" or "ollama")
        **kwargs: Additional arguments for provider initialization
    
    Returns:
        Embedding provider instance
    """
    if provider_type.lower() == "openai":
        return OpenAIEmbeddingProvider(**kwargs)
    elif provider_type.lower() == "ollama":
        return OllamaEmbeddingProvider(**kwargs)
    else:
        raise ValueError(f"Unsupported provider type: {provider_type}")


def prepare_product_entries(df, include_enhanced_features: bool = True) -> List[str]:
    """
    Prepare product entries for embedding generation.
    
    Args:
        df: Product catalog DataFrame
        include_enhanced_features: Whether to include enhanced feature extraction
    
    Returns:
        List of formatted product entries
    """
    entries = []
    
    for _, row in df.iterrows():
        # Calculate discount information
        if pd.notnull(row.get('Retail Price')) and row['Retail Price'] > 0:
            discount_percentage = ((row['Retail Price'] - row['Better Home Price']) / row['Retail Price']) * 100
            discount_text = f"Better Home Price is {discount_percentage:.2f}% less than Retail Price."
        else:
            discount_text = "No discount available."

        # Extract basic product information
        product_type = row.get('Product Type', 'Not Available')
        brand = row.get('Brand', 'Not Available')
        title = row.get('title', 'Not Available')
        price = row.get('Better Home Price', 'Not Available')
        retail_price = row.get('Retail Price', 'Not Available')
        warranty = row.get('Warranty', 'Not Available')
        features = row.get('Features', 'Not Available')
        description = row.get('Description', 'Not Available')
        url = row.get('url', 'Not Available')
        
        # Enhanced feature extraction (if enabled)
        enhanced_features = ""
        if include_enhanced_features:
            # Age-appropriate features
            age_features = ""
            if 'fan' in product_type.lower():
                if 'BLDC' in title or 'Brushless' in title:
                    age_features += "Energy efficient, suitable for all ages. "
                if 'remote' in description.lower() or 'remote' in title.lower():
                    age_features += "Remote control, convenient for elderly users. "
                if 'child' in description.lower() or 'child' in title.lower():
                    age_features += "Child-safe design. "
            
            # Room-specific features
            room_features = ""
            if 'kitchen' in description.lower() or 'kitchen' in title.lower():
                room_features += "Kitchen-friendly. "
            if 'bedroom' in description.lower() or 'bedroom' in title.lower():
                room_features += "Bedroom-optimized. "
            if 'living room' in description.lower() or 'living room' in title.lower():
                room_features += "Living room suitable. "
            
            # Energy efficiency features
            energy_features = ""
            if 'energy' in description.lower() or 'efficient' in description.lower():
                energy_features += "Energy efficient. "
            if 'power' in description.lower() and 'low' in description.lower():
                energy_features += "Low power consumption. "
            
            enhanced_features = f"Age Features: {age_features} Room Features: {room_features} Energy Features: {energy_features} "

        # Create main product entry
        entry = (
            f"Product Type: {product_type}. "
            f"Brand: {brand}. "
            f"Title: {title}. "
            f"Better Home Price: {price} INR. "
            f"Retail Price: {retail_price} INR. "
            f"{discount_text} "
            f"Warranty: {warranty}. "
            f"Features: {features}. "
            f"{enhanced_features}"
            f"Description: {description}. "
            f"Product URL: {url}."
        )
        entries.append(entry)

    return entries


def generate_embeddings_batch(texts: List[str], provider: EmbeddingProvider, batch_size: int = 10) -> List[Optional[np.ndarray]]:
    """
    Generate embeddings for a list of texts using a provider.
    
    Args:
        texts: List of texts to embed
        provider: Embedding provider to use
        batch_size: Batch size for processing
    
    Returns:
        List of embedding vectors
    """
    if not texts:
        print("Error: No texts provided for embedding generation.")
        return []
    
    print(f"Generating embeddings for {len(texts)} texts...")
    embeddings = provider.get_batch_embeddings(texts, batch_size)
    
    # Filter out None embeddings and report statistics
    valid_embeddings = [e for e in embeddings if e is not None]
    print(f"Successfully generated {len(valid_embeddings)}/{len(texts)} embeddings.")
    
    if valid_embeddings:
        print(f"Embedding dimension: {valid_embeddings[0].shape}")
    
    return embeddings


def validate_embeddings(embeddings: List[np.ndarray], expected_dim: int = None) -> bool:
    """
    Validate a list of embeddings.
    
    Args:
        embeddings: List of embedding vectors
        expected_dim: Expected dimension (if None, uses first embedding's dimension)
    
    Returns:
        True if all embeddings are valid
    """
    if not embeddings:
        print("No embeddings to validate")
        return False
    
    # Get expected dimension from first embedding if not provided
    if expected_dim is None:
        expected_dim = len(embeddings[0])
    
    # Check all embeddings have the same dimension
    for i, embedding in enumerate(embeddings):
        if embedding is None:
            print(f"Embedding {i} is None")
            return False
        if len(embedding) != expected_dim:
            print(f"Embedding {i} has dimension {len(embedding)}, expected {expected_dim}")
            return False
    
    print(f"All {len(embeddings)} embeddings are valid with dimension {expected_dim}")
    return True