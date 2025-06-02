import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import streamlit as st
from concurrent.futures import ThreadPoolExecutor
import faiss
import time
import os
import openai
import requests

# ==========================
# Configuration
# ==========================
#CSV_FILE_PATH = 'cleaned_products.csv'
CSV_FILE_PATH = 'cleaned_products_10.csv'
EMBEDDINGS_FILE_PATH = 'embeddings.json'
INDEX_FILE_PATH = 'faiss_index.index'

# Embedding provider configuration
USE_OPENAI = True  # Set to False to use Ollama instead
OLLAMA_URL = "http://localhost:11434/api/embeddings"
OLLAMA_MODEL = "llama2"

# Expected embedding dimensions
OPENAI_EMBEDDING_DIM = 1536  # text-embedding-3-small has 1536 dimensions
OLLAMA_EMBEDDING_DIM = 4096  # llama2 has 4096 dimensions

# Initialize OpenAI client if using OpenAI
if USE_OPENAI:
    openai.api_key = os.environ.get("OPENAI_API_KEY")
    # Set the API base URL if needed
    # openai.api_base = "https://api.openai.com/v1"

# ==========================
# Step 1: Load Product Catalog
# ==========================
def load_product_catalog(file_path):
    """Load and preprocess the product catalog."""
    try:
        # Read the CSV file
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
    except Exception as e:
        print(f"Error loading product catalog: {str(e)}")
        return None

# ==========================
# Step 2: Prepare Entries (Enhanced with Product Type and Brand)
# ==========================
def prepare_entries(df):
    entries = []
    product_type_entries = []
    brand_entries = []
    
    for _, row in df.iterrows():
        if pd.notnull(row.get('Retail Price')) and row['Retail Price'] > 0:
            discount_percentage = ((row['Retail Price'] - row['Better Home Price']) / row['Retail Price']) * 100
            discount_text = f"Better Home Price is {discount_percentage:.2f}% less than Retail Price."
        else:
            discount_text = "No discount available."

        # Extract key features for recommendation
        product_type = row.get('Product Type', 'Not Available')
        brand = row.get('Brand', 'Not Available')
        title = row.get('title', 'Not Available')
        price = row.get('Better Home Price', 'Not Available')
        retail_price = row.get('Retail Price', 'Not Available')
        warranty = row.get('Warranty', 'Not Available')
        features = row.get('Features', 'Not Available')
        description = row.get('Description', 'Not Available')
        url = row.get('url', 'Not Available')
        
        # Extract age-appropriate features
        age_features = ""
        if 'fan' in product_type.lower():
            if 'BLDC' in title or 'Brushless' in title:
                age_features += "Energy efficient, suitable for all ages. "
            if 'remote' in description.lower() or 'remote' in title.lower():
                age_features += "Remote control, convenient for elderly users. "
            if 'child' in description.lower() or 'child' in title.lower():
                age_features += "Child-safe design. "
        
        # Extract room-specific features
        room_features = ""
        if 'kitchen' in description.lower() or 'kitchen' in title.lower():
            room_features += "Kitchen-friendly. "
        if 'bedroom' in description.lower() or 'bedroom' in title.lower():
            room_features += "Bedroom-optimized. "
        if 'living room' in description.lower() or 'living room' in title.lower():
            room_features += "Living room suitable. "
        
        # Extract energy efficiency features
        energy_features = ""
        if 'energy' in description.lower() or 'efficient' in description.lower():
            energy_features += "Energy efficient. "
        if 'power' in description.lower() and 'low' in description.lower():
            energy_features += "Low power consumption. "
        
        # Main product entry with enhanced feature emphasis
        entry = (
            f"Product Type: {product_type}. "
            f"Brand: {brand}. "
            f"Title: {title}. "
            f"Better Home Price: {price} INR. "
            f"Retail Price: {retail_price} INR. "
            f"{discount_text} "
            f"Warranty: {warranty}. "
            f"Features: {features}. "
            f"Age Features: {age_features} "
            f"Room Features: {room_features} "
            f"Energy Features: {energy_features} "
            f"Description: {description}. "
            f"Product URL: {url}."
        )
        entries.append(entry)

    return entries

# ==========================
# Step 3: Generate Embeddings with OpenAI or Ollama
# ==========================
def generate_embeddings(entries, batch_size=10, max_retries=3):
    """Generate embeddings using either OpenAI's API or Ollama."""
    embeddings = []
    
    if not entries:
        print("Error: No entries available for generating embeddings.")
        return embeddings
    
    def generate_batch_embeddings(batch):
        for retry in range(max_retries):
            try:
                print(f"Attempt {retry+1}/{max_retries} for batch of size {len(batch)}")
                
                if USE_OPENAI:
                    # Call OpenAI API for embeddings
                    batch_embeddings = []
                    for text in batch:
                        response = openai.Embedding.create(
                            model="text-embedding-3-small",
                            input=text
                        )
                        if 'data' in response and len(response['data']) > 0:
                            embedding = response['data'][0]['embedding']
                            batch_embeddings.append(embedding)
                else:
                    # Call Ollama API for embeddings
                    batch_embeddings = []
                    for text in batch:
                        response = requests.post(
                            OLLAMA_URL,
                            json={
                                "model": OLLAMA_MODEL,
                                "prompt": text
                            }
                        )
                        if response.status_code == 200:
                            embedding = response.json().get('embedding', [])
                            if embedding:
                                batch_embeddings.append(embedding)
                
                # Validate embedding dimensions
                if batch_embeddings:
                    embedding_dim = len(batch_embeddings[0])
                    expected_dim = OPENAI_EMBEDDING_DIM if USE_OPENAI else OLLAMA_EMBEDDING_DIM
                    
                    if embedding_dim != expected_dim:
                        print(f"Warning: Unexpected embedding dimension: {embedding_dim}, expected: {expected_dim}")
                        print("This may cause issues with FAISS indexing. Proceeding anyway.")
                    else:
                        print(f"Generated embedding with correct dimension: {embedding_dim}")
                    
                    # Add to global embeddings list
                    embeddings.extend(batch_embeddings)
                    
                    return batch_embeddings
                else:
                    raise Exception("No embeddings returned from API")
                
            except Exception as e:
                print(f"Error generating embeddings: {str(e)}")
                if retry < max_retries - 1:
                    print(f"Retrying in 2 seconds...")
                    time.sleep(2)
                    continue
                return []
        
        return []
    
    # Process in batches with progress bar
    total_entries = len(entries)
    print(f"Processing {total_entries} entries in batches of {batch_size}")
    
    with ThreadPoolExecutor(max_workers=1) as executor:
        batches = [entries[i:i + batch_size] for i in range(0, len(entries), batch_size)]
        results = list(tqdm(executor.map(generate_batch_embeddings, batches), total=len(batches), desc="Generating Embeddings"))
        
        # Count successful embeddings
        successful_embeddings = sum(1 for batch_embeddings in results if batch_embeddings)
        print(f"Successfully generated {successful_embeddings} out of {len(batches)} batches")
        
        # Add successful embeddings to the result
        for batch_embeddings in results:
            if batch_embeddings:
                embeddings.extend(batch_embeddings)

    if not embeddings:
        print("Error: No embeddings were generated.")
    else:
        print(f"Successfully generated {len(embeddings)} embeddings")

    return embeddings

# ==========================
# Step 4: Save Embeddings
# ==========================
def save_embeddings(embeddings_dict, file_name):
    with open(file_name, 'w') as f:
        json.dump(embeddings_dict, f)
    print(f"Embeddings saved successfully to {file_name}.")

# ==========================
# Step 5: Build & Save FAISS Index
# ==========================
def build_faiss_index(embeddings, index_file_path):
    if not embeddings or not all(isinstance(e, list) for e in embeddings):
        print("Error: No embeddings to build the index.")
        return None

    try:
        dimension = len(embeddings[0])
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings))
        faiss.write_index(index, index_file_path)
        print(f"FAISS index built and saved successfully at {index_file_path}.")
        return index
    except Exception as e:
        print(f"Error building FAISS index: {str(e)}")
        return None

# ==========================
# Step 6: Generate User Profile Embeddings
# ==========================
def generate_user_profile_embeddings(user_profiles, batch_size=10, max_retries=3):
    """Generate embeddings for user profiles using either OpenAI or Ollama."""
    embeddings = []
    
    if not user_profiles:
        print("Error: No user profiles available for generating embeddings.")
        return embeddings
    
    # Prepare user profile entries
    entries = []
    for profile in user_profiles:
        entry = (
            f"Age Group: {profile.get('age_group', 'Not Available')}. "
            f"Room Type: {profile.get('room_type', 'Not Available')}. "
            f"Preferences: {', '.join(profile.get('preferences', []))}. "
            f"Budget: {profile.get('budget', 'Not Available')}."
        )
        entries.append(entry)
    
    def generate_batch_embeddings(batch):
        for retry in range(max_retries):
            try:
                print(f"Attempt {retry+1}/{max_retries} for user profile batch of size {len(batch)}")
                
                if USE_OPENAI:
                    # Call OpenAI API for embeddings
                    batch_embeddings = []
                    for text in batch:
                        response = openai.Embedding.create(
                            model="text-embedding-3-small",
                            input=text
                        )
                        if 'data' in response and len(response['data']) > 0:
                            embedding = response['data'][0]['embedding']
                            batch_embeddings.append(embedding)
                else:
                    # Call Ollama API for embeddings
                    batch_embeddings = []
                    for text in batch:
                        response = requests.post(
                            OLLAMA_URL,
                            json={
                                "model": OLLAMA_MODEL,
                                "prompt": text
                            }
                        )
                        if response.status_code == 200:
                            embedding = response.json().get('embedding', [])
                            if embedding:
                                batch_embeddings.append(embedding)
                
                # Validate embedding dimensions
                if batch_embeddings:
                    embedding_dim = len(batch_embeddings[0])
                    expected_dim = OPENAI_EMBEDDING_DIM if USE_OPENAI else OLLAMA_EMBEDDING_DIM
                    
                    if embedding_dim != expected_dim:
                        print(f"Warning: Unexpected user profile embedding dimension: {embedding_dim}, expected: {expected_dim}")
                        print("This may cause issues with FAISS indexing. Proceeding anyway.")
                    else:
                        print(f"Generated user profile embedding with correct dimension: {embedding_dim}")
                    
                    # Add to global embeddings list
                    embeddings.extend(batch_embeddings)
                    
                    return batch_embeddings
                else:
                    raise Exception("No embeddings returned from API")
                
            except Exception as e:
                print(f"Error generating user profile embeddings: {str(e)}")
                if retry < max_retries - 1:
                    print(f"Retrying in 2 seconds...")
                    time.sleep(2)
                    continue
                return []
        
        return []
    
    # Process in batches with progress bar
    total_entries = len(entries)
    print(f"Processing {total_entries} user profile entries in batches of {batch_size}")
    
    with ThreadPoolExecutor(max_workers=1) as executor:
        batches = [entries[i:i + batch_size] for i in range(0, len(entries), batch_size)]
        results = list(tqdm(executor.map(generate_batch_embeddings, batches), total=len(batches), desc="Generating User Profile Embeddings"))
        
        # Count successful embeddings
        successful_embeddings = sum(1 for batch_embeddings in results if batch_embeddings)
        print(f"Successfully generated {successful_embeddings} out of {len(batches)} user profile batches")
        
        # Add successful embeddings to the result
        for batch_embeddings in results:
            if batch_embeddings:
                embeddings.extend(batch_embeddings)

    if not embeddings:
        print("Error: No user profile embeddings were generated.")
    else:
        print(f"Successfully generated {len(embeddings)} user profile embeddings")

    return embeddings

# ==========================
# Main Function
# ==========================
def main():
    print("Starting embedding generation process...")
    print(f"Using {'OpenAI' if USE_OPENAI else 'Ollama'} for embeddings")
    
    # Load product catalog
    df = load_product_catalog(CSV_FILE_PATH)
    if df.empty:
        print("Product catalog could not be loaded. Exiting.")
        return

    # Prepare entries
    entries = prepare_entries(df)
    if not entries:
        print("No valid entries were found. Exiting.")
        return

    print(f"Prepared {len(entries)} entries for embedding generation")
    
    # Try to generate embeddings with retries
    max_attempts = 3
    for attempt in range(max_attempts):
        print(f"\nAttempt {attempt+1}/{max_attempts} to generate embeddings")
        
        # Generate product embeddings
        embeddings = generate_embeddings(entries)
        if not embeddings:
            print(f"Failed to generate product embeddings on attempt {attempt+1}")
            if attempt < max_attempts - 1:
                print("Waiting 10 seconds before retrying...")
                time.sleep(10)
                continue
            else:
                print("All attempts to generate product embeddings failed. Exiting.")
                return

        # Generate product type embeddings
        product_type_entries = [f"Product Type: {row.get('Product Type', 'Not Available')}" for _, row in df.iterrows()]
        product_type_embeddings = generate_embeddings(product_type_entries)
        if not product_type_embeddings:
            print(f"Failed to generate product type embeddings on attempt {attempt+1}")
            if attempt < max_attempts - 1:
                print("Waiting 10 seconds before retrying...")
                time.sleep(10)
                continue
            else:
                print("All attempts to generate product type embeddings failed. Exiting.")
                return
        
        # Generate brand embeddings
        brand_entries = [f"Brand: {row.get('Brand', 'Not Available')}" for _, row in df.iterrows()]
        brand_embeddings = generate_embeddings(brand_entries)
        if not brand_embeddings:
            print(f"Failed to generate brand embeddings on attempt {attempt+1}")
            if attempt < max_attempts - 1:
                print("Waiting 10 seconds before retrying...")
                time.sleep(10)
                continue
            else:
                print("All attempts to generate brand embeddings failed. Exiting.")
                return
        
        # Generate sample user profiles
        sample_user_profiles = [
            {
                'age_group': 'elderly',
                'room_type': 'bedroom',
                'preferences': ['quiet operation', 'remote control', 'energy efficient'],
                'budget': '₹5000-₹10000'
            },
            {
                'age_group': 'children',
                'room_type': 'bedroom',
                'preferences': ['child-safe', 'quiet operation'],
                'budget': '₹3000-₹8000'
            },
            {
                'age_group': 'adult',
                'room_type': 'living room',
                'preferences': ['energy efficient', 'stylish design'],
                'budget': '₹8000-₹15000'
            }
        ]
        
        # Generate user profile embeddings
        user_profile_embeddings = generate_user_profile_embeddings(sample_user_profiles)
        if not user_profile_embeddings:
            print(f"Failed to generate user profile embeddings on attempt {attempt+1}")
            if attempt < max_attempts - 1:
                print("Waiting 10 seconds before retrying...")
                time.sleep(10)
                continue
            else:
                print("All attempts to generate user profile embeddings failed. Exiting.")
                return
        
        # If we get here, all embeddings were generated successfully
        print("\nAll embeddings generated successfully!")

        # Save embeddings
        embeddings_dict = {
            'product_embeddings': embeddings,
            'product_type_embeddings': product_type_embeddings,
            'brand_embeddings': brand_embeddings,
            'user_profile_embeddings': user_profile_embeddings,
            'metadata': {
                'total_products': len(df),
                'unique_product_types': df['Product Type'].nunique(),
                'unique_brands': df['Brand'].nunique(),
                'user_profiles': sample_user_profiles,
                'embedding_provider': 'OpenAI' if USE_OPENAI else 'Ollama'
            }
        }
        
        try:
            save_embeddings(embeddings_dict, EMBEDDINGS_FILE_PATH)
            print(f"Embeddings saved to {EMBEDDINGS_FILE_PATH}")

            # Build and save separate FAISS indexes
            build_faiss_index(embeddings, 'faiss_index.index_product')
            build_faiss_index(product_type_embeddings, 'faiss_index.index_type')
            build_faiss_index(brand_embeddings, 'faiss_index.index_brand')
            build_faiss_index(user_profile_embeddings, 'faiss_index.index_user_profile')
            print("FAISS indexes built and saved successfully")
            
            # If we get here, everything was successful
            print("\nEmbedding generation completed successfully!")
            return
            
        except Exception as e:
            print(f"Error saving embeddings or building indexes: {str(e)}")
            if attempt < max_attempts - 1:
                print("Waiting 10 seconds before retrying...")
                time.sleep(10)
                continue
            else:
                print("All attempts to save embeddings failed. Exiting.")
                return
    
    # If we get here, all attempts failed
    print("All attempts to generate and save embeddings failed. Exiting.")

if __name__ == "__main__":
    main()
