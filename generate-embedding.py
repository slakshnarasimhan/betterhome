from ollama import Client as Ollama
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import streamlit as st
from concurrent.futures import ThreadPoolExecutor
import faiss

# ==========================
# Configuration
# ==========================
MODEL_NAME = "llama3.2"  # Use your installed Ollama model
#CSV_FILE_PATH = 'cleaned_products.csv'
CSV_FILE_PATH = 'cleaned_products.csv'
EMBEDDINGS_FILE_PATH = 'embeddings.json'
INDEX_FILE_PATH = 'faiss_index.index'

# ==========================
# Step 1: Load Product Catalog
# ==========================
def load_product_catalog(file_path):
    df = pd.read_csv(file_path)
    print(f"Successfully loaded product catalog with {len(df)} entries.")
    df['Better Home Price'] = pd.to_numeric(df['Better Home Price'], errors='coerce')
    df['Retail Price'] = pd.to_numeric(df['Retail Price'], errors='coerce')
    return df

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
# Step 3: Generate Embeddings with Ollama
# ==========================
def generate_local_embeddings(entries, batch_size=1):
    embeddings = []
    client = Ollama()
    
    if not entries:
        print("Error: No entries available for generating embeddings.")
        return embeddings
    
    def generate_batch_embeddings(batch):
        try:
            print("Sending batch to Ollama...")
            response = client.embed(model=MODEL_NAME, input=batch)
            print("Received response:", response)
            if isinstance(response, dict) and 'embeddings' in response:
                print("vaalid embeddingg..")
                for item in response["embeddings"]:
                    embeddings.append(item)
                return response['embeddings']
                #return response['embeddings'][0]
            else:
                print(f"Failed to extract embeddings for batch. Response: {response}")
                return []

        except Exception as e:
            print(f"Error generating embeddings for batch: {str(e)}")
            return []
    
    with ThreadPoolExecutor() as executor:
        batches = [entries[i:i + batch_size] for i in range(0, len(entries), batch_size)]
        results = list(tqdm(executor.map(generate_batch_embeddings, batches), total=len(batches), desc="Generating Embeddings"))
        for batch_embeddings in results:
            embeddings.extend(batch_embeddings)

    if embeddings:
        print(f"Successfully generated embeddings for {len(embeddings)} entries.")
        print(f"Embedding dimension: {len(embeddings[0])}.")
    else:
        print("Error: No embeddings were generated.")

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
def generate_user_profile_embeddings(user_profiles, batch_size=1):
    """
    Generate embeddings for user profiles to improve personalized recommendations.
    
    Args:
        user_profiles: List of user profile dictionaries
        batch_size: Number of profiles to process in each batch
        
    Returns:
        List of embeddings for user profiles
    """
    profile_entries = []
    
    for profile in user_profiles:
        # Extract user profile information
        age_group = profile.get('age_group', 'Not Available')
        room_type = profile.get('room_type', 'Not Available')
        preferences = profile.get('preferences', [])
        budget = profile.get('budget', 'Not Available')
        
        # Create a structured profile entry
        entry = (
            f"Age Group: {age_group}. "
            f"Room Type: {room_type}. "
            f"Preferences: {', '.join(preferences)}. "
            f"Budget: {budget}."
        )
        profile_entries.append(entry)
    
    # Use the existing embedding function to generate embeddings
    return generate_local_embeddings(profile_entries, batch_size)

# ==========================
# Main Function
# ==========================
def main():
    df = load_product_catalog(CSV_FILE_PATH)
    if df.empty:
        print("Product catalog could not be loaded. Exiting.")
        return

    entries = prepare_entries(df)
    if not entries:
        print("No valid entries were found. Exiting.")
        return

    embeddings = generate_local_embeddings(entries)
    if not embeddings:
        print("No embeddings were generated. Exiting.")
        return

    # Separate embeddings for product type, brand, and main product entry
    product_type_entries = [f"Product Type: {row.get('Product Type', 'Not Available')}" for _, row in df.iterrows()]
    brand_entries = [f"Brand: {row.get('Brand', 'Not Available')}" for _, row in df.iterrows()]

    print(f"Generating embeddings for {len(product_type_entries)} product types.")
    product_type_embeddings = generate_local_embeddings(product_type_entries)
    print(f"Generated {len(product_type_embeddings)} product type embeddings.")
    if product_type_embeddings:
        print(f"Product type embedding dimension: {len(product_type_embeddings[0])}.")

    print(f"Generating embeddings for {len(brand_entries)} brands.")
    brand_embeddings = generate_local_embeddings(brand_entries)
    print(f"Generated {len(brand_embeddings)} brand embeddings.")
    if brand_embeddings:
        print(f"Brand embedding dimension: {len(brand_embeddings[0])}.")
    
    # Generate sample user profiles for testing
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
    
    print("Generating embeddings for user profiles.")
    user_profile_embeddings = generate_user_profile_embeddings(sample_user_profiles)
    print(f"Generated {len(user_profile_embeddings)} user profile embeddings.")
    if user_profile_embeddings:
        print(f"User profile embedding dimension: {len(user_profile_embeddings[0])}.")

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
            'user_profiles': sample_user_profiles
        }
    }
    print("Saving embeddings to file.")
    save_embeddings(embeddings_dict, EMBEDDINGS_FILE_PATH)

    # Build and save separate FAISS indexes with specific file names
    build_faiss_index(embeddings, 'faiss_index.index_product')
    build_faiss_index(product_type_embeddings, 'faiss_index.index_type')
    build_faiss_index(brand_embeddings, 'faiss_index.index_brand')
    build_faiss_index(user_profile_embeddings, 'faiss_index.index_user_profile')

if __name__ == "__main__":
    main()
