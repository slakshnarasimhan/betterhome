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
MODEL_NAME = "nomic-embed-text"  # Use your installed Ollama model
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

        # Main product entry with emphasis on Product Type and Brand
        entry = (
            f"Product Type: {row.get('Product Type', 'Not Available')}. "
            f"Brand: {row.get('Brand', 'Not Available')}. "
            f"Title: {row.get('title', 'Not Available')}. "
            f"Better Home Price: {row.get('Better Home Price', 'Not Available')} INR. "
            f"Retail Price: {row.get('Retail Price', 'Not Available')} INR. "
            f"{discount_text} "
            f"Warranty: {row.get('Warranty', 'Not Available')}. "
            f"Features: {row.get('Features', 'Not Available')}. "
            f"Description: {row.get('Description', 'Not Available')}. "
            f"Product URL: {row.get('url', 'Not Available')}."
        )
        entries.append(entry)

    return entries

# ==========================
# Step 3: Generate Embeddings with Ollama
# ==========================

def generate_local_embeddings(entries, batch_size=1):
    embeddings = []
    client = Ollama(host='http://localhost:11434')  # Ensure it's explicit

    if not entries:
        print("Error: No entries available for generating embeddings.")
        return embeddings

    def generate_batch_embeddings(batch):
        try:
            response = client.embed(model=MODEL_NAME, input=batch)
            if isinstance(response, dict) and 'embeddings' in response:
                return response['embeddings']
            else:
                print(f"[{MODEL_NAME}] Invalid response: {response}")
                return []
        except Exception as e:
            print(f"[{MODEL_NAME}] Batch failed: {e}. Retrying each item...")
            results = []
            for entry in batch:
                try:
                    single_resp = client.embed(model=MODEL_NAME, input=[entry])
                    if isinstance(single_resp, dict) and 'embeddings' in single_resp:
                        results.extend(single_resp['embeddings'])
                    else:
                        print(f"Failed on single entry: {entry[:60]}...")
                except Exception as se:
                    print(f"Error on single entry: {se}")
            return results

    batches = [entries[i:i + batch_size] for i in range(0, len(entries), batch_size)]
    for batch in tqdm(batches, desc="Generating Embeddings"):
        batch_embeddings = generate_batch_embeddings(batch)
        embeddings.extend(batch_embeddings)

    if embeddings:
        print(f"Successfully generated embeddings for {len(embeddings)} entries.")
        print(f"Embedding dimension: {len(embeddings[0])}.")
    else:
        print("Error: No embeddings were generated.")

    return embeddings

    
    def generate_batch_embeddings(batch):
        try:
            response = client.embed(model=MODEL_NAME, input=batch)
            if isinstance(response, dict) and 'embeddings' in response:
                return response['embeddings']
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
    if not embeddings:
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

    # Save embeddings
    embeddings_dict = {
        'product_embeddings': embeddings,
        'product_type_embeddings': product_type_embeddings,
        'brand_embeddings': brand_embeddings,
        'metadata': {
            'total_products': len(df),
            'unique_product_types': df['Product Type'].nunique(),
            'unique_brands': df['Brand'].nunique()
        }
    }
    print("Saving embeddings to file.")
    save_embeddings(embeddings_dict, EMBEDDINGS_FILE_PATH)

    # Build and save separate FAISS indexes with specific file names
    build_faiss_index(embeddings, 'faiss_index.index_product')
    build_faiss_index(product_type_embeddings, 'faiss_index.index_type')
    build_faiss_index(brand_embeddings, 'faiss_index.index_brand')

if __name__ == "__main__":
    main()
