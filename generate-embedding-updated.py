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
CSV_FILE_PATH = 'cleaned_products.csv'  # Use the cleaned data file
EMBEDDINGS_FILE_PATH = 'embeddings.json'
INDEX_FILE_PATH = 'faiss_index.index'

# ==========================
# Step 1: Load Product Catalog
# ==========================
def load_product_catalog(file_path):
    df = pd.read_csv(file_path)
    
    # Verify Image Src column exists
    if 'Image Src' not in df.columns:
        print("Error: 'Image Src' column not found in CSV file")
        return None
    
    df['Better Home Price'] = pd.to_numeric(df['Better Home Price'], errors='coerce')
    df['Retail Price'] = pd.to_numeric(df['Retail Price'], errors='coerce')
    return df

# ==========================
# Step 2: Prepare Entries
# ==========================
def prepare_entries(df):
    entries = []
    product_type_entries = []
    brand_entries = []
    image_entries = []
    
    for _, row in df.iterrows():
        # Calculate discount if applicable
        if pd.notnull(row.get('Retail Price')) and row['Retail Price'] > 0:
            discount_percentage = ((row['Retail Price'] - row['Better Home Price']) / row['Retail Price']) * 100
            discount_text = f"Better Home Price is {discount_percentage:.2f}% less than Retail Price."
        else:
            discount_text = "No discount available."

        # Main product entry
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
        
        # Create image entry only if image source exists
        if pd.notna(row.get('Image Src')):
            image_entry = (
                f"Product: {row.get('title', 'Not Available')}. "
                f"Brand: {row.get('Brand', 'Not Available')}. "
                f"Type: {row.get('Product Type', 'Not Available')}. "
                f"Image Source: {row.get('Image Src')}"
            )
            image_entries.append(image_entry)

        # Product type entry
        if pd.notna(row.get('Product Type')):
            product_type_entries.append(f"Product Type: {row.get('Product Type')}")
        
        # Brand entry
        if pd.notna(row.get('Brand')):
            brand_entries.append(f"Brand: {row.get('Brand')}")

    return entries, product_type_entries, brand_entries, image_entries

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
                print(f"Error: Invalid response format from Ollama: {response}")
                return []
        except Exception as e:
            print(f"Error generating embeddings for batch: {str(e)}")
            print("First entry in failing batch:", batch[0][:100] + "..." if batch else "Empty batch")
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
    total_batches = len(batches)
    successful_batches = 0
    
    for batch in tqdm(batches, desc="Generating Embeddings"):
        batch_embeddings = generate_batch_embeddings(batch)
        if batch_embeddings:
            embeddings.extend(batch_embeddings)
            successful_batches += 1

    if embeddings:
        print(f"Successfully generated embeddings for {successful_batches}/{total_batches} batches")
        print(f"Embedding dimension: {len(embeddings[0])}")
    else:
        print(f"Failed to generate any embeddings. Processed {total_batches} batches.")

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
    # Load product catalog
    df = load_product_catalog(CSV_FILE_PATH)
    if df is None or df.empty:
        print("Error: Product catalog could not be loaded.")
        return

    # Prepare all entries
    entries, product_type_entries, brand_entries, image_entries = prepare_entries(df)
    if not entries:
        print("Error: No valid entries were found.")
        return

    print(f"Starting embedding generation for {len(entries)} entries...")
    
    # Generate embeddings with smaller batch size
    embeddings = generate_local_embeddings(entries, batch_size=1)
    if not embeddings:
        print("Error: Failed to generate main product embeddings.")
        return

    # Generate remaining embeddings only if main embeddings succeeded
    image_embeddings = generate_local_embeddings(image_entries, batch_size=1) if image_entries else []
    product_type_embeddings = generate_local_embeddings(product_type_entries, batch_size=1) if product_type_entries else []
    brand_embeddings = generate_local_embeddings(brand_entries, batch_size=1) if brand_entries else []

    # Save all embeddings
    embeddings_dict = {
        'product_embeddings': embeddings,
        'product_type_embeddings': product_type_embeddings,
        'brand_embeddings': brand_embeddings,
        'image_embeddings': image_embeddings,
        'metadata': {
            'total_products': len(df),
            'products_with_images': len(image_entries),
            'unique_product_types': df['Product Type'].nunique(),
            'unique_brands': df['Brand'].nunique()
        }
    }
    
    try:
        save_embeddings(embeddings_dict, EMBEDDINGS_FILE_PATH)
        print(f"Successfully saved embeddings to {EMBEDDINGS_FILE_PATH}")
    except Exception as e:
        print(f"Error saving embeddings: {str(e)}")
        return

    # Build and save FAISS indexes
    try:
        if embeddings:
            build_faiss_index(embeddings, 'faiss_index.index_product')
        if product_type_embeddings:
            build_faiss_index(product_type_embeddings, 'faiss_index.index_type')
        if brand_embeddings:
            build_faiss_index(brand_embeddings, 'faiss_index.index_brand')
        if image_embeddings:
            build_faiss_index(image_embeddings, 'faiss_index.index_image')
        print("Successfully built FAISS indexes")
    except Exception as e:
        print(f"Error building FAISS indexes: {str(e)}")
        return

    print("Embedding generation completed successfully")

if __name__ == "__main__":
    main()
