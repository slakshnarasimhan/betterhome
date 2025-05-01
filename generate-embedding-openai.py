import openai
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import streamlit as st
from concurrent.futures import ThreadPoolExecutor
import faiss
from typing import Dict, List

# ==========================
# Configuration
# ==========================
MODEL_NAME = "text-embedding-3-small"
CSV_FILE_PATH = 'cleaned_products.csv'
EMBEDDINGS_FILE_PATH = 'embeddings.json'
PRODUCT_CATALOG_PATH = 'web_app/product_catalog.json'

# FAISS index output files
INDEX_FILE_PRODUCT = 'faiss_index.index_product'
INDEX_FILE_TYPE = 'faiss_index.index_type'
INDEX_FILE_BRAND = 'faiss_index.index_brand'
INDEX_FILE_IMAGE = 'faiss_index.index_image'

# ==========================
# Step 1: Load Product Catalog
# ==========================
def load_product_catalog(file_path):
    df = pd.read_csv(file_path)
    print(f"Successfully loaded product catalog with {len(df)} entries.")
    df['Better Home Price'] = pd.to_numeric(df['Better Home Price'], errors='coerce')
    df['Retail Price'] = pd.to_numeric(df['Retail Price'], errors='coerce')
    # Ensure 'Features' is included
    if 'Features (product.metafields.custom.features)' in df.columns:
        df.rename(columns={'Features (product.metafields.custom.features)': 'Features'}, inplace=True)

    # Fill missing 'Features' for rows with the same 'Handle'
    if 'Handle' in df.columns and 'Features' in df.columns:
        for handle, group in df.groupby('Handle'):
            if group['Features'].isnull().any():
                filled_value = group['Features'].dropna().iloc[0] if not group['Features'].dropna().empty else 'Not Available'
                df.loc[df['Handle'] == handle, 'Features'] = df.loc[df['Handle'] == handle, 'Features'].fillna(filled_value)

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
        if pd.notnull(row.get('Retail Price')) and row['Retail Price'] > 0:
            discount_percentage = ((row['Retail Price'] - row['Better Home Price']) / row['Retail Price']) * 100
            discount_text = f"Better Home Price is {discount_percentage:.2f}% less than Retail Price."
        else:
            discount_text = "No discount available."

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
# Step 3: Save Product Catalog
# ==========================
def parse_features(features_str: str) -> List[str]:
    """Convert features string (separated by '|') into a list of features."""
    # Handle potential non-string types gracefully
    if not isinstance(features_str, str):
        return []
    # Split by '|' and strip whitespace from each feature
    return [feature.strip() for feature in features_str.split('|') if feature.strip()]

def save_product_catalog(df, file_path=PRODUCT_CATALOG_PATH):
    catalog = []
    for _, row in df.iterrows():
        # Debug: Print 'Features' field before parsing
        print(f"[DEBUG] Raw Features: {row.get('Features', 'Not Available')}")

        # Parse features
        parsed_features = parse_features(row.get('Features', ''))

        # Debug: Print parsed features
        print(f"[DEBUG] Parsed Features: {parsed_features}")

        product = {
            'product_type': row.get('Product Type', 'Not Available'),
            'brand': row.get('Brand', 'Not Available'),
            'title': row.get('title', 'Not Available'),
            'better_home_price': row.get('Better Home Price', 'Not Available'),
            'retail_price': row.get('Retail Price', 'Not Available'),
            'warranty': row.get('Warranty', 'Not Available'),
            'features': parsed_features,  # Use parsed features
            'description': row.get('Description', 'Not Available'),
            'url': row.get('url', 'Not Available'),
            'image_src': row.get('Image Src', 'Not Available')
        }
        # Debug statement to print Features data before writing to JSON
        print(f"[DEBUG] Features to JSON: {product['features']}")
        catalog.append(product)
    with open(file_path, 'w') as f:
        json.dump({'products': catalog}, f, indent=2)
    print(f"Product catalog saved successfully to {file_path}.")

# ==========================
# Step 4: Generate Embeddings using OpenAI
# ==========================
def generate_openai_embeddings(entries, batch_size=10):
    embeddings = []
    openai.api_key = st.secrets["OPENAI_API_KEY"]

    def generate_batch(batch):
        try:
            response = openai.Embedding.create(
                model=MODEL_NAME,
                input=batch
            )
            return [e.embedding for e in response.data]
        except Exception as e:
            print(f"Error generating batch embeddings: {str(e)}")
            return []

    with ThreadPoolExecutor() as executor:
        batches = [entries[i:i + batch_size] for i in range(0, len(entries), batch_size)]
        results = list(tqdm(executor.map(generate_batch, batches), total=len(batches), desc="Generating Embeddings"))
        for batch_embeddings in results:
            embeddings.extend(batch_embeddings)

    return embeddings

# ==========================
# Step 5: Save Embeddings
# ==========================
def save_embeddings(embeddings_dict, file_name):
    with open(file_name, 'w') as f:
        json.dump(embeddings_dict, f)
    print(f"Embeddings saved successfully to {file_name}.")

# ==========================
# Step 6: Build FAISS Index
# ==========================
def build_faiss_index(embeddings, index_file_path):
    if not embeddings:
        print("Error: No embeddings to build the index.")
        return None

    try:
        dimension = len(embeddings[0])
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings).astype('float32'))
        faiss.write_index(index, index_file_path)
        print(f"FAISS index saved at {index_file_path}")
        return index
    except Exception as e:
        print(f"Error building FAISS index: {str(e)}")
        return None

# ==========================
# Main Execution
# ==========================
def main():
    df = load_product_catalog(CSV_FILE_PATH)
    if df.empty:
        print("Product catalog could not be loaded. Exiting.")
        return

    # Save the product catalog with image URLs
    print("Saving product catalog...")
    save_product_catalog(df)

    # Prepare all entries
    print("Preparing entries...")
    product_entries, product_type_entries, brand_entries, image_entries = prepare_entries(df)
    
    # Generate embeddings for all types
    print("Generating embeddings...")
    product_embeddings = generate_openai_embeddings(product_entries)
    product_type_embeddings = generate_openai_embeddings(product_type_entries)
    brand_embeddings = generate_openai_embeddings(brand_entries)
    image_embeddings = generate_openai_embeddings(image_entries)

    # Save all embeddings to one file
    embeddings_dict = {
        'product_embeddings': product_embeddings,
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
    save_embeddings(embeddings_dict, EMBEDDINGS_FILE_PATH)

    # Build FAISS indexes
    print("Building FAISS indexes...")
    build_faiss_index(product_embeddings, INDEX_FILE_PRODUCT)
    build_faiss_index(product_type_embeddings, INDEX_FILE_TYPE)
    build_faiss_index(brand_embeddings, INDEX_FILE_BRAND)
    build_faiss_index(image_embeddings, INDEX_FILE_IMAGE)

    print("\nEmbedding generation completed successfully!")

if __name__ == "__main__":
    main()

