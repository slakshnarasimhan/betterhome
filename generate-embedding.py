from openai import OpenAI
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import streamlit as st

# ==========================
# Configuration
# ==========================
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ==========================
# Step 1: Load Product Catalog
# ==========================
def load_product_catalog(file_path):
    df = pd.read_csv(file_path)
    # No need to modify column names as they're already cleaned
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

        # Separate entry for Product Type with context
        product_type_entry = (
            f"Product Type: {row.get('Product Type', 'Not Available')}. "
            f"Title: {row.get('title', 'Not Available')}. "
            f"Brand: {row.get('Brand', 'Not Available')}. "
            f"Category: {row.get('Category', 'Not Available')}."
        )
        product_type_entries.append(product_type_entry)

        # Separate entry for Brand with context
        brand_entry = (
            f"Brand: {row.get('Brand', 'Not Available')}. "
            f"Product Type: {row.get('Product Type', 'Not Available')}. "
            f"Title: {row.get('title', 'Not Available')}. "
            f"Category: {row.get('Category', 'Not Available')}."
        )
        brand_entries.append(brand_entry)

    return entries, product_type_entries, brand_entries

# ==========================
# Step 3: Generate Embeddings
# ==========================
def generate_openai_embeddings(entries):
    embeddings = []
    for entry in tqdm(entries, desc="Generating Embeddings"):
        response = client.embeddings.create(model="text-embedding-ada-002", input=[entry])
        embeddings.append(response.data[0].embedding)
    return embeddings

# ==========================
# Step 4: Save Embeddings
# ==========================
def save_embeddings(embeddings_dict, file_name):
    with open(file_name, 'w') as f:
        json.dump(embeddings_dict, f)

# ==========================
# Main Function
# ==========================
def main():
    # Load the cleaned CSV file
    file_path = 'cleaned_products.csv'
    df = load_product_catalog(file_path)
    
    # Prepare different types of entries
    entries, product_type_entries, brand_entries = prepare_entries(df)

    # Generate embeddings for each type
    print("Generating main product embeddings...")
    product_embeddings = generate_openai_embeddings(entries)
    
    print("Generating product type embeddings...")
    product_type_embeddings = generate_openai_embeddings(product_type_entries)
    
    print("Generating brand embeddings...")
    brand_embeddings = generate_openai_embeddings(brand_entries)

    # Create a dictionary to store all embeddings
    embeddings_dict = {
        'product_embeddings': product_embeddings,
        'product_type_embeddings': product_type_embeddings,
        'brand_embeddings': brand_embeddings,
        'metadata': {
            'total_products': len(df),
            'unique_product_types': df['Product Type'].nunique(),
            'unique_brands': df['Brand'].nunique()
        }
    }

    # Save embeddings to a file
    save_embeddings(embeddings_dict, 'embeddings.json')
    print("Embeddings generated and saved successfully.")
    print(f"Total products processed: {len(df)}")
    print(f"Unique product types: {df['Product Type'].nunique()}")
    print(f"Unique brands: {df['Brand'].nunique()}")

if __name__ == "__main__":
    main()
