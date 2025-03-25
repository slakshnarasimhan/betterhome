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
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['compare_at_price'] = pd.to_numeric(df['compare_at_price'], errors='coerce')
    return df

# ==========================
# Step 2: Prepare Entries (Enhanced with URL and Price Comparison)
# ==========================
def prepare_entries(df):
    entries = []
    for _, row in df.iterrows():
        if pd.notnull(row.get('compare_at_price')) and row['compare_at_price'] > 0:
            discount_percentage = ((row['compare_at_price'] - row['price']) / row['compare_at_price']) * 100
            discount_text = f"Better Home Price is {discount_percentage:.2f}% less than Retail Price."
        else:
            discount_text = "No discount available."

        product_url = f"https://betterhomeapp.com/products/{row.get('handle', 'unknown')}"

        entry = (
            f"Product: {row.get('title', 'Not Available')}. "
            f"Brand: {row.get('brand', 'Not Available')}. "
            f"Better Home Price: {row.get('price', 'Not Available')} INR. "
            f"Retail Price: {row.get('compare_at_price', 'Not Available')} INR. "
            f"{discount_text} "
            f"Warranty: {row.get('warranty', 'Not Available')}. "
            f"Features: {row.get('features', 'Not Available')}. "
            f"Description: {row.get('description', 'Not Available')}. "
            f"Product URL: {product_url}."
        )
        entries.append(entry)
    return entries

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
def save_embeddings(embeddings, file_name):
    with open(file_name, 'w') as f:
        json.dump(embeddings, f)

# ==========================
# Main Function
# ==========================
def main():
    # Load the cleaned CSV file
    file_path = 'cleaned-products.csv'  # Adjusted to the correct file name
    df = load_product_catalog(file_path)
    entries = prepare_entries(df)

    # Generate embeddings
    embeddings = generate_openai_embeddings(entries)

    # Save embeddings to a file
    save_embeddings(embeddings, 'embeddings.json')
    print("Embeddings generated and saved successfully.")

if __name__ == "__main__":
    main()
