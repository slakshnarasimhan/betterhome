import openai
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import streamlit as st

# ==========================
# Configuration
# ==========================
openai.api_key=st.secrets["OPENAI_API_KEY"]


# ==========================
# Step 1: Load Product Catalog
# ==========================

def load_product_catalog(file_path):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    df['price'] = df['price'].astype(str) + ' INR'  # Convert price to string and add INR
    return df


# ==========================
# Step 2: Prepare Entries (Focused Approach)
# ==========================

def prepare_entries(df):
    entries = []
    for _, row in df.iterrows():
        entry = (
            f"Product: {row.get('title', 'Not Available')}. "
            f"Brand: {row.get('brand', 'Not Available')}. "
            f"Price: {row.get('price', 'Not Available')} INR. "
            f"Warranty: {row.get('warranty', 'Not Available')}. "
            f"Features: {row.get('features', 'Not Available')}. "
            f"Description: {row.get('description', 'Not Available')}."
        )
        entries.append(entry)
    return entries


# ==========================
# Step 3: Generate Embeddings
# ==========================
# ==========================
# Step 3: Generate Embeddings (Fixed for New API)
# ==========================

def generate_openai_embeddings(entries):
    embeddings = []
    for entry in tqdm(entries, desc="Generating Embeddings"):
        response = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=[entry]  # The input must be a list of strings
        )
        embeddings.append(response.data[0].embedding)  # Access embeddings correctly
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
    file_path = 'products-clean-cleaned.csv'  # Make sure this file is present
    df = load_product_catalog(file_path)
    entries = prepare_entries(df)

    # Generate embeddings
    embeddings = generate_openai_embeddings(entries)

    # Save embeddings to a file
    save_embeddings(embeddings, 'embeddings.json')
    print("Embeddings generated and saved successfully.")


if __name__ == "__main__":
    main()

