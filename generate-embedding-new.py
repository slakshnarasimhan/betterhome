import json
import pandas as pd
import numpy as np
from openai import OpenAI
import time
import streamlit as st

# ==========================
# Configuration
# ==========================
OPENAI_API_KEY=st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=OPENAI_API_KEY)
CSV_FILE_PATH = 'cleaned_products.csv'  # The cleaned product catalog
EMBEDDINGS_FILE_PATH = 'embeddings.json'
LOG_FILE_PATH = 'processed_products_log.json'  # Log file for tracking processed products

# ==========================
# Generate Embeddings
# ==========================

def get_openai_embedding(text):
    response = client.embeddings.create(model="text-embedding-ada-002", input=text)
    embedding = response.data[0].embedding

    if len(embedding) != 1536:  # Ensure correct embedding dimension
        raise ValueError(f"Expected embedding of dimension 1536 but got {len(embedding)}")

    return embedding

def generate_embeddings():
    df = pd.read_csv(CSV_FILE_PATH)
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')  # Standardize column names

    entries = []
    embeddings = []
    processed_products = []

    for index, row in df.iterrows():
        try:
            bh_price = row.get('better_home_price')
            retail_price = row.get('retail_price')
            title = row['title']
            brand = row.get('brand', 'Unknown')
            warranty = row.get('warranty', 'Unknown')
            product_url = row['url']
            category = row.get('category', 'Unknown')
            category_hierarchy = row.get('category_hierarchy', 'Unknown')
            tags = row['tags']
            description = row.get('description', '')

            if pd.notna(bh_price) and pd.notna(retail_price) and retail_price > 0:
                discount_percentage = ((retail_price - bh_price) / retail_price) * 100
                discount_text = f"Better Home Price: {bh_price} INR (Retail Price: {retail_price} INR). Better Home is {discount_percentage:.2f}% cheaper."
            else:
                discount_text = f"Better Home Price: {bh_price} INR. No discount available."

            text_entry = f"Title: {title}. Brand: {brand}. Warranty: {warranty}. Description: {description}. Tags: {tags}. Category: {category}. Category Hierarchy: {category_hierarchy}. {discount_text} URL: {product_url}."
            
            embedding = get_openai_embedding(text_entry)
            entries.append(text_entry)
            embeddings.append(embedding)

            processed_products.append({
                "title": title,
                "brand": brand,
                "warranty": warranty,
                "category": category,
                "category_hierarchy": category_hierarchy
            })

        except Exception as e:
            print(f"Error processing row {index}: {e}")

    with open(EMBEDDINGS_FILE_PATH, 'w') as f:
        json.dump(embeddings, f)

    with open(LOG_FILE_PATH, 'w') as log_file:
        json.dump(processed_products, log_file, indent=4)
    print("Embeddings and logs saved successfully.")

if __name__ == "__main__":
    generate_embeddings()

