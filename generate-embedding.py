import faiss
import numpy as np
import json
from openai import OpenAI
import pandas as pd
import streamlit as st


# ==========================
# Configuration
# ==========================
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]  # Replace with your OpenAI API Key if not using Streamlit Secrets
client = OpenAI(api_key=OPENAI_API_KEY)


# ==========================
# Step 1: Load Product Catalog
# ==========================

def load_product_catalog(uploaded_file):
    # Read CSV file from the uploaded file object
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    df['price'] = df['price'].astype(str) + ' INR'  # Convert price to string and add INR
    return df


# ==========================
# Step 2: Prepare Entries (Including Brand Names)
# ==========================

def prepare_entries(df):
    entries = []
    for _, row in df.iterrows():
        entry = (
            f"Title: {row['title']}. Price: {row['price']}. Description: {row['description']}. Features: {row['features']}. "
            f"Brand: {row['brand']}. Type: {row['type']}. Tags: {row['tags']}. Warranty: {row['warranty']}"
        )
        entries.append(entry)
    return entries


# ==========================
# Step 3: Generate Embeddings
# ==========================
# ==========================
# Step 3: Generate Embeddings
# ==========================
def generate_openai_embeddings(entries):
    embeddings = []
    for entry in entries:
        response = client.embeddings.create(model="text-embedding-ada-002", input=entry)
        embeddings.append(response.data[0].embedding)  # Corrected access method
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
    st.title('Generate Product Embeddings')
    uploaded_file = st.file_uploader("Upload your product catalog CSV file", type="csv")

    if uploaded_file:
        df = load_product_catalog(uploaded_file)
        entries = prepare_entries(df)
        embeddings = generate_openai_embeddings(entries)

        # Save the embeddings to a file
        save_embeddings(embeddings, 'embeddings.json')
        st.success("Embeddings generated and saved successfully as 'embeddings.json'.")


if __name__ == "__main__":
    main()

