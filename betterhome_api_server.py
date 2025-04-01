
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import faiss
import json
import os
from ollama import Client as Ollama

app = FastAPI()

# ====== Config =======
CSV_FILE_PATH = 'cleaned_products.csv'
EMBEDDINGS_FILE_PATH = 'embeddings.json'
PRODUCT_INDEX_FILE_PATH = 'faiss_index.index_product'

# ====== Load Data and Models =======
df = pd.read_csv(CSV_FILE_PATH)
with open(EMBEDDINGS_FILE_PATH, 'r') as f:
    embedding_data = json.load(f)

product_embeddings = np.array(embedding_data['product_embeddings']).astype('float32')
index = faiss.IndexFlatL2(product_embeddings.shape[1])
index.add(product_embeddings)
ollama_client = Ollama()

# ====== Helper: Create embedding for query =======
def get_query_embedding(text):
    try:
        result = ollama_client.embed(model="llama3.2", input=[text])
        return np.array(result['embeddings'][0], dtype='float32')
    except Exception as e:
        print(f"[Embedding Error] {e}")
        return np.random.rand(index.d).astype('float32')  # fallback

# ====== Helper: Retrieve top product entries =======
def search_catalog(query, top_k=5):
    q_emb = get_query_embedding(query).reshape(1, -1)

    # Ensure embedding matches FAISS index dimension
    query_dim = q_emb.shape[1]
    index_dim = index.d
    if query_dim > index_dim:
        q_emb = q_emb[:, :index_dim]
    elif query_dim < index_dim:
        padding = np.zeros((1, index_dim - query_dim), dtype='float32')
        q_emb = np.hstack((q_emb, padding))

    D, I = index.search(q_emb, top_k)
    return df.iloc[I[0]].to_dict(orient='records')

# ====== Helper: Format response =======
def format_answer(products, query):
    response = f"Found {len(products)} products matching your query: **{query}**\n\n"
    for p in products:
        title = p.get('title', 'N/A')
        price = p.get('Better Home Price', 'N/A')
        retail = p.get('Retail Price', 'N/A')
        brand = p.get('Brand', 'N/A')
        url = p.get('url', '#')
        response += f"### {title}\n- Brand: {brand}\n- Better Home Price: ₹{price}\n- Retail Price: ₹{retail}\n[Click here to buy]({url})\n\n"
    return response

# ====== API Model =======
class QueryInput(BaseModel):
    query: str

# ====== API Endpoint =======
@app.post("/api/ask")
async def ask_question(payload: QueryInput):
    try:
        query = payload.query
        results = search_catalog(query)
        response = format_answer(results, query)
        return {"query": query, "results": response}
    except Exception as e:
        return {"error": f"Internal error: {str(e)}"}

