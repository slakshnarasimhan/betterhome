import ijson  # Add ijson for efficient JSON parsing
import json
import numpy as np
from betterhome.common.embeddings import get_query_embedding

# Path to the blog embeddings file
BLOG_EMBEDDINGS_FILE_PATH = 'blog_embeddings.json'

# Function to calculate cosine similarity
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Extract chimney-related articles and calculate similarity scores
def extract_and_calculate_similarity(query):
    query_embedding = get_query_embedding(query)
    chimney_articles = []

    with open(BLOG_EMBEDDINGS_FILE_PATH, 'r') as f:
        # Use ijson to parse the file efficiently
        for article in ijson.items(f, 'item'):
            print(f"Processing article: {article.get('title', 'No Title')}")  # Debug: Print each article being processed
            if 'chimney' in article.get('title', '').lower() or 'chimney' in article.get('content', '').lower():
                chimney_articles.append(article)
                print(f"Found chimney article: {article.get('title', 'No Title')}")  # Debug: Print found article

    if not chimney_articles:
        print("No chimney-related articles found.")  # Debug: Print if no articles are found

    for article in chimney_articles:
        article_embedding = np.array(article['embedding'])
        similarity = cosine_similarity(query_embedding, article_embedding)
        print(f"Title: {article['title']}, Similarity: {similarity}")

# Run the extraction and similarity calculation
extract_and_calculate_similarity("how to choose right chimney")
