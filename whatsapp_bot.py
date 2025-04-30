import os
from flask import Flask, request
import json
import requests
from datetime import datetime
import faiss
import numpy as np
from process_blogs import extract_blog_content, fetch_blog_urls
import re

app = Flask(__name__)

# WhatsApp API Configuration
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN")
WHATSAPP_API_URL = "https://graph.facebook.com/v17.0"
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID")

# Load blog embeddings and metadata
def load_blog_data():
    try:
        with open('blog_embeddings.json', 'r') as f:
            data = json.load(f)
            return data['blog_embeddings'], data['metadata']
    except Exception as e:
        print(f"Error loading blog data: {e}")
        return [], []

# Load FAISS index
def load_faiss_index():
    try:
        index = faiss.read_index('blog_faiss_index.index')
        return index
    except Exception as e:
        print(f"Error loading FAISS index: {e}")
        return None

def send_whatsapp_message(phone_number_id, recipient_number, message):
    """Send a message to a WhatsApp user"""
    url = f"{WHATSAPP_API_URL}/{phone_number_id}/messages"
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json",
    }
    data = {
        "messaging_product": "whatsapp",
        "to": recipient_number,
        "type": "text",
        "text": {"body": message}
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return True
    except Exception as e:
        print(f"Error sending WhatsApp message: {e}")
        return False

def format_product_response(products, is_single_result=False):
    """Format product information for WhatsApp message"""
    if not products:
        return "I couldn't find any products matching your query."
    
    if is_single_result:
        product = products[0]
        response = f"🏷️ {product['title']}\n"
        response += f"💰 Price: ₹{product['price']:,.2f}\n"
        if product.get('savings'):
            response += f"💵 Potential Savings: ₹{product['savings']:,.2f}\n"
        response += f"🔗 Buy Now: {product['url']}"
        return response
    
    response = "Here are the products I found:\n\n"
    for i, product in enumerate(products, 1):
        response += f"{i}. {product['title']}\n"
        response += f"   Price: ₹{product['price']:,.2f}\n"
        if product.get('savings'):
            response += f"   Potential Savings: ₹{product['savings']:,.2f}\n"
        response += f"   Buy Now: {product['url']}\n\n"
    return response

def handle_product_query(query, blog_embeddings, metadata, index):
    """Handle product-related queries"""
    # Generate query embedding (you'll need to implement this)
    query_embedding = generate_query_embedding(query)
    
    # Search for relevant blogs
    k = 3  # Number of results to return
    distances, indices = index.search(query_embedding.reshape(1, -1), k)
    
    # Get relevant blog content
    relevant_blogs = [metadata[i] for i in indices[0]]
    
    # Extract product information
    products = []
    for blog in relevant_blogs:
        # Extract product information from blog content
        # This is a simplified example - you'll need to implement proper product extraction
        product = {
            'title': blog['title'],
            'price': extract_price_from_content(blog['content']),
            'url': blog['url']
        }
        products.append(product)
    
    return products

def generate_query_embedding(query):
    """Generate embedding for the query"""
    # Implement query embedding generation
    # This should match the embedding generation method used in process_blogs.py
    return np.random.rand(3072)  # Placeholder

def extract_price_from_content(content):
    """Extract price information from content"""
    # Implement price extraction logic
    # This is a placeholder
    return 0.0

@app.route("/webhook", methods=["GET"])
def verify_webhook():
    """Verify the webhook for WhatsApp"""
    mode = request.args.get("hub.mode")
    token = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")

    if mode and token:
        if mode == "subscribe" and token == VERIFY_TOKEN:
            print("Webhook verified!")
            return challenge
        else:
            return "Invalid verification token"
    return "Invalid request"

@app.route("/webhook", methods=["POST"])
def webhook():
    """Handle incoming WhatsApp messages"""
    try:
        data = request.get_json()
        
        if data["object"] == "whatsapp_business_account":
            for entry in data["entry"]:
                for change in entry["changes"]:
                    if change["value"]["messages"]:
                        message = change["value"]["messages"][0]
                        recipient_number = message["from"]
                        message_text = message["text"]["body"]
                        
                        # Process the message
                        response = process_message(message_text)
                        
                        # Send response back to user
                        send_whatsapp_message(PHONE_NUMBER_ID, recipient_number, response)
        
        return "OK"
    except Exception as e:
        print(f"Error in webhook: {e}")
        return "Error", 500

def process_message(message):
    """Process incoming message and generate response"""
    # Load necessary data
    blog_embeddings, metadata = load_blog_data()
    index = load_faiss_index()
    
    if not blog_embeddings or not index:
        return "I'm sorry, but I'm having trouble accessing the product database right now. Please try again later."
    
    # Check if it's a follow-up question
    if is_follow_up_question(message):
        return handle_follow_up_question(message, blog_embeddings, metadata, index)
    
    # Handle product query
    products = handle_product_query(message, blog_embeddings, metadata, index)
    
    # Format response
    is_single_result = any(keyword in message.lower() for keyword in ['cheapest', 'most expensive'])
    return format_product_response(products, is_single_result)

def is_follow_up_question(message):
    """Check if the message is a follow-up question"""
    follow_up_patterns = [
        r'which of these',
        r'cheaper',
        r'price',
        r'cost',
        r'expensive'
    ]
    return any(re.search(pattern, message.lower()) for pattern in follow_up_patterns)

def handle_follow_up_question(message, blog_embeddings, metadata, index):
    """Handle follow-up questions about products"""
    # Implement follow-up question handling logic
    # This should match the logic in your existing code
    return "I'm processing your follow-up question..."

def handle_message(message, product_terms, blog_embeddings, blog_metadata, index):
    """
    Handle an incoming message and return relevant blog articles.
    
    Args:
        message (str): The user's message
        product_terms (dict): Dictionary of product terms and their alternatives
        blog_embeddings (list): List of blog article embeddings
        blog_metadata (list): List of blog article metadata
        index (faiss.Index): FAISS index for similarity search
    
    Returns:
        tuple: (response_message, search_results)
    """
    # Convert message to lowercase for case-insensitive matching
    message = message.lower()
    
    # Check for general appliance queries
    general_terms = ["home appliances", "appliances", "household appliances", "kitchen appliances", "bathroom appliances"]
    if any(term in message for term in general_terms):
        # Generate a general query embedding
        query_embedding = np.zeros(len(blog_embeddings[0]))
        for product, info in product_terms.items():
            # Add 1 to positions corresponding to all products
            for alt in info['alternatives']:
                idx = hash(alt) % len(query_embedding)
                query_embedding[idx] += 1
        
        # Normalize the query embedding
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Search for similar articles
        k = 5  # Return more results for general queries
        distances, indices = index.search(query_embedding.reshape(1, -1).astype('float32'), k)
        
        # Format results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(blog_metadata):
                article = blog_metadata[idx]
                results.append({
                    'title': article['title'],
                    'url': article['url'],
                    'content': article['content'],
                    'score': float(distances[0][i])
                })
        
        # Generate response message
        response = "Here are some articles about home appliances:\n\n"
        for i, result in enumerate(results, 1):
            response += f"{i}. {result['title']}\n"
            response += f"   Read more: {result['url']}\n\n"
        
        return response, results
    
    # Find matching product terms with partial matching
    matching_products = []
    query_words = set(message.split())
    
    for product, info in product_terms.items():
        product_words = set(product.lower().split())
        # Check if any word from the query matches with the product name
        if any(word in product_words for word in query_words):
            matching_products.append(product)
            continue
            
        # Check alternatives
        for alt in info['alternatives']:
            alt_words = set(alt.lower().split())
            if any(word in alt_words for word in query_words):
                matching_products.append(product)
                break
                
        # Check categories
        for cat in info['categories']:
            cat_words = set(cat.lower().split())
            if any(word in cat_words for word in query_words):
                matching_products.append(product)
                break
    
    if not matching_products:
        return "I couldn't find any products matching your query. Could you please try again with a different product name?", None
    
    # Generate query embedding with improved weighting
    query_embedding = np.zeros(len(blog_embeddings[0]))
    for product in matching_products:
        if product in product_terms:
            # Add 2 to positions corresponding to matching products (higher weight)
            for alt in product_terms[product]['alternatives']:
                idx = hash(alt) % len(query_embedding)
                query_embedding[idx] += 2
            # Add 1 to positions corresponding to categories (lower weight)
            for cat in product_terms[product]['categories']:
                idx = hash(cat) % len(query_embedding)
                query_embedding[idx] += 1
    
    # Normalize the query embedding
    query_embedding = query_embedding / np.linalg.norm(query_embedding)
    
    # Search for similar articles with more results
    k = 5  # Increased number of results
    distances, indices = index.search(query_embedding.reshape(1, -1).astype('float32'), k)
    
    # Format results
    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(blog_metadata):
            article = blog_metadata[idx]
            # Only include results that are relevant to the query
            content_lower = article['content'].lower()
            if any(product.lower() in content_lower or 
                   any(alt.lower() in content_lower for alt in product_terms[product]['alternatives'])
                   for product in matching_products):
                results.append({
                    'title': article['title'],
                    'url': article['url'],
                    'content': article['content'],
                    'score': float(distances[0][i])
                })
    
    if not results:
        return "I couldn't find any relevant articles about your query. Could you please try again with a different product name?", None
    
    # Generate response message
    response = f"Here are some articles about {', '.join(matching_products)}:\n\n"
    for i, result in enumerate(results, 1):
        response += f"{i}. {result['title']}\n"
        response += f"   Read more: {result['url']}\n\n"
    
    return response, results

def handle_follow_up_question(message, last_query, last_results, product_terms, blog_embeddings, blog_metadata, index):
    """
    Handle follow-up questions about previous search results.
    
    Args:
        message (str): The user's follow-up question
        last_query (str): The previous query
        last_results (list): Results from the previous query
        product_terms (dict): Dictionary of product terms and their alternatives
        blog_embeddings (list): List of blog article embeddings
        blog_metadata (list): List of blog article metadata
        index (faiss.Index): FAISS index for similarity search
    
    Returns:
        str: Response message
    """
    message = message.lower()
    
    # Handle price-related follow-up questions
    if "cheapest" in message or "lowest" in message:
        # Sort results by price (if available) and return the cheapest
        return f"Based on your previous query about '{last_query}', here's the most affordable option:\n\n{last_results[0]['title']}\n{last_results[0]['url']}"
    
    elif "expensive" in message or "highest" in message:
        # Sort results by price (if available) and return the most expensive
        return f"Based on your previous query about '{last_query}', here's the most expensive option:\n\n{last_results[-1]['title']}\n{last_results[-1]['url']}"
    
    elif "all" in message or "show all" in message:
        # Show all results again
        response = "Here are all the results from your previous query:\n\n"
        for i, result in enumerate(last_results, 1):
            response += f"{i}. {result['title']}\n"
            response += f"   Read more: {result['url']}\n\n"
        return response
    
    else:
        # Default response for unrecognized follow-up questions
        return "I'm not sure how to answer that follow-up question. Could you please try asking about the cheapest or most expensive options, or ask a new question?"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True) 