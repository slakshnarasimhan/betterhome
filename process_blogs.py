import pandas as pd
import numpy as np
import json
import openai
import streamlit as st
from datetime import datetime

# Blog data structure
blogs = [
    {
        "title": "Soft closing systems are safe & trendy: in doors, cupboards, drawers and even toilets",
        "date": "2025-02-06",
        "author": "Balaji Kalyansundaram",
        "content": "For your wardrobes, cupboards, drawers, and even toilet seats, please use a soft close system. You won't hurt yourself.",
        "categories": ["Hardware", "Door Fitting", "Bathroom Accessories"],
        "url": "https://betterhomeapp.com/blogs/articles/soft-closing-systems-safe-trendy"
    },
    {
        "title": "Toughened glass saves action heroes and your family ✅",
        "date": "2025-02-06",
        "author": "Balaji Kalyansundaram",
        "content": "If the glass in your home is not a toughened glass like this, throw it away today. It could even cause death. Let's see what is a toughened glass. Normal glass breaks into big sharp pieces which cut our arms and legs. It could even be fatal.But in cinemas, the hero and villain will...",
        "categories": ["Glass and Mirror", "Glass"],
        "url": "https://betterhomeapp.com/blogs/articles/toughened-glass-saves-action-heroes"
    },
    {
        "title": "Switch to BLDC Fans in 2025: Save the Planet, Save 1,500 every year!",
        "date": "2024-12-31",
        "author": "Balaji Kalyansundaram",
        "content": "In 2025, let's make a conscious effort to reduce our carbon footprint and lighten our electricity bills by ₹1,500 every year. Switching to BLDC (Brushless Direct Current) fans is a simple yet impactful step towards achieving both these goals.",
        "categories": ["Electrical", "Ceiling Fan"],
        "url": "https://betterhomeapp.com/blogs/articles/switch-to-bldc-fans-2025"
    },
    {
        "title": "Termite worms can damage your modular kitchen made of low quality plywoods",
        "date": "2024-10-29",
        "author": "Balaji Kalyansundaram",
        "content": "If your modular kitchen is not made with good plywood, do you know it can be damaged by termite worms? As it is exposed to water again and again, the plywood will decompose and be damaged by worms. Water leakage happens mainly due to bad sealing around the sink. So definitely buy a quality...",
        "categories": ["Kitchen", "Timber", "Plywood"],
        "url": "https://betterhomeapp.com/blogs/articles/termite-worms-damage-modular-kitchen"
    }
]

def generate_blog_embeddings(blogs, openai_api_key):
    """Generate embeddings for blog articles using OpenAI API"""
    openai.api_key = openai_api_key
    blog_embeddings = []
    
    for blog in blogs:
        # Combine title and content for embedding
        text = f"{blog['title']}\n{blog['content']}"
        try:
            response = openai.Embedding.create(
                model="text-embedding-ada-002",
                input=text
            )
            embedding = response['data'][0]['embedding']
            blog_embeddings.append({
                'embedding': embedding,
                'title': blog['title'],
                'date': blog['date'],
                'categories': blog['categories'],
                'url': blog['url']
            })
        except Exception as e:
            print(f"Error generating embedding for blog: {blog['title']}, Error: {e}")
    
    return blog_embeddings

def save_blog_embeddings(blog_embeddings, output_file='blog_embeddings.json'):
    """Save blog embeddings to a JSON file"""
    with open(output_file, 'w') as f:
        json.dump({
            'blog_embeddings': [item['embedding'] for item in blog_embeddings],
            'metadata': [{
                'title': item['title'],
                'date': item['date'],
                'categories': item['categories'],
                'url': item['url']
            } for item in blog_embeddings]
        }, f)

def main():
    # Get API key from Streamlit secrets
    openai_api_key = st.secrets["OPENAI_API_KEY"]
    
    # Generate embeddings
    blog_embeddings = generate_blog_embeddings(blogs, openai_api_key)
    
    # Save embeddings
    save_blog_embeddings(blog_embeddings)
    
    print("Blog embeddings generated and saved successfully!")

if __name__ == "__main__":
    main() 