import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import json
import openai
import time
from datetime import datetime
import streamlit as st
import traceback
import faiss
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from youtube_transcript_api import YouTubeTranscriptApi
import re
import os

# Configure OpenAI
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
openai.api_key = OPENAI_API_KEY
MODEL_NAME = "text-embedding-3-small"

# Define blog sources
BLOG_SOURCES = [
    {
        "name": "Better Home",
        "base_url": "https://betterhomeapp.com/blogs/articles",
        "article_url_pattern": "/blogs/articles/",
        "domain": "betterhomeapp.com",
        "max_pages": 3,
        "title_selector": "h3",
        "content_selectors": ["div.article-content", "div.rte", "article"],
        "date_selector": "time",
        "author_selector": "span.article-author",
        "tags_selector": "ul.tag-list.list-unstyled.clearfix span.text"
    },
    {
        "name": "Kitchen Brand Store",
        "base_url": "https://in.kitchenbrandstore.com/blog",
        "article_url_pattern": "/blog/",
        "domain": "in.kitchenbrandstore.com",
        "max_pages": 5,
        "title_selector": "h2, h3, h4",
        "content_selectors": ["div.blog-content", "div.entry-content", "article"],
        "date_selector": "time, span.date",
        "author_selector": "span.author, a.author",
        "tags_selector": "div.tags a, ul.tags li"
    },
    {
        "name": "The Optimal Zone",
        "base_url": "https://theoptimalzone.in",
        "article_url_pattern": "/",
        "domain": "theoptimalzone.in",
        "max_pages": 5,
        "title_selector": "h1, h2",
        "content_selectors": ["div.entry-content", "article", "div.content"],
        "date_selector": "time, span.date",
        "author_selector": "span.author, a.author",
        "tags_selector": "div.tags a, ul.tags li"
    },
    {
        "name": "Baltra",
        "base_url": "https://baltra.in/blog/",
        "article_url_pattern": "/blog/",
        "domain": "baltra.in",
        "max_pages": 5,
        "title_selector": "h1, h2",
        "content_selectors": ["div.blog-content", "div.entry-content", "article"],
        "date_selector": "time, span.date",
        "author_selector": "span.author, a.author",
        "tags_selector": "div.tags a, ul.tags li"
    },
    {
        "name": "Crompton",
        "base_url": "https://www.crompton.co.in/pages/blog",
        "article_url_pattern": "/pages/blog/",
        "domain": "www.crompton.co.in",
        "max_pages": 5,
        "title_selector": "h1, h2, h3",
        "content_selectors": ["div.blog-content", "div.entry-content", "article", "div.content"],
        "date_selector": "time, span.date",
        "author_selector": "span.author, a.author",
        "tags_selector": "div.tags a, ul.tags li"
    },
    {
        "name": "Atomberg",
        "base_url": "https://atomberg.com/blog",
        "article_url_pattern": "/blog/",
        "domain": "atomberg.com",
        "max_pages": 5,
        "title_selector": "h1, h2",
        "content_selectors": ["div.blog-content", "div.entry-content", "article"],
        "date_selector": "time, span.date",
        "author_selector": "span.author, a.author",
        "tags_selector": "div.tags a, ul.tags li"
    }
]

def generate_blog_embeddings(blog_articles):
    embeddings = []

    def generate_batch(batch):
        try:
            response = openai.Embedding.create(
                model=MODEL_NAME,
                input=batch
            )
            return [e["embedding"] for e in response["data"]]
        except Exception as e:
            print(f"Error generating batch embeddings: {str(e)}")
            return [np.random.rand(1536).tolist() for _ in batch]

    texts = [f"{article['title']} {article['content']}" for article in blog_articles]
    batch_size = 10
    batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]

    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(generate_batch, batches), total=len(batches), desc="Generating Embeddings"))
        for batch_embeddings in results:
            embeddings.extend(batch_embeddings)

    return embeddings

def fetch_blog_urls(source_config):
    base_url = source_config["base_url"]
    blog_urls = []
    page = 1
    article_url_pattern = source_config["article_url_pattern"]
    domain = source_config["domain"]
    max_pages = source_config.get("max_pages", 3)
    title_selector = source_config.get("title_selector", "h3")

    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
        }

        while True:
            page_url = f"{base_url}?page={page}" if page > 1 else base_url
            print(f"\nFetching URLs from {source_config['name']} page {page}: {page_url}")

            response = requests.get(page_url, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            # Find all links that match the article pattern
            article_links = soup.find_all('a', href=lambda href: href and 
                                        article_url_pattern in href and 
                                        not any(x in href for x in ['?page=', '/tagged/', '#']))

            page_urls = []
            seen_urls = set()
            for link in article_links:
                # Handle both relative and absolute URLs
                if link['href'].startswith('http'):
                    url = link['href']
                else:
                    url = f"https://{domain}{link['href']}" if not link['href'].startswith('/') else f"https://{domain}{link['href']}"
                
                if url not in seen_urls and not url.endswith('/blog') and not url.endswith('/blogs'):
                    page_urls.append(url)
                    seen_urls.add(url)
                    
                    # Try to find the title using the configured selector
                    title_tag = None
                    for selector in title_selector.split(','):
                        title_tag = link.find(selector.strip()) or link.find_next(selector.strip())
                        if title_tag:
                            break
                    
                    title = title_tag.get_text().strip() if title_tag else 'Unknown Title'
                    print(f"Found article: {title}")

            if not page_urls:
                break

            blog_urls.extend(page_urls)
            print(f"Added {len(page_urls)} articles from {source_config['name']} page {page}")

            if page >= max_pages:
                break

            page += 1
            time.sleep(1)

        blog_urls = list(set(blog_urls))
        print(f"\nTotal unique blog articles found from {source_config['name']}: {len(blog_urls)}")
        return blog_urls

    except Exception as e:
        print(f"Error fetching blog URLs from {source_config['name']}: {e}")
        traceback.print_exc()
        return []

def extract_youtube_id(url):
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?]+)',
        r'youtube\.com\/shorts\/([^&\n?]+)'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def get_video_transcript(video_id, max_retries=3):
    for attempt in range(max_retries):
        try:
            print(f"Attempting to get transcript for video {video_id} (attempt {attempt + 1})")
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            return ' '.join([entry['text'] for entry in transcript])
        except Exception as e:
            print(f"Attempt {attempt + 1} failed for video {video_id}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                print(f"Failed to get transcript for video {video_id} after {max_retries} attempts")
                return None

def extract_blog_content(url, product_terms, source_config):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        slug = url.split('/')[-1] if '/' in url else ''
        
        # Find title using configured selector
        title = None
        for selector in source_config.get("title_selector", "h1").split(','):
            title = soup.find(selector.strip())
            if title:
                break
        
        title_text = title.get_text().strip() if title else f"Article about {slug.replace('-', ' ')}"

        # Find date and author using configured selectors
        date, author = '', ''
        
        date_selector = source_config.get("date_selector", "")
        if date_selector:
            for selector in date_selector.split(','):
                date_elem = soup.find(selector.strip())
                if date_elem:
                    date = date_elem.get_text().strip()
                    break
        
        author_selector = source_config.get("author_selector", "")
        if author_selector:
            for selector in author_selector.split(','):
                author_elem = soup.find(selector.strip())
                if author_elem:
                    author = author_elem.get_text().strip()
                    break

        # Extract content using configured selectors
        content = ''
        for selector in source_config.get("content_selectors", ["div.article-content", "div.rte", "article"]):
            if content:
                break
                
            content_div = soup.find(selector)
            if content_div:
                # Remove unwanted elements
                for unwanted in content_div.find_all(['script', 'style', 'iframe', 'noscript']):
                    unwanted.decompose()
                
                content_elements = content_div.find_all(['p', 'h2', 'h3', 'h4', 'li'])
                content = ' '.join([elem.get_text().strip() for elem in content_elements])

        # If no content found with selectors, try a more generic approach
        if not content:
            article = soup.find('article')
            if article:
                for tag in article.find_all(['header', 'footer', 'script', 'style', 'iframe', 'noscript']):
                    tag.decompose()
                content_elements = article.find_all(['p', 'h2', 'h3', 'h4', 'li'])
                content = ' '.join([elem.get_text().strip() for elem in content_elements])

        # Extract video transcripts
        video_transcripts = []
        video_ids = set()
        for link in soup.find_all('a', href=True):
            video_id = extract_youtube_id(link['href'])
            if video_id and video_id not in video_ids:
                video_ids.add(video_id)
                transcript = get_video_transcript(video_id)
                if transcript:
                    video_transcripts.append(transcript)

        for iframe in soup.find_all('iframe'):
            video_id = extract_youtube_id(iframe.get('src', ''))
            if video_id and video_id not in video_ids:
                video_ids.add(video_id)
                transcript = get_video_transcript(video_id)
                if transcript:
                    video_transcripts.append(transcript)

        if video_transcripts:
            content += "\n\nVideo Transcripts:\n" + "\n\n".join(video_transcripts)

        # Extract tags using configured selector
        tags = []
        tags_selector = source_config.get("tags_selector", "")
        if tags_selector:
            for selector in tags_selector.split(','):
                tag_elements = soup.find_all(selector.strip())
                if tag_elements:
                    tags = [tag.get_text().strip() for tag in tag_elements]
                    break

        # Find related product terms
        related_words = set()
        for product, info in product_terms.items():
            if product.lower() in content.lower():
                related_words.add(product)
            for alt in info['alternatives']:
                if alt.lower() in content.lower():
                    related_words.add(product)
            for cat in info['categories']:
                if cat.lower() in content.lower():
                    related_words.add(product)

        blog_data = {
            'title': title_text,
            'date': date,
            'author': author,
            'content': content,
            'categories': tags,
            'related_words': list(related_words),
            'url': url,
            'slug': slug,
            'has_videos': len(video_transcripts) > 0,
            'video_ids': list(video_ids),
            'source': source_config['name']
        }

        return blog_data
    except Exception as e:
        print(f"Error extracting content from {url}: {str(e)}")
        traceback.print_exc()
        return None

def save_blog_embeddings(blog_embeddings, blog_articles, output_file='blog_embeddings.json'):
    if len(blog_embeddings) != len(blog_articles):
        print(f"Warning: Mismatch between embeddings ({len(blog_embeddings)}) and articles ({len(blog_articles)})")
        min_length = min(len(blog_embeddings), len(blog_articles))
        blog_embeddings = blog_embeddings[:min_length]
        blog_articles = blog_articles[:min_length]

    metadata = []
    for article in blog_articles:
        meta = {
            'title': article.get('title', 'Untitled'),
            'url': article.get('url', ''),
            'date': article.get('date', ''),
            'author': article.get('author', ''),
            'categories': article.get('categories', []),
            'content': article.get('content', '')[:500],
            'source': article.get('source', 'Unknown')
        }
        metadata.append(meta)

    print(f"Saving {len(blog_embeddings)} embeddings with full metadata to {output_file}")
    with open(output_file, 'w') as f:
        json.dump({'blog_embeddings': blog_embeddings, 'metadata': metadata}, f)
    print(f"Saved embeddings and metadata to {output_file}")

    try:
        print("Creating FAISS index for blog embeddings...")
        embeddings_array = np.array(blog_embeddings).astype('float32')
        dimension = embeddings_array.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings_array)
        print(f"Created FAISS index with dimension {index.d}")
        faiss.write_index(index, 'blog_faiss_index.index')
        print("Saved FAISS index")
    except Exception as e:
        print(f"Error creating FAISS index: {e}")
        traceback.print_exc()

def main(sources=None):
    print("Starting blog crawling and embedding generation...")
    
    # Load product terms
    try:
        with open('product_terms.json', 'r') as f:
            product_terms = json.load(f)
    except FileNotFoundError:
        print("Warning: product_terms.json not found. Creating empty product terms.")
        product_terms = {}
    
    all_blogs = []
    
    # Use provided sources or all sources
    sources_to_process = sources if sources else BLOG_SOURCES
    
    # Process each blog source
    for source_config in sources_to_process:
        print(f"\nProcessing blog source: {source_config['name']}")
        
        blog_urls = fetch_blog_urls(source_config)
        if not blog_urls:
            print(f"No blog URLs found for {source_config['name']}. Skipping.")
            continue
        
        # Limit the number of articles if specified
        max_articles = source_config.get("max_articles_per_source")
        if max_articles and len(blog_urls) > max_articles:
            print(f"Limiting to {max_articles} articles for {source_config['name']}")
            blog_urls = blog_urls[:max_articles]
        
        source_blogs = []
        for url in tqdm(blog_urls, desc=f"Extracting content from {source_config['name']}"):
            blog_data = extract_blog_content(url, product_terms, source_config)
            if blog_data:
                source_blogs.append(blog_data)
        
        print(f"Extracted {len(source_blogs)} articles from {source_config['name']}")
        all_blogs.extend(source_blogs)
    
    if not all_blogs:
        print("No blog contents extracted from any source. Exiting.")
        return
    
    print(f"\nExtracted {len(all_blogs)} blog articles in total")
    print("\nGenerating embeddings...")
    blog_embeddings = generate_blog_embeddings(all_blogs)
    
    if not blog_embeddings:
        print("No embeddings generated. Exiting.")
        return
    
    print("\nSaving embeddings...")
    save_blog_embeddings(blog_embeddings, all_blogs)
    
    print("\nBlog embeddings generated and saved successfully!")
    print(f"Total blogs processed: {len(blog_embeddings)}")
    
    # Print summary by source
    source_counts = {}
    for blog in all_blogs:
        source = blog.get('source', 'Unknown')
        source_counts[source] = source_counts.get(source, 0) + 1
    
    print("\nSummary by source:")
    for source, count in source_counts.items():
        print(f"{source}: {count} articles")

if __name__ == "__main__":
    main()

