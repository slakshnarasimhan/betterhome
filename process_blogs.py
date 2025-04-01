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

# Configure OpenAI
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
openai.api_key = OPENAI_API_KEY
MODEL_NAME = "text-embedding-3-small"

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

def fetch_blog_urls():
    base_url = "https://betterhomeapp.com/blogs/articles"
    blog_urls = []
    page = 1

    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
        }

        while True:
            page_url = f"{base_url}?page={page}" if page > 1 else base_url
            print(f"\nFetching URLs from page {page}: {page_url}")

            response = requests.get(page_url, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            article_links = soup.find_all('a', href=lambda href: href and 
                                        '/blogs/articles/' in href and 
                                        not any(x in href for x in ['?page=', '/tagged/', '#']))

            page_urls = []
            seen_urls = set()
            for link in article_links:
                url = f"https://betterhomeapp.com{link['href']}"
                if url not in seen_urls and not url.endswith('/articles'):
                    page_urls.append(url)
                    seen_urls.add(url)
                    title_tag = link.find('h3') or link.find_next('h3')
                    title = title_tag.get_text().strip() if title_tag else 'Unknown Title'
                    print(f"Found article: {title}")

            if not page_urls:
                break

            blog_urls.extend(page_urls)
            print(f"Added {len(page_urls)} articles from page {page}")

            if page >= 3:
                break

            page += 1
            time.sleep(1)

        blog_urls = list(set(blog_urls))
        print(f"\nTotal unique blog articles found: {len(blog_urls)}")
        return blog_urls

    except Exception as e:
        print(f"Error fetching blog URLs: {e}")
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

def extract_blog_content(url, product_terms):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        slug = url.split('/')[-1] if '/' in url else ''
        title = soup.find('h1', class_='title') or soup.find('h1')
        title_text = title.get_text().strip() if title else f"Article about {slug.replace('-', ' ')}"

        info_div = soup.find('div', class_='info')
        date, author = '', ''
        if info_div:
            date_span = info_div.find('time')
            author_span = info_div.find('span', class_='article-author')
            date = date_span.get_text().strip() if date_span else ''
            author = author_span.get_text().strip() if author_span else ''

        content = ''
        for div_class in ['article-content', 'rte']:
            if content:
                break
            content_div = soup.find('div', class_=div_class)
            if content_div:
                content_elements = content_div.find_all(['p', 'h2', 'h3', 'h4', 'li'])
                content = ' '.join([elem.get_text().strip() for elem in content_elements])

        if not content:
            article = soup.find('article')
            if article:
                for tag in article.find_all(['header', 'footer']):
                    tag.decompose()
                content_elements = article.find_all(['p', 'h2', 'h3', 'h4', 'li'])
                content = ' '.join([elem.get_text().strip() for elem in content_elements])

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

        tags = []
        tag_ul = soup.find('ul', class_='tag-list list-unstyled clearfix')
        if tag_ul:
            tag_spans = tag_ul.find_all('span', class_='text')
            tags = [tag.get_text().strip() for tag in tag_spans]

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
            'video_ids': list(video_ids)
        }

        return blog_data
    except Exception as e:
        print(f"Error extracting content from {url}: {str(e)}")
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
            'content': article.get('content', '')[:500]
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

def main():
    print("Starting blog crawling and embedding generation...")
    with open('product_terms.json', 'r') as f:
        product_terms = json.load(f)

    blog_urls = fetch_blog_urls()
    if not blog_urls:
        print("No blog URLs found. Exiting.")
        return

    blogs = []
    for url in blog_urls:
        blog_data = extract_blog_content(url, product_terms)
        if blog_data:
            blogs.append(blog_data)

    if not blogs:
        print("No blog contents extracted. Exiting.")
        return

    print(f"\nExtracted {len(blogs)} blog articles")
    print("\nGenerating embeddings...")
    blog_embeddings = generate_blog_embeddings(blogs)

    if not blog_embeddings:
        print("No embeddings generated. Exiting.")
        return

    print("\nSaving embeddings...")
    save_blog_embeddings(blog_embeddings, blogs)

    print("\nBlog embeddings generated and saved successfully!")
    print(f"Total blogs processed: {len(blog_embeddings)}")

if __name__ == "__main__":
    main()

