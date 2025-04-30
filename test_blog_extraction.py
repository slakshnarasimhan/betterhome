import requests
from bs4 import BeautifulSoup
import json
import time
import traceback
from process_blogs import BLOG_SOURCES, extract_blog_content

def test_blog_extraction():
    """
    Test the blog extraction functionality for each source.
    This script will attempt to extract content from one article from each blog source
    and save the results to a JSON file for inspection.
    """
    print("Testing blog extraction for each source...")
    
    # Load product terms if available
    try:
        with open('product_terms.json', 'r') as f:
            product_terms = json.load(f)
    except FileNotFoundError:
        print("Warning: product_terms.json not found. Using empty product terms.")
        product_terms = {}
    
    results = {}
    
    for source_config in BLOG_SOURCES:
        print(f"\nTesting extraction from {source_config['name']}...")
        
        try:
            # Fetch the first page to find an article URL
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Connection': 'keep-alive',
            }
            
            response = requests.get(source_config["base_url"], headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find the first article link
            article_links = soup.find_all('a', href=lambda href: href and 
                                        source_config["article_url_pattern"] in href and 
                                        not any(x in href for x in ['?page=', '/tagged/', '#']))
            
            if not article_links:
                print(f"No article links found for {source_config['name']}")
                results[source_config['name']] = {"error": "No article links found"}
                continue
            
            # Get the first article URL
            link = article_links[0]
            if link['href'].startswith('http'):
                url = link['href']
            else:
                url = f"https://{source_config['domain']}{link['href']}" if not link['href'].startswith('/') else f"https://{source_config['domain']}{link['href']}"
            
            print(f"Testing extraction from URL: {url}")
            
            # Extract content
            blog_data = extract_blog_content(url, product_terms, source_config)
            
            if blog_data:
                # Save a summary of the extracted data
                summary = {
                    "title": blog_data.get("title", ""),
                    "url": blog_data.get("url", ""),
                    "content_length": len(blog_data.get("content", "")),
                    "has_videos": blog_data.get("has_videos", False),
                    "categories": blog_data.get("categories", []),
                    "related_words": blog_data.get("related_words", [])
                }
                results[source_config['name']] = summary
                print(f"Successfully extracted content from {source_config['name']}")
            else:
                results[source_config['name']] = {"error": "Failed to extract content"}
                print(f"Failed to extract content from {source_config['name']}")
                
        except Exception as e:
            print(f"Error testing {source_config['name']}: {e}")
            traceback.print_exc()
            results[source_config['name']] = {"error": str(e)}
        
        # Add a delay between requests
        time.sleep(2)
    
    # Save results to a file
    with open('blog_extraction_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nTest results saved to blog_extraction_test_results.json")
    
    # Print summary
    print("\nTest Summary:")
    for source, result in results.items():
        if "error" in result:
            print(f"{source}: ❌ {result['error']}")
        else:
            print(f"{source}: ✅ Successfully extracted content (title: {result['title'][:50]}...)")

if __name__ == "__main__":
    test_blog_extraction() 