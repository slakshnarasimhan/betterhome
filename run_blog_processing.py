import argparse
import time
import traceback
from process_blogs import main, BLOG_SOURCES

def run_blog_processing(sources=None, max_articles_per_source=None):
    """
    Run the blog processing for specified sources or all sources.
    
    Args:
        sources (list): List of source names to process. If None, process all sources.
        max_articles_per_source (int): Maximum number of articles to process per source. If None, process all.
    """
    print("Starting blog processing...")
    
    # Filter sources if specified
    if sources:
        filtered_sources = [s for s in BLOG_SOURCES if s["name"] in sources]
        if not filtered_sources:
            print(f"Error: No valid sources found in {sources}")
            return
        print(f"Processing sources: {', '.join(sources)}")
    else:
        filtered_sources = BLOG_SOURCES
        print("Processing all sources")
    
    # Modify max_pages if max_articles_per_source is specified
    if max_articles_per_source:
        for source in filtered_sources:
            source["max_pages"] = 1  # Start with 1 page
            source["max_articles_per_source"] = max_articles_per_source
    
    # Run the main processing
    try:
        main(filtered_sources)
    except Exception as e:
        print(f"Error during blog processing: {e}")
        traceback.print_exc()

def main_wrapper():
    parser = argparse.ArgumentParser(description="Process blogs from various sources")
    parser.add_argument("--sources", nargs="+", help="Specific sources to process")
    parser.add_argument("--max-articles", type=int, help="Maximum articles to process per source")
    parser.add_argument("--test", action="store_true", help="Run in test mode (process only one article per source)")
    
    args = parser.parse_args()
    
    if args.test:
        print("Running in test mode (one article per source)")
        from test_blog_extraction import test_blog_extraction
        test_blog_extraction()
    else:
        run_blog_processing(args.sources, args.max_articles)

# Add Samsung home appliances buying guide URL
BLOG_SOURCES.append({
    "name": "Samsung Home Appliances Buying Guide",
    "url": "https://www.samsung.com/in/home-appliances/buying-guide/",
    "base_url": "https://www.samsung.com/in/home-appliances/buying-guide/",
    "article_url_pattern": "/home-appliances/buying-guide/",
    "max_pages": 1  # Assuming we want to process one page
})

# Add blog URLs for the brands
BLOG_SOURCES.extend([
    {
        "name": "AO Smith Blog",
        "url": "https://www.aosmith.com/resources/blog/",
        "max_pages": 1
    },
    {
        "name": "Ashirvad Blog",
        "url": "https://www.ashirvad.com/blog/",
        "max_pages": 1
    },
    {
        "name": "Astral Blog",
        "url": "https://www.astralpipes.com/blog/",
        "max_pages": 1
    },
    {
        "name": "Better Home Blog",
        "url": "https://betterhomeapp.com/blogs/articles",
        "max_pages": 3
    }
])

if __name__ == "__main__":
    main_wrapper() 