# Blog Processing for Better Home

This system extracts content from various home appliance and kitchen blogs, generates embeddings, and creates a searchable index for providing better product guidance to users.

## Overview

The system consists of three main scripts:

1. `process_blogs.py` - Core functionality for extracting blog content and generating embeddings
2. `test_blog_extraction.py` - Test script to verify extraction from each blog source
3. `run_blog_processing.py` - Command-line interface for running the blog processing

## Supported Blog Sources

The system currently supports the following blog sources:

1. Better Home (betterhomeapp.com)
2. Kitchen Brand Store (in.kitchenbrandstore.com)
3. The Optimal Zone (theoptimalzone.in)
4. Baltra (baltra.in)
5. Crompton (www.crompton.co.in)
6. Atomberg (atomberg.com)

## Requirements

- Python 3.7+
- Required packages: requests, beautifulsoup4, pandas, numpy, openai, streamlit, faiss, tqdm, selenium, youtube_transcript_api

## Usage

### Testing Blog Extraction

To test the extraction from each blog source (extracts one article from each source):

```bash
python run_blog_processing.py --test
```

### Processing All Blog Sources

To process all blog sources:

```bash
python run_blog_processing.py
```

### Processing Specific Sources

To process only specific blog sources:

```bash
python run_blog_processing.py --sources "Better Home" "Kitchen Brand Store"
```

### Limiting Articles Per Source

To limit the number of articles processed per source:

```bash
python run_blog_processing.py --max-articles 10
```

## Output Files

The system generates the following output files:

1. `blog_embeddings.json` - Contains the blog embeddings and metadata
2. `blog_faiss_index.index` - FAISS index for efficient similarity search
3. `blog_extraction_test_results.json` - Results from the test extraction (when running in test mode)

## Adding New Blog Sources

To add a new blog source, add a new entry to the `BLOG_SOURCES` list in `process_blogs.py` with the following structure:

```python
{
    "name": "Source Name",
    "base_url": "https://example.com/blog",
    "article_url_pattern": "/blog/",
    "domain": "example.com",
    "max_pages": 5,
    "title_selector": "h1, h2",
    "content_selectors": ["div.blog-content", "div.entry-content", "article"],
    "date_selector": "time, span.date",
    "author_selector": "span.author, a.author",
    "tags_selector": "div.tags a, ul.tags li"
}
```

## Troubleshooting

If you encounter issues with specific blog sources:

1. Check the HTML structure of the blog site
2. Update the selectors in the `BLOG_SOURCES` configuration
3. Run the test script to verify the extraction
4. Check the logs for specific error messages

## Integration with Better Home API

The generated embeddings and FAISS index can be used with the Better Home API to provide better product guidance to users. The API can use these embeddings to find relevant blog articles based on user queries. 