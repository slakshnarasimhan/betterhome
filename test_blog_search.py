import json
import numpy as np
import faiss
import traceback
import os
import openai

# Configuration
BLOG_EMBEDDINGS_FILE_PATH = 'blog_embeddings.json'
BLOG_INDEX_FILE_PATH = 'faiss_index.index_blog'

# Set up OpenAI API key
try:
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
    if not OPENAI_API_KEY:
        with open('openai_api_key.txt', 'r') as f:
            OPENAI_API_KEY = f.read().strip()
    openai.api_key = OPENAI_API_KEY
except Exception as e:
    print(f"Error loading OpenAI API key: {str(e)}")
    OPENAI_API_KEY = "not-set"
    openai.api_key = OPENAI_API_KEY

def get_openai_embedding(text, model='text-embedding-ada-002'):
    try:
        response = openai.Embedding.create(
            model=model,
            input=text
        )
        embedding = response['data'][0]['embedding']
        return np.array(embedding).reshape(1, -1).astype('float32')
    except Exception as e:
        print(f"Error generating embedding with OpenAI: {e}")
        return np.random.rand(1, 1536).astype('float32')

def build_or_load_faiss_index(embeddings, dimension, index_path):
    if os.path.exists(index_path):
        try:
            index = faiss.read_index(index_path)
            if index.d != dimension:
                print(f"Dimension mismatch in existing index. Rebuilding index with correct dimensions ({dimension})")
                os.remove(index_path)
                index = faiss.IndexFlatL2(dimension)
                index.add(embeddings)
                faiss.write_index(index, index_path)
        except Exception as e:
            print(f"Error reading index: {str(e)}. Rebuilding index.")
            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings)
            faiss.write_index(index, index_path)
    else:
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        faiss.write_index(index, index_path)
    return index

def load_blog_embeddings(file_path):
    try:
        print(f"Loading blog embeddings from {file_path}")
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Validate the format of the loaded data
        if 'blog_embeddings' not in data or 'metadata' not in data:
            print(f"Invalid blog embeddings file format. Expected 'blog_embeddings' and 'metadata' keys. Found: {list(data.keys())}")
            return None
        
        # Check if there are any blog embeddings
        if len(data['blog_embeddings']) == 0:
            print("Blog embeddings array is empty")
            return None
            
        # Check if there is metadata for each embedding
        if len(data['blog_embeddings']) != len(data['metadata']):
            print(f"Mismatch between embeddings ({len(data['blog_embeddings'])}) and metadata ({len(data['metadata'])})")
        
        # Check if metadata has the necessary fields (title, url, etc.)
        if data['metadata'] and len(data['metadata']) > 0:
            sample_metadata = data['metadata'][0]
            print(f"Sample metadata fields: {list(sample_metadata.keys())}")
            
            # Check for URL field
            url_field_present = any('url' in item for item in data['metadata'])
            if not url_field_present:
                print("WARNING: No 'url' field found in blog metadata. URLs may not display correctly.")
            
            # Check for title field
            title_field_present = any('title' in item for item in data['metadata'])
            if not title_field_present:
                print("WARNING: No 'title' field found in blog metadata. Titles may not display correctly.")
        
        print(f"Successfully loaded {len(data['blog_embeddings'])} blog embeddings with {len(data['metadata'])} metadata entries")
        return {
            'blog_embeddings': np.array(data['blog_embeddings']),
            'metadata': data['metadata']
        }
    except FileNotFoundError:
        print(f"Blog embeddings file not found: {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error decoding JSON from blog embeddings file: {file_path}")
        return None
    except Exception as e:
        print(f"Error loading blog embeddings: {str(e)}")
        traceback.print_exc()
        return None

def search_relevant_blogs(query, blog_embeddings_dict, k=3, similarity_threshold=0.3):
    print("Starting blog search...")
    print(f"Query: {query}")
    print(f"Similarity threshold: {similarity_threshold}")
    
    # Debug: Check if blog_embeddings_dict is valid
    if not blog_embeddings_dict:
        print("ERROR: blog_embeddings_dict is None or empty")
        return []
    
    # Debug: Check the structure of blog_embeddings_dict
    print(f"Keys in blog_embeddings_dict: {list(blog_embeddings_dict.keys())}")
    
    blog_embeddings = blog_embeddings_dict['blog_embeddings']
    print(f"Blog embeddings type: {type(blog_embeddings)}")
    print(f"Blog embeddings shape: {blog_embeddings.shape if hasattr(blog_embeddings, 'shape') else 'No shape attribute'}")
    print(f"Number of blog embeddings: {len(blog_embeddings)}")
    
    # Generate query embedding using OpenAI for consistency
    try:
        query_embedding = get_openai_embedding(query)
        print(f"Generated query embedding using OpenAI: {query_embedding.shape}")
    except Exception as e:
        print(f"Error generating query embedding: {str(e)}")
        traceback.print_exc()
        return []
    
    # Build or load FAISS index for blog search
    try:
        blog_index = build_or_load_faiss_index(
            blog_embeddings,
            query_embedding.shape[1],  # Use the same dimension as the query embedding
            BLOG_INDEX_FILE_PATH
        )
        print(f"Successfully built/loaded FAISS index with {blog_index.d} dimensions")
    except Exception as e:
        print(f"Error building blog index: {str(e)}")
        traceback.print_exc()
        return []
    
    # Only search if we have blog embeddings
    if len(blog_embeddings) > 0:
        # Search for more blog posts than needed so we can filter
        search_k = min(k * 10, len(blog_embeddings_dict['metadata']))  # Increased from k * 5 to k * 10
        print(f"Searching for top {search_k} blog articles")
        D, I = blog_index.search(query_embedding, search_k)
        print(f"Found {len(I[0])} initial matching blog articles with distances: {D[0]}")
        
        # Get the metadata for found articles
        results = []
        query_lower = query.lower()
        is_chimney_query = "chimney" in query_lower
        
        # First pass: collect all potential matches with their scores
        potential_matches = []
        for idx, (distance, i) in enumerate(zip(D[0], I[0])):
            if i < len(blog_embeddings_dict['metadata']):
                metadata = blog_embeddings_dict['metadata'][i]
                
                # Debug: Print the title of the article being processed
                title = metadata.get('title', 'Untitled Article')
                print(f"Processing blog {i}: {title}")
                
                # Calculate base similarity score
                similarity_score = 1.0 / (1.0 + distance)  # Convert distance to similarity (0-1 scale)
                print(f"Blog {i}: Base similarity score = {similarity_score:.3f}")
                
                # Check if query matches related words or categories
                matches_related_words = any(word.lower() in query_lower for word in metadata.get('related_words', []))
                matches_categories = any(tag.lower() in query_lower for tag in metadata.get('categories', []))
                
                # For chimney queries, check if the title or content contains "chimney"
                matches_chimney = False
                if is_chimney_query:
                    title_lower = metadata.get('title', '').lower()
                    content_lower = metadata.get('content', '').lower()
                    matches_chimney = "chimney" in title_lower or "chimney" in content_lower
                    
                    # Boost score for articles with "chimney" in title or content
                    if matches_chimney:
                        if "chimney" in title_lower:
                            similarity_score *= 2.0  # Double the score for title matches
                            print(f"Blog {i}: Score doubled due to chimney in title")
                        else:
                            similarity_score *= 1.5  # 1.5x score for content matches
                            print(f"Blog {i}: Score boosted by 1.5x due to chimney in content")
                    
                    print(f"Blog {i}: Matches chimney = {matches_chimney}")
                
                print(f"Blog {i}: Matches related words = {matches_related_words}, Matches categories = {matches_categories}")
                print(f"Blog {i}: Final similarity score = {similarity_score:.3f}")
                
                # Only include if similarity score is above threshold or matches criteria
                if similarity_score > similarity_threshold or matches_related_words or matches_categories or matches_chimney:
                    metadata['_similarity_score'] = similarity_score
                    potential_matches.append(metadata)
                    print(f"Added blog {i} to potential matches with similarity score {similarity_score:.3f}")
                else:
                    print(f"Skipping blog {i} due to low similarity score: {similarity_score:.3f}")
            else:
                print(f"Index {i} is out of bounds for metadata array of length {len(blog_embeddings_dict['metadata'])}")
        
        # Sort by similarity score and limit to k results
        if potential_matches:
            # Sort by similarity score
            potential_matches.sort(key=lambda x: x.get('_similarity_score', 0), reverse=True)
            
            # For chimney queries, ensure chimney-related articles appear first
            if is_chimney_query:
                chimney_articles = [article for article in potential_matches 
                                  if "chimney" in article.get('title', '').lower() 
                                  or "chimney" in article.get('content', '').lower()]
                other_articles = [article for article in potential_matches 
                                if article not in chimney_articles]
                results = chimney_articles + other_articles
            else:
                results = potential_matches
            
            # Limit to k results
            results = results[:k]
            print(f"Returning {len(results)} most relevant blog articles")
            
            # Debug: Print the titles of the returned articles
            for i, result in enumerate(results):
                print(f"Result {i+1}: {result.get('title', 'Untitled Article')} (Score: {result.get('_similarity_score', 0):.3f})")
            return results
        else:
            print("No sufficiently relevant blog articles found")
            return []
    else:
        print("No blog embeddings to search")
        return []

def format_blog_response(blog_results, query=None):
    if not blog_results or len(blog_results) == 0:
        return None
    
    # Extract query topic for header if available
    topic_header = ""
    if query:
        # Try to extract the subject of the query
        query_words = query.lower().split()
        
        # Simplified logic to extract likely topic words
        stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'is', 'are', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 
                      'about', 'like', 'of', 'do', 'does', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 
                      'those', 'list', 'show', 'tell', 'me', 'get', 'can', 'could', 'would', 'should', 'how'}
        
        # Get potential topic words (non-stop words)
        topic_words = [word for word in query_words if word not in stop_words and len(word) > 3]
        
        if topic_words:
            topic_header = f" About {' '.join(topic_words[:3]).title()}"
    
    # Concise header
    response = f"### 📚 Articles{topic_header}\n\n"
    
    # Limit to 2 blog articles maximum for brevity
    blog_count = 0
    for blog in blog_results:
        if blog_count >= 2:
            break
            
        title = blog.get('title', 'Untitled Article')
        url = blog.get('url', '#')
        
        # Make sure we're not displaying generic titles like "Article X"
        if not title or title.startswith('Article '):
            title = "Blog Article (Click to read)"
        
        # Ensure we have a valid URL
        if not url or url == '#':
            # If no URL is available, try to construct one from other metadata
            if 'slug' in blog:
                url = f"https://betterhomeapp.com/blogs/articles/{blog['slug']}"
            else:
                url = "https://betterhomeapp.com/blogs/articles"
        
        # Create a very brief excerpt if content is available
        content = blog.get('content', '')
        
        # Improved excerpt that focuses on relevant parts if possible
        excerpt = ""
        if content and query:
            # Try to find most relevant sentence containing query words
            query_words = set(query.lower().split()) - {'a', 'an', 'the', 'and', 'or', 'but', 'is', 'are', 'in', 'on', 'at'}
            sentences = content.replace('\n', ' ').split('. ')
            
            # Score sentences by how many query words they contain
            sentence_scores = [(s, sum(1 for w in query_words if w in s.lower())) for s in sentences if len(s) > 20]
            
            if sentence_scores:
                # Get the most relevant sentence (highest score)
                best_sentence = max(sentence_scores, key=lambda x: x[1])[0]
                excerpt = best_sentence.strip()
                
                # Truncate if too long - make it shorter for WhatsApp
                if len(excerpt) > 100:
                    excerpt = excerpt[:100] + "..."
            else:
                # No good match, use beginning of content
                excerpt = content[:100] + "..." if len(content) > 100 else content
        elif content:
            # No query provided, just use the beginning
            excerpt = content[:100] + "..." if len(content) > 100 else content
        
        # Format the article entry with minimal formatting
        response += f"**{title}**\n"
        
        # Add very brief excerpt if available
        if excerpt:
            response += f"{excerpt}\n"
        
        # Add a more prominent link
        response += f"📖 [Read Article]({url})\n\n"
        
        blog_count += 1
    
    # Add a note about clicking the links
    response += "*Click on 'Read Article' to view the full article.*\n"
    
    return response

def test_blog_search():
    """
    Test the blog search functionality with a chimney-related query.
    """
    print("Testing blog search functionality...")
    
    # Load blog embeddings
    try:
        print(f"Loading blog embeddings from {BLOG_EMBEDDINGS_FILE_PATH}")
        blog_embeddings_dict = load_blog_embeddings(BLOG_EMBEDDINGS_FILE_PATH)
        if blog_embeddings_dict and blog_embeddings_dict['blog_embeddings'].shape[0] > 0:
            print(f"Successfully loaded {blog_embeddings_dict['blog_embeddings'].shape[0]} blog embeddings")
            
            # Test with a chimney-related query
            query = "how to choose right chimney"
            print(f"\nTesting with query: '{query}'")
            
            # Search for relevant blogs
            blog_results = search_relevant_blogs(query, blog_embeddings_dict, k=3, similarity_threshold=0.3)
            
            if blog_results:
                print(f"\nFound {len(blog_results)} relevant blog articles")
                
                # Format the response
                response = format_blog_response(blog_results, query)
                print("\nFormatted response:")
                print(response)
            else:
                print("\nNo relevant blog articles found")
        else:
            print("Failed to load blog embeddings or no embeddings found")
    except Exception as e:
        print(f"Error testing blog search: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    test_blog_search() 