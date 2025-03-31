import json

def print_blog_tags(file_path='blog_embeddings.json'):
    try:
        # Load the JSON data from the file
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Extract metadata
        metadata = data.get('metadata', [])
        
        # Print the tags for each blog article
        for article in metadata:
            title = article.get('title', 'Untitled')
            url = article.get('url', 'No URL')
            tags = article.get('categories', [])
            print(f"Title: {title}")
            print(f"URL: {url}")
            print(f"Tags: {', '.join(tags) if tags else 'No tags found'}")
            print("-" * 40)
    
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except json.JSONDecodeError:
        print(f"Error decoding JSON from the file: {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    print_blog_tags()
