import json

# Path to the blog embeddings file
BLOG_EMBEDDINGS_FILE_PATH = 'blog_embeddings.json'

# Function to read and print the first few entries
def read_first_entries(file_path, num_entries=5):
    with open(file_path, 'r') as f:
        data = json.load(f)
        
        # Check if the data is a list
        if isinstance(data, list):
            for i, article in enumerate(data[:num_entries]):
                print(f"Entry {i+1}: {article}")
                if i >= num_entries - 1:
                    break
        # If the data is a dictionary, iterate over its keys
        elif isinstance(data, dict):
            for key, value in data.items():
                print(f"Key: {key}, Value: {value}")
                # If the value is a list, print the first few entries
                if isinstance(value, list):
                    for i, article in enumerate(value[:num_entries]):
                        print(f"Entry {i+1}: {article}")
                        if i >= num_entries - 1:
                            break
                break  # Remove this break if you want to check more keys

# Run the function to read and print entries
read_first_entries(BLOG_EMBEDDINGS_FILE_PATH)