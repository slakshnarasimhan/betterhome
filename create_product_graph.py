import pandas as pd
import networkx as nx

# Load the product catalog
catalog_file_path = 'cleaned_products.csv'  # Update with actual path
catalog_df = pd.read_csv(catalog_file_path)

# Initialize a directed graph
G = nx.DiGraph()

# Add nodes and edges to the graph
for _, row in catalog_df.iterrows():
    # Use a combination of title and brand as the unique identifier
    product_id = f"{row['title']}_{row['Brand']}"
    product_title = row['title']
    product_type = row['Product Type']
    brand = row['Brand']
    
    # Add product node
    G.add_node(product_id, title=product_title, type='product')
    
    # Add category node and edge
    if product_type:
        G.add_node(product_type, type='category')
        G.add_edge(product_id, product_type, relation='belongs_to')
    
    # Add brand node and edge
    if brand:
        G.add_node(brand, type='brand')
        G.add_edge(product_id, brand, relation='manufactured_by')

# Save the graph to a file
import pickle
with open('product_graph.gpickle', 'wb') as f:
    pickle.dump(G, f)

print("Graph created and saved as 'product_graph.gpickle'") 