import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import re
from collections import defaultdict
import pickle

def create_knowledge_graph(csv_file):
    """
    Create a knowledge graph from product data CSV file
    """
    print(f"Loading data from {csv_file}")
    df = pd.read_csv(csv_file)
    
    # Initialize a directed graph
    G = nx.DiGraph()
    
    # Track unique entities for better visualization
    unique_categories = set()
    unique_brands = set()
    unique_features = set()
    unique_price_ranges = set()
    
    print(f"Processing {len(df)} products...")
    
    # Process each product
    for idx, row in df.iterrows():
        if idx % 100 == 0:
            print(f"Processed {idx} products...")
            
        # Extract basic product info
        product_id = idx
        product_title = row.get('title', f'Unknown Product {idx}')
        product_type = row.get('Product Type', 'Unknown Category')
        brand = row.get('Brand', 'Unknown Brand')
        price = row.get('Better Home Price', 0)
        description = str(row.get('Description', ''))
        
        # Create price ranges (budget, mid-range, premium)
        if price <= 10000:
            price_range = 'Budget'
        elif price <= 30000:
            price_range = 'Mid-Range'
        else:
            price_range = 'Premium'
            
        # Extract features from description or other fields
        features = extract_features(description)
        
        # Add product node
        G.add_node(f"product_{product_id}", 
                   type='product', 
                   title=product_title, 
                   price=price)
        
        # Add category node and edge
        if product_type:
            unique_categories.add(product_type)
            G.add_node(f"category_{product_type}", type='category', name=product_type)
            G.add_edge(f"product_{product_id}", f"category_{product_type}", relation='belongs_to')
        
        # Add brand node and edge
        if brand:
            unique_brands.add(brand)
            G.add_node(f"brand_{brand}", type='brand', name=brand)
            G.add_edge(f"product_{product_id}", f"brand_{brand}", relation='manufactured_by')
        
        # Add price range node and edge
        unique_price_ranges.add(price_range)
        G.add_node(f"price_{price_range}", type='price_range', name=price_range)
        G.add_edge(f"product_{product_id}", f"price_{price_range}", relation='priced_as')
        
        # Add feature nodes and edges
        for feature in features:
            unique_features.add(feature)
            G.add_node(f"feature_{feature}", type='feature', name=feature)
            G.add_edge(f"product_{product_id}", f"feature_{feature}", relation='has_feature')
        
        # Add related products (same category)
        for other_idx, other_row in df.iterrows():
            if idx != other_idx and other_row.get('Product Type') == product_type:
                G.add_edge(f"product_{product_id}", f"product_{other_idx}", relation='related_to')
    
    print(f"Knowledge graph created with {len(G.nodes())} nodes and {len(G.edges())} edges")
    print(f"Unique categories: {len(unique_categories)}")
    print(f"Unique brands: {len(unique_brands)}")
    print(f"Unique features: {len(unique_features)}")
    print(f"Unique price ranges: {len(unique_price_ranges)}")
    
    return G, {
        'categories': unique_categories,
        'brands': unique_brands,
        'features': unique_features,
        'price_ranges': unique_price_ranges
    }

def extract_features(description):
    """
    Extract product features from description
    """
    features = set()
    
    # Common appliance features to look for
    feature_keywords = [
        'energy efficient', 'energy saving', 'power saving',
        'high performance', 'quiet', 'silent', 'noise reduction',
        'durable', 'warranty', 'smart', 'automatic', 'digital',
        'inverter', 'star rating', 'eco friendly', 'environment friendly',
        'waterproof', 'water resistant', 'rust proof', 'corrosion resistant',
        'easy clean', 'auto clean', 'led', 'lcd', 'display', 'touch control',
        'remote control', 'wifi', 'bluetooth', 'wireless', 'app control',
        'timer', 'programmable', 'memory', 'preset', 'adjustable',
        'compact', 'lightweight', 'portable', 'heavy duty', 'industrial',
        'stainless steel', 'metal', 'plastic', 'glass', 'ceramic',
        'heat resistant', 'cold resistant', 'temperature control',
        'child lock', 'safety', 'anti-bacterial', 'hygienic'
    ]
    
    # Check for each feature keyword in the description
    for keyword in feature_keywords:
        if keyword in description.lower():
            features.add(keyword)
    
    # Look for specific patterns like X litres, Y watts, etc.
    capacity_match = re.search(r'(\d+)\s*litres?', description.lower())
    if capacity_match:
        features.add(f"{capacity_match.group(1)}_litres")
    
    power_match = re.search(r'(\d+)\s*watts?', description.lower())
    if power_match:
        features.add(f"{power_match.group(1)}_watts")
    
    return features

def visualize_graph(G, metadata, output_file='knowledge_graph.png'):
    """
    Visualize the knowledge graph
    """
    plt.figure(figsize=(15, 15))
    
    # Define node colors by type
    color_map = {
        'product': 'skyblue',
        'category': 'lightgreen',
        'brand': 'lightcoral',
        'feature': 'lightyellow',
        'price_range': 'lightpink'
    }
    
    # Assign colors to nodes
    node_colors = [color_map.get(G.nodes[node].get('type', 'unknown'), 'gray') for node in G.nodes()]
    
    # Improve layout by increasing separation
    pos = nx.spring_layout(G, k=0.15, iterations=50)
    
    # Draw the graph
    nx.draw(G, pos, 
            node_color=node_colors,
            node_size=100,
            font_size=8,
            with_labels=False,
            alpha=0.8,
            width=0.5,
            edge_color='gray')
    
    # Draw labels for important nodes only
    category_labels = {node: G.nodes[node]['name'] for node in G.nodes() 
                      if G.nodes[node].get('type') == 'category'}
    brand_labels = {node: G.nodes[node]['name'] for node in G.nodes() 
                   if G.nodes[node].get('type') == 'brand'}
    price_labels = {node: G.nodes[node]['name'] for node in G.nodes() 
                   if G.nodes[node].get('type') == 'price_range'}
    
    nx.draw_networkx_labels(G, pos, labels=category_labels, font_size=10, font_color='darkgreen')
    nx.draw_networkx_labels(G, pos, labels=brand_labels, font_size=10, font_color='darkred')
    nx.draw_networkx_labels(G, pos, labels=price_labels, font_size=10, font_color='purple')
    
    plt.title('Product Knowledge Graph')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Graph visualization saved as {output_file}")
    plt.close()

def get_product_recommendations(G, query_product_id, max_recommendations=5):
    """
    Get product recommendations based on the knowledge graph
    """
    recommendations = []
    product_node = f"product_{query_product_id}"
    
    if product_node not in G:
        return recommendations
    
    # Get product category
    category_nodes = [n for u, n in G.out_edges(product_node) 
                    if G.nodes[n].get('type') == 'category']
    
    # Get product features
    feature_nodes = [n for u, n in G.out_edges(product_node) 
                    if G.nodes[n].get('type') == 'feature']
    
    # Get product brand
    brand_nodes = [n for u, n in G.out_edges(product_node) 
                  if G.nodes[n].get('type') == 'brand']
    
    # Find similar products based on shared category, features and brand
    similarity_scores = defaultdict(float)
    
    # Get all product nodes
    product_nodes = [n for n in G.nodes() if G.nodes[n].get('type') == 'product' and n != product_node]
    
    for other_product in product_nodes:
        # Check if same category
        other_categories = [n for u, n in G.out_edges(other_product) 
                          if G.nodes[n].get('type') == 'category']
        
        category_match = len(set(category_nodes).intersection(set(other_categories)))
        if category_match > 0:
            similarity_scores[other_product] += 1.0  # Base similarity for same category
            
            # Check feature overlap
            other_features = [n for u, n in G.out_edges(other_product) 
                            if G.nodes[n].get('type') == 'feature']
            
            feature_overlap = len(set(feature_nodes).intersection(set(other_features)))
            similarity_scores[other_product] += 0.2 * feature_overlap  # Add score for each shared feature
            
            # Check if same brand
            other_brands = [n for u, n in G.out_edges(other_product) 
                          if G.nodes[n].get('type') == 'brand']
            
            brand_match = len(set(brand_nodes).intersection(set(other_brands)))
            similarity_scores[other_product] += 0.5 * brand_match  # Add score for brand match
    
    # Sort products by similarity score
    sorted_products = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Get top recommendations
    for product_node, score in sorted_products[:max_recommendations]:
        product_title = G.nodes[product_node].get('title', 'Unknown Product')
        recommendations.append({
            'product_id': product_node.replace('product_', ''),
            'title': product_title,
            'similarity_score': score
        })
    
    return recommendations

def save_graph(G, filename='product_knowledge_graph.gpickle'):
    """
    Save the knowledge graph to a file
    """
    with open(filename, 'wb') as f:
        pickle.dump(G, f)
    print(f"Knowledge graph saved as {filename}")

def load_graph(filename='product_knowledge_graph.gpickle'):
    """
    Load the knowledge graph from a file
    """
    with open(filename, 'rb') as f:
        G = pickle.load(f)
    print(f"Knowledge graph loaded from {filename}")
    return G

def main():
    csv_file = 'cleaned_products.csv'
    
    # Create knowledge graph
    G, metadata = create_knowledge_graph(csv_file)
    
    # Visualize graph
    visualize_graph(G, metadata)
    
    # Save graph
    save_graph(G)
    
    # Example: Get recommendations for a product
    # Uncomment to test with a specific product ID
    # product_id = 0  # Replace with actual product ID
    # recommendations = get_product_recommendations(G, product_id)
    # print(f"Recommendations for product {product_id}:")
    # for rec in recommendations:
    #     print(f"  - {rec['title']} (Score: {rec['similarity_score']:.2f})")

if __name__ == "__main__":
    main() 