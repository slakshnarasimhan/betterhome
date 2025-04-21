import networkx as nx
import matplotlib.pyplot as plt
import argparse
from collections import defaultdict

def visualize_knowledge_graph(graph_file='product_knowledge_graph.gpickle', output_file='knowledge_graph_visualization.png'):
    """
    Visualize the knowledge graph
    """
    print(f"Loading knowledge graph from {graph_file}")
    G = nx.read_gpickle(graph_file)
    
    print(f"Graph has {len(G.nodes())} nodes and {len(G.edges())} edges")
    
    # Count node types
    node_type_counts = defaultdict(int)
    for node, data in G.nodes(data=True):
        node_type = data.get('type', 'unknown')
        node_type_counts[node_type] += 1
    
    print("Node types:")
    for node_type, count in node_type_counts.items():
        print(f"  - {node_type}: {count}")
    
    # Count edge relations
    edge_relation_counts = defaultdict(int)
    for u, v, data in G.edges(data=True):
        relation = data.get('relation', 'unknown')
        edge_relation_counts[relation] += 1
    
    print("Edge relations:")
    for relation, count in edge_relation_counts.items():
        print(f"  - {relation}: {count}")
    
    # Limit graph size for visualization
    if len(G) > 1000:
        print("Graph is too large for full visualization, sampling nodes...")
        # Extract representative nodes for each type
        sampled_nodes = []
        
        # Take samples of each node type
        for node_type in node_type_counts.keys():
            nodes_of_type = [n for n, d in G.nodes(data=True) if d.get('type') == node_type]
            
            # Take up to 100 nodes of each type, or all if fewer
            sample_size = min(100, len(nodes_of_type))
            sampled_nodes.extend(nodes_of_type[:sample_size])
        
        # Create a subgraph with the sampled nodes
        G = G.subgraph(sampled_nodes)
        print(f"Created subgraph with {len(G.nodes())} nodes and {len(G.edges())} edges")
    
    plt.figure(figsize=(20, 20))
    
    # Define node colors by type
    color_map = {
        'product': 'skyblue',
        'category': 'lightgreen',
        'brand': 'lightcoral',
        'feature': 'lightyellow',
        'price_range': 'lightpink'
    }
    
    # Assign colors and sizes to nodes
    node_colors = []
    node_sizes = []
    
    for node in G.nodes():
        node_type = G.nodes[node].get('type', 'unknown')
        node_colors.append(color_map.get(node_type, 'gray'))
        
        # Make non-product nodes larger for better visibility
        if node_type != 'product':
            node_sizes.append(300)
        else:
            node_sizes.append(50)
    
    # Improve layout
    print("Computing layout...")
    pos = nx.spring_layout(G, k=0.15, iterations=50)
    
    # Draw the graph
    print("Drawing graph...")
    nx.draw(G, pos, 
            node_color=node_colors,
            node_size=node_sizes,
            with_labels=False,
            alpha=0.8,
            width=0.5,
            edge_color='gray')
    
    # Draw labels for important nodes only
    category_labels = {node: G.nodes[node].get('name', '') 
                      for node in G.nodes() 
                      if G.nodes[node].get('type') == 'category'}
    
    brand_labels = {node: G.nodes[node].get('name', '')
                   for node in G.nodes() 
                   if G.nodes[node].get('type') == 'brand' and node in pos}
    
    price_labels = {node: G.nodes[node].get('name', '')
                   for node in G.nodes() 
                   if G.nodes[node].get('type') == 'price_range' and node in pos}
    
    # Filter to show only some important brand labels if there are too many
    if len(brand_labels) > 20:
        brand_labels = dict(list(brand_labels.items())[:20])
    
    nx.draw_networkx_labels(G, pos, labels=category_labels, font_size=12, font_color='darkgreen')
    nx.draw_networkx_labels(G, pos, labels=brand_labels, font_size=10, font_color='darkred')
    nx.draw_networkx_labels(G, pos, labels=price_labels, font_size=12, font_color='purple')
    
    print(f"Creating legend...")
    # Create a legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                                 label=node_type, markersize=10) 
                      for node_type, color in color_map.items()]
    
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.title('Product Knowledge Graph')
    plt.axis('off')
    plt.tight_layout()
    
    print(f"Saving visualization to {output_file}")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Graph visualization completed and saved as {output_file}")
    return True

def main():
    parser = argparse.ArgumentParser(description='Visualize a product knowledge graph')
    parser.add_argument('--graph', default='product_knowledge_graph.gpickle', 
                        help='Path to the knowledge graph file')
    parser.add_argument('--output', default='knowledge_graph_visualization.png',
                        help='Path to save the visualization')
    
    args = parser.parse_args()
    visualize_knowledge_graph(args.graph, args.output)

if __name__ == "__main__":
    main() 