import networkx as nx
import matplotlib.pyplot as plt
import argparse
from collections import defaultdict

def visualize_knowledge_graph(graph_file='product_knowledge_graph.gpickle', output_file='knowledge_graph_visualization.png'):
    """
    Visualize the knowledge graph - optimized for speed with large graphs
    """
    import pickle
    print(f"Loading knowledge graph from {graph_file}")
    
    # Load graph with pickle instead of NetworkX for speed
    try:
        with open(graph_file, 'rb') as f:
            G = pickle.load(f)
        print(f"Graph loaded with pickle")
    except:
        G = nx.read_gpickle(graph_file)
        print(f"Graph loaded with NetworkX")
    
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
    print("Counting edge types (sampled)...")
    edge_relation_counts = defaultdict(int)
    # Only sample a subset of edges for speed
    edge_sample = list(G.edges(data=True))[:min(1000, len(G.edges()))]
    for u, v, data in edge_sample:
        relation = data.get('relation', 'unknown')
        edge_relation_counts[relation] += 1
    
    print("Edge relations (from sample):")
    for relation, count in edge_relation_counts.items():
        print(f"  - {relation}: {count}")
    
    # Create a more manageable subgraph for visualization
    print("Creating visualization subgraph...")
    max_nodes = 500  # Maximum number of nodes for visualization
    
    if len(G) > max_nodes:
        # Extract representative nodes for each type
        sampled_nodes = []
        
        # Calculate how many nodes to include of each type
        total_nodes_to_include = max_nodes
        type_allocation = {}
        
        # Allocate nodes by type with minimum guarantees
        # At least 20 of each type, rest proportional
        min_per_type = 20
        remaining_allocation = total_nodes_to_include - min_per_type * len(node_type_counts)
        
        if remaining_allocation < 0:
            # If we can't guarantee minimums, just do equal distribution
            for node_type in node_type_counts:
                type_allocation[node_type] = total_nodes_to_include // len(node_type_counts)
        else:
            # Allocate minimum, then distribute remaining proportionally
            for node_type, count in node_type_counts.items():
                type_allocation[node_type] = min_per_type
                
                # Calculate proportional allocation of remaining nodes
                proportion = count / sum(node_type_counts.values())
                additional = int(remaining_allocation * proportion)
                type_allocation[node_type] += additional
        
        # Make sure we don't exceed actual counts
        for node_type, count in node_type_counts.items():
            type_allocation[node_type] = min(type_allocation[node_type], count)
            
        print("Node allocation for visualization:")
        for node_type, allocation in type_allocation.items():
            print(f"  - {node_type}: {allocation}")
            
        # Take samples of each node type
        for node_type, allocation in type_allocation.items():
            # Get all nodes of this type
            nodes_of_type = [n for n, d in G.nodes(data=True) if d.get('type') == node_type]
            
            # If we need categories, brands or features, prefer ones with more connections
            if node_type in ['category', 'brand', 'feature', 'price_range']:
                # Sort by number of connections
                nodes_of_type.sort(key=lambda n: len(list(G.neighbors(n))), reverse=True)
                
            # Take the allocated number
            sampled_nodes.extend(nodes_of_type[:allocation])
        
        # Create a subgraph with the sampled nodes
        viz_graph = G.subgraph(sampled_nodes)
        print(f"Created visualization subgraph with {len(viz_graph.nodes())} nodes and {len(viz_graph.edges())} edges")
    else:
        viz_graph = G
    
    plt.figure(figsize=(20, 20))
    
    # Define node colors by type
    color_map = {
        'product': 'skyblue',
        'category': 'lightgreen',
        'brand': 'lightcoral',
        'feature': 'lightyellow',
        'price_range': 'lightpink'
    }
    
    # Assign colors and sizes to nodes - do this more efficiently
    node_colors = []
    node_sizes = []
    
    for node in viz_graph.nodes():
        node_type = viz_graph.nodes[node].get('type', 'unknown')
        node_colors.append(color_map.get(node_type, 'gray'))
        
        # Make non-product nodes larger for better visibility
        if node_type != 'product':
            node_sizes.append(300)
        else:
            node_sizes.append(50)
    
    # Use faster layout algorithm for large graphs
    print("Computing layout...")
    if len(viz_graph) > 200:
        # For larger graphs, use faster layout algorithms
        pos = nx.kamada_kawai_layout(viz_graph)
    else:
        # For smaller graphs, spring layout works well
        pos = nx.spring_layout(viz_graph, k=0.15, iterations=30)
    
    # Draw the graph
    print("Drawing graph...")
    nx.draw(viz_graph, pos, 
            node_color=node_colors,
            node_size=node_sizes,
            with_labels=False,
            alpha=0.8,
            width=0.5,
            edge_color='gray')
    
    # Draw labels for important nodes only
    category_labels = {node: viz_graph.nodes[node].get('name', '') 
                      for node in viz_graph.nodes() 
                      if viz_graph.nodes[node].get('type') == 'category'}
    
    brand_labels = {node: viz_graph.nodes[node].get('name', '')
                   for node in viz_graph.nodes() 
                   if viz_graph.nodes[node].get('type') == 'brand' and node in pos}
    
    price_labels = {node: viz_graph.nodes[node].get('name', '')
                   for node in viz_graph.nodes() 
                   if viz_graph.nodes[node].get('type') == 'price_range' and node in pos}
    
    # Limit label count for readability and performance
    if len(category_labels) > 20:
        # Sort by importance (number of connections)
        sorted_categories = sorted(category_labels.keys(), 
                                  key=lambda n: len(list(viz_graph.neighbors(n))),
                                  reverse=True)
        category_labels = {k: category_labels[k] for k in sorted_categories[:20]}
    
    if len(brand_labels) > 15:
        sorted_brands = sorted(brand_labels.keys(),
                              key=lambda n: len(list(viz_graph.neighbors(n))),
                              reverse=True)
        brand_labels = {k: brand_labels[k] for k in sorted_brands[:15]}
    
    print("Adding labels...")
    nx.draw_networkx_labels(viz_graph, pos, labels=category_labels, font_size=12, font_color='darkgreen')
    nx.draw_networkx_labels(viz_graph, pos, labels=brand_labels, font_size=10, font_color='darkred')
    nx.draw_networkx_labels(viz_graph, pos, labels=price_labels, font_size=12, font_color='purple')
    
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