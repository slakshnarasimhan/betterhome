
import networkx as nx
import matplotlib.pyplot as plt
import pickle

# Load the graph
with open('product_graph.gpickle', 'rb') as f:
    graph = pickle.load(f)

# Remove invalid nodes
graph.remove_nodes_from([n for n in graph.nodes if n != n or not isinstance(n, str)])

# Generate layout
pos = nx.spring_layout(graph, seed=42)

# Draw the graph
plt.figure(figsize=(12, 12))
nx.draw(graph, pos, with_labels=True, node_size=500, node_color='lightblue', font_size=8, font_weight='bold')
plt.title("Product Graph Visualization")
plt.show()
