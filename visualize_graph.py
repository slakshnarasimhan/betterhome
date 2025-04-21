from pyvis.network import Network
import pickle

# Load the graph
with open("product_graph.gpickle", "rb") as f:
    G = pickle.load(f)

# Initialize the PyVis network
net = Network(height="750px", width="100%", bgcolor="#ffffff", font_color="black")

# Add nodes with type-based colors
for node, data in G.nodes(data=True):
    node_type = data.get("type", "unknown")
    color = {
        "product": "skyblue",
        "brand": "lightgreen",
        "category": "orange"
    }.get(node_type, "gray")
    net.add_node(node, label=node, title=node_type, color=color)

# Add edges
for src, dst, data in G.edges(data=True):
    net.add_edge(src, dst, title=data.get("relation", ""))

# Generate and open the interactive HTML
net.show("product_graph_interactive.html", notebook=False)

