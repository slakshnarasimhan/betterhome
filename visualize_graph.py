import pickle
from pyvis.network import Network

# Load the graph
with open("product_graph.gpickle", "rb") as f:
    G = pickle.load(f)

# Create the network
net = Network(height="800px", width="100%", bgcolor="#ffffff", font_color="black")

# Collect unique values
brands, categories, node_types = set(), set(), set()

# Add nodes
for node, data in G.nodes(data=True):
    node_type = data.get("type", "unknown")
    node_types.add(node_type)

    if node_type == "brand":
        brands.add(node)
    elif node_type == "category":
        categories.add(node)

    net.add_node(
        node,
        label=node,
        title=node_type,
        group=node_type,
        color={
            "product": "skyblue",
            "brand": "lightgreen",
            "category": "orange"
        }.get(node_type, "gray")
    )

# Add edges
for src, dst, data in G.edges(data=True):
    net.add_edge(src, dst, title=data.get("relation", ""))

# Save base HTML
net.write_html("product_graph_interactive_filtered.html")


# Inject filtering dropdowns
with open("product_graph_interactive_filtered.html", "r") as f:
    html = f.read()

# Dropdown and filter logic
dropdown_html = f"""
<div style="padding: 10px;">
  <label for="brandFilter">Brand:</label>
  <select id="brandFilter" onchange="filterGraph()">
    <option value="">All</option>
    {''.join(f'<option value="{b}">{b}</option>' for b in sorted(brands))}
  </select>

  <label for="categoryFilter">Product Type:</label>
  <select id="categoryFilter" onchange="filterGraph()">
    <option value="">All</option>
    {''.join(f'<option value="{c}">{c}</option>' for c in sorted(categories))}
  </select>

  <label for="typeFilter">Node Type:</label>
  <select id="typeFilter" onchange="filterGraph()">
    <option value="">All</option>
    <option value="product">Product</option>
    <option value="brand">Brand</option>
    <option value="category">Category</option>
  </select>
</div>

<script>
function filterGraph() {{
  var brand = document.getElementById('brandFilter').value;
  var category = document.getElementById('categoryFilter').value;
  var type = document.getElementById('typeFilter').value;

  network.body.data.nodes.update(network.body.data.nodes.map(n => {{
    let show = true;
    if (type && n.group !== type) show = false;
    if (brand && n.label !== brand && n.group === 'brand') show = false;
    if (category && n.label !== category && n.group === 'category') show = false;
    return {{id: n.id, hidden: !show}};
  }}));

  network.body.data.edges.update(network.body.data.edges.map(e => {{
    const from = network.body.data.nodes.get(e.from);
    const to = network.body.data.nodes.get(e.to);
    return {{id: e.id, hidden: from.hidden || to.hidden}};
  }}));
}}
</script>
"""

# Inject it into the body
html = html.replace("<body>", f"<body>\n{dropdown_html}")

# Save final version
with open("product_graph_interactive_filtered.html", "w") as f:
    f.write(html)

print("✅ Interactive graph with filters saved as 'product_graph_interactive_filtered.html'")
