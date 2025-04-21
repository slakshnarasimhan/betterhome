
import pickle
from pyvis.network import Network
from collections import defaultdict

# Load the product graph
with open("product_graph.gpickle", "rb") as f:
    G = pickle.load(f)

# Extract filterable fields
brands = sorted([n for n, d in G.nodes(data=True) if d.get("type") == "brand"])
categories = sorted([n for n, d in G.nodes(data=True) if d.get("type") == "category"])
node_types = sorted(set(d.get("type", "unknown") for _, d in G.nodes(data=True)))

# Generate color map for product categories
category_colors = {}
color_palette = ["#f94144", "#f3722c", "#f8961e", "#f9c74f", "#90be6d", "#43aa8b", "#577590", "#277da1"]
for i, category in enumerate(categories):
    category_colors[category] = color_palette[i % len(color_palette)]

# Initialize pyvis network
net = Network(height="800px", width="100%", bgcolor="#ffffff", font_color="black", directed=True)

# Add all nodes with metadata
for node, data in G.nodes(data=True):
    node_type = data.get("type", "unknown")
    color = "gray"
    if node_type == "product":
        connected = list(G.successors(node))
        category = next((n for n in connected if G.nodes[n].get("type") == "category"), None)
        if category and category in category_colors:
            color = category_colors[category]
    elif node_type == "brand":
        color = "lightgreen"
    elif node_type == "category":
        color = "orange"

    net.add_node(
        node,
        label=node,
        title=node_type,
        group=node_type,
        color=color
    )

# Add all edges with basic info
for src, dst, data in G.edges(data=True):
    relation = data.get("relation", "")
    net.add_edge(src, dst, title=relation)

# Save HTML and inject dropdown filters
html_path = "product_graph_filtered_dynamic.html"
net.save_graph(html_path)

# Inject advanced filters and JS
with open(html_path, "r") as f:
    html = f.read()

# Create multi-select dropdowns and filtering logic
filter_script = f"""
<div style="padding:10px;">
  <label>Brand:</label>
  <select id="brandFilter" multiple onchange="filterGraph()">
    {''.join(f'<option value="{b}">{b}</option>' for b in brands)}
  </select>

  <label>Category:</label>
  <select id="categoryFilter" multiple onchange="filterGraph()">
    {''.join(f'<option value="{c}">{c}</option>' for c in categories)}
  </select>

  <label>Type:</label>
  <select id="typeFilter" multiple onchange="filterGraph()">
    {''.join(f'<option value="{t}">{t}</option>' for t in node_types)}
  </select>

  <button onclick="resetGraph()">Reset</button>
</div>

<script>
function getSelectedValues(id) {{
  return Array.from(document.getElementById(id).selectedOptions).map(o => o.value);
}}

function filterGraph() {{
  const brands = getSelectedValues('brandFilter');
  const cats = getSelectedValues('categoryFilter');
  const types = getSelectedValues('typeFilter');

  const visibleNodes = new Set();

  nodes.forEach(n => {{
    let show = true;
    if (types.length && !types.includes(n.group)) show = false;
    if (brands.length && n.group === 'brand' && !brands.includes(n.label)) show = false;
    if (cats.length && n.group === 'category' && !cats.includes(n.label)) show = false;

    if (show) visibleNodes.add(n.id);
  }});

  edges.forEach(e => {{
    if (visibleNodes.has(e.from) || visibleNodes.has(e.to)) {{
      visibleNodes.add(e.from);
      visibleNodes.add(e.to);
    }}
  }});

  nodes.update(nodes.map(n => {{
    return {{id: n.id, hidden: !visibleNodes.has(n.id)}}
  }}));

  edges.update(edges.map(e => {{
    return {{id: e.id, hidden: !(visibleNodes.has(e.from) && visibleNodes.has(e.to))}}
  }}));
}}

function resetGraph() {{
  nodes.update(nodes.map(n => {{ return {{id: n.id, hidden: false}} }}));
  edges.update(edges.map(e => {{ return {{id: e.id, hidden: false}} }}));
}}
</script>
"""

# Inject filter script
html = html.replace("<body>", "<body>\n" + filter_script)

# Save final HTML
with open(html_path, "w") as f:
    f.write(html)

print("✅ Interactive filtered graph saved to:", html_path)
