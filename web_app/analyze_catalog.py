import json
from collections import Counter
from pathlib import Path

def analyze_product_types():
    # Read the catalog file
    catalog_path = Path(__file__).parent / 'product_catalog.json'
    with open(catalog_path, 'r') as f:
        catalog = json.load(f)
    
    # Get products list from the 'products' key
    products = catalog.get('products', [])
    
    # Count product types
    product_types = Counter()
    for product in products:
        if isinstance(product, dict):
            product_type = product.get('product_type', 'Unknown')
            product_types[product_type] += 1
    
    # Print summary
    print("\nProduct Type Summary:")
    print("-" * 50)
    print(f"{'Product Type':<30} {'Count':>10}")
    print("-" * 50)
    
    # Sort by count in descending order
    for product_type, count in sorted(product_types.items(), key=lambda x: x[1], reverse=True):
        print(f"{product_type:<30} {count:>10}")
    
    print("-" * 50)
    print(f"Total unique product types: {len(product_types)}")
    print(f"Total products: {sum(product_types.values())}")

if __name__ == "__main__":
    analyze_product_types() 