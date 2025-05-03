import csv
import re
from fuzzywuzzy import fuzz

def read_csv_file(file_path):
    """Read a CSV file and return a list of dictionaries"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data

def clean_title(title):
    """Clean product title for better matching"""
    # Convert to lowercase
    title = title.lower()
    # Remove size/dimension information
    title = re.sub(r'\d+\s*(?:mm|cm|inch|inches|feet|ft|litre|ltr|l)\b', '', title)
    # Remove model numbers
    title = re.sub(r'(?:model|series)\s*[\w-]+', '', title)
    # Remove color information
    title = re.sub(r'(?:white|black|silver|grey|gray|brown|blue|red|green|yellow)\b', '', title)
    # Remove special characters and extra spaces
    title = re.sub(r'[^\w\s]', ' ', title)
    title = re.sub(r'\s+', ' ', title).strip()
    return title

def get_product_keywords(title):
    """Extract important keywords from a product title"""
    # Remove common words and retain only significant words
    words = clean_title(title).split()
    stopwords = {'with', 'for', 'and', 'the', 'in', 'of', 'a', 'an', 'from', 'to', 'by', 'on', 'at'}
    keywords = [word for word in words if word not in stopwords and len(word) > 2]
    return keywords

def categorize_by_keywords(title):
    """Determine product type based on keywords in the title"""
    title_lower = title.lower()
    
    # Bathroom fixtures
    if any(keyword in title_lower for keyword in ['wc', 'toilet', 'wall hung']):
        return 'Sanitary Ware', 'Bathroom'
    
    if any(keyword in title_lower for keyword in ['basin', 'sink']):
        return 'Sanitary Ware', 'Bathroom'
    
    if any(keyword in title_lower for keyword in ['shower', 'faucet', 'tap', 'cock', 'mixer']):
        return 'Bathroom Fittings', 'Bathroom'
    
    # Mirrors and glass
    if 'mirror' in title_lower:
        return 'Bathroom Accessories', 'Bathroom'
    
    if 'glass' in title_lower and ('partition' in title_lower or 'door' in title_lower):
        return 'Home Improvement', 'Glass & Mirrors'
    
    # Appliances
    if 'fan' in title_lower:
        if 'ceiling' in title_lower:
            return 'Ceiling Fan', 'Fans'
        return 'Fan', 'Fans'
    
    if 'ac' in title_lower or 'air conditioner' in title_lower:
        return 'Air Conditioner', 'Air Conditioners'
    
    if 'water heater' in title_lower or 'geyser' in title_lower:
        return 'Water Heater', 'Water Heaters'
    
    if 'water purifier' in title_lower or 'ro' in title_lower:
        return 'Water Purifier', 'Water Purifiers'
    
    if 'hob' in title_lower or 'burner' in title_lower:
        return 'Hob', 'Kitchen Appliances'
    
    if 'mixer grinder' in title_lower:
        return 'Mixer Grinder', 'Kitchen Appliances'
    
    if 'tumble dryer' in title_lower or 'dryer' in title_lower:
        return 'Clothes Dryer', 'Laundry Appliances'
    
    # Paints and waterproofing
    if 'paint' in title_lower or 'waterproofing' in title_lower or 'dampproof' in title_lower:
        return 'Paint & Waterproofing', 'Home Improvement'
    
    # Lighting
    if any(keyword in title_lower for keyword in ['downlighter', 'light', 'led']):
        return 'Lighting', 'Electrical'
    
    # Wood panels
    if 'wood panel' in title_lower or 'hdwr' in title_lower:
        return 'Wood Panel', 'Home Improvement'
    
    # Default
    return '', ''

def find_similar_products(missing_product, reference_products, threshold=75):
    """Find similar products in the reference data using fuzzy matching"""
    missing_title_clean = clean_title(missing_product['title'])
    missing_keywords = get_product_keywords(missing_product['title'])
    
    matches = []
    
    for ref_product in reference_products:
        ref_title_clean = clean_title(ref_product['title'])
        
        # Calculate similarity ratio between titles
        similarity = fuzz.token_sort_ratio(missing_title_clean, ref_title_clean)
        
        # Check keyword overlap
        ref_keywords = get_product_keywords(ref_product['title'])
        keyword_overlap = len(set(missing_keywords).intersection(set(ref_keywords)))
        
        # Boost score based on keyword overlap
        adjusted_similarity = similarity + (keyword_overlap * 5)
        
        if adjusted_similarity >= threshold:
            matches.append({
                'product': ref_product,
                'similarity': adjusted_similarity
            })
    
    # Sort matches by similarity score
    matches.sort(key=lambda x: x['similarity'], reverse=True)
    return matches

def determine_product_info(missing_product, reference_products):
    """Determine product type and category for a missing product"""
    # First try categorizing by keywords
    product_type, category = categorize_by_keywords(missing_product['title'])
    
    # If we couldn't determine by keywords, try fuzzy matching
    if not product_type or not category:
        matches = find_similar_products(missing_product, reference_products)
        
        if matches:
            top_match = matches[0]['product']
            product_type = top_match['Product Type']
            category = top_match['Category']
            
            # If no category in the match, make a guess based on product type
            if not category and product_type:
                if 'fan' in product_type.lower():
                    category = 'Fans'
                elif 'ac' in product_type.lower() or 'air conditioner' in product_type.lower():
                    category = 'Air Conditioners'
                elif 'water heater' in product_type.lower():
                    category = 'Water Heaters'
                elif 'toilet' in product_type.lower() or 'basin' in product_type.lower():
                    category = 'Bathroom'
    
    # Check if we have a brand in the title that might help determine type
    if not product_type or not category:
        brand_keywords = {
            'jaquar': ('Bathroom Fittings', 'Bathroom'),
            'bosch': ('Home Appliances', 'Appliances'),
            'lg': ('Home Appliances', 'Appliances'),
            'ao smith': ('Water Heater', 'Water Heaters'),
            'atomberg': ('Ceiling Fan', 'Fans'),
            'venus': ('Water Heater', 'Water Heaters'),
            'asian paints': ('Paint & Waterproofing', 'Home Improvement'),
            'nippon paint': ('Paint & Waterproofing', 'Home Improvement')
        }
        
        for brand, (pt, cat) in brand_keywords.items():
            if brand in missing_product['title'].lower() or brand == missing_product['Brand'].lower():
                if not product_type:
                    product_type = pt
                if not category:
                    category = cat
                break
    
    return product_type, category

def main():
    # Read data from both CSV files
    missing_products = read_csv_file('missing_bestsellers.csv')
    reference_products = read_csv_file('cleaned_products.csv')
    
    print(f"Read {len(missing_products)} missing products")
    print(f"Read {len(reference_products)} reference products")
    
    # Initialize counters
    updated_count = 0
    already_categorized = 0
    
    # Process each missing product
    for product in missing_products:
        # Skip if already has both product type and category
        if product['Product Type'] and product['Category']:
            already_categorized += 1
            continue
        
        # Determine product type and category
        product_type, category = determine_product_info(product, reference_products)
        
        # Update product information
        if product_type and not product['Product Type']:
            product['Product Type'] = product_type
            updated_count += 1
        
        if category and not product['Category']:
            product['Category'] = category
            updated_count += 1
    
    print(f"Updated {updated_count} fields")
    print(f"{already_categorized} products already had complete information")
    
    # Save updated data to new CSV
    output_file = 'categorized_bestsellers.csv'
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        fieldnames = missing_products[0].keys()
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(missing_products)
    
    print(f"Saved updated data to {output_file}")
    
    # Print a summary of categories
    print("\nCategory distribution:")
    categories = {}
    for product in missing_products:
        cat = product['Category']
        if cat:
            categories[cat] = categories.get(cat, 0) + 1
    
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")
    
    print("\nProduct Type distribution:")
    product_types = {}
    for product in missing_products:
        pt = product['Product Type']
        if pt:
            product_types[pt] = product_types.get(pt, 0) + 1
    
    for pt, count in sorted(product_types.items()):
        print(f"  {pt}: {count}")

if __name__ == "__main__":
    main() 