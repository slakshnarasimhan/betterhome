import pandas as pd
import json
import re

def clean_features(description):
    """Extract features from description text"""
    if not description or pd.isna(description):
        return []
    
    # Try to find bullet points first
    bullet_points = re.findall(r'(?:•|\d+\.)\s*(.*?)(?=(?:•|\d+\.)|$)', description)
    if bullet_points:
        features = [point.strip() for point in bullet_points if point.strip() and len(point.strip()) > 10]
        if features:
            return features[:5]  # Limit to 5 most relevant features
    
    # If no bullet points, try to find key features by looking for sentences with keywords
    sentences = re.split(r'[.!?]+', description)
    features = []
    keywords = ['feature', 'include', 'come with', 'equipped', 'built-in', 'technology', 'system', 'design', 'power', 'capacity', 'efficiency', 'function', 'smart', 'advanced', 'innovative']
    
    # Clean up sentences and remove duplicates
    unique_sentences = set()
    for sentence in sentences:
        sentence = sentence.strip()
        # Clean up the sentence
        sentence = re.sub(r'\s+', ' ', sentence)  # Remove extra spaces
        sentence = re.sub(r'^[^a-zA-Z]*', '', sentence)  # Remove leading non-letters
        sentence = re.sub(r'Buy original.*?Better Home.*?$', '', sentence, flags=re.IGNORECASE)  # Remove purchase text
        
        if sentence and len(sentence) > 10 and sentence not in unique_sentences:
            if any(keyword in sentence.lower() for keyword in keywords):
                features.append(sentence)
                unique_sentences.add(sentence)
    
    # If still no features found, use the first few relevant sentences
    if not features:
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20 and not any(skip in sentence.lower() for skip in ['buy', 'price', 'better home', 'wholesale']):
                features.append(sentence)
                if len(features) >= 3:  # Limit to 3 features if using generic sentences
                    break
    
    return features[:5]  # Limit to 5 most relevant features

def extract_colors(text):
    """Extract color information from text"""
    if not text or pd.isna(text):
        return []
    
    # Common color words
    color_words = {
        'black': ['black', 'ebony', 'onyx', 'charcoal'],
        'white': ['white', 'ivory', 'pearl', 'snow'],
        'silver': ['silver', 'platinum', 'chrome', 'metallic'],
        'grey': ['grey', 'gray', 'slate', 'smoke'],
        'blue': ['blue', 'navy', 'azure', 'cobalt'],
        'red': ['red', 'crimson', 'ruby', 'scarlet'],
        'brown': ['brown', 'bronze', 'copper', 'mahogany'],
        'gold': ['gold', 'golden', 'brass', 'amber'],
        'steel': ['steel', 'stainless steel', 'metal', 'metallic'],
        'copper': ['copper', 'bronze', 'brass']
    }
    
    colors = set()
    
    # Look for color patterns in text
    color_patterns = [
        r'(?:available in|color options?:?|finish:?|colours?:?)\s*((?:[a-zA-Z]+(?:\s+[a-zA-Z]+)*(?:,\s*|\s+and\s+|\s+&\s+))*[a-zA-Z]+(?:\s+[a-zA-Z]+)*)',
        r'\b(black|white|silver|grey|gray|blue|red|brown|gold|bronze|steel|copper)\b'
    ]
    
    text_lower = text.lower()
    
    # Check for specific color mentions
    for color, variants in color_words.items():
        if any(variant in text_lower for variant in variants):
            colors.add(color)
    
    # Check for color patterns
    for pattern in color_patterns:
        matches = re.finditer(pattern, text_lower)
        for match in matches:
            if ',' in match.group(1):
                colors.update(c.strip() for c in match.group(1).split(','))
            elif ' and ' in match.group(1):
                colors.update(c.strip() for c in match.group(1).split(' and '))
            else:
                colors.add(match.group(1).strip())
    
    # Standardize colors
    standardized_colors = set()
    for color in colors:
        color = color.lower().strip()
        for standard_color, variants in color_words.items():
            if color in variants or color == standard_color:
                standardized_colors.add(standard_color)
                break
        else:
            standardized_colors.add(color)
    
    return list(standardized_colors)

def extract_dimensions(text):
    """Extract dimensions from text"""
    if not text or pd.isna(text):
        return None
    
    # Look for dimension patterns like "600x500x850mm" or "60cm x 50cm x 85cm"
    dim_patterns = [
        r'(\d+(?:\.\d+)?)\s*(?:mm|cm|m)\s*[x×]\s*(\d+(?:\.\d+)?)\s*(?:mm|cm|m)\s*[x×]\s*(\d+(?:\.\d+)?)\s*(?:mm|cm|m)',
        r'(?:dimensions?|size):\s*(?:w|width)?\s*(\d+(?:\.\d+)?)\s*(?:mm|cm|m)\s*(?:[x×]|by)\s*(?:d|depth)?\s*(\d+(?:\.\d+)?)\s*(?:mm|cm|m)\s*(?:[x×]|by)\s*(?:h|height)?\s*(\d+(?:\.\d+)?)\s*(?:mm|cm|m)',
        r'(\d+(?:\.\d+)?)\s*(?:mm|cm|m)\s*(?:width|w)\s*(?:x|×|by)\s*(\d+(?:\.\d+)?)\s*(?:mm|cm|m)\s*(?:depth|d)\s*(?:x|×|by)\s*(\d+(?:\.\d+)?)\s*(?:mm|cm|m)\s*(?:height|h)'
    ]
    
    for pattern in dim_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            w, d, h = match.groups()
            # Convert all dimensions to cm
            def to_cm(val, unit_text):
                val = float(val)
                if 'mm' in unit_text.lower():
                    return val / 10
                elif 'm' in unit_text.lower() and 'cm' not in unit_text.lower():
                    return val * 100
                return val
            
            unit = 'cm' if 'cm' in text else ('mm' if 'mm' in text else 'm')
            return {
                "width": to_cm(w, unit),
                "depth": to_cm(d, unit),
                "height": to_cm(h, unit)
            }
    
    return None

def extract_capacity(title):
    """Extract capacity from title if present"""
    capacity_match = re.search(r'(\d+(?:\.\d+)?)\s*(L|l|kg|KG|Kg|KGS|kgs|Kgs|Ton|TON|ton)', title)
    if capacity_match:
        value, unit = capacity_match.groups()
        # Standardize units
        unit = unit.lower()
        if unit in ['ton', 'tons']:
            return f"{value} Ton"
        elif unit in ['l']:
            return f"{value}L"
        elif unit in ['kg', 'kgs']:
            return f"{value} Kg"
    return None

def extract_sweep_size(title):
    """Extract sweep size for fans"""
    sweep_match = re.search(r'(\d+)\s*(?:mm|MM|cm|CM)', title)
    if sweep_match:
        size = sweep_match.group(1)
        return f"{size}mm"
    return None

def convert_to_json():
    # Read the CSV file
    df = pd.read_csv('cleaned_products.csv')
    
    # Initialize the catalog structure
    catalog = {
        "refrigerator": [],
        "washing_machine": [],
        "dishwasher": [],
        "chimney": [],
        "geyser": [],
        "ceiling_fan": [],
        "bathroom_exhaust": [],
        "shower_system": [],
        "ac": [],
        "gas_stove": [],
        "small_fan": [],
        "wash_basin": [],
        "tiles": [],
        "faucet": [],
        "plywood": []
    }
    
    # Map product types with more variations
    product_type_mapping = {
        # Refrigerators
        "Refrigerator": "refrigerator",
        "Fridge": "refrigerator",
        "Refrigerator Freezer": "refrigerator",
        "Fridge Freezer": "refrigerator",
        
        # Washing Machines
        "Washing Machine": "washing_machine",
        "Washer": "washing_machine",
        "Laundry Machine": "washing_machine",
        "Clothes Washer": "washing_machine",
        "Dishwasher": "dishwasher",
        
        # Chimneys
        "Chimney": "chimney",
        "Cooker Hood": "chimney",
        "Kitchen Hood": "chimney",
        
        # Water Heaters
        "Water Heater": "geyser",
        "Storage Water Heater": "geyser",
        "Instant Water Heater": "geyser",
        "Geyser": "geyser",
        
        # Fans
        "Ceiling Fan": "ceiling_fan",
        "Exhaust Fan": "bathroom_exhaust",
        "Bathroom Exhaust": "bathroom_exhaust",
        "Ventilation Fan": "bathroom_exhaust",
        "Wall Fan": "small_fan",
        "Table Fan": "small_fan",
        "Pedestal Fan": "small_fan",
        "Personal Fan": "small_fan",
        
        # Showers
        "Shower": "shower_system",
        "Body Shower": "shower_system",
        "Hand Shower": "shower_system",
        "Shower System": "shower_system",
        
        # Air Conditioners
        "Air Conditioner": "ac",
        "Air Conditioners": "ac",
        "AC": "ac",
        "Split AC": "ac",
        "Window AC": "ac",
        
        # Kitchen Appliances
        "Gas Stove": "gas_stove",
        "Hob Top": "gas_stove",
        "Cooktop": "gas_stove",
        "Stove": "gas_stove",
        
        # Sanitaryware
        "Wash Basin": "wash_basin",
        "Basin": "wash_basin",
        "Sink": "wash_basin",
        
        # Tiles
        "Tiles": "tiles",
        "Floor Tiles": "tiles",
        "Wall Tiles": "tiles",
        
        # Faucets
        "Faucet": "faucet",
        "Mixer": "faucet",
        "Tap": "faucet",
        
        # Wood
        "Plywood": "plywood",
        "Veneer": "plywood",
        "Timber": "plywood"
    }
    
    # Process each row
    for _, row in df.iterrows():
        product_type = row['Product Type']
        if pd.isna(product_type):
            continue
            
        # Find matching category
        category = None
        # First check for exact matches to handle special cases like dishwashers
        if product_type.lower() == "dishwasher":
            category = "dishwasher"
        else:
            # Then check for partial matches for other categories
            for key, value in product_type_mapping.items():
                if key.lower() in product_type.lower():
                    category = value
                    break
                
        if not category:
            continue
            
        # Extract product details
        product = {
            "brand": row['Brand'],
            "model": row['title'],
            "price": float(row['Better Home Price']),
            "features": clean_features(row['Description']),
            "warranty": row['Warranty'],
            "color_options": extract_colors(row['Color']),
            "in_stock": True,
            "delivery_time": "2-4 days",
            "url": row['url']
        }
        
        # Add type-specific attributes
        if category == "ac":
            product["type"] = "Split" if "split" in row['title'].lower() else "Window"
            product["capacity"] = extract_capacity(row['title'])
        elif category == "refrigerator":
            product["type"] = "Double Door" if "double" in row['title'].lower() else "Single Door"
            product["capacity"] = extract_capacity(row['title'])
        elif category == "washing_machine":
            product["type"] = "Front Load" if "front" in row['title'].lower() else "Top Load"
            product["capacity"] = extract_capacity(row['title'])
        elif category in ["ceiling_fan", "bathroom_exhaust", "small_fan"]:
            product["type"] = "Standard"
            product["sweep_size"] = extract_sweep_size(row['title'])
        elif category == "chimney":
            product["type"] = "Wall Mounted" if "wall" in row['title'].lower() else "Island"
            product["suction_power"] = "1200 m³/hr"  # Default value
        elif category == "geyser":
            product["type"] = "Storage" if "storage" in row['title'].lower() else "Instant"
            product["capacity"] = extract_capacity(row['title'])
        
        # Add dimensions if available
        dimensions = extract_dimensions(row['Description'])
        if dimensions:
            product["dimensions"] = dimensions
            
        # Add to catalog
        catalog[category].append(product)
    
    # Save to JSON file
    with open('product_catalog.json', 'w') as f:
        json.dump(catalog, f, indent=4)

if __name__ == "__main__":
    convert_to_json()