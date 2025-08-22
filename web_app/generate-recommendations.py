import pandas as pd
import json
import yaml
from typing import Dict, Any, List
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
import sys
from reportlab.pdfgen import canvas
import requests
import os
from urllib.parse import urlparse, quote_plus
import re # Add import for regex
from reportlab.platypus import Image
from reportlab.lib.colors import HexColor
from reportlab.platypus.flowables import HRFlowable
from datetime import datetime
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
import pprint
from s3_config import S3Handler
from os.path import splitext
import sys
import argparse

# Update the read_best_sellers_csv function to read from Google Sheets CSV export
BEST_SELLERS_SHEET_CSV_URL = 'https://docs.google.com/spreadsheets/d/18Z8nJdJstXKGgmExWnXqtB5dCOgWelp36AKG9DOGkXs/export?format=csv'



def read_best_sellers_csv(csv_path: str) -> List[Dict[str, Any]]:
    """Read and parse the best-sellers.csv file into a list of product dicts with only required fields."""
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    df.columns = [col.strip() for col in df.columns]
    products = []
    for _, row in df.iterrows():
        def parse_price(val):
            val = str(val).replace(',', '').replace('"', '').strip()
            try:
                return float(val) if val else 0.0
            except Exception:
                return 0.0
        def parse_int(val):
            try:
                return int(str(val).strip()) if str(val).strip() else 0
            except Exception:
                return 0
        product = {
            'sku': row.get('SKU Code', '').strip(),
            'category': row.get('Category', '').strip(),
            'bh_price': parse_price(row.get('BH Price', '')),
            'standard_premium': row.get('Standard/Premium', '').strip(),
            'brand': row.get('Brand', '').strip(),
            'title': row.get('Title', '').strip(),
            'priority': parse_int(row.get('Priority', '')),
            'image_src': row.get('Product Image URL', '').strip(),  # <-- Add this line
            'top_benefits': row.get('Top Benefits', '').strip(),
        }
        products.append(product)
    return products

def get_recommended_products(products: List[Dict[str, Any]], category: str, budget: float) -> List[Dict[str, Any]]:
    """Filter and sort products for a given category and budget."""
    # Budget logic: <400000 = Standard, >=400000 = Premium
    if budget < 400000:
        filter_type = 'Standard'
    else:
        filter_type = 'Premium'
    filtered = [p for p in products if p['category'].lower() == category.lower() and p['standard_premium'].lower() == filter_type.lower()]
    # Sort by priority (ascending), then by price (ascending)
    filtered.sort(key=lambda x: (x['priority'], x['bh_price'] or x['retail_price'] or x['mrp']))
    return filtered  # Return top 3 by priority


def load_product_catalog_json(json_path: str) -> List[Dict[str, Any]]:
    """Load the full product catalog from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data.get('products', [])


def get_recommended_products_with_fallback(
    best_seller_products: List[Dict[str, Any]],
    catalog_products: List[Dict[str, Any]],
    category: str,
    budget: float
) -> List[Dict[str, Any]]:
    """
    Prefer products from best-sellers.csv, but fall back to catalog if none found.
    Adds a 'source' field to each result: 'best-seller' or 'catalog'.
    """
    # Try best-sellers first
    best = get_recommended_products(best_seller_products, category, budget)
    for p in best:
        p['source'] = 'best-seller'
    if best:
        return best
    # Fallback to catalog
    # Budget logic: <400000 = Standard, >=400000 = Premium (use price fields)
    if budget < 400000:
        price_field = 'better_home_price' if 'better_home_price' in catalog_products[0] else 'retail_price'
        price_limit = 40000  # Example threshold for 'Standard' (adjust as needed)
    else:
        price_field = 'better_home_price' if 'better_home_price' in catalog_products[0] else 'retail_price'
        price_limit = 1000000  # High value for 'Premium'
    filtered = [p for p in catalog_products if str(p.get('product_type', '')).lower() == category.lower() and float(p.get(price_field, 0)) <= price_limit]
    # Sort by price ascending, then by title
    filtered.sort(key=lambda x: (float(x.get(price_field, 0)), x.get('title', '')))
    for p in filtered:
        p['source'] = 'catalog'
    return filtered


def analyze_user_requirements(excel_file: str):
    try:
        # Read the Excel file
        print(f"Reading Excel file: {excel_file}")
        df = pd.read_excel(excel_file)
        
        print(f"Excel file loaded, columns: {df.columns.tolist()}")
        
        # Clean up column names by removing newlines and extra spaces
        df.columns = [col.split('\n')[0].strip() for col in df.columns]
        
        print(f"Cleaned column names: {df.columns.tolist()}")
        
        if df.empty:
            print("ERROR: Excel file has no data rows")
            return None
            
        # Check for required columns
        required_columns = [
            'Name', 
            'Mobile Number (Preferably on WhatsApp)', 
            'E-mail', 
            'Apartment Address',
            'What is your overall budget for home appliances?',
            'Number of bedrooms',
            'Number of bathrooms',
            'Adults (between the age 18 to 50)',
            'Elders (above the age 60)',
            'Kids (below the age 18)'
        ]
        
        # Check which required columns are missing
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"ERROR: Missing required columns: {missing_columns}")
            return None
        
        row = df.iloc[0]
        print(f"First row: {row.head()}")
        
        # Use safer data access with error handling
        try:
            # Handle potentially empty or non-numeric fields
            def safe_int(val):
                try:
                    if pd.isna(val):
                        return 0
                    return int(val)
                except (ValueError, TypeError):
                    print(f"Warning: Could not convert {val} to int, using 0")
                    return 0
            def safe_float(val):
                try:
                    if pd.isna(val):
                        return 0.0
                    return float(val)
                except (ValueError, TypeError):
                    print(f"Warning: Could not convert {val} to float, using 0.0")
                    return 0.0
            def safe_str(val):
                if pd.isna(val):
                    return ""
                return str(val).replace('\n', ' ')
            # Convert DataFrame to dictionary with safer access
            user_data = {
                'name': safe_str(df.iloc[0]['Name']),
                'mobile': safe_str(df.iloc[0]['Mobile Number (Preferably on WhatsApp)']),
                'email': safe_str(df.iloc[0]['E-mail']),
                'address': safe_str(df.iloc[0]['Apartment Address']),
                'total_budget': safe_float(df.iloc[0]['What is your overall budget for home appliances?']),
                'num_bedrooms': safe_int(df.iloc[0]['Number of bedrooms']),
                'num_bathrooms': safe_int(df.iloc[0]['Number of bathrooms']),
                'demographics': {
                    'adults': safe_int(df.iloc[0]['Adults (between the age 18 to 50)']),
                    'elders': safe_int(df.iloc[0]['Elders (above the age 60)']),
                    'kids': safe_int(df.iloc[0]['Kids (below the age 18)'])
                }
            }
        except Exception as e:
            print(f"Error reading user information: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
        # Clean up NaN values
        def clean_dict(d):
            for k, v in d.items():
                if isinstance(v, dict):
                    clean_dict(v)
                elif pd.isna(v):
                    d[k] = None
                elif isinstance(v, str) and v.lower() == 'nan':
                    d[k] = None
        clean_dict(user_data)
        # Extract room requirements
        requirements = {
            'hall': {
                'fans': int(df.iloc[0]['Hall: Fan(s)?']),
                'ac': df.iloc[0]['Hall: Air Conditioner (AC)?'] == 'Yes',
                'color_theme': df.iloc[0]['Hall: Colour theme?'],
                'size_sqft': float(df.iloc[0].get('Hall: What is the square feet ?', 150.0)),  # Updated column name
                'is_for_kids': df.iloc[0].get('Hall: Is this for kids above', 'No') == 'Yes'  # Add is_for_kids field
            },
            'kitchen': {
                'chimney_width': df.iloc[0]['Kitchen: Chimney width?'],
                'gas_stove_type': df.iloc[0]['Kitchen: Gas stove type?'],
                'num_burners': int(df.iloc[0]['Kitchen: Number of burners?']),
                'small_fan': df.iloc[0]['Kitchen: Do you need a small fan?'] == 'Yes',
                'color_theme': None,  # No color theme specified for kitchen
                'refrigerator_type': df.iloc[0].get('Kitchen: Refrigerator type?', None), # Add refrigerator type
                'refrigerator_capacity': df.iloc[0].get('Kitchen: Refrigerator capacity?', None), # Add refrigerator capacity
                'dishwasher_capacity': df.iloc[0].get('Kitchen: Dishwasher capacity?', None), # Add dishwasher capacity
                'size_sqft': float(df.iloc[0].get('Kitchen: Size (square feet)', 100.0)),  # Default to 100 sq ft if not specified
                'is_for_kids': df.iloc[0].get('Kitchen: Is this for kids above', 'No') == 'Yes'  # Add is_for_kids field
            },
            'master_bedroom': {
                'ac': df.iloc[0]['Master: Air Conditioner (AC)?'] == 'Yes',
                'bathroom': {
                    'water_heater_type': df.iloc[0]['Master: How do you bath with the hot & cold water?'],
                    'exhaust_fan_size': df.iloc[0]['Master: Exhaust fan size?'],
                    'water_heater_ceiling': df.iloc[0]['Master: Is the water heater going to be inside the false ceiling in the bathroom?'],
                    'led_mirror': df.iloc[0]['Master: Would you like to have a LED Mirror?'] == 'Yes',  # Add LED mirror preference
                    'glass_partition': df.iloc[0].get('Master: Do you want a Glass Partition in the bathroom?') == 'Yes'  # Add glass partition preference
                },
                'color_theme': df.iloc[0]['Master: What is the colour theme?'],
                'size_sqft': float(df.iloc[0].get('Master: What is the area of the bedroom in square feet?', 140.0)),  # Updated column name
                'is_for_kids': df.iloc[0].get('Master: Is this for kids above', 'No') == 'Yes'  # Add is_for_kids field
            },
            'bedroom_2': {
                'ac': df.iloc[0]['Bedroom 2: Air Conditioner (AC)?'] == 'Yes',
                'bathroom': {
                    'water_heater_type': df.iloc[0]['Bedroom 2: How do you bath with the hot & cold water?'],
                    'exhaust_fan_size': df.iloc[0]['Bedroom 2: Exhaust fan size?'],
                    'water_heater_ceiling': df.iloc[0]['Bedroom 2: Is the water heater going to be inside the false ceiling in the bathroom?'],
                    'led_mirror': df.iloc[0]['Bedroom 2: Would you like to have a LED Mirror?'] == 'Yes',  # Add LED mirror preference
                    'glass_partition': df.iloc[0].get('Bedroom 2: Do you want a Glass Partition in the bathroom?') == 'Yes'  # Add glass partition preference
                },
                'color_theme': df.iloc[0]['Bedroom 2: What is the colour theme?'],
                'size_sqft': float(df.iloc[0].get('Bedroom 2: What is the area of the bedroom in square feet?', 120.0)),  # Updated column name
                'is_for_kids': df.iloc[0].get('Bedroom 2: Is this for kids above', 'No') == 'Yes'  # Add is_for_kids field
            },
            'bedroom_3': {
                'ac': df.iloc[0]['Bedroom 3: Air Conditioner (AC)?'] == 'Yes',
                'bathroom': {
                    'water_heater_type': df.iloc[0]['Bedroom 3: How do you bath with the hot & cold water?'],
                    'exhaust_fan_size': df.iloc[0]['Bedroom 3: Exhaust fan size?'],
                    'water_heater_ceiling': df.iloc[0]['Bedroom 3: Is the water heater going to be inside the false ceiling in the bathroom?'],
                    'led_mirror': df.iloc[0]['Bedroom 3: Would you like to have a LED Mirror?'] == 'Yes',  # Add LED mirror preference
                    'glass_partition': df.iloc[0].get('Bedroom 3: Do you want a Glass Partition in the bathroom?') == 'Yes'  # Add glass partition preference
                },
                'color_theme': df.iloc[0]['Bedroom 3: What is the colour theme?'],
                'size_sqft': float(df.iloc[0].get('Bedroom 3: What is the area of the bedroom in square feet?', 120.0)),  # Updated column name
                'is_for_kids': df.iloc[0].get('Bedroom 3: Is this for kids above', 'No') == 'Yes'  # Add is_for_kids field
            },
            'laundry': {
                'washing_machine_type': df.iloc[0]['Laundry: Washing Machine?'],
                'dryer_type': df.iloc[0]['Laundry: Dryer?'],
                'color_theme': None,  # No color theme specified for laundry
                'size_sqft': float(df.iloc[0].get('Laundry: Size (square feet)', 50.0)),  # Default to 50 sq ft if not specified
                'is_for_kids': df.iloc[0].get('Laundry: Is this for kids above', 'No') == 'Yes'  # Add is_for_kids field
            },
            'dining': {
                'fan_size': df.iloc[0].get('Dining: Fan', None),
                'fans': safe_int(df.iloc[0].get('Dining: Fan(s)?', 1)),  # Use safe_int to handle non-numeric values
                'ac': df.iloc[0].get('Dining: Air Conditioner (AC)?', 'No') == 'Yes',
                'color_theme': df.iloc[0].get('Dining: Colour theme?', None),
                'size_sqft': safe_float(df.iloc[0].get('Dining: What is the square feet?', 120.0)),  # Use safe_float
                'is_for_kids': df.iloc[0].get('Dining: Is this for kids above', 'No') == 'Yes'  # Add is_for_kids field
            }
        }
        # Merge requirements into user_data
        user_data.update(requirements)
        # Debug: Log the first few rows of the DataFrame
        print("First few rows of the DataFrame:")
        print(df.head())
        # Debug: Log the extracted user data
        print("Extracted user data:", user_data)
        # Debug: Log the extracted requirements
        print("Extracted requirements:", requirements)
        return user_data
    except Exception as e:
        print(f"Error reading user information: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def main(user_xlsx: str, best_sellers_csv: str):
    # Step 1: Read user data
    user_data = analyze_user_requirements(user_xlsx)
    if not user_data:
        print("Failed to read user data from Excel.")
        return

    # Step 2: Read best-sellers products
    products = read_best_sellers_csv(best_sellers_csv)
    if not products:
        print("No products found in best-sellers.csv.")
        return

    # Step 3: For each required appliance, generate recommendations
    appliance_map = {
        'ac': 'Air Conditioner',
        'ceiling_fan': 'Ceiling Fan',
        'water_heater': 'Storage Water Heater',
        'instant_water_heater': 'Instant Water Heater',
        'chimney': 'Chimney',
        'hob': 'Hob',
        'cooktop': 'Cooktop',
        'dishwasher': 'Dishwasher',
        'washing_machine': 'Washing Machine',
        'refrigerator': 'Refrigerator',
        'microwave': 'Microwave Oven',
        'otg': 'OTG',
        'pedestal_fan': 'Pedestal Fan',
        'wall_fan': 'Wall Mounted Fan',
        'exhaust_fan': 'Exhaust Fan',
        'vacuum_cleaner': 'Vacuum Cleaner',
        'led_mirror': 'LED Mirror',
    }
    recommendations = {}
    total_budget = user_data.get('total_budget', 0)
    # Hall
    recommendations['hall'] = {}
    if user_data['hall'].get('ac'):
        ac_recs = get_ac_recommendations(products, total_budget)
        recommendations['hall']['ac'] = ac_recs
    if user_data['hall'].get('fans', 0) > 0:
        fan_recs = get_appliance_recommendations(products, appliance_map['ceiling_fan'], total_budget, user_data['hall']['fans'])
        recommendations['hall']['fans'] = fan_recs

    # Kitchen
    recommendations['kitchen'] = {}
    if user_data['kitchen'].get('chimney_width'):
        chimney_recs = get_appliance_recommendations(products, appliance_map['chimney'], total_budget)
        recommendations['kitchen']['chimney'] = chimney_recs
    if user_data['kitchen'].get('dishwasher_capacity'):
        dishwasher_recs = get_appliance_recommendations(products, appliance_map['dishwasher'], total_budget)
        recommendations['kitchen']['dishwasher'] = dishwasher_recs
    if user_data['kitchen'].get('refrigerator_type'):
        refrigerator_recs = get_appliance_recommendations(products, appliance_map['refrigerator'], total_budget)
        recommendations['kitchen']['refrigerator'] = refrigerator_recs
    if user_data['kitchen'].get('num_burners', 0) > 0:
        hob_recs = get_appliance_recommendations(products, appliance_map['hob'], total_budget)
        recommendations['kitchen']['hob'] = hob_recs

    # Master Bedroom
    recommendations['master_bedroom'] = {}
    if user_data['master_bedroom'].get('ac'):
        ac_recs = get_ac_recommendations(products, total_budget)
        recommendations['master_bedroom']['ac'] = ac_recs
    if user_data['master_bedroom'].get('bathroom', {}).get('water_heater_type'):
        water_heater_recs = get_appliance_recommendations(products, appliance_map['water_heater'], total_budget)
        recommendations['master_bedroom']['water_heater'] = water_heater_recs

    # Bedroom 2
    recommendations['bedroom_2'] = {}
    if user_data['bedroom_2'].get('ac'):
        ac_recs = get_ac_recommendations(products, total_budget)
        recommendations['bedroom_2']['ac'] = ac_recs
    if user_data['bedroom_2'].get('bathroom', {}).get('water_heater_type'):
        water_heater_recs = get_appliance_recommendations(products, appliance_map['water_heater'], total_budget)
        recommendations['bedroom_2']['water_heater'] = water_heater_recs

    # Bedroom 3
    recommendations['bedroom_3'] = {}
    if user_data['bedroom_3'].get('ac'):
        ac_recs = get_ac_recommendations(products, total_budget)
        recommendations['bedroom_3']['ac'] = ac_recs
    if user_data['bedroom_3'].get('bathroom', {}).get('water_heater_type'):
        water_heater_recs = get_appliance_recommendations(products, appliance_map['water_heater'], total_budget)
        recommendations['bedroom_3']['water_heater'] = water_heater_recs

    # Laundry
    recommendations['laundry'] = {}
    if user_data['laundry'].get('washing_machine_type'):
        washing_machine_recs = get_appliance_recommendations(products, appliance_map['washing_machine'], total_budget)
        recommendations['laundry']['washing_machine'] = washing_machine_recs

    # Dining
    recommendations['dining'] = {}
    if user_data['dining'].get('ac'):
        ac_recs = get_ac_recommendations(products, total_budget)
        recommendations['dining']['ac'] = ac_recs
    if user_data['dining'].get('fans', 0) > 0:
        fan_recs = get_appliance_recommendations(products, appliance_map['ceiling_fan'], total_budget, user_data['dining']['fans'])
        recommendations['dining']['fans'] = fan_recs

    # Enrich best-seller products with catalog info
    catalog_products = load_product_catalog_json("product_catalog.json")


    recommendations = enrich_recommendations(recommendations, catalog_products)

    # Step 4: Write HTML output
    base, _ = splitext(user_xlsx)
    html_filename = base + ".html"
    generate_html_file(user_data, recommendations, html_filename, default_mode=True)
    # Assuming generate_html_file is available or needs to be copied
    # For now, we'll just print the filename
    print(f"Generated HTML recommendations: {html_filename}")

# Example usage (uncomment and set file paths to run as script):
# if __name__ == "__main__":
#     main("user_input.xlsx", "web_app/best-sellers.csv")

def enrich_recommendations(recs, catalog_products):
    if isinstance(recs, dict):
        for k, v in recs.items():
            recs[k] = enrich_recommendations(v, catalog_products)
        return recs
    elif isinstance(recs, list):
        for i, prod in enumerate(recs):
            recs[i] = enrich_best_seller_product(prod, catalog_products)
        return recs
    else:
        return recs
    
import re

def render_top_benefits(top_benefits):
    if not top_benefits:
        return ""
    # Split on numbered points (e.g., 1. ... 2. ... 3. ...)
    points = re.split(r'(?:^|\\n|\\r|;|\\.)\\s*(\\d+\\.)', top_benefits)
    # The split will keep the numbers as separate elements, so we need to recombine them
    items = []
    i = 1
    while i < len(points):
        # points[i] is the number (e.g., '1.')
        # points[i+1] is the text
        if i+1 < len(points):
            text = points[i+1].strip()
            if text:
                items.append(text)
        i += 2
    # Fallback: if nothing found, just show the whole string as one item
    if not items:
        items = [top_benefits.strip()]
    html = "<ul class='top-benefits-list'>"
    for item in items:
        html += f"<li>{item}</li>"
    html += "</ul>"
    return html


def calculate_total_cost(recommendations):
    """Calculate total cost by summing BH price for all selected items including quantities.

    - For list entries, sum the price of each product in the list (duplicates represent quantity).
    - For nested dict entries (e.g., bathrooms), also sum each product in the nested lists.
    - Prefer 'bh_price'; if missing/zero, fall back to 'better_home_price' then 'market_price_1'.
    """
    def extract_price(product: dict) -> float:
        try:
            price = float(product.get('bh_price', 0) or 0)
            if price and price > 0:
                return price
        except Exception:
            pass
        # Fallbacks
        for fld in ('better_home_price', 'market_price_1'):
            try:
                val = float(product.get(fld, 0) or 0)
                if val and val > 0:
                    return val
            except Exception:
                continue
        return 0.0

    total_cost = 0.0
    for room, products in recommendations.items():
        if not isinstance(products, dict):
            continue
        for product_type, options in products.items():
            if isinstance(options, dict):
                # Nested categories (e.g., bathroom)
                for nested_type, nested_options in options.items():
                    if not isinstance(nested_options, list):
                        continue
                    for product in nested_options:
                        if isinstance(product, dict):
                            total_cost += extract_price(product)
            elif isinstance(options, list):
                for product in options:
                    if isinstance(product, dict):
                        total_cost += extract_price(product)
    return total_cost



def get_room_description(room: str, user_data: Dict[str, Any]) -> str:
    """Generate a description for each room based on user requirements, calling out if it's for kids or elders."""
    # Helper to get label
    def get_special_label(room_data):
        if room_data.get('is_for_kids', False):
            return 'For Kids'
        elif room_data.get('is_for_elders', False):
            return 'For Elders'
        return None

    if room == 'hall':
        room_size = user_data['hall'].get('size_sqft', 150.0)
        ac_info = ""
        if user_data['hall'].get('ac', False):
            recommended_tonnage = determine_ac_tonnage(room_size, 'hall')
            ac_info = f"an AC ({recommended_tonnage} Ton recommended)"
        else:
            ac_info = "no AC"
        return f"A welcoming space of approximately {room_size} sq ft with {user_data['hall'].get('fans', 'no')} fan(s) and {ac_info}, " \
               f"complemented by a {user_data['hall'].get('color_theme', 'neutral')} color theme."
    
    elif room == 'kitchen':
        room_size = user_data['kitchen'].get('size_sqft', 100.0)
        return f"A functional kitchen of {room_size} sq ft with a {user_data['kitchen'].get('chimney_width', 'standard')} chimney, " \
               f"{user_data['kitchen'].get('stove_type', 'standard')} with {user_data['kitchen'].get('num_burners', '4')} burners, " \
               f"and {'a small fan' if user_data['kitchen'].get('small_fan', False) else 'no fan'}."
    
    elif room == 'master_bedroom':
        room_data = user_data['master_bedroom']
        label = get_special_label(room_data)
        room_size = room_data.get('size_sqft', 140.0)
        ac_info = ""
        if room_data.get('ac', False):
            recommended_tonnage = determine_ac_tonnage(room_size, 'master_bedroom')
            ac_info = f"an AC ({recommended_tonnage} Ton recommended)"
        else:
            ac_info = "no AC"
        prefix = "Master bedroom"
        if label:
            prefix += f" - {label}"
        return f"{prefix} of {room_size} sq ft with {room_data.get('color_theme', 'neutral')} theme, " \
               f"{ac_info}, " \
               f"and a bathroom equipped with {room_data.get('bathroom', {}).get('water_heater_type', 'standard')} water heating."
    
    elif room == 'bedroom_2':
        room_data = user_data['bedroom_2']
        label = get_special_label(room_data)
        room_size = room_data.get('size_sqft', 120.0)
        ac_info = ""
        if room_data.get('ac', False):
            recommended_tonnage = determine_ac_tonnage(room_size, 'bedroom_2')
            ac_info = f"an AC ({recommended_tonnage} Ton recommended)"
        else:
            ac_info = "no AC"
        prefix = "Second bedroom"
        if label:
            prefix += f" - {label}"
        return f"{prefix} of {room_size} sq ft with {room_data.get('color_theme', 'neutral')} theme, " \
               f"{ac_info}, " \
               f"and a bathroom equipped with {room_data.get('bathroom', {}).get('water_heater_type', 'standard')} water heating."

    elif room == 'bedroom_3':
        room_data = user_data['bedroom_3']
        label = get_special_label(room_data)
        room_size = room_data.get('size_sqft', 120.0)
        ac_info = ""
        if room_data.get('ac', False):
            recommended_tonnage = determine_ac_tonnage(room_size, 'bedroom_3')
            ac_info = f"an AC ({recommended_tonnage} Ton recommended)"
        else:
            ac_info = "no AC"
        prefix = "Third bedroom"
        if label:
            prefix += f" - {label}"
        return f"{prefix} of {room_size} sq ft with {room_data.get('color_theme', 'neutral')} theme, " \
               f"{ac_info}, " \
               f"and a bathroom equipped with {room_data.get('bathroom', {}).get('water_heater_type', 'standard')} water heating."
    
    elif room == 'laundry':
        room_size = user_data['laundry'].get('size_sqft', 50.0)
        return f"Laundry area of {room_size} sq ft equipped with a {user_data['laundry'].get('washing_machine_type', 'standard')} washing machine" \
               f"{' and a dryer' if user_data['laundry'].get('dryer_type', '').lower() == 'yes' else ''}."
    
    return ""

# Function to generate an HTML file with recommendations
def generate_html_file(user_data: Dict[str, Any], final_list: Dict[str, Any], html_filename: str, default_mode: bool = False) -> None:
    #import pprint
    #pprint.pprint(final_list)
    # print("[DEBUG HTML] final_list keys:", list(final_list.keys()))
    
    # Get current date for the footer
    current_date = pd.Timestamp.now().strftime("%Y-%m-%d")
    current_year = pd.Timestamp.now().year
    
    # Check if logo exists in multiple possible locations
    possible_logo_paths = [
        os.path.join(os.path.dirname(__file__), 'better_home_logo.png'),
        os.path.join(os.path.dirname(__file__), 'static', 'better_home_logo.png'),
        'better_home_logo.png'
    ]
    
    logo_path = None
    logo_exists = False
    for path in possible_logo_paths:
        if os.path.exists(path):
            logo_path = path
            logo_exists = True
            print(f"Found logo at: {path}")
            break
    
    if not logo_exists:
        print("Logo not found in any of the expected locations")
    
    # Always use a relative URL path that will be handled by Flask
    logo_html = ""
    if logo_exists:
        logo_html = '<img src="/static/better_home_logo.png" alt="BetterHome Logo" class="logo">'
    
    # Helper to render Top Benefits as a clean list with separators (no auto numbers)
    def render_benefits(benefits_text: str) -> str:
        if not benefits_text:
            return ""
        text = str(benefits_text).strip()
        if not text:
            return ""
        items = []
        # Try splitting by common separators
        for sep in ["\n", "•", ";", "|", " — ", " - ", "·"]:
            parts = [p.strip() for p in text.split(sep) if p and p.strip()]
            if len(parts) >= 2:
                items = parts
                break
        # Fallback to sentence split
        if not items:
            parts = re.split(r"\.(?:\s+|$)", text)
            items = [p.strip() for p in parts if p and p.strip()]
        if not items:
            items = [text]
        lis = ''.join([f'<li>{p}</li>' for p in items])
        return f'<ul class="benefits-list">{lis}</ul>'

    # Create HTML header (CSS part)
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>BetterHome Product Recommendations</title>
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap" rel="stylesheet">
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
        <!-- Excel libraries with integrity checks and error handling -->
        <script src="/static/xlsx.full.min.js"></script>
        <script src="/static/FileSaver.min.js"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/html2pdf.js/0.10.1/html2pdf.bundle.min.js" integrity="sha512-GsLlZN/3F2ErC5ifS5QtgpiJtWd43JWSuIgh7mbzZ8zBps+dvLusV+eNQATqgA/HdeKFVgA5v3S/cIrLF7QnIg==" crossorigin="anonymous" referrerpolicy="no-referrer"></script>
        <style>
            /* Modern typography and base styles */
            body {
                font-family: 'Poppins', 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
                margin: 0;
                padding: 0;
                line-height: 1.6;
                color: #333;
                background-color: #f9f9f9;
            }
            
            .container {
                max-width: 1100px;
                margin: 0 auto;
                padding: 30px;
            }
            
            header {
                text-align: center;
                margin-bottom: 40px;
                border-bottom: 1px solid #eaeaea;
                padding-bottom: 30px;
                background: linear-gradient(to right, #f8f9fa, #e9ecef);
                border-radius: 8px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            }
            
            .logo {
                max-width: 200px;
                margin-bottom: 15px;
                transition: transform 0.3s ease;
            }
            
            .logo:hover {
                transform: scale(1.05);
            }
            
            /* Client information styling */
            .client-info {
                margin: 30px 0;
                padding: 20px;
                border-radius: 8px;
                background-color: #fff;
                box-shadow: 0 2px 10px rgba(0,0,0,0.05);
                display: grid;
                grid-template-columns: 1fr; /* Mobile-first: single column */
                gap: 12px;
            }
            
            @media (min-width: 600px) {
                .client-info { grid-template-columns: repeat(2, minmax(0, 1fr)); }
            }
            @media (min-width: 900px) {
                .client-info { grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); }
            }
            
            .client-info-item { margin-bottom: 0; min-width: 0; }
            
            .client-info-label {
                font-weight: 500;
                color: #666;
                margin-bottom: 4px;
            }
            
            .client-info-value {
                font-weight: 600;
                color: #333;
                word-break: break-word;
                overflow-wrap: anywhere;
                min-width: 0;
            }

            @media (max-width: 480px) {
                .container { padding: 20px 16px; }
            }
            
            /* Product card and grid styling */
            .products-grid {
                display: grid;
                grid-template-columns: 1fr;
                gap: 30px;
                margin-top: 20px;
            }
            
            @media (min-width: 768px) {
                .products-grid {
                       grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
                }
            }
            
            .product-card {
                background: #fff;
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 3px 10px rgba(0,0,0,0.1);
                display: flex;
                flex-direction: column;
                transition: transform 0.3s ease;
                position: relative;
            }

            .product-card.best-product {
                border: 2px solid #3498db;  // Emphasize the best product
                transform: scale(1.05);  // Slightly enlarge the best product
            }
            
            .product-selection {
                position: absolute;
                top: 10px;
                left: 10px;
                z-index: 10;
            }
            
            .product-checkbox {
                width: 20px;
                height: 20px;
                cursor: pointer;
            }
            
            .product-checkbox:checked + label {
                font-weight: bold;
                color: #3498db;
            }
            
            .selection-label {
                background: #3498db;
                color: white;
                padding: 4px 8px;
                border-radius: 4px;
                font-size: 12px;
                margin-left: 5px;
                display: none;
            }
            
            .product-checkbox:checked ~ .selection-label {
                display: inline-block;
            }
            
            .product-image-container {
                position: relative;
                height: 200px;
                overflow: hidden;
            }
            
            .product-image {
                width: 100%;
                height: 100%;
                object-fit: contain;
                transition: transform 0.3s ease;
            }
            
            .product-card:hover .product-image {
                transform: scale(1.05);
            }
            
            /* Badge styling */
            .bestseller-badge, .recommended-badge {
                position: absolute;
                top: 10px;
                right: 10px;
                color: white;
                padding: 5px 10px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 12px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.2);
                display: flex;
                align-items: center;
                gap: 5px;
            }
            
            .bestseller-badge {
                background-color: #ff6b00;
            }
            
            .recommended-badge {
                background-color: #27ae60;
            }
            
            .bestseller-badge i, .recommended-badge i {
                font-size: 14px;
            }
            
            /* Product details styling */
            .product-details {
                padding: 20px;
                flex-grow: 1;
                display: flex;
                flex-direction: column;
            }
            
            .product-type {
                font-size: 12px;
                text-transform: uppercase;
                color: #666;
                margin-bottom: 5px;
                letter-spacing: 1px;
            }
            
            .product-title {
                font-size: 18px;
                font-weight: 600;
                margin: 0 0 15px;
                color: #333;
            }
            
            .price-container {
                margin-bottom: 15px;
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
                align-items: baseline;
            }
            
            .current-price {
                font-size: 20px;
                font-weight: 700;
                color: #e74c3c;
            }
            
            .retail-price {
                font-size: 16px;
                color: #7f8c8d;
                text-decoration: line-through;
            }
            
            .savings {
                font-size: 14px;
                color: #27ae60;
                font-weight: 500;
            }
            
            .product-info-item {
                margin-bottom: 10px;
                font-size: 14px;
                color: #555;
            }
            
            .product-info-label {
                font-weight: 600;
                color: #333;
            }
            
            .reasons-list {
                list-style: none;
                padding: 0;
                margin: 0 0 20px;
            }
            
            .reasons-list li {
                margin-bottom: 8px;
                display: flex;
                align-items: flex-start;
                gap: 8px;
            }
            
            .reasons-list i {
                color: #3498db;
                margin-top: 3px;
            }
            
            .benefits-list {
                list-style: none;
                margin: 8px 0 0 0;
                padding: 0;
                color: #444;
            }
            .benefits-list li {
                padding: 8px 0;
                border-top: 1px solid #eee;
                line-height: 1.4;
            }
            .benefits-list li:first-child { border-top: none; }
            
            .buy-button {
                display: inline-block;
                background-color: #3498db;
                color: white;
                padding: 10px 20px;
                margin-top: auto;
                text-align: center;
                text-decoration: none;
                border-radius: 4px;
                font-weight: 500;
                transition: background-color 0.3s;
            }
            
            .buy-button:hover {
                background-color: #2980b9;
            }
            
            /* Budget summary styling */
            .budget-summary {
                margin: 30px 0;
                padding: 25px;
                border-radius: 8px;
                background-color: #fff;
                box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            }
            
            .budget-summary h2 {
                margin-top: 0;
                margin-bottom: 20px;
                color: #333;
                font-weight: 600;
            }
            
            .budget-info {
                display: grid;
                grid-template-columns: 1fr;
                gap: 15px;
                margin-bottom: 20px;
            }
            
            @media (min-width: 768px) {
                .budget-info {
                       grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
                }
            }
            
            .budget-item {
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 6px;
            }
            
            .budget-item-label {
                font-size: 14px;
                color: #666;
                margin-bottom: 8px;
            }
            
            .budget-item-value {
                font-size: 18px;
                font-weight: 600;
                color: #333;
            }
            
            .budget-status {
                padding: 12px 15px;
                border-radius: 6px;
                font-weight: 500;
            }
            
            .budget-status.good {
                background-color: #d4edda;
                color: #155724;
            }
            
            .budget-status.warning {
                background-color: #fff3cd;
                color: #856404;
            }
            
            /* Room styling */
            .room-section {
                margin: 40px 0;
            }
            
            .room-section h2 {
                font-size: 24px;
                color: #2c3e50;
                margin-bottom: 10px;
                position: relative;
                padding-bottom: 10px;
            }
            
            .room-section h2:after {
                content: '';
                position: absolute;
                bottom: 0;
                left: 0;
                width: 50px;
                height: 3px;
                background-color: #3498db;
            }
            
            .room-description {
                font-size: 16px;
                color: #555;
                margin-bottom: 20px;
                line-height: 1.5;
            }
            
            /* Generate Recommendation Button */
            .generate-container {
                text-align: center;
                margin: 40px 0;
                padding: 20px;
                background-color: #f8f9fa;
                border-radius: 8px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }
            
            .generate-button {
                display: inline-block;
                background-color: #2ecc71;
                color: white;
                padding: 15px 30px;
                border-radius: 6px;
                font-size: 18px;
                font-weight: 600;
                text-decoration: none;
                transition: background-color 0.3s;
                border: none;
                cursor: pointer;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }
            
            .generate-button:hover {
                background-color: #27ae60;
                transform: translateY(-2px);
                box-shadow: 0 6px 8px rgba(0,0,0,0.15);
            }
            
            .generate-button:active {
                transform: translateY(0);
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            
            .download-button {
                background-color: #3498db;
                margin-left: 10px;
            }
            
            .download-button:hover {
                background-color: #2980b9;
            }

            /* Footer styling */
            footer {
                margin-top: 50px;
                padding-top: 20px;
                border-top: 1px solid #eaeaea;
                text-align: center;
                color: #7f8c8d;
                font-size: 14px;
            }
            
            /* Print styles */
            @media print {
                body {
                    background-color: white;
                }
                
                .container {
                    max-width: 100%;
                    padding: 0;
                    margin: 0;
                }
                
                .generate-container,
                .product-selection,
                .buy-button {
                    display: none !important;
                }
                
                .product-card {
                    break-inside: avoid;
                    page-break-inside: avoid;
                    box-shadow: none;
                    border: 1px solid #eaeaea;
                }
                
                .room-section {
                    page-break-before: always;
                }
                
                .room-section:first-child {
                    page-break-before: avoid;
                }
                
                header {
                    page-break-after: avoid;
                }
                
                .client-info, .budget-summary {
                    page-break-inside: avoid;
                }
            }
            
            /* Accordion styles */
            .accordion {
                background-color: #f1f1f1;
                color: #333;
                cursor: pointer;
                padding: 18px;
                width: 100%;
                border: none;
                text-align: left;
                outline: none;
                font-size: 18px;
                transition: background-color 0.2s;
                border-radius: 8px 8px 0 0;
                margin-bottom: 0;
            }
            
            .accordion.active, .accordion:hover {
                background-color: #e2e6ea;
            }
            
            .panel {
                padding: 0 18px 18px 18px;
                background-color: white;
                display: none;
                overflow: hidden;
                border-radius: 0 0 8px 8px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.04);
            }
            
            .panel[style*='display: block'] {
                display: block;
            }

            .features-toggle.open .arrow {
                transform: rotate(90deg);
            }
        </style>
        <script>
            // Wait for the document to be fully loaded
            window.addEventListener('load', function() {
                console.log('Document fully loaded');
                
                // Debug helper to check if elements exist
                function checkElement(id) {
                    const element = document.getElementById(id);
                    console.log(`Element ${id} exists: ${!!element}`);
                    return element;
                }
                // Ensure only one product per category can be selected
                const setupCategorySelections = () => {
                    const categories = {};
                    const processedProducts = new Set(); // To prevent duplicate handling of products
                    
                    // Group checkboxes by category
                    document.querySelectorAll('.product-checkbox').forEach(checkbox => {
                        const category = checkbox.getAttribute('data-category');
                        const room = checkbox.getAttribute('data-room');
                        const productId = checkbox.getAttribute('data-product-id');
                        const key = `${room}-${category}`;
                        
                        // Skip if already processed (for dupicate products)
                        if (processedProducts.has(productId)) {
                            return;
                        }
                        processedProducts.add(productId);
                        
                        if (!categories[key]) {
                            categories[key] = [];
                        }
                        categories[key].push(checkbox);
                        
                        // Add change listener
                        checkbox.addEventListener('change', function() {
                            const isChecked = this.checked;
                            
                            // Update the 'selected' class for this product's card
                            const productCard = this.closest('.product-card');
                            if (isChecked) {
                                productCard.classList.add('selected');
                                
                                // Uncheck other checkboxes in the same category (maintain single selection per category)
                                categories[key].forEach(cb => {
                                    if (cb !== this) {
                                        cb.checked = false;
                                        cb.closest('.product-card').classList.remove('selected');
                                    }
                                });
                            } else {
                                productCard.classList.remove('selected');
                            }
                        });
                    });
                };
                
                // Handle generate final recommendation
                const setupGenerateButton = () => {
                    console.log('Setting up generate button');
                    const generateButton = checkElement('generate-final');
                    if (!generateButton) {
                        console.error('Generate button not found in the DOM');
                        return;
                    }
                    
                    // Automatically run initial selection for best products
                    // This ensures initial visual state matches pre-selected products
                    document.querySelectorAll('.product-checkbox:checked').forEach(checkbox => {
                        const productCard = checkbox.closest('.product-card');
                        if (productCard) {
                            productCard.classList.add('selected');
                        }
                    });
                    
                    generateButton.addEventListener('click', function(e) {
                        console.log('Generate button clicked');
                        e.preventDefault();
                        
                        // Collect all selected products
                        const selectedProducts = [];
                        document.querySelectorAll('.product-checkbox:checked').forEach(checkbox => {
                            const productId = checkbox.getAttribute('data-product-id');
                            const room = checkbox.getAttribute('data-room');
                            const category = checkbox.getAttribute('data-category');
                            const brand = checkbox.getAttribute('data-brand');
                            const model = checkbox.getAttribute('data-model');
                            const price = checkbox.getAttribute('data-price');
                            const image = checkbox.getAttribute('data-image');
                            
                            selectedProducts.push({
                                id: productId,
                                room: room,
                                category: category,
                                brand: brand,
                                model: model,
                                price: price,
                                image: image
                            });
                        });
                        
                        if (selectedProducts.length === 0) {
                            alert("Please select at least one product before generating the final recommendation.");
                            return;
                        }
                        
                        // Store selected products in local storage
                        localStorage.setItem('selectedProducts', JSON.stringify(selectedProducts));
                        
                        // Calculate total price
                        const totalPrice = selectedProducts.reduce((sum, product) => sum + parseFloat(product.price), 0);
                        localStorage.setItem('totalPrice', totalPrice);
                        
                        // Hide recommendation page and show final page
                        document.querySelector('.container').style.display = 'none';
                        const finalPage = document.getElementById('final-recommendation-page');
                        finalPage.style.display = 'block';
                        
                        // Populate selected products
                        displayFinalRecommendation(selectedProducts, totalPrice);
                    });
                    
                    // Function to display final recommendation
                    function displayFinalRecommendation(products, totalPrice) {
                        const container = document.getElementById('selected-products-container');
                        container.innerHTML = '';
                        
                        // Group products by room
                        const roomProducts = {};
                        products.forEach(product => {
                            if (!roomProducts[product.room]) {
                                roomProducts[product.room] = [];
                            }
                            roomProducts[product.room].push(product);
                        });
                        
                        // Create room sections
                        for (const [room, roomItems] of Object.entries(roomProducts)) {
                            const roomTitle = room.replace('_', ' ').toUpperCase();
                            const roomSection = document.createElement('div');
                            roomSection.className = 'room-section';
                            roomSection.innerHTML = `<h2>${roomTitle}</h2>`;
                            
                            // Create product grid
                            const productGrid = document.createElement('div');
                            productGrid.className = 'products-grid';
                            
                            // Add products
                            roomItems.forEach(product => {
                                const productCard = document.createElement('div');
                                productCard.className = 'product-card';
                                
                                const categoryTitle = product.category.replace('_', ' ').toUpperCase();
                                const price = parseFloat(product.price).toLocaleString('en-IN', {
                                    style: 'currency',
                                    currency: 'INR',
                                    maximumFractionDigits: 2
                                });
                                
                                productCard.innerHTML = `
                                    <div class="product-image-container">
                                        <img class="product-image" src="${product.image || 'https://via.placeholder.com/300x300?text=No+Image+Available'}" alt="${product.brand} ${product.model}">
                                    </div>
                                    <div class="product-details">
                                        <span class="product-type">${categoryTitle}</span>
                                        <h3 class="product-title">${product.brand} ${product.model}</h3>
                                        <div class="price-container">
                                            <span class="current-price">${price}</span>
                                        </div>
                                    </div>
                                `;
                                
                                productGrid.appendChild(productCard);
                            });
                            
                            roomSection.appendChild(productGrid);
                            container.appendChild(roomSection);
                        }
                        
                        // Update budget information
                        const totalElement = document.getElementById('final-total-cost');
                        totalElement.textContent = totalPrice.toLocaleString('en-IN', {
                            style: 'currency',
                            currency: 'INR',
                            maximumFractionDigits: 2
                        });
                        
                        // Calculate budget utilization
                        const budget = parseFloat(totalElement.nextElementSibling.nextElementSibling.textContent.replace(/[^0-9.]/g, ''));
                        const utilization = (totalPrice / budget) * 100;
                        document.getElementById('final-budget-utilization').textContent = `${utilization.toFixed(1)}%`;
                        
                        // Update budget status
                        const budgetStatus = document.getElementById('final-budget-status');
                        if (utilization > 100) {
                            budgetStatus.className = 'budget-status warning';
                            budgetStatus.textContent = '⚠ The total cost exceeds your budget. Consider reviewing your selections.';
                        }
                    }
                };
                
                // Function to export to Excel
                const setupExportButton = () => {
                    console.log('Setting up export button');
                    const exportButton = checkElement('export-excel');
                    if (!exportButton) {
                        console.error('Export button not found in the DOM');
                        return;
                    }
                    
                    exportButton.addEventListener('click', function() {
                        console.log('Export button clicked');
                        const selectedProducts = JSON.parse(localStorage.getItem('selectedProducts') || '[]');
                        if (selectedProducts.length === 0) {
                            alert('Please select at least one product first');
                            return;
                        }
                        
                        // Prepare data for Excel
                        const data = [
                            ['Room', 'Category', 'Brand', 'Model', 'Price']
                        ];
                        
                        selectedProducts.forEach(product => {
                            data.push([
                                product.room.replace('_', ' ').toUpperCase(),
                                product.category.replace('_', ' ').toUpperCase(),
                                product.brand,
                                product.model,
                                product.price
                            ]);
                        });
                        
                        // Add total row
                        const totalPrice = selectedProducts.reduce((sum, product) => sum + parseFloat(product.price), 0);
                        data.push(['', '', '', 'TOTAL', totalPrice.toFixed(2)]);
                        
                        try {
                            console.log('Creating Excel file with data:', data);
                            
                            // Check if XLSX is available
                            if (typeof XLSX === 'undefined') {
                                console.error('XLSX library not loaded');
                                alert('Excel export library not loaded. Please check your internet connection.');
                                return;
                            }
                            
                            // Create worksheet
                            const ws = XLSX.utils.aoa_to_sheet(data);
                            
                            // Create workbook
                            const wb = XLSX.utils.book_new();
                            XLSX.utils.book_append_sheet(wb, ws, 'Recommendations');
                            
                            // Save file
                            XLSX.writeFile(wb, 'BetterHome_Recommendations.xlsx');
                            console.log('Excel file created successfully');
                        } catch (error) {
                            console.error('Error creating Excel file:', error);
                            alert('Failed to create Excel file. Error: ' + error.message);
                        }
                    });
                };
                
                // Setup the download button for final recommendation
                const setupFinalDownloadButton = () => {
                    console.log('Setting up final download button');
                    const downloadButton = checkElement('download-final-excel');
                    if (!downloadButton) {
                        console.error('Final download button not found in the DOM');
                        return;
                    }
                    
                    downloadButton.addEventListener('click', function() {
                        console.log('Download final button clicked');
                        const selectedProducts = [];
                        
                        // Collect all products from the selected-products-container
                        const roomSections = document.querySelectorAll('#selected-products-container .room-section');
                        roomSections.forEach(section => {
                            const roomName = section.querySelector('h2').textContent;
                            const productCards = section.querySelectorAll('.product-card');
                            
                            productCards.forEach(card => {
                                const category = card.querySelector('.product-type').textContent;
                                const title = card.querySelector('.product-title').textContent;
                                const price = card.querySelector('.current-price').textContent;
                                
                                // Split title into brand and model - assumes format "Brand Model"
                                const titleParts = title.split(' ');
                                const brand = titleParts[0];
                                const model = titleParts.slice(1).join(' ');
                                
                                selectedProducts.push({
                                    room: roomName,
                                    category: category,
                                    brand: brand,
                                    model: model,
                                    price: price.replace(/[^0-9.]/g, '') // Remove currency symbols
                                });
                            });
                        });
                        
                        // Prepare data for Excel
                        const data = [
                            ['Room', 'Category', 'Brand', 'Model', 'Price']
                        ];
                        
                        selectedProducts.forEach(product => {
                            data.push([
                                product.room,
                                product.category,
                                product.brand,
                                product.model,
                                product.price
                            ]);
                        });
                        
                        // Add total price
                        const totalElement = document.getElementById('final-total-cost');
                        const totalPrice = totalElement.textContent.replace(/[^0-9.]/g, '');
                        data.push(['', '', '', 'TOTAL', totalPrice]);
                        
                        // Add client information
                        data.push([]);
                        data.push(['Client Information']);
                        const clientInfoItems = document.querySelectorAll('.client-info .client-info-item');
                        clientInfoItems.forEach(item => {
                            const label = item.querySelector('.client-info-label').textContent;
                            const value = item.querySelector('.client-info-value').textContent;
                            data.push([label, value]);
                        });
                        
                        // Create worksheet
                        const ws = XLSX.utils.aoa_to_sheet(data);
                        
                        // Create workbook
                        const wb = XLSX.utils.book_new();
                        XLSX.utils.book_append_sheet(wb, ws, 'Final Recommendations');
                        
                        // Save file
                        XLSX.writeFile(wb, 'BetterHome_Final_Recommendations.xlsx');
                    });
                    
                    // Setup print button
                    console.log('Setting up print button');
                    const printButton = checkElement('print-final');
                    if (!printButton) {
                        console.error('Print button not found in the DOM');
                        return;
                    }
                    
                    printButton.addEventListener('click', function() {
                        console.log('Print button clicked');
                        window.print();
                    });
                };
                
                // Setup accordion for room sections
                // Use a dedicated function for the accordion that will be called when DOM is ready
                function setupAccordion() {
                    console.log('Setting up accordion functionality');
                    const accordionButtons = document.querySelectorAll('.accordion');
                    console.log('Found accordion buttons:', accordionButtons.length);
                    
                    accordionButtons.forEach((btn, idx) => {
                        // Mark the first one as active
                        if (idx === 0) btn.classList.add('active');
                        
                        btn.onclick = function(e) {
                            e.preventDefault();
                            e.stopPropagation();
                            const prevY = window.scrollY;
                            console.log('Accordion button clicked');
                            const panel = this.nextElementSibling;
                            console.log('Panel element:', panel);
                            const isOpen = panel.style.display === 'block';
                            console.log('Is panel open?', isOpen);
                            
                            // Close all panels and remove active class
                            document.querySelectorAll('.panel').forEach(p => p.style.display = 'none');
                            document.querySelectorAll('.accordion').forEach(b => b.classList.remove('active'));
                            
                            // If it wasn't open before, open it now
                            if (!isOpen) {
                                panel.style.display = 'block';
                                this.classList.add('active');
                                console.log('Panel opened');
                            }
                            // Keep viewport position and remove focus to avoid jumps
                            this.blur();
                            // Use double RAF to ensure layout is settled before restoring scroll
                            requestAnimationFrame(() => {
                                requestAnimationFrame(() => {
                                    window.scrollTo(0, prevY);
                                });
                            });
                        };
                    });
                }
                
                // Add setupAccordion to window load and also call it directly
                window.addEventListener('DOMContentLoaded', setupAccordion);
                window.addEventListener('load', setupAccordion);

                // Initialize all event handlers
                function initializeEventHandlers() {
                    console.log('Initializing all event handlers');
                    
                    // Add a small delay to ensure DOM is fully processed
                    setTimeout(() => {
                        try {
                            setupCategorySelections();
                            setupGenerateButton();
                            setupExportButton();
                            setupFinalDownloadButton();
                            console.log('All event handlers initialized successfully');
                        } catch (error) {
                            console.error('Error initializing event handlers:', error);
                        }
                    }, 500);
                }
                
                // Call initialization
                initializeEventHandlers();
            });
        </script>
    </head>
    <body>
        <div class="container">
    """
    
    # Add header section with explicit f-string
    header_section = f"""
            <header>
                {logo_html}
                <h1>Your Personalized Home Appliance Recommendations</h1>
                <p>Specially curated for {user_data['name']}</p>
            </header>
    """
            
    # Add client info section with explicit f-string
    # Determine budget display: in default mode, show calculated total instead of provided budget
    # We'll compute total_cost a bit later; set a placeholder that we'll replace
    client_info_section = f"""
            <div class="client-info">
                <div class="client-info-item">
                    <div class="client-info-label">Name</div>
                    <div class="client-info-value">{user_data['name']}</div>
                </div>
                
                <div class="client-info-item">
                    <div class="client-info-label">Mobile</div>
                    <div class="client-info-value">{user_data['mobile']}</div>
                </div>
                
                <div class="client-info-item">
                    <div class="client-info-label">Email</div>
                    <div class="client-info-value">{user_data['email']}</div>
                </div>
                
                <div class="client-info-item">
                    <div class="client-info-label">Address</div>
                    <div class="client-info-value">{user_data['address']}</div>
                </div>
                
                <div class="client-info-item">
                    <div class="client-info-label">Total Budget</div>
                    <div class="client-info-value" id="client-total-budget">₹{user_data['total_budget']:,.2f}</div>
                </div>
            </div>
    """
    html_content += client_info_section

    # Add budget summary
    total_cost = calculate_total_cost(final_list)
    # If in default mode, also update the client info budget to reflect total cost
    if default_mode:
        html_content = html_content.replace(
            f"id=\"client-total-budget\">₹{user_data['total_budget']:,.2f}",
            f"id=\"client-total-budget\">₹{total_cost:,.2f}"
        )
        # Ensure user_data has the updated value to avoid downstream math issues
        try:
            user_data['total_budget'] = float(total_cost)
        except Exception:
            user_data['total_budget'] = total_cost
    budget_utilization = (total_cost / user_data['total_budget']) * 100
    # Calculate total_savings for the default recommended set
    total_savings = 0
    for room, products in final_list.items():
        if not isinstance(products, dict):
            continue
        for product_type, options in products.items():
            if isinstance(options, dict):
                for nested_type, nested_options in options.items():
                    if not nested_options or not isinstance(nested_options, list):
                        continue
                    best_product = max(nested_options, key=lambda x: x.get('feature_match_score', 0), default=None)
                    if best_product:
                        #print(f"[DEBUG] Room: {room}, Type: {product_type}, Subtype: {nested_type}, Title: {best_product.get('title')}, Savings: {best_product.get('savings', 0)}, Retail: {best_product.get('retail_price')}, BH: {best_product.get('better_home_price')}")
                        total_savings += best_product.get('savings', 0)
            elif isinstance(options, list) and options:
                best_product = max(options, key=lambda x: x.get('feature_match_score', 0), default=None)
                if best_product:
                    #print(f"[DEBUG] Room: {room}, Type: {product_type}, Title: {best_product.get('title')}, Savings: {best_product.get('savings', 0)}, Retail: {best_product.get('retail_price')}, BH: {best_product.get('better_home_price')}")
                    total_savings += best_product.get('savings', 0)
    if not default_mode:
        budget_summary_section = f"""
            <div class="budget-summary">
                <h2>Budget Analysis</h2>
                <div class="budget-info">
                    <div class="budget-item">
                        <div class="budget-item-label">Total Recommended Products</div>
                        <div class="budget-item-value">₹{total_cost:,.2f}</div>
                    </div>
                    <div class="budget-item">
                        <div class="budget-item-label">Your Budget</div>
                        <div class="budget-item-value">₹{user_data['total_budget']:,.2f}</div>
                    </div>
                    <div class="budget-item">
                        <div class="budget-item-label">Budget Utilization</div>
                        <div class="budget-item-value">{budget_utilization:.1f}%</div>
                    </div>
                </div>
                <div class="budget-item">
                    <div class="budget-item-label">Total Savings</div>
                    <div class="budget-item-value">₹{int(total_savings):,}</div>
                </div>
        """
        html_content += budget_summary_section
        
        if budget_utilization <= 100:
            html_content += """
                <div class="budget-status good">
                    ✓ Your selected products fit comfortably within your budget!
                </div>
            """
        else:
            html_content += """
                <div class="budget-status warning">
                    ⚠ The total cost slightly exceeds your budget. Consider reviewing options if needed.
                </div>
            """
        
        html_content += """
            </div>
        """

    # Debug: Print the final list for kitchen before generating HTML
    # print("[DEBUG FINAL LIST] Kitchen hob tops:", final_list['kitchen']['hob_top'])

    # Process each room in specified order
    room_idx = 0
    
    # Define the room order
    room_order = ['hall', 'kitchen', 'dining', 'laundry', 'master_bedroom', 'bedroom_2', 'bedroom_3']
    
    # Process rooms in the specified order
    for room in room_order:
        # Skip if room doesn't exist in final_list
        if room not in final_list:
            continue
        appliances = final_list[room]
        if room == 'summary':
            continue
        # Check if the room has any products before creating the section
        has_products = False
        for appliance_type, products in appliances.items():
            if isinstance(products, list) and products:
                has_products = True
                break
            if isinstance(products, dict):
                for sub_products in products.values():
                    if isinstance(sub_products, list) and sub_products:
                        has_products = True
                        break
        if not has_products:
            continue
        room_title = room.replace('_', ' ').title()
        html_content += f"""
            <div class=\"room-section\">
                <button class='accordion' type='button'>{room_title}</button>
                <div class='panel' style='display: {'block' if room_idx == 0 else 'none'};'>
        """
        room_desc = None
        if not default_mode:
            room_desc = get_room_description(room, user_data)
        if room_desc:
            html_content += f'                    <div class="room-description">{room_desc}</div>\n'
        # Group by appliance type (and sub-type), ensuring room items appear before bathroom/nested items
        ordered_items = [
            (k, v) for k, v in appliances.items() if not isinstance(v, dict)
        ] + [
            (k, v) for k, v in appliances.items() if isinstance(v, dict)
        ]
        for appliance_type, products in ordered_items:
            # Handle nested appliance groups (e.g., bathroom)
            if isinstance(products, dict):
                for sub_appliance_type, sub_products in products.items():
                    if not isinstance(sub_products, list) or not sub_products:
                        continue
                    # Only allow duplication for certain types
                    allow_duplication = sub_appliance_type not in ['glass_partition', 'partition', 'shower_partition']
                    grouped_products = sub_products
                    # Sort by feature match score if available
                    grouped_products.sort(key=lambda x: -x.get('feature_match_score', 0))
                    # Section heading for sub-type
                    sub_type_title = sub_appliance_type.replace('_', ' ').title()
                    html_content += f'<h4 style="margin-top:20px;">{sub_type_title}</h4>'
                    html_content += '<div class="products-grid">'
                    # Identify the best product (highest feature_match_score)
                    best_product = grouped_products[0] if grouped_products else None
                    # Track which product keys have been checked for this room-category
                    checked_product_keys = set()
                    for idx, product in enumerate(grouped_products):
                        brand = product.get('brand', 'Unknown Brand')
                        model = product.get('model', product.get('title', 'Unknown Model'))
                        image_src = product.get('image_src', 'https://via.placeholder.com/300x300?text=No+Image+Available')
                        description = product.get('description', 'No description available')
                        better_home_price = float(product.get('bh_price', 0.0))
                        original_price = float(product.get('market_price_1', 0.0))
                        if original_price <= 0:
                            original_price = better_home_price * 1.25
                        discount = original_price - better_home_price
                        warranty = product.get('warranty', 'Standard warranty applies')
                        delivery_time = product.get('delivery_time', 'Contact store for details')
                        purchase_url = product.get('url', '#')
                        product_type_title = sub_appliance_type.replace('_', ' ').title()
                        # top_benefits_html = render_top_benefits(product.get('top_benefits', ''))
                        # if top_benefits_html:
                        #     html_content += top_benefits_html
                        reason_text = get_product_recommendation_reason(
                            product, 
                            sub_appliance_type, 
                            room, 
                            user_data['demographics'],
                            user_data['total_budget'],
                            {},
                            user_data
                        )
                        # For default mode, avoid an external reasons block before the card
                        if not default_mode:
                            html_content += f'<ul class="reasons-list"><li>{reason_text}</li></ul>'
                        # Prefer Top Benefits as the description if present (render as numbered list)
                        top_benefits = product.get('top_benefits', '')
                        benefits_html = render_benefits(top_benefits)
                        concise_description = product.get('concise_description') or product.get('description', 'No description available')
                        description_html = benefits_html if benefits_html else f'<div class="product-info-item">{concise_description}</div>'
                        # Debug print
                        if top_benefits:
                            print(f"DEBUG: Product {product.get('title', 'Unknown')} has top_benefits: {top_benefits[:100]}...")
                        # Badges
                        badges = ""
                        if product.get('is_bestseller', False):
                            badges += '<div class="bestseller-badge"><i class="fas fa-star"></i> BESTSELLER</div>'
                        # Checkbox and selection
                        product_id = f"{room}-{sub_appliance_type}-{idx}"
                        # Use a unique key for the product (brand+model)
                        product_key = f"{brand}::{model}"
                        # Only check the first occurrence of a product
                        if product_key not in checked_product_keys and product == best_product:
                            checked = 'checked'
                            checked_product_keys.add(product_key)
                        else:
                            checked = ''
                        selected_class = ' selected' if checked else ''
                        selection_html = '' if default_mode else f'''<div class="product-selection">
                                <input type="checkbox"
                                    id="{product_id}"
                                    class="product-checkbox"
                                    name="{room}-{sub_appliance_type}"
                                    data-product-id="{product_id}"
                                    data-room="{room}"
                                    data-category="{sub_appliance_type}"
                                    data-brand="{brand}"
                                    data-model="{model}"
                                    data-price="{better_home_price}"
                                    data-image="{image_src}"
                                    {checked}>
                                <label for="{product_id}"></label>
                                <span class="selection-label">Selected</span>
                            </div>'''
                        html_content += f'''<div class="product-card{selected_class}">
                            {selection_html}
                            <div class="product-image-container">
                                <a href="{purchase_url}" target="_blank" rel="noopener noreferrer">
                                    <img class="product-image" src="{image_src}" alt="{brand} {model}">
                                </a>
                                {badges}
                            </div>
                            <div class="product-details">
                                <span class="product-type">{product_type_title}</span>
                                <h3 class="product-title">{brand} {model}</h3>
                                <div class="price-container">
                                    <span class="current-price">₹{int(better_home_price):,}</span>
                                    <span class="retail-price">₹{int(original_price):,}</span>
                                    <span class="savings">Save ₹{int(discount):,}</span>
                                </div>
                                <div class="product-info-item"><span class="product-info-label">Warranty:</span> {warranty}</div>
                                <div class="product-info-item"><span class="product-info-label">Delivery:</span> {delivery_time}</div>
                                <ul class="reasons-list"><li>{reason_text}</li></ul>
                                {description_html}
                                
                            </div>
                        </div>'''
                    html_content += '</div>'
            elif isinstance(products, list) and products:
                allow_duplication = appliance_type not in ['glass_partition', 'partition', 'shower_partition']
                grouped_products = products
                grouped_products.sort(key=lambda x: -x.get('feature_match_score', 0))
                type_title = appliance_type.replace('_', ' ').title()
                html_content += f'<h4 style="margin-top:20px;">{type_title}</h4>'
                html_content += '<div class="products-grid">'
                best_product = grouped_products[0] if grouped_products else None
                checked_product_keys = set()
                for idx, product in enumerate(grouped_products):
                    brand = product.get('brand', 'Unknown Brand')
                    model = product.get('model', product.get('title', 'Unknown Model'))
                    image_src = product.get('image_src', 'https://via.placeholder.com/300x300?text=No+Image+Available')
                    description = product.get('description', 'No description available')
                    better_home_price = float(product.get('bh_price', 0.0))
                    original_price = float(product.get('market_price_1', 0.0))
                    if original_price <= 0:
                        original_price = better_home_price * 1.25
                    discount = original_price - better_home_price
                    warranty = product.get('warranty', 'Standard warranty applies')
                    delivery_time = product.get('delivery_time', 'Contact store for details')
                    purchase_url = product.get('url', '#')
                    product_type_title = appliance_type.replace('_', ' ').title()
                    # top_benefits_html = render_top_benefits(product.get('top_benefits', ''))
                    # if top_benefits_html:
                    #     html_content += top_benefits_html
                    reason_text = get_product_recommendation_reason(
                        product, 
                        appliance_type, 
                        room, 
                        user_data['demographics'],
                        user_data['total_budget'],
                        {},
                        user_data
                    )
                    # For default mode, avoid an external reasons block before the card
                    if not default_mode:
                        html_content += f'<ul class="reasons-list"><li>{reason_text}</li></ul>'
                    # Prefer Top Benefits as the description if present (render as numbered list)
                    top_benefits = product.get('top_benefits', '')
                    benefits_html = render_benefits(top_benefits)
                    concise_description = product.get('concise_description') or product.get('description', 'No description available')
                    description_html = benefits_html if benefits_html else f'<div class="product-info-item">{concise_description}</div>'
                    # Debug print
                    if top_benefits:
                        print(f"DEBUG: Product {product.get('title', 'Unknown')} has top_benefits: {top_benefits[:100]}...")
                    badges = ""
                    if product.get('is_bestseller', False):
                        badges += '<div class="bestseller-badge"><i class="fas fa-star"></i> BESTSELLER</div>'
                    
                    product_id = f"{room}-{appliance_type}-{idx}"
                    product_key = f"{brand}::{model}"
                    if product_key not in checked_product_keys and product == best_product:
                        checked = 'checked'
                        checked_product_keys.add(product_key)
                    else:
                        checked = ''
                    selected_class = ' selected' if checked else ''
                    features = product.get('features', {})
                    parsed_features = features.get('parsed_features', {}) if isinstance(features, dict) else {}
                    if isinstance(parsed_features, dict):
                        features_html = ''.join([f'<li><span class="product-info-label">{k}:</span> {v}</li>' for k, v in parsed_features.items()])
                    elif isinstance(parsed_features, list):
                        features_html = ''.join([f'<li>{f}</li>' for f in parsed_features])
                    else:
                        features_html = ''
                    selection_html = '' if default_mode else f'''<div class="product-selection">
                            <input type="checkbox"
                                id="{product_id}"
                                class="product-checkbox"
                                name="{room}-{appliance_type}"
                                data-product-id="{product_id}"
                                data-room="{room}"
                                data-category="{appliance_type}"
                                data-brand="{brand}"
                                data-model="{model}"
                                data-price="{better_home_price}"
                                data-image="{image_src}"
                                {checked}>
                            <label for="{product_id}"></label>
                            <span class="selection-label">Selected</span>
                        </div>'''
                    html_content += f'''<div class="product-card{selected_class}">
                        {selection_html}
                        <div class="product-image-container">
                            <a href="{purchase_url}" target="_blank" rel="noopener noreferrer">
                                <img class="product-image" src="{image_src}" alt="{brand} {model}">
                            </a>
                            {badges}
                        </div>
                        <div class="product-details">
                            <span class="product-type">{product_type_title}</span>
                            <h3 class="product-title">{brand} {model}</h3>
                            <div class="price-container">
                                <span class="current-price">₹{int(better_home_price):,}</span>
                                <span class="retail-price">₹{int(original_price):,}</span>
                                <span class="savings">Save ₹{int(discount):,}</span>
                            </div>
                            {description_html}
                        </div>
                    </div>'''
                html_content += '</div>'
        html_content += '</div></div>'
        room_idx += 1

    if default_mode:
        # Force single-column grid via CSS override and hide generate buttons
        html_content += """
                <style>
                    .products-grid { grid-template-columns: 1fr !important; }
                    @media (min-width: 768px) { .products-grid { grid-template-columns: 1fr !important; } }
                </style>
            """
    if default_mode:
        # Add floating Customize button linking back to landing page, with prefilled query params when available
        # Pass bedrooms/bathrooms guess based on num_bedrooms/num_bathrooms when present
        customize_query = f"name={quote_plus(str(user_data.get('name','')))}&mobile={quote_plus(str(user_data.get('mobile','')))}&email={quote_plus(str(user_data.get('email','')))}&address={quote_plus(str(user_data.get('address','')))}&bedrooms={quote_plus(str(user_data.get('num_bedrooms','')))}&bathrooms={quote_plus(str(user_data.get('num_bathrooms','')))}"
        html_content += f"""
                <div class="generate-container">
                    <div style="display:flex; gap:12px; flex-wrap:wrap; justify-content:center;">
                        <button id="download-pdf" class="generate-button" onclick="downloadRecommendationsPdf();">Download PDF</button>
                    </div>
                </div>
                <a href="/" class="customize-fab" id="customizeFab">Customize</a>
                <style>
                    .customize-fab {{
                        position: fixed;
                        right: 16px;
                        bottom: 16px;
                        background: #0d6efd;
                        color: #fff;
                        padding: 12px 16px;
                        border-radius: 999px;
                        text-decoration: none;
                        font-weight: 600;
                        box-shadow: 0 4px 12px rgba(13,110,253,0.3);
                        z-index: 1000;
                    }}
                    .customize-fab:hover {{ background: #0b5ed7; color: #fff; }}
                </style>
                <script>
                    (function() {{
                        var fab = document.getElementById('customizeFab');
                        if (fab) {{
                            var q = '{customize_query}';
                            // Only append if at least one value exists
                            if (q.replace(/(name=|mobile=|email=|address=)/g,'').replace(/&/g,'').trim() !== '') {{
                                fab.href = '/?' + q;
                            }}
                        }}
                    }})();
                </script>
        """
        html_content += f"""
                <footer>
                    <p>This product recommendation brochure was created on {current_date}</p>
                    <p> © {current_year} BetterHome.</p>
                </footer>
            </div>
            """
    else:
        html_content += f"""
                <div class="generate-container">
                    <h2>Select Your Preferred Products</h2>
                    <p>Please select one product from each category above that best suits your needs.</p>
                    <div style="display:flex; gap:12px; flex-wrap:wrap; justify-content:center;">
                        <button id="generate-final" class="generate-button" onclick="generateFinalRecommendation()">Generate Final Recommendations</button>
                        <button id="download-pdf" class="generate-button" onclick="downloadRecommendationsPdf();">Download PDF</button>
                    </div>
                </div>
        """

    # Always ensure accordion behavior is active (both modes)
    html_content += """
            <script>
            window.addEventListener('load', function() {
                var buttons = document.querySelectorAll('.accordion');
                for (var i = 0; i < buttons.length; i++) {
                    (function(btn){
                        btn.addEventListener('click', function(e){
                            e.preventDefault();
                            e.stopPropagation();
                            var prevY = window.scrollY;
                            var panel = this.nextElementSibling;
                            var isOpen = panel && panel.style.display === 'block';
                            var panels = document.querySelectorAll('.panel');
                            for (var j = 0; j < panels.length; j++) { panels[j].style.display = 'none'; }
                            var accs = document.querySelectorAll('.accordion');
                            for (var k = 0; k < accs.length; k++) { accs[k].classList.remove('active'); }
                            if (!isOpen && panel) { panel.style.display = 'block'; this.classList.add('active'); }
                            this.blur();
                            requestAnimationFrame(function(){ window.scrollTo(0, prevY); });
                        });
                    })(buttons[i]);
                }
            });
            </script>
    """

    # Add the generate final recommendation button and JavaScript
    if not default_mode:
        html_content += f"""
            <div class="generate-container">
                <h2>Select Your Preferred Products</h2>
                <p>Please select one product from each category above that best suits your needs.</p>
                <div style=\"display:flex; gap:12px; flex-wrap:wrap; justify-content:center;\">
                    <button id=\"generate-final\" class=\"generate-button\" onclick=\"generateFinalRecommendation()\">Generate Final Recommendations</button>
                    <button id=\"download-pdf\" class=\"generate-button\" onclick=\"downloadRecommendationsPdf();\">Download PDF</button>
                </div>
            </div>
            
            <footer>
                <p>This product recommendation brochure was created for {user_data['name']} on {current_date}</p>
                <p> © {current_year} BetterHome. All recommendations are personalized based on your specific requirements.</p>
            </footer>
        </div>
        
        <!-- Create the final recommendation page -->
        <div id=\"final-recommendation-page\" style=\"display: none;\">
         <div id=\"final-recommendation-content\"></div>
            <div class=\"container\">
                <header>
                    {logo_html}
                    <h1>Your Final Product Selections</h1>
                    <p>Specially curated for {user_data['name']}</p>
                </header>
                <div id=\"selected-products-container\"></div>
                <div class=\"budget-summary\">
                    <h2>Budget Summary</h2>
                    <p>Total Cost: <span id=\"total-cost\">{total_cost}</span></p>
                    <p>Total Savings: <span id=\"total-savings\">{total_savings}</span></p>
                    <p id=\"budget-utilization\"></p>
                </div>
                <div style=\"display: flex; justify-content: center; margin-top: 24px;\">
                    <button onclick=\"backToSelection()\" class=\"generate-button\">Back to Selection</button>
                </div>
            </div>
        </div>
        
        <script>
            function backToSelection() {{
                window.location.reload();

            }}

            function generateFinalRecommendation() {{
                const selectedProducts = [];
                let totalCost = 0;
                let totalRetailCost = 0;
                let totalSavings = 0;

                // Get all selected products (no room-category deduplication)
                document.querySelectorAll('.product-card.selected').forEach(card => {{
                    const room = card.getAttribute('data-room');
                    const category = card.getAttribute('data-category');
                    const infoItems = Array.from(card.querySelectorAll('.product-info-item'));
                    const findByLabel = (label) => {{
                        const el = infoItems.find(it => (it.querySelector('.product-info-label')?.textContent || '').trim().startsWith(label));
                        return el ? el.textContent.replace(label, '').trim() : '';
                    }};
                    const warrantyText = findByLabel('Warranty:');
                    const deliveryText = findByLabel('Delivery:');
                    const product = {{
                        room: room,
                        category: category,
                        name: card.querySelector('.product-title').textContent,
                        price: parseFloat(card.querySelector('.current-price').textContent.replace('₹', '').replace(/,/g, '')),
                        retailPrice: parseFloat(card.querySelector('.retail-price').textContent.replace('₹', '').replace(/,/g, '')),
                        savings: parseFloat(card.querySelector('.savings').textContent.replace('Save ₹', '').replace(/,/g, '')),
                        image: card.querySelector('.product-image').src,
                        warranty: warrantyText,
                        delivery: deliveryText,
                        reason: card.querySelector('.reasons-list li').textContent,
                        purchaseUrl: card.querySelector('.buy-button').href
                    }};
                    selectedProducts.push(product);
                    totalCost += product.price;
                    totalRetailCost += product.retailPrice;
                    totalSavings += product.savings;
                }});

                if (selectedProducts.length === 0) {{
                    alert('Please select at least one product before generating final recommendations.');
                    return;
                }}

                // Generate HTML for selected products
                let finalHtml = '';
                selectedProducts.forEach(product => {{
                    finalHtml += `
                        <div class="final-product-card">
                            <div class="final-product-image">
                                <img src="${{product.image}}" alt="${{product.name}}">
                            </div>
                            <div class="final-product-details">
                                <h3>${{product.name}}</h3>
                                <div class="final-product-info">
                                    <p><strong>Room:</strong> ${{product.room}}</p>
                                    <p><strong>Category:</strong> ${{product.category}}</p>
                                    <p><strong>Price:</strong> ₹${{product.price.toLocaleString('en-IN')}}</p>
                                    <p><strong>Warranty:</strong> ${{product.warranty}}</p>
                                    <p><strong>Delivery:</strong> ${{product.delivery}}</p>
                                    <p><strong>Why this product:</strong> ${{product.reason}}</p>
                                </div>
                                <a href="${{product.purchaseUrl}}" class="buy-button" target="_blank">Buy Now</a>
                            </div>
                        </div>
                    `;
                }});

                // Update the final recommendation page
                const selectedProductsContainer = document.getElementById('selected-products-container');
                const totalCostElement = document.getElementById('total-cost');
                const totalSavingsElement = document.getElementById('total-savings');
                const budgetUtilizationElement = document.getElementById('budget-utilization');
                const selectionPage = document.getElementById('product-selection-page');
                const finalPage = document.getElementById('final-recommendation-page');

                if (selectedProductsContainer) selectedProductsContainer.innerHTML = finalHtml;
                if (totalCostElement) totalCostElement.textContent = `₹${{totalCost.toLocaleString('en-IN')}}`;
                if (totalSavingsElement) totalSavingsElement.textContent = `₹${{totalSavings.toLocaleString('en-IN')}}`;
                if (budgetUtilizationElement) {{
                    let budgetText = '';
                    const userBudget = {user_data['total_budget']};
                    if (totalCost > userBudget) {{
                        budgetText = `Budget exceeded by ₹${{(totalCost - userBudget).toLocaleString('en-IN')}}`;
                    }} else {{
                        budgetText = `Budget utilized: ₹${{totalCost.toLocaleString('en-IN')}} of ₹{user_data['total_budget']:,.2f}`;
                    }}
                    budgetUtilizationElement.textContent = budgetText;
                }}

                // Show final recommendation page
                if (selectionPage && finalPage) {{
                    selectionPage.style.display = 'none';
                    finalPage.style.display = 'block';
                }}
            }}

            function downloadRecommendationsPdf() {{
                // Expand all panels
                const panels = document.querySelectorAll('.panel');
                const accordions = document.querySelectorAll('.accordion');
                
                for (let i = 0; i < panels.length; i++) {{ 
                    panels[i].style.display = 'block'; 
                }}
                for (let j = 0; j < accordions.length; j++) {{ 
                    accordions[j].classList.add('active'); 
                }}
                
                // Wait for layout then print
                setTimeout(function() {{
                    window.print();
                }}, 500);
            }}
            
            // Make function globally available
            window.downloadRecommendationsPdf = downloadRecommendationsPdf;

            // Set up accordion functionality
            (function() {{
                const accordionButtons = document.querySelectorAll('.accordion');
                console.log('DIRECT: Found accordion buttons:', accordionButtons.length);
                
                for (let i = 0; i < accordionButtons.length; i++) {{
                    const btn = accordionButtons[i];
                    
                    // Set active class on first button
                    if (i === 0) {{
                        btn.classList.add('active');
                    }}
                    
                    // Direct event binding
                    btn.addEventListener('click', function(e) {{
                        e.preventDefault();
                        e.stopPropagation();
                        
                        const panel = this.nextElementSibling;
                        const isOpen = panel.style.display === 'block';
                        
                        // Close all panels and remove active class
                        const allPanels = document.querySelectorAll('.panel');
                        for (let j = 0; j < allPanels.length; j++) {{
                            allPanels[j].style.display = 'none';
                        }}
                        
                        const allButtons = document.querySelectorAll('.accordion');
                        for (let j = 0; j < allButtons.length; j++) {{
                            allButtons[j].classList.remove('active');
                        }}
                        
                        // If it wasn't open before, open it now
                        if (!isOpen) {{
                            panel.style.display = 'block';
                            this.classList.add('active');
                            console.log('DIRECT: Panel opened');
                        }}
                        
                        return false;
                    }});
                    console.log('DIRECT: Setup click handler for:', btn.textContent);
                }}
                
                console.log('DIRECT: Setting up accordion - complete');
            }})();
        </script>
        """
    html_content += """
        <script>
        function toggleFeatures(btn) {{
            var content = btn.nextElementSibling;
            if (content.style.display === "none" || content.style.display === "") {{
                content.style.display = "block";
                btn.querySelector('.arrow').innerHTML = "&#9660;"; // Down arrow
                btn.classList.add('open');
            }} else {{
                content.style.display = "none";
                btn.querySelector('.arrow').innerHTML = "&#9654;"; // Right arrow
                btn.classList.remove('open');
            }}
        }}
        
        function downloadRecommendationsPdf() {{
            // Expand all panels
            const panels = document.querySelectorAll('.panel');
            const accordions = document.querySelectorAll('.accordion');
            
            for (let i = 0; i < panels.length; i++) {{ 
                panels[i].style.display = 'block'; 
            }}
            for (let j = 0; j < accordions.length; j++) {{ 
                accordions[j].classList.add('active'); 
            }}
            
            // Wait for layout then print
            setTimeout(function() {{
                window.print();
            }}, 500);
        }}
        
        // Make function globally available
        window.downloadRecommendationsPdf = downloadRecommendationsPdf;
        </script>
      
    </body>
    </html>
    """

    # Write the HTML content to a file
    with open(html_filename, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    # Debug: Check if any unprocessed variables remain in the HTML content
    if '{logo_html}' in html_content or "{user_data['name']}" in html_content:
        print("WARNING: Some template variables were not properly substituted!")
        print("Sample of unprocessed content:")
        sample_start = html_content.find('{logo_html}') if '{logo_html}' in html_content else html_content.find("{user_data")
        if sample_start >= 0:
            sample_end = min(sample_start + 200, len(html_content))
            print(html_content[sample_start:sample_end])
    
    print(f"Generated professional HTML brochure: {html_filename}")
    
    # After generating the HTML file, upload it to S3
    files = {
        'html': html_filename
    }


def get_product_recommendation_reason(
    product: Dict[str, Any],
    appliance_type: str,
    room: str,
    demographics: Dict[str, int],
    total_budget: float,
    required_features: Dict[str, str] = None,
    user_data: Dict[str, Any] = None
) -> str:
    """
    Generate a personalized recommendation reason for a product, highlighting matching features.
    If the product is from best-sellers.csv, use the 'top_benefits' field.
    """
    # Prefer Top Benefits if present (for best-sellers)
    if 'top_benefits' in product and product['top_benefits']:
        return product['top_benefits']
    # Fallback: use description or features
    if 'description' in product and product['description']:
        return product['description']
    # Fallback: use features
    if 'features' in product and isinstance(product['features'], dict):
        features = product['features']
        if 'parsed_features' in features and isinstance(features['parsed_features'], dict):
            return "; ".join([f"{k}: {v}" for k, v in features['parsed_features'].items()])
    return "Recommended based on your requirements."


def determine_ac_tonnage(square_feet: float, room_type: str = None) -> float:
    """
    Estimate the required AC tonnage based on room size.
    """
    if square_feet < 120:
        return 1.0
    elif square_feet < 180:
        return 1.2
    elif square_feet < 250:
        return 1.5
    else:
        return 2.0


def process_features(features_str: str) -> list:
    """
    Parse a features string into a list of features.
    """
    if not features_str:
        return []
    return [f.strip() for f in features_str.split(',') if f.strip()]


def parse_product_feature(feature_str: str) -> dict:
    """
    Parse a product feature string into a dictionary.
    """
    if not feature_str:
        return {}
    features = {}
    for item in feature_str.split(','):
        if ':' in item:
            k, v = item.split(':', 1)
            features[k.strip()] = v.strip()
        else:
            features[item.strip()] = True
    return features


def enrich_best_seller_product(best_seller, catalog_products):
    sku = best_seller.get('sku', '').strip()
    match = None
    if sku:
        match = next((p for p in catalog_products if p.get('sku', '').strip() == sku), None)
    # Use Top Benefits as description if present
    if best_seller.get('top_benefits'):
        best_seller['description'] = best_seller['top_benefits']
    elif match:
        # Use only concise_description from catalog, never description
        best_seller['description'] = match.get('concise_description', '')
    # Prefer image_src from CSV, but fallback to catalog if missing
    if not best_seller.get('image_src') and match and 'image_src' in match:
        best_seller['image_src'] = match['image_src']
    # Always set url from catalog if available
    if match and 'url' in match:
        best_seller['url'] = match['url']
    # Set concise_description if not present
    if 'concise_description' not in best_seller or not best_seller.get('concise_description'):
        best_seller['concise_description'] = match.get('concise_description', '') if match else ''
    # Set title if not present
    if 'title' not in best_seller or not best_seller.get('title'):
        if match and 'title' in match:
            best_seller['title'] = match['title']
    return best_seller



def get_ac_recommendations(products, budget, top_n=3):
    """
    Prioritize ACs by Priority and Standard/Premium. If budget >= 400000, prefer Premium, else Standard.
    If not enough in preferred type, fill with fallback type.
    """
    if budget < 400000:
        preferred_type = 'Standard'
        fallback_type = 'Premium'
    else:
        preferred_type = 'Premium'
        fallback_type = 'Standard'
    # Filter for ACs
    acs = [p for p in products if p['category'].strip().lower() == 'air conditioner']
    # Preferred type, sorted by priority
    preferred = [p for p in acs if p['standard_premium'].strip().lower() == preferred_type.lower()]
    preferred.sort(key=lambda x: x['priority'])
    # Fallback type, sorted by priority
    fallback = [p for p in acs if p['standard_premium'].strip().lower() == fallback_type.lower()]
    fallback.sort(key=lambda x: x['priority'])
    # Combine, but only enough to fill top_n
    result = preferred[:top_n]
    if len(result) < top_n:
        result += fallback[:top_n - len(result)]
    return result[:top_n]


def get_appliance_recommendations(products, category, budget, top_n=3):
    """
    Prioritize appliances by Priority and Standard/Premium. If budget >= 400000, prefer Premium, else Standard.
    If not enough in preferred type, fill with fallback type. Remove duplicates by SKU or brand+model.
    """
    if budget < 400000:
        preferred_type = 'Standard'
        fallback_type = 'Premium'
    else:
        preferred_type = 'Premium'
        fallback_type = 'Standard'
    # Filter for category
    items = [p for p in products if p['category'].strip().lower() == category.strip().lower()]
    # Preferred type, sorted by priority
    preferred = [p for p in items if p['standard_premium'].strip().lower() == preferred_type.lower()]
    preferred.sort(key=lambda x: x['priority'])
    # Fallback type, sorted by priority
    fallback = [p for p in items if p['standard_premium'].strip().lower() == fallback_type.lower()]
    fallback.sort(key=lambda x: x['priority'])
    # Combine, but only enough to fill top_n
    result = preferred[:top_n]
    if len(result) < top_n:
        result += fallback[:top_n - len(result)]
    # Remove duplicates by SKU or brand+model
    seen = set()
    unique = []
    for p in result:
        key = p.get('sku') or (p.get('brand', '').lower(), p.get('title', '').lower())
        if key and key not in seen:
            unique.append(p)
            seen.add(key)
    return unique[:top_n]

# New function to generate default recommendations for 2BHK and 3BHK

def generate_default_recommendations(
    csv_path: str,
    catalog_path: str,
    bhk_choice: str | None = None,  # '2BHK' or '3BHK'
    name: str | None = None,
    address: str | None = None,
    mobile: str | None = None,
    email: str | None = None,
    user_budget: float | None = None,
):
    # Load budget config
    cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'budget_config.yaml')
    cfg = {}
    try:
        with open(cfg_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f) or {}
        print(f"[defaults] Loaded budget config from {cfg_path}: {cfg}")
    except Exception as e:
        print(f"[defaults] Failed to load budget config {cfg_path}: {e}")
        cfg = {}

    # Read the CSV and filter curated rows using new semantics
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    df.columns = [col.strip() for col in df.columns]
    dl_col = df.get('Default List', '').astype(str)
    # Curated rows are those explicitly tagged for any list (Standard/Premium) or legacy 'Y'
    mask_any_list = (
        dl_col.str.upper().eq('Y') |
        dl_col.str.contains('Standard List', case=False, na=False) |
        dl_col.str.contains('Premium List', case=False, na=False)
    )
    default_items = df[mask_any_list].copy()
    print('[defaults] Filtered curated rows (any list):', len(default_items))
    products = []

    def parse_price(val):
        val = str(val).replace(',', '').replace('"', '').strip()
        try:
            return float(val) if val else 0.0
        except Exception:
            return 0.0
    def parse_int(val):
        try:
            return int(str(val).strip()) if str(val).strip() else 0
        except Exception:
            return 0
    for _, row in default_items.iterrows():
        product = {
            'sku': row.get('SKU Code', '').strip(),
            'category': row.get('Category', '').strip(),
            'bh_price': parse_price(row.get('BH Price', '')),
            'standard_premium': row.get('Standard/Premium', '').strip(),
            'brand': row.get('Brand', '').strip(),
            'title': row.get('Title', '').strip(),
            'priority': parse_int(row.get('Priority', '')),
            'image_src': row.get('Product Image URL', '').strip(),
            'url': row.get('Product URL', '').strip() if 'Product URL' in row else '',
            'top_benefits': row.get('Top Benefits', '').strip(),
            'default_list_raw': row.get('Default List', '').strip(),
        }
        # Parse Default List membership flags
        dl_raw_upper = product['default_list_raw'].upper()
        dl_raw_lower = product['default_list_raw'].lower()
        in_standard_list = ('standard list' in dl_raw_lower) or (dl_raw_upper == 'Y')
        in_premium_list = ('premium list' in dl_raw_lower) or (dl_raw_upper == 'Y')
        product['in_standard_list'] = in_standard_list
        product['in_premium_list'] = in_premium_list
        products.append(product)
    # Load catalog for enrichment
    catalog_products = load_product_catalog_json(catalog_path)
    enriched_products = [enrich_best_seller_product(p, catalog_products) for p in products]
    # Room mapping (customize as needed)
    # Top-level categories mapped to one or more rooms
    category_to_rooms = {
        'Air Conditioner': ['hall', 'master_bedroom', 'bedroom_2', 'bedroom_3'],
        'Ceiling Fan': ['hall', 'dining','master_bedroom', 'bedroom_2', 'bedroom_3'],
        'Chimney': ['kitchen'],
        'Cooktop': ['kitchen'],
        'Hob': ['kitchen'],
        'Dishwasher': ['kitchen'],
        'Instant Water Heater': ['master_bedroom', 'bedroom_2', 'bedroom_3'],
        'Storage Water Heater': ['master_bedroom', 'bedroom_2', 'bedroom_3'],
        'LED Mirror': ['master_bedroom', 'bedroom_2', 'bedroom_3'],
        'Water Purifier': ['kitchen'],
        'Refrigerator': ['kitchen'],
        'Exhaust Fan': ['kitchen','master_bedroom', 'bedroom_2', 'bedroom_3'],
        'Washing Machine': ['laundry'],
    }
    # Categories that should be shown inside the bathroom section of bedrooms
    bathroom_nested_categories = ['Storage Water Heater', 'LED Mirror']

    bhk_list = [bhk_choice] if bhk_choice in ('2BHK', '3BHK') else ['2BHK', '3BHK']
    for bhk in bhk_list:
        # Decide Standard vs Premium using config thresholds
        threshold = None
        try:
            tier_cfg = cfg.get('home_appliance', {})
            # Use 'standard' threshold for decision as per instruction: if budget >= standard threshold -> premium
            if bhk == '2BHK':
                threshold = float(tier_cfg.get('standard', {}).get('2BHK')) if tier_cfg else None
            else:
                threshold = float(tier_cfg.get('standard', {}).get('3BHK')) if tier_cfg else None
        except Exception:
            threshold = None
        selected_tier = None
        if user_budget is not None and threshold is not None:
            selected_tier = 'Premium' if user_budget >= threshold else 'Standard'
        print(f"[defaults] Tier decision → BHK={bhk}, user_budget={user_budget}, threshold={threshold}, selected_tier={selected_tier}")
        # Fall back to CSV Standard/Premium mix if not determined

        user_data = {
            'name': (name or f'Default {bhk} User'),
            'mobile': (mobile or ''),
            'email': (email or ''),
            'address': (address or ''),
            'total_budget': float(user_budget) if user_budget is not None else (600000.0 if bhk == '2BHK' else 900000.0),
            'num_bedrooms': 2 if bhk == '2BHK' else 3,
            'num_bathrooms': 2 if bhk == '2BHK' else 3,
            'demographics': {'adults': 2, 'elders': 0, 'kids': 0},
        }
        # Provide minimal room data expected by get_room_description and HTML
        user_data.update({
            'hall': {'size_sqft': 200.0, 'fans': 1, 'ac': True, 'color_theme': 'White', 'is_for_kids': False},
            'kitchen': {'size_sqft': 100.0, 'chimney_width': '60 cm', 'num_burners': 3, 'small_fan': True, 'color_theme': 'Grey', 'is_for_kids': False},
            'dining': {'size_sqft': 120.0, 'fans': 1, 'ac': False, 'color_theme': 'Grey', 'is_for_kids': False},
            'laundry': {'size_sqft': 50.0, 'washing_machine_type': 'Front-Load', 'dryer_type': 'No', 'is_for_kids': False},
            'master_bedroom': {
                'size_sqft': 140.0, 'ac': True, 'color_theme': 'White', 'is_for_kids': False,
                'bathroom': {'water_heater_type': 'Shower', 'exhaust_fan_size': '150mm', 'water_heater_ceiling': 'No', 'led_mirror': True, 'glass_partition': False}
            },
            'bedroom_2': {
                'size_sqft': 120.0, 'ac': True, 'color_theme': 'White', 'is_for_kids': False,
                'bathroom': {'water_heater_type': 'Shower', 'exhaust_fan_size': '150mm', 'water_heater_ceiling': 'No', 'led_mirror': True, 'glass_partition': False}
            },
            'bedroom_3': {
                'size_sqft': 120.0, 'ac': True, 'color_theme': 'White', 'is_for_kids': False,
                'bathroom': {'water_heater_type': 'Shower', 'exhaust_fan_size': '150mm', 'water_heater_ceiling': 'No', 'led_mirror': True, 'glass_partition': False}
            },
        })

        # Initialize recommendations with all expected rooms
        recommendations = {
            'hall': {}, 'kitchen': {}, 'dining': {}, 'laundry': {},
            'master_bedroom': {'bathroom': {}},
            'bedroom_2': {'bathroom': {}},
            'bedroom_3': {'bathroom': {}},
        }

        # If tier is chosen, apply Default List semantics with per-category handling
        if selected_tier:
            # Group by category for fine-grained fallback behavior
            by_category: Dict[str, List[Dict[str, Any]]] = {}
            for p in enriched_products:
                cat = p.get('category', '').strip() or 'Uncategorized'
                by_category.setdefault(cat, []).append(p)

            tiered_products: List[Dict[str, Any]] = []
            if selected_tier.lower() == 'standard':
                # Include items tagged for Standard List (includes shared Standard+Premium)
                for cat, items in by_category.items():
                    std_items = [x for x in items if x.get('in_standard_list')]
                    tiered_products.extend(std_items)
            else:
                # Premium: Prefer Premium-only items; if none exist for a category, include shared Standard+Premium items
                for cat, items in by_category.items():
                    premium_only = [x for x in items if x.get('in_premium_list') and not x.get('in_standard_list')]
                    if premium_only:
                        tiered_products.extend(premium_only)
                        continue
                    shared_premium = [x for x in items if x.get('in_premium_list')]
                    tiered_products.extend(shared_premium)
            print(f"[defaults] Tier={selected_tier}, candidates after Default List filtering: {len(tiered_products)}")
            # Legacy fallback: if nothing made it through, fall back to Standard tier by Standard/Premium column
            if not tiered_products:
                print(f"[defaults] No products after Default List filtering for tier {selected_tier}. Falling back to Standard tier via Standard/Premium column.")
            tiered_products = [p for p in enriched_products if p.get('standard_premium', '').strip().lower() == 'standard']
        else:
            # No selected tier (no threshold decision) → keep all curated products
            tiered_products = list(enriched_products)
            print(f"[defaults] Tier=None, using all curated products: {len(tiered_products)}")

        for p in tiered_products:
            category_name = p.get('category', '').strip()
            if not category_name:
                continue
            # Bathroom-nested categories go under each bedroom's bathroom
            if category_name in bathroom_nested_categories:
                for room_key in ['master_bedroom', 'bedroom_2', 'bedroom_3']:
                    bathroom = recommendations[room_key].setdefault('bathroom', {})
                    bathroom.setdefault(category_name, []).append(p)
                continue
            # Top-level categories mapped to one or more rooms
            target_rooms = category_to_rooms.get(category_name)
            if target_rooms:
                for room_key in target_rooms:
                    recommendations[room_key].setdefault(category_name, []).append(p)
            else:
                # If no mapping, place under hall by default
                recommendations['hall'].setdefault(category_name, []).append(p)

        # Enforce per-room instance counts regardless of how many items exist
        def get_required_count(room_key: str, category_name: str) -> int:
            if category_name == 'Air Conditioner':
                return 1
            if category_name == 'Ceiling Fan':
                return 2 if room_key == 'hall' else 1
            if category_name == 'Chimney':
                return 1 if room_key == 'kitchen' else 0
            if category_name == 'Washing Machine':
                return 1 if room_key == 'laundry' else 0
            if category_name == 'Dishwasher':
                return 1 if room_key == 'kitchen' else 0
            if category_name == 'Exhaust Fan':
                return 1
            return -1  # -1 means leave as-is

        for room_key, room_cats in list(recommendations.items()):
            if not isinstance(room_cats, dict):
                continue
            for category_name, items in list(room_cats.items()):
                required = get_required_count(room_key, category_name)
                if required == -1:
                    continue
                if required == 0:
                    # Remove categories that should not appear in this room
                    del room_cats[category_name]
                    continue
                if not isinstance(items, list) or len(items) == 0:
                    # Nothing to choose from
                    continue
                # Pick the single highest-priority item (priority ascending)
                def priority_of(prod):
                    try:
                        return int(prod.get('priority', 999999) or 999999)
                    except Exception:
                        return 999999
                items.sort(key=priority_of)
                best = items[0]
                # Duplicate the best item to match required count when needed (e.g., 2 fans for hall)
                room_cats[category_name] = [best for _ in range(required)]

        # Enforce counts for bathroom-nested categories (1 per category in each bathroom)
        def best_by_priority(items_list):
            def priority_of(prod):
                try:
                    return int(prod.get('priority', 999999) or 999999)
                except Exception:
                    return 999999
            items_list.sort(key=priority_of)
            return items_list[0]

        for room_key in ['master_bedroom', 'bedroom_2', 'bedroom_3']:
            bathroom = recommendations.get(room_key, {}).get('bathroom')
            if not isinstance(bathroom, dict):
                continue
            for cat in list(bathroom.keys()):
                items = bathroom.get(cat)
                if not isinstance(items, list) or not items:
                    continue
                bathroom[cat] = [best_by_priority(items)]  # exactly 1 per bathroom category

        # Determine output filename based on selected tier for clarity
        tier_for_filename = (selected_tier or 'Standard').lower()
        output_filename = f'{tier_for_filename}_default_recommendations_{bhk}.html'
        generate_html_file(user_data, recommendations, output_filename, default_mode=True)
        print(f"Generated {selected_tier or 'Standard'} default recommendations for {bhk}: {output_filename}")

# Update main block to handle --generate-defaults
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('user_xlsx', nargs='?', default=None)
    parser.add_argument('best_sellers_csv', nargs='?', default=None)
    parser.add_argument('--generate-defaults', action='store_true')
    parser.add_argument('--catalog', default='product_catalog.json')
    args = parser.parse_args()
    if args.generate_defaults:
        generate_default_recommendations(args.best_sellers_csv, args.catalog)
    elif args.user_xlsx and args.best_sellers_csv:
        main(args.user_xlsx, args.best_sellers_csv)
    else:
        print("Usage: python generate-recommendations.py <user_input.xlsx> <best_sellers_csv> [--generate-defaults] [--catalog <catalog_path>]")

