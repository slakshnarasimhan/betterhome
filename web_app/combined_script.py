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
from urllib.parse import urlparse
import re # Add import for regex
from reportlab.platypus import Image
from reportlab.lib.colors import HexColor
from reportlab.platypus.flowables import HRFlowable
from datetime import datetime
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
import pprint

# Function to format currency
def format_currency(amount: float) -> str:
    """Format amount in Indian Rupees"""
    return f"₹{amount:,.2f}"

# Function to load product catalog
def load_product_catalog() -> Dict[str, Any]:
    """Load product catalog from JSON file"""
    try:
        with open('product_catalog.json', 'r') as f:
            catalog = json.load(f)
            return catalog
    except FileNotFoundError:
        print("Product catalog file not found")
        return {}
    except Exception as e:
        print(f"[DEBUG] Error loading product_catalog.json: {e}")
        return {}

# Function to load configuration
def load_config() -> Dict[str, Any]:
    """Load configuration from YAML file"""
    try:
        with open('config.yaml', 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print("Configuration file not found")
        return {}

# Function to analyze user requirements
def analyze_user_requirements(excel_file: str):
    try:
        # Read the Excel file
        df = pd.read_excel(excel_file)
        
        # Clean up column names by removing newlines and extra spaces
        df.columns = [col.split('\n')[0].strip() for col in df.columns]
        row = df.iloc[0]
        
        # Convert DataFrame to dictionary
        user_data = {
            'name': df.iloc[0]['Name'],
            'mobile': df.iloc[0]['Mobile Number (Preferably on WhatsApp)'],
            'email': df.iloc[0]['E-mail'],
            'address': df.iloc[0]['Apartment Address'].replace('\n', ' '),
            'total_budget': float(df.iloc[0]['What is your overall budget for home appliances?']),
            'num_bedrooms': int(df.iloc[0]['Number of bedrooms']),
            'num_bathrooms': int(df.iloc[0]['Number of bathrooms']),
            'demographics': {
                'adults': int(df.iloc[0]['Adults (between the age 18 to 50)']),
                'elders': int(df.iloc[0]['Elders (above the age 60)']),
                'kids': int(df.iloc[0]['Kids (below the age 18)'])
            }
        }
        
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
                'size_sqft': float(df.iloc[0].get('Hall: What is the square feet ?', 150.0))  # Updated column name
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
                'size_sqft': float(df.iloc[0].get('Kitchen: Size (square feet)', 100.0))  # Default to 100 sq ft if not specified
            },
            'master_bedroom': {
                'ac': df.iloc[0]['Master: Air Conditioner (AC)?'] == 'Yes',
                'bathroom': {
                    'water_heater_type': df.iloc[0]['Master: How do you bath with the hot & cold water?'],
                    'exhaust_fan_size': df.iloc[0]['Master: Exhaust fan size?'],
                    'water_heater_ceiling': df.iloc[0]['Master: Is the water heater going to be inside the false ceiling in the bathroom?'],
                    'led_mirror': df.iloc[0]['Master: Would you like to have a LED Mirror?'] == 'Yes'  # Add LED mirror preference
                },
                'color_theme': df.iloc[0]['Master: What is the colour theme?'],
                'size_sqft': float(df.iloc[0].get('Master: What is the area of the bedroom in square feet?', 140.0))  # Updated column name
            },
            'bedroom_2': {
                'ac': df.iloc[0]['Bedroom 2: Air Conditioner (AC)?'] == 'Yes',
                'bathroom': {
                    'water_heater_type': df.iloc[0]['Bedroom 2: How do you bath with the hot & cold water?'],
                    'exhaust_fan_size': df.iloc[0]['Bedroom 2: Exhaust fan size?'],
                    'water_heater_ceiling': df.iloc[0]['Bedroom 2: Is the water heater going to be inside the false ceiling in the bathroom?'],
                    'led_mirror': df.iloc[0]['Bedroom 2: Would you like to have a LED Mirror?'] == 'Yes'  # Add LED mirror preference
                },
                'color_theme': df.iloc[0]['Bedroom 2: What is the colour theme?'],
                'size_sqft': float(df.iloc[0].get('Bedroom 2: What is the area of the bedroom in square feet?', 120.0))  # Updated column name
            },
            'bedroom_3': {
                'ac': df.iloc[0]['Bedroom 3: Air Conditioner (AC)?'] == 'Yes',
                'bathroom': {
                    'water_heater_type': df.iloc[0]['Bedroom 3: How do you bath with the hot & cold water?'],
                    'exhaust_fan_size': df.iloc[0]['Bedroom 3: Exhaust fan size?'],
                    'water_heater_ceiling': df.iloc[0]['Bedroom 3: Is the water heater going to be inside the false ceiling in the bathroom?'],
                    'led_mirror': df.iloc[0]['Bedroom 3: Would you like to have a LED Mirror?'] == 'Yes'  # Add LED mirror preference
                },
                'color_theme': df.iloc[0]['Bedroom 3: What is the colour theme?'],
                'size_sqft': float(df.iloc[0].get('Bedroom 3: What is the area of the bedroom in square feet?', 120.0))  # Updated column name
            },
            'laundry': {
                'washing_machine_type': df.iloc[0]['Laundry: Washing Machine?'],
                'dryer_type': df.iloc[0]['Laundry: Dryer?'],
                'color_theme': None,  # No color theme specified for laundry
                'size_sqft': float(df.iloc[0].get('Laundry: Size (square feet)', 50.0))  # Default to 50 sq ft if not specified
            },
            'dining': {
                'fan_size': df.iloc[0].get('Dining: Fan', None),
                'fans': int(df.iloc[0].get('Dining: Fan(s)?', 0)),
                'ac': df.iloc[0].get('Dining: Air Conditioner (AC)?', 'No') == 'Yes',
                'color_theme': df.iloc[0].get('Dining: Colour theme?', None),
                'size_sqft': float(df.iloc[0].get('Dining: What is the square feet?', 120.0))  # Default to 120 sq ft if not specified
            }
        }
        
        # Merge requirements into user_data
        user_data.update(requirements)
        
        return user_data
    except Exception as e:
        print(f"Error reading user information: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def calculate_total_cost(recommendations):
    """Calculate total cost of recommendations"""
    total_cost = 0
    processed_types = set()  # Track which product types we've already counted
    
    for room, products in recommendations.items():
        if not isinstance(products, dict):
            continue
        
        for product_type, options in products.items():
            if not options or not isinstance(options, list):
                continue
            
            # Only count the highest-priced option for each product type
            if product_type not in processed_types:
                try:
                    prices = [float(option.get('retail_price', option.get('price', option.get('better_home_price', 0)))) for option in options if isinstance(option, dict)]
                    if prices:
                        max_price = max(prices)
                        total_cost += max_price
                    processed_types.add(product_type)
                except (ValueError, TypeError):
                    continue
    
    return total_cost 

# Function to get budget category
def get_budget_category(total_budget: float, appliance_type: str, quantity: int = 1) -> str:
    """Determine budget category based on total budget and appliance type"""
    # Define priority and allocation for each appliance type
    appliance_priorities = {
        'ac': {'priority': 1, 'allocation': 0.25},  # Essential for Chennai climate
        'refrigerator': {'priority': 1, 'allocation': 0.30},  # Essential appliance
        'washing_machine': {'priority': 1, 'allocation': 0.20},  # Essential appliance
        'chimney': {'priority': 2, 'allocation': 0.15},  # Important but secondary
        'ceiling_fan': {'priority': 2, 'allocation': 0.05},  # Important for air circulation
        'gas_stove': {'priority': 2, 'allocation': 0.10},  # Important for cooking
        'hob_top': {'priority': 2, 'allocation': 0.12},  # Important for cooking
        'geyser': {'priority': 3, 'allocation': 0.08},  # Comfort appliance
        'bathroom_exhaust': {'priority': 3, 'allocation': 0.02},  # Comfort appliance
        'small_fan': {'priority': 3, 'allocation': 0.02},  # Optional appliance
        'led_mirror': {'priority': 3, 'allocation': 0.05},  # Optional appliance
    }

    # Get appliance priority and allocation
    appliance_info = appliance_priorities.get(appliance_type, {'priority': 3, 'allocation': 0.10})
    allocation = appliance_info['allocation']
    priority = appliance_info['priority']

    # Calculate allocated budget, adjusted for quantity
    # For multiple units, we increase the allocation proportionally but with a slight discount
    # to account for bulk purchases
    quantity_factor = 1 + (quantity - 1) * 0.8  # 20% discount for additional units
    allocated_budget = total_budget * allocation * quantity_factor

    # For high priority appliances (AC, refrigerator, washing machine)
    if priority == 1:
        if total_budget >= 500000:  # High total budget
            return "premium"
        elif total_budget >= 300000:  # Medium total budget
            if allocated_budget >= 40000 * quantity_factor:
                return "premium"
            else:
                return "mid"
        else:  # Lower total budget (<=300000)
            if allocated_budget >= 25000 * quantity_factor:  # More conservative threshold
                return "mid"
            else:
                return "budget"

    # For medium priority appliances (chimney, ceiling fan, gas stove)
    elif priority == 2:
        if total_budget >= 400000:  # High total budget
            return "premium"
        elif total_budget >= 200000:  # Medium total budget
            if allocated_budget >= 30000 * quantity_factor:
                return "premium"
            else:
                return "mid"
        else:  # Lower total budget
            if allocated_budget >= 15000 * quantity_factor:  # More conservative threshold
                return "mid"
            else:
                return "budget"

    # For lower priority appliances
    else:
        if total_budget >= 300000:  # High total budget
            if allocated_budget >= 20000 * quantity_factor:
                return "premium"
            else:
                return "mid"
        else:  # Lower total budget
            if allocated_budget >= 10000 * quantity_factor:  # More conservative threshold
                return "mid"
            else:
                return "budget"

# Function to get budget category for a product
def get_budget_category_for_product(price: float, appliance_type: str) -> str:
    """Determine budget category based on product price and appliance type"""
    categories = {
        'refrigerator': {'budget': 40000, 'mid': 80000},  # Above 80000 is premium
        'washing_machine': {'budget': 30000, 'mid': 50000},
        'chimney': {'budget': 20000, 'mid': 35000},
        'geyser': {'budget': 10000, 'mid': 20000},
        'ceiling_fan': {'budget': 4000, 'mid': 6000},  # Updated to match actual product prices
        'bathroom_exhaust': {'budget': 2000, 'mid': 4000},
        'ac': {'budget': 75000, 'mid': 100000, 'premium': 150000},  # Increased thresholds to prioritize proper tonnage
        'dishwasher': {'budget': 30000, 'mid': 50000},
        'dryer': {'budget': 50000, 'mid': 75000},  # Updated dryer thresholds
        'shower_system': {'budget': 30000, 'mid': 50000},
        'gas_stove': {'budget': 15000, 'mid': 25000}, # Add gas stove
        'hob_top': {'budget': 20000, 'mid': 40000}    # Add hob top
    }
    
    # Default ranges if appliance type is not in the categories
    default_ranges = {'budget': 20000, 'mid': 40000}
    ranges = categories.get(appliance_type, default_ranges)
    
    if price <= ranges['budget']:
        return 'budget'
    elif price <= ranges['mid']:
        return 'mid'
    else:
        return 'premium'

# Helper function to parse product feature string
def parse_product_feature(feature_str: str) -> Dict[str, str]:
    """Parses a feature string like 'Key: Value Unit' into a dict."""
    if not isinstance(feature_str, str) or ':' not in feature_str:
        return {} 
    key, value_part = feature_str.split(':', 1)
    key = key.strip().lower()
    value_part = value_part.strip()
    # Try to extract numeric value and unit
    match = re.match(r"([\d.]+)\s*(\w*)", value_part, re.IGNORECASE)
    if match:
        return {'key': key, 'value': match.group(1), 'unit': match.group(2).lower() if match.group(2) else None, 'raw_value': value_part}
    else:
        # Handle cases like "Key: Value" or "Key: Yes/No"
        return {'key': key, 'value': value_part, 'unit': None, 'raw_value': value_part}

# Function to compare features (basic numerical comparison for now)
def compare_features(req_value_str: str, prod_feature: Dict[str, str]) -> bool:
    """Compares a required feature value string with a parsed product feature."""
    if not isinstance(req_value_str, str) or not prod_feature:
        return False
        
    # Attempt numerical comparison if possible
    req_match = re.match(r"([\d.]+)\s*(\w*)", req_value_str, re.IGNORECASE)
    prod_value_num = None
    try:
        prod_value_num = float(prod_feature.get('value', 'not a number'))
    except ValueError:
        pass

    if req_match and prod_value_num is not None:
        req_val = float(req_match.group(1))
        req_unit = req_match.group(2).lower() if req_match.group(2) else None
        prod_unit = prod_feature.get('unit')
        
        # Compare numbers, ignore units if only one is present or if they mismatch slightly (cm vs cms)
        unit_match = (req_unit == prod_unit) or \
                     (req_unit and prod_unit and req_unit.startswith(prod_unit)) or \
                     (req_unit and prod_unit and prod_unit.startswith(req_unit)) or \
                     (req_unit is None or prod_unit is None) # Allow comparison if one lacks unit

        # Use a small tolerance for float comparison
        TOLERANCE = 1e-6
        if unit_match and abs(req_val - prod_value_num) < TOLERANCE:
             return True
             
    # Fallback to case-insensitive raw string comparison (handles non-numeric values)
    match = req_value_str.strip().lower() == prod_feature.get('raw_value', '').lower()
    return match

def is_horizontal_water_heater(product: Dict[str, Any]) -> bool:
    """Check if a water heater is horizontal based on title and features."""
    title = product.get('title', '').lower()
    features = [f.lower() for f in product.get('features', [])]
    
    # Check title for horizontal keywords
    title_is_horizontal = any(keyword in title for keyword in ['horizontal', 'slim', 'rhs', 'flat'])
    
    # Check features for horizontal keywords
    features_is_horizontal = any('horizontal' in f for f in features)
    
    # Check dimensions in features for horizontal orientation
    dimensions_is_horizontal = False
    for feature in features:
        if 'dimensions' in feature or 'size' in feature:
            # If height is less than width/depth, it's likely horizontal
            if 'x' in feature:
                try:
                    # Extract dimensions, handling various formats like "HxWxD" or "H x W x D"
                    dims = [d.strip() for d in feature.split('x')]
                    if len(dims) >= 2:
                        # Extract first number from each dimension
                        h = float(''.join(filter(str.isdigit, dims[0].split()[0])))
                        w = float(''.join(filter(str.isdigit, dims[1].split()[0])))
                        # If width is greater than height, it's likely horizontal
                        if w > h:
                            dimensions_is_horizontal = True
                except (ValueError, IndexError):
                    pass
    
    is_horizontal = title_is_horizontal or features_is_horizontal or dimensions_is_horizontal

    return is_horizontal

# Function to get specific product recommendations
def get_specific_product_recommendations(
    appliance_type: str, 
    target_budget_category: str, 
    demographics: Dict[str, int], 
    room_color_theme: str = None, 
    user_data: Dict[str, Any] = None,
    required_features: Dict[str, str] = None,  # Added parameter
    room: str = None  # Add room parameter
) -> List[Dict[str, Any]]:
    """Get specific product recommendations based on appliance type, budget category, demographics, color theme, and specific features."""
    
    required_features = required_features or {} # Ensure it's a dict
    catalog = load_product_catalog()
    recommendations = []
    product_groups = {}  # Dictionary to group products by model
    
    # Define budget ranges for each appliance type
    ranges = {
        'refrigerator': {'budget': 40000, 'mid': 80000},  # Above 80000 is premium
        'washing_machine': {'budget': 30000, 'mid': 50000},
        'chimney': {'budget': 20000, 'mid': 35000},
        'geyser': {'budget': 10000, 'mid': 20000},
        'ceiling_fan': {'budget': 4000, 'mid': 6000},  # Updated to match actual product prices
        'bathroom_exhaust': {'budget': 2000, 'mid': 4000},
        'ac': {'budget': 75000, 'mid': 100000, 'premium': 150000},  # Increased thresholds to prioritize proper tonnage
        'dishwasher': {'budget': 30000, 'mid': 50000},
        'dryer': {'budget': 50000, 'mid': 75000},  # Updated dryer thresholds
        'shower_system': {'budget': 30000, 'mid': 50000},
        'gas_stove': {'budget': 15000, 'mid': 25000}, # Add gas stove
        'hob_top': {'budget': 20000, 'mid': 40000}    # Add hob top
    }
    
    # Get the ranges for this appliance type, or use default ranges
    ranges = ranges.get(appliance_type, {'budget': 20000, 'mid': 40000})
    
    # Process available products
    if catalog and "products" in catalog:
        if appliance_type == 'geyser':
            all_types = set(str(p.get('product_type', 'No type')) for p in catalog['products'])
            # print(f"[DEBUG][Geyser] All product types in catalog: {sorted(all_types)}")
        norm_type = appliance_type.lower().replace('_', ' ')
        # Adjust filtering logic to handle special cases
        filtered_products = []
        for p in catalog["products"]:
            if isinstance(p.get("product_type", ""), str):
                product_type_norm = p.get("product_type", "").lower().replace('_', ' ')
                matches = False
                # Special case: AC can be labeled as "ac" or "air conditioner"
                if norm_type == 'ac':
                    matches = product_type_norm == 'ac' or product_type_norm == 'air conditioner'
                # Special case: geyser can be labeled as 'geyser', 'water heater', 'instant water heater', or 'storage water heater'
                elif norm_type == 'geyser':
                    matches = any(t in product_type_norm for t in ['geyser', 'water heater', 'instant water heater', 'storage water heater'])
                    # print(f"[DEBUG][Geyser] Found water heater: {p.get('title')}")
                    # If false ceiling installation is required, only include horizontal water heaters
                    if matches and required_features.get('water_heater_ceiling', '').lower() == 'yes':
                        # print(f"[DEBUG][Geyser] False ceiling required, checking orientation")
                        matches = is_horizontal_water_heater(p)
                        # print(f"[DEBUG][Geyser] After orientation check: {matches}")
                    elif matches and required_features.get('water_heater_ceiling', '').lower() == 'no':
                        # print(f"[DEBUG][Geyser] No false ceiling, checking for vertical orientation")
                        matches = not is_horizontal_water_heater(p)
                        # print(f"[DEBUG][Geyser] After orientation check: {matches}")
                # Special case: bathroom_exhaust can be labeled as 'exhaust fan', 'bathroom exhaust', or 'ventilation fan'
                elif norm_type == 'bathroom exhaust':
                    matches = any(t in product_type_norm for t in ['exhaust fan', 'bathroom exhaust', 'ventilation fan'])
                # Special case: dryer can be labeled as 'cloth dryer'
                elif norm_type == 'dryer':
                    matches = 'cloth dryer' in product_type_norm or 'dryer' == product_type_norm
                else:
                    matches = product_type_norm == norm_type
                if matches:
                    filtered_products.append(p)

        # Further filter for washing machine type if specified
        if norm_type == 'washing machine' and required_features.get('type'):
            user_requested_wm_type_str = str(required_features.get('type', '')).strip()
            if user_requested_wm_type_str: # Ensure it's not empty
                # Normalize user's preference: 'Front-Load' -> 'front load'
                normalized_user_type = user_requested_wm_type_str.replace('-', ' ').lower()
                
                temp_filtered_products = []
                for product in filtered_products:
                    # Get catalog product type: product['features']['parsed_features']['Type']
                    # Example: "Front Load"
                    catalog_wm_type_str = product.get('features', {}).get('parsed_features', {}).get('Type', '').strip()
                    if catalog_wm_type_str:
                        # Normalize catalog type: "Front Load" -> "front load"
                        normalized_catalog_type = catalog_wm_type_str.lower() 
                        # Also handle if catalog might have hyphens, though less likely for 'Front Load'
                        normalized_catalog_type = normalized_catalog_type.replace('-', ' ')

                        if normalized_catalog_type == normalized_user_type:
                            temp_filtered_products.append(product)
                filtered_products = temp_filtered_products

        if appliance_type == 'hob_top':
            # print(f"[DEBUG HOB] required_features: {required_features}")
            # print(f"[DEBUG HOB] Titles before burner filter:")
            # for p in filtered_products:
            #     print(f"  - {p.get('title', 'No title')}")
            num_burners = None
            if required_features and 'burners' in required_features:
                try:
                    num_burners = int(required_features['burners'])
                except Exception:
                    num_burners = None
            if num_burners:
                def has_burner_count(p):
                    title = p.get('title', '').lower().replace('-', ' ')
                    return f'{num_burners} burner' in title
                
                # Apply the filter
                filtered_products = [p for p in filtered_products if has_burner_count(p)]
                
                # print(f"[DEBUG HOB] Titles after burner filter:")
                # for p in filtered_products:
                #     print(f"  - {p.get('title', 'No title')}")

        # Simple debug prints after filtering, specific types
        if appliance_type == 'refrigerator':
            # print(f"[DEBUG] Filtered products count immediately after filtering for '{appliance_type}': {len(filtered_products)}")
            if not filtered_products:
                 all_product_types = [p.get('product_type', 'No type') for p in catalog.get('products', [])]
                 # print(f"[DEBUG] All product types found: {all_product_types}")
        elif appliance_type == 'ac': 
            # print(f"[DEBUG AC FILTERING] Required features: {required_features}")
            # print(f"[DEBUG AC FILTERING] Filtered products count immediately after filtering for '{appliance_type}': {len(filtered_products)}")
            if not filtered_products:
                 all_product_types = [p.get('product_type', 'No type') for p in catalog.get('products', [])]
                 # print(f"[DEBUG AC FILTERING] All product types found (checking for AC): {all_product_types}")
        elif appliance_type == 'hob_top':
            # print(f"[DEBUG HOB] Filtered products count for 'hob_top': {len(filtered_products)}")
            if not filtered_products:
                 all_product_types = sorted(set([p.get('product_type', 'No type') for p in catalog.get('products', [])]))
                 # print(f"[DEBUG HOB] All product types found: {all_product_types}")

        matching_products_data = [] # Store product data along with scores
        
        # DEBUG: Print initial list of ACs after type filter
        if appliance_type == 'ac':
            # print("[DEBUG AC DETAILS] Initial AC candidates before budget/feature check:")
            for idx, p in enumerate(filtered_products):
                try:
                    p_tonnage = "Unknown"
                    # First, try extracting from features
                    for feat in p.get('features', []):
                        match = re.search(r'(\d+\.?\d*)\s*ton', feat.lower())
                        if match:
                            try:
                                p_tonnage = float(match.group(1))
                                break # Found in features, exit loop
                            except ValueError:
                                p_tonnage = "Unknown" # Handle potential conversion error

                    # If not found in features, try extracting from the title
                    if p_tonnage == "Unknown":
                        title = p.get('title', '')
                        if title:
                            match = re.search(r'(\d+\.?\d*)\s*ton', title.lower())
                            if match:
                                try:
                                    p_tonnage = float(match.group(1))
                                except ValueError:
                                    p_tonnage = "Unknown" # Handle potential conversion error
                                    
                    # print(f"  {idx+1}. {p.get('brand', 'N/A')} - {p.get('title', 'N/A')} - Tonnage: {p_tonnage} - Price: {p.get('retail_price', p.get('price', p.get('better_home_price', 0)))}")
                except Exception as e:
                     print(f"  Error printing details for product index {idx}: {e}")

        for product in filtered_products:
            # Use retail_price as primary, fallback to price or better_home_price
            try:
                price = float(product.get('retail_price', product.get('price', product.get('better_home_price', 0))))
            except (ValueError, TypeError):
                price = 0.0
            
            # Determine if product matches budget category and type requirements
            product_matches_budget = False
            if target_budget_category == 'premium':
                product_matches_budget = True
            elif target_budget_category == 'mid':
                product_matches_budget = (price <= ranges.get('mid', float('inf')))
            else:  # budget category
                product_matches_budget = (price <= ranges.get('budget', float('inf')))
            
            # Add matching products to list if budget matches
            if not product_matches_budget:
                continue

            # --- Determine Actual Product Tonnage (checking features and title) ---
            actual_product_tonnage = None
            if appliance_type == 'ac':
                # Try features first
                for feat in product.get('features', []):
                    match = re.search(r'(\d+\.?\d*)\s*ton', feat.lower())
                    if match:
                        try:
                            actual_product_tonnage = float(match.group(1))
                            break
                        except ValueError:
                            pass # Ignore conversion errors in features list
                # Try title if not found in features
                if actual_product_tonnage is None:
                    title = product.get('title', '')
                    if title:
                        match = re.search(r'(\d+\.?\d*)\s*ton', title.lower())
                        if match:
                            try:
                                actual_product_tonnage = float(match.group(1))
                            except ValueError:
                                pass # Ignore conversion errors in title

            # Calculate feature match score
            feature_match_score = 0
            product_features_list = product.get('features', []) # Already a list from json
            
            # DEBUG: Specific check for tonnage feature match
            tonnage_feature_score = 0
            product_tonnage_value_for_debug = "N/A" # Keep this for the debug print only

            if required_features: # Check if required_features is not None or empty
                for req_key, req_value in required_features.items():
                    req_key_norm = req_key.strip().lower()
                    found_match_for_req = False

                    # --- Handle Tonnage Matching Separately ---
                    if appliance_type == 'ac' and req_key_norm == 'tonnage':
                        tonnage_requirement_met = False
                        if actual_product_tonnage is not None:
                            # Parse required tonnage value (e.g., "1.5 Ton")
                            req_match = re.match(r"([\d.]+)", str(req_value).strip())
                            if req_match:
                                try:
                                    required_tonnage_num = float(req_match.group(1))
                                    # Compare with actual product tonnage (allow small tolerance)
                                    TOLERANCE = 0.01 # Allow very small float differences
                                    # Allow slightly higher tonnage as well (e.g., 1.5 Ton required, 1.6 Ton product is ok)
                                    if actual_product_tonnage >= required_tonnage_num - TOLERANCE:
                                        feature_match_score += 5 # Bonus for matching or exceeding
                                        tonnage_feature_score = 5
                                        found_match_for_req = True
                                        tonnage_requirement_met = True
                                except ValueError:
                                    pass # Ignore conversion errors for required value
                            # Update debug value
                            product_tonnage_value_for_debug = f"{actual_product_tonnage} Ton"
                        
                        # Penalize if tonnage requirement was present but not met
                        if not tonnage_requirement_met:
                            penalty = -100 # Define explicit penalty value
                            feature_match_score += penalty # Apply penalty
                            tonnage_feature_score = penalty # Set tonnage score to penalty
                            # --- Temporary Debug Print --- 
                            # print(f"[TEMP DEBUG] Tonnage Mismatch! Product: {product.get('title', 'N/A')[:30]}... Penalty Applied. New FeatScore: {feature_match_score}, New TonnageScore: {tonnage_feature_score}")
                            # --- End Temporary Debug Print ---
                            
                        # Even if no match, we handled the tonnage requirement check
                        continue # Move to the next required feature

                    # --- Handle Other Feature Matching (using existing logic) ---
                    if product_features_list:  # Only iterate if features list exists
                        for feature_str in product_features_list:
                            parsed_prod_feature = parse_product_feature(feature_str)
                            # Extract product tonnage for debug print (only, not for matching logic)
                            if appliance_type == 'ac' and parsed_prod_feature.get('key') == 'tonnage':
                                product_tonnage_value_for_debug = parsed_prod_feature.get('raw_value', 'N/A')
                            if parsed_prod_feature.get('key') == req_key_norm:
                                if compare_features(req_value, parsed_prod_feature):
                                    feature_match_score += 5  # Significant boost for matching feature
                                    found_match_for_req = True
                                    break  # Stop checking this product's features for this required key

            # DEBUG: Print tonnage match result (using the specific debug variable)
            if appliance_type == 'ac':
                req_tonnage = required_features.get('tonnage', 'Not Specified') if required_features else 'Not Specified'
                # Use the actual tonnage if found, otherwise fallback to the debug value from features
                display_tonnage = f"{actual_product_tonnage} Ton" if actual_product_tonnage is not None else product_tonnage_value_for_debug
                # print(f"[DEBUG AC TONNAGE MATCH] Product: {product.get('brand', 'N/A')} {product.get('title', 'N/A')} - Required: {req_tonnage}, Product has: {display_tonnage}, Tonnage Feature Score: {tonnage_feature_score}")

            # Calculate relevance score (existing logic)
            relevance_score = 0
            product_data = product.copy() # Work on a copy
            product_data['color_match'] = False # Initialize color match flag
            product_data['matched_color'] = None # Initialize matched_color
            
            if product_data.get('color_options'):
                product_data['color_options'] = list(set(product_data.get('color_options', [])))
                if room_color_theme:
                    room_colors = room_color_theme.lower().split()
                    product_colors_original_case = product_data.get('color_options', []) # Keep original case for storing
                    product_colors_lower = [c.lower() for c in product_colors_original_case]
                    
                    for room_color in room_colors: # User's theme color
                        for i, pc_lower in enumerate(product_colors_lower): # Product's available colors (lowercase)
                            if room_color in pc_lower: # If user's theme color is found in a product color
                                product_data['color_match'] = True
                                product_data['matched_color'] = product_colors_original_case[i] # Store the original case product color
                                relevance_score += 2 # Or some other score adjustment
                                break # Found a match for this room_color, move to next room_color or exit outer loop
                        if product_data['color_match']:
                            break # Match found for one of the room_colors, no need to check other room_colors
            else:
                product_data['color_options'] = []
            
            features_str_list = product_data.get('features', [])
            features_str_combined = '|'.join(features_str_list)
            if 'BLDC' in features_str_combined.upper():
                relevance_score += 1
            if 'remote' in features_str_combined.lower():
                relevance_score += 1
            if target_budget_category == 'premium':
                relevance_score += 2

            # Prioritize sound level for bedrooms and elder rooms
            if appliance_type == 'ac' and room in ['master_bedroom', 'bedroom_2']:
                sound_level = None
                for feature in product_features_list:
                    if 'sound level' in feature.lower():
                        match = re.search(r'(\d+\.?\d*)\s*dB', feature)
                        if match:
                            sound_level = float(match.group(1))
                            break
                if sound_level is not None:
                    # Lower sound level is better
                    relevance_score += max(0, 10 - sound_level / 10)  # Example scoring

            # Prioritize power consumption for hall
            if appliance_type == 'ac' and room == 'hall':
                power_consumption = None
                for feature in product_features_list:
                    if 'power consumption' in feature.lower():
                        match = re.search(r'(\d+\.?\d*)\s*W', feature)
                        if match:
                            power_consumption = float(match.group(1))
                            break
                if power_consumption is not None:
                    # Lower power consumption is better
                    relevance_score += max(0, 10 - power_consumption / 100)  # Example scoring

            # Store product with scores
            product_data['feature_match_score'] = feature_match_score
            product_data['relevance_score'] = relevance_score
            # Ensure price fields are present and numeric before adding
            product_data['price'] = price # Store the calculated price used for budget check
            product_data['retail_price'] = float(product.get('retail_price', price * 1.2)) # Estimate retail if missing
            product_data['better_home_price'] = float(product.get('better_home_price', price / 1.2 if price > 0 else 0)) # Estimate BH if missing
            
            # Normalize best seller field
            product_data['is_bestseller'] = str(product.get('best_seller', '')).strip().lower() == 'yes'
            
            matching_products_data.append(product_data)
        if appliance_type == 'geyser':
            # print(f"[DEBUG][Geyser] Products after budget filter: {len(matching_products_data)}")
            pass

        # Track unique combinations of brand and tonnage (or other relevant features)
        unique_combinations = set()

        # Sort by feature match score, then is_bestseller flag, then relevance score, then price
        matching_products_data.sort(key=lambda x: (
            -x.get('feature_match_score', 0), 
            -x.get('is_bestseller', False),  # Prioritize bestsellers after feature match
            -x.get('relevance_score', 0), 
            -float(x.get('price', 0))
        ))

        # DEBUG: Print top N products after sorting
        # if appliance_type == 'ac':
        #     print("[DEBUG AC SORTING] Top 5 AC candidates after sorting:")
        #     for idx, p_data in enumerate(matching_products_data[:5]):
        #         print(f"  {idx+1}. {p_data.get('brand', 'N/A')} - {p_data.get('model', 'N/A')} "
        #               f"(Price: {p_data.get('price', 0):.2f}, FeatScore: {p_data.get('feature_match_score', 0)}, RelScore: {p_data.get('relevance_score', 0)})")

        # Group top products by model (using existing logic, but apply to sorted list)
        # Take the top N unique models after sorting
        final_recommendations = []
        seen_models = set()
        limit = 3 # Number of unique models to recommend
        
        # Select the top 'limit' unique models based on the primary sort order
        for product_data in matching_products_data:
            # Ensure product_data is a dictionary before proceeding
            if not isinstance(product_data, dict):
                continue
                
            # Create a unique key based on brand and title to avoid duplicates of the *same* model
            model_key = f"{product_data.get('brand', 'UnknownBrand')}_{product_data.get('title', 'UnknownModel')}"
            
            # Only add if we haven't seen this exact model key yet
            if model_key not in seen_models:
                # Format the recommendation dict (ensure all keys are accessed safely)
                recommendation = {
                    'brand': product_data.get('brand', 'UnknownBrand'),
                    'model': product_data.get('title', 'UnknownModel'),
                    'title': product_data.get('title', 'UnknownModel'),  # <-- Add this line
                    'price': float(product_data.get('price', 0.0)),
                    'retail_price': float(product_data.get('retail_price', 0.0)),
                    'better_home_price': float(product_data.get('better_home_price', 0.0)),
                    'features': product_data.get('features', []),
                    'description': product_data.get('concise_description', 'No description available'), # Use actual description from product data
                    'color_options': product_data.get('color_options', []),
                    'color_match': product_data.get('color_match', False),
                    'warranty': product_data.get('warranty', 'Standard warranty applies'),
                    'in_stock': product_data.get('in_stock', True),
                    'delivery_time': product_data.get('delivery_time', 'Contact store for details'),
                    'url': product_data.get('url', 'https://betterhomeapp.com'),
                    'relevance_score': product_data.get('relevance_score', 0),
                    'feature_match_score': product_data.get('feature_match_score', 0),
                    'energy_rating': product_data.get('energy_rating', None),
                    'capacity': product_data.get('capacity', ''), # Consider populating this from actual_tonnage if AC
                    'type': product_data.get('type', ''),
                    'suction_power': product_data.get('suction_power', ''),
                    'image_src': product_data.get('image_src', 'https://via.placeholder.com/300x300?text=No+Image+Available'),
                    'is_bestseller': product_data.get('is_bestseller', False), # Preserve bestseller flag
                }
                
                final_recommendations.append(recommendation)
                seen_models.add(model_key)
                
                # Stop once we have enough recommendations
                if len(final_recommendations) >= limit:
                    break
                    

        # If no products match the budget or feature requirements, select the best available product
        if not final_recommendations:
            # Sort by price (ascending) and feature match score (descending) to find the best available product
            matching_products_data.sort(key=lambda x: (
                float(x.get('price', float('inf'))),  # Sort by price ascending
                -x.get('feature_match_score', 0)  # Sort by feature match score descending
            ))
            # Take the top product if available
            if matching_products_data:
                # Append the formatted recommendation, not the raw product_data
                top_product_data = matching_products_data[0]
                recommendation = {
                    'brand': top_product_data.get('brand', 'UnknownBrand'),
                    'model': top_product_data.get('title', 'UnknownModel'),
                    'title': top_product_data.get('title', 'UnknownModel'),  # <-- Add this line
                    'price': top_product_data.get('price', 0.0),
                    'retail_price': top_product_data.get('retail_price', 0.0),
                    'better_home_price': top_product_data.get('better_home_price', 0.0),
                    'features': top_product_data.get('features', []),
                    'description': f"{top_product_data.get('type', '')} {top_product_data.get('capacity', '')}",
                    'color_options': top_product_data.get('color_options', []),
                    'color_match': top_product_data.get('color_match', False),
                    'warranty': top_product_data.get('warranty', 'Standard warranty applies'),
                    'in_stock': top_product_data.get('in_stock', True),
                    'delivery_time': top_product_data.get('delivery_time', 'Contact store for details'),
                    'url': top_product_data.get('url', 'https://betterhomeapp.com'),
                    'relevance_score': top_product_data.get('relevance_score', 0),
                    'feature_match_score': top_product_data.get('feature_match_score', 0),
                    'energy_rating': top_product_data.get('energy_rating', None),
                    'capacity': top_product_data.get('capacity', ''),
                    'type': top_product_data.get('type', ''),
                    'suction_power': top_product_data.get('suction_power', ''),
                    'image_src': top_product_data.get('image_src', 'https://via.placeholder.com/300x300?text=No+Image+Available'),
                    'is_bestseller': top_product_data.get('is_bestseller', False), # Preserve bestseller flag
                }
                final_recommendations.append(recommendation)

        if appliance_type == 'hob_top':
            # print(f"[DEBUG HOB] required_features: {required_features}")
            # print(f"[DEBUG HOB] Titles before burner filter:")
            # for p in filtered_products:
            #     print(f"  - {p.get('title', 'No title')}")
            num_burners = None
            if required_features and 'burners' in required_features:
                try:
                    num_burners = int(required_features['burners'])
                except Exception:
                    num_burners = None
            if num_burners:
                def has_burner_count(p):
                    title = p.get('title', '').lower().replace('-', ' ')
                    return f'{num_burners} burner' in title
                
                # Apply the filter
                filtered_products = [p for p in filtered_products if has_burner_count(p)]
                
                # print(f"[DEBUG HOB] Titles after burner filter:")
                # for p in filtered_products:
                #     print(f"  - {p.get('title', 'No title')}")

        # After initial filtering for type, add this for bathroom_exhaust:
        if appliance_type == 'bathroom_exhaust' and required_features:

            
            # --- Size Matching (blade length in cm or mm) ---
            size_val = str(required_features.get('dimensions', '') or '').strip().lower()
            size_num = None
            size_unit = None
            if size_val:
                match = re.match(r'(\d+\.?\d*)\s*(mm|cm)?', size_val)
                if match:
                    size_num = float(match.group(1))
                    size_unit = match.group(2) or 'cm'  # Default to cm if not specified
            print(f"Parsed size: {size_num} {size_unit}")
            
            def blade_length_matches(product):
                features = product.get('features', {})
                numeric = features.get('numeric_features', {}) if isinstance(features, dict) else {}
                # Try to get blade_length from numeric_features
                blade = numeric.get('blade_length') or numeric.get('blade length')
                if blade and isinstance(blade, dict):
                    val = blade.get('value')
                    unit = (blade.get('unit') or '').lower()
                    if val is not None:
                        # Convert to input unit
                        if size_unit == 'mm' and unit == 'cm':
                            val = val * 10
                        elif size_unit == 'cm' and unit == 'mm':
                            val = val / 10
                        # Allow small tolerance (±2mm or ±0.2cm)
                        tolerance = 2 if size_unit == 'mm' else 0.2
                        return abs(val - size_num) <= tolerance
                return False
            
            if size_num is not None:
                filtered_products = [p for p in filtered_products if blade_length_matches(p)]
                print(f"Products after size filtering: {len(filtered_products)}")
            
            # --- Color Matching ---
            color_val = str(required_features.get('color', '') or '').strip().lower()
            print(f"Required color: {color_val}")
            
            def color_matches(product):
                # Check color field, then search description and features
                product_color = str(product.get('color', '') or '').lower()
                if color_val and product_color:
                    return color_val in product_color
                return True  # If no color specified, don't filter
            
            if color_val:
                filtered_products = [p for p in filtered_products if color_matches(p)]
            

        # For ceiling fans, filter by blade_length if fan_size_cm is specified
        if appliance_type == 'ceiling_fan' and 'fan_size_cm' in required_features and required_features['fan_size_cm']:
            fan_size_cm = required_features['fan_size_cm']
            def matches_blade_length(product):
                # First, try numeric_features dict
                features = product.get('features', {})
                if isinstance(features, dict):
                    numeric = features.get('numeric_features', {})
                    blade = numeric.get('blade_length') or numeric.get('blade length')
                    if blade and isinstance(blade, dict):
                        val = blade.get('value')
                        # print(f"[DEBUG] Checking {product.get('title', 'No Title')} (numeric_features): blade_length={val}, required={fan_size_cm}")
                        if val is not None and abs(val - fan_size_cm) <= 2:
                            # print(f"[DEBUG] -> MATCH (numeric_features)")
                            return True
                # Fallback: parse features list of strings
                features_list = product.get('features', [])
                if isinstance(features_list, list):
                    for feat in features_list:
                        feat_lower = feat.lower()
                        # Look for 'blade length' or 'fan size'
                        if 'blade length' in feat_lower or 'fan size' in feat_lower:
                            match = re.search(r'(\d+)\s*(mm|cm)?', feat_lower)
                            if match:
                                val = int(match.group(1))
                                unit = match.group(2)
                                # Convert mm to cm if needed
                                if unit == 'mm':
                                    val = val // 10
                                # print(f"[DEBUG] Checking {product.get('title', 'No Title')} (features list): {feat} -> blade_length={val}, required={fan_size_cm}")
                                if abs(val - fan_size_cm) <= 2:
                                    # print(f"[DEBUG] -> MATCH (features list)")
                                    return True
                return False
            # Only print debug info for dining room ceiling fans
            if appliance_type == 'ceiling_fan' and room == 'dining':
                # print("[DEBUG] Filtered ceiling fans before final_recommendations:")
                for p in filtered_products:
                    if p.get('product_type', '').lower() != 'ceiling fan':
                        continue
                    blade_length = None
                    features = p.get('features', {})
                    if isinstance(features, dict):
                        numeric = features.get('numeric_features', {})
                        blade = numeric.get('blade_length') or numeric.get('blade length')
                        if blade and isinstance(blade, dict):
                            blade_length = blade.get('value')
                    if not blade_length:
                        features_list = p.get('features', [])
                        if isinstance(features_list, list):
                            for feat in features_list:
                                if 'blade length' in feat.lower() or 'fan size' in feat.lower():
                                    match = re.search(r'(\d+)', feat)
                                    if match:
                                        blade_length = int(match.group(1))
                                        break
                    # print(f"  - {p.get('title', 'No Title')} - blade_length={blade_length}")

        # After all filtering and sorting, before building final_recommendations
        # print("[DEBUG] Filtered ceiling fans before final_recommendations1:")
        for p in filtered_products:
            blade_length = None
            features = p.get('features', {})
            if isinstance(features, dict):
                numeric = features.get('numeric_features', {})
                blade = numeric.get('blade_length') or numeric.get('blade length')
                if blade and isinstance(blade, dict):
                    blade_length = blade.get('value')
            if not blade_length:
                features_list = p.get('features', [])
                if isinstance(features_list, list):
                    for feat in features_list:
                        if 'blade length' in feat.lower() or 'fan size' in feat.lower():
                            match = re.search(r'(\d+)', feat)
                            if match:
                                blade_length = int(match.group(1))
                                break
            if appliance_type == 'ceiling_fan' and room == 'dining':
                # print(f"  - {p.get('title', 'No Title')} - blade_length={blade_length}")
                pass

        # --- Room Type Filtering Helper ---
        def matches_room_type(product, required_room_type):
            features = product.get('features', {})
            parsed = features.get('parsed_features', {}) if isinstance(features, dict) else {}
            # Prefer parsed_features if available
            room_type_val = parsed.get('room type')
            if room_type_val:
                # Normalize and check if required room type is present
                room_types = [r.strip().lower() for r in room_type_val.split(',')]
                return required_room_type.lower() in room_types
            # Fallback: search in raw_features or feature strings
            raw_features = features.get('raw_features', []) if isinstance(features, dict) else []
            for feat in raw_features:
                if 'room type' in feat.lower() and required_room_type.lower() in feat.lower():
                    return True
            # Fallback: search in features list of strings
            features_list = product.get('features', [])
            if isinstance(features_list, list):
                for feat in features_list:
                    if 'room type' in feat.lower() and required_room_type.lower() in feat.lower():
                        return True
            return False

        # Apply room type filtering if required
        if required_features and required_features.get('room_type'):
            filtered_products = [p for p in filtered_products if matches_room_type(p, required_features['room_type'])]

        return final_recommendations

    else: # No catalog loaded or no products key
      return []


def get_room_description(room: str, user_data: Dict[str, Any]) -> str:
    """Generate a description for each room based on user requirements"""
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
        room_size = user_data['master_bedroom'].get('size_sqft', 140.0)
        ac_info = ""
        if user_data['master_bedroom'].get('ac', False):
            recommended_tonnage = determine_ac_tonnage(room_size, 'master_bedroom')
            ac_info = f"an AC ({recommended_tonnage} Ton recommended)"
        else:
            ac_info = "no AC"
            
        return f"Master bedroom of {room_size} sq ft with {user_data['master_bedroom'].get('color_theme', 'neutral')} theme, " \
               f"{ac_info}, " \
               f"and a bathroom equipped with {user_data['master_bedroom'].get('bathroom', {}).get('water_heater_type', 'standard')} water heating."
    
    elif room == 'bedroom_2':
        room_size = user_data['bedroom_2'].get('size_sqft', 120.0)
        ac_info = ""
        if user_data['bedroom_2'].get('ac', False):
            recommended_tonnage = determine_ac_tonnage(room_size, 'bedroom_2')
            ac_info = f"an AC ({recommended_tonnage} Ton recommended)"
        else:
            ac_info = "no AC"
            
        return f"Second bedroom of {room_size} sq ft with {user_data['bedroom_2'].get('color_theme', 'neutral')} theme, " \
               f"{ac_info}, " \
               f"and a bathroom equipped with {user_data['bedroom_2'].get('bathroom', {}).get('water_heater_type', 'standard')} water heating."
    ######
    elif room == 'bedroom_3':
        room_size = user_data['bedroom_3'].get('size_sqft', 120.0)
        ac_info = ""
        if user_data['bedroom_3'].get('ac', False):
            recommended_tonnage = determine_ac_tonnage(room_size, 'bedroom_3')
            ac_info = f"an AC ({recommended_tonnage} Ton recommended)"
        else:
            ac_info = "no AC"
            
        return f"Third bedroom of {room_size} sq ft with {user_data['bedroom_3'].get('color_theme', 'neutral')} theme, " \
               f"{ac_info}, " \
               f"and a bathroom equipped with {user_data['bedroom_3'].get('bathroom', {}).get('water_heater_type', 'standard')} water heating."
    
    elif room == 'laundry':
        room_size = user_data['laundry'].get('size_sqft', 50.0)
        return f"Laundry area of {room_size} sq ft equipped with a {user_data['laundry'].get('washing_machine_type', 'standard')} washing machine" \
               f"{' and a dryer' if user_data['laundry'].get('dryer_type', '').lower() == 'yes' else ''}."
    
    return ""

def get_user_information(excel_filename: str) -> Dict[str, Any]:
    """Read user information from the Excel file"""
    try:
        # Read the Excel file
        df = pd.read_excel(excel_filename)
        
        # Clean up column names by removing newlines and extra spaces
        df.columns = [col.split('\n')[0].strip() for col in df.columns]
        row = df.iloc[0]
        
        # Convert DataFrame to dictionary
        user_data = {
            'name': df.iloc[0]['Name'],
            'mobile': df.iloc[0]['Mobile Number (Preferably on WhatsApp)'],
            'email': df.iloc[0]['E-mail'],
            'address': df.iloc[0]['Apartment Address'],
            'total_budget': float(df.iloc[0]['What is your overall budget for home appliances?']),
            'num_bedrooms': int(df.iloc[0]['Number of bedrooms']),
            'num_bathrooms': int(df.iloc[0]['Number of bathrooms']),
            'demographics': {
                'adults': int(df.iloc[0]['Adults (between the age 18 to 50)']),
                'elders': int(df.iloc[0]['Elders (above the age 60)']),
                'kids': int(df.iloc[0]['Kids (below the age 18)'])
            }
        }
        
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
        
        return user_data
    except Exception as e:
        print(f"Error reading user information: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# Function to get product recommendation reason
def get_product_recommendation_reason(product: Dict[str, Any], appliance_type: str, room: str, demographics: Dict[str, int], total_budget: float, required_features: Dict[str, str] = None) -> str:
    """Generate a personalized recommendation reason for a product, highlighting matching features. Only use actual product features."""
    reasons = []
    required_features = required_features or {}
    
    # Check if product is a bestseller
    if product.get('is_bestseller', False):
        reasons.append("Top-rated bestseller with excellent customer reviews")
    
    # Budget consideration
    budget_saved = product.get('retail_price', 0) - product.get('price', 0)
    if budget_saved > 0:
        reasons.append(f"Save {format_currency(budget_saved)} compared to retail price - exceptional value for money")

    # Color matching
    if product.get('color_match', False):
        reasons.append(f"Stylish color options ({', '.join(product.get('color_options', []))}) perfectly match your room's theme")

    # Energy efficiency
    if product.get('energy_rating') in ['5 Star', '4 Star']:
        reasons.append(f"Premium {product['energy_rating']} energy rating - significantly reduces your electricity bills")

    # Highlight matching features
    matching_features = []
    product_features_list = product.get('features', [])
    for req_key, req_value in required_features.items():
        for feature_str in product_features_list:
            parsed_prod_feature = parse_product_feature(feature_str)
            if parsed_prod_feature.get('key') == req_key and compare_features(req_value, parsed_prod_feature):
                matching_features.append(f"{parsed_prod_feature['key'].capitalize()}: {parsed_prod_feature['raw_value']}")
                break
    if matching_features:
        reasons.append("Premium features that match your needs: " + ", ".join(matching_features))

    # Room and appliance specific reasons (only use actual product features)
    if appliance_type == 'ceiling_fan':
        if any('bldc' in f.lower() for f in product.get('features', [])):
            reasons.append("Advanced BLDC motor technology - 50% more energy efficient than conventional fans")
        if room == 'hall':
            reasons.append("Perfect for your hall - provides powerful air circulation for large spaces")
    
    elif appliance_type == 'ac':
        # Only mention tonnage if present in product features
        product_tonnage = None
        for feature in product.get('features', []):
            if 'ton' in feature.lower():
                tonnage_match = re.search(r'(\d+\.?\d*)\s*ton', feature.lower())
                if tonnage_match:
                    product_tonnage = float(tonnage_match.group(1))
                    break
        if product_tonnage is not None:
            reasons.append(f"{product_tonnage} Ton capacity - suitable for medium to large rooms")
        # Inverter technology
        if any('inverter' in feature.lower() for feature in product.get('features', [])):
            reasons.append("Advanced inverter technology - up to 40% more energy efficient")
        # Energy rating
        energy_rating = product.get('energy_rating')
        if energy_rating:
            reasons.append(f"Premium {energy_rating} rating - maximum energy savings")
    
    elif appliance_type == 'bathroom_exhaust':
        if demographics.get('elders', 0) > 0 and any('humidity sensor' in f.lower() for f in product.get('features', [])):
            reasons.append("Smart humidity sensing - automatically maintains ideal bathroom conditions")
        reasons.append("Essential for Chennai's climate - prevents mold and maintains freshness")
    
    elif appliance_type == 'geyser':
        if demographics.get('elders', 0) > 0:
            if any('temperature control' in f.lower() for f in product.get('features', [])):
                reasons.append("Advanced temperature control - ensures safe water temperature for elderly")
        if product.get('capacity', '').lower().endswith('l'):
            capacity = int(product.get('capacity', '0L')[:-1])
            family_size = sum(demographics.values())
            if capacity >= family_size * 5:
                reasons.append(f"Large {product['capacity']} capacity - perfect for your family of {family_size}")
    
    elif appliance_type == 'refrigerator':
        # Robust handling for capacity
        capacity_str = str(product.get('capacity', '')).strip().lower()
        family_size = sum(demographics.values())
        capacity_found = False
        if capacity_str.endswith('l'):
            try:
                capacity = int(re.sub(r'[^\d]', '', capacity_str))
                capacity_found = True
                if capacity >= family_size * 100:
                    reasons.append(f"Spacious {product.get('capacity', '')} capacity - ideal for your family of {family_size}")
                else:
                    reasons.append(f"{product.get('capacity', '')} capacity - suitable for your family of {family_size}")
            except Exception:
                pass
        if not capacity_found:
            reasons.append("Spacious and reliable refrigerator for your family")
        # Highlight only 2 string key features (not dicts/lists)
        features = product.get('features', [])
        key_features = []
        if features:
            if isinstance(features, list):
                key_features = [f for f in features if isinstance(f, str)][:2]
            elif isinstance(features, dict):
                key_features = [str(v) for v in list(features.values()) if isinstance(v, str)][:2]
        for feature in key_features:
            reasons.append(f"Key feature: {feature}")
    elif appliance_type == 'washing_machine':
        family_size = sum(demographics.values())
        if product.get('capacity', '').lower().endswith('kg'):
            capacity = float(product.get('capacity', '0kg')[:-2])
            reasons.append(f"Large {product['capacity']} capacity - perfect for family laundry")
        if any('anti-allergen' in f.lower() for f in product.get('features', [])):
            reasons.append("Anti-allergen technology - gentle on sensitive skin")
    
    elif appliance_type == 'chimney':
        if 'auto-clean' in product.get('type', '').lower():
            reasons.append("Smart auto-clean technology - minimal maintenance required")
        if product.get('suction_power', '').lower().endswith('m³/hr'):
            power = int(product.get('suction_power', '0 m³/hr').split()[0])
            if power >= 1200:
                reasons.append(f"Powerful {power} m³/hr suction - effectively removes cooking fumes")
                
    elif appliance_type == 'gas_stove':
        if any('burner' in feature.lower() for feature in product.get('features', [])):
            burner_count = [int(s) for s in re.findall(r'\d+\s*burner', ' '.join(product.get('features', [])).lower())]
            if burner_count and burner_count[0] >= 3:
                family_size = sum(demographics.values())
                if family_size >= 4 and burner_count[0] >= 4:
                    reasons.append(f"Premium {burner_count[0]}-burner design - perfect for family cooking")
                else:
                    reasons.append(f"Versatile {burner_count[0]}-burner configuration - ideal for multiple dishes")
        if required_features and required_features.get('type'):
            preferred_type = required_features.get('type').lower()
            if preferred_type in ' '.join(product.get('features', [])).lower():
                reasons.append(f"Premium {preferred_type} design - matches your cooking style")
                
    elif appliance_type == 'hob_top':
        reasons.append("Sleek modern design - enhances your kitchen's aesthetics")
        
        burner_count = [int(s) for s in re.findall(r'\d+\s*burner', ' '.join(product.get('features', [])).lower())]
        if burner_count:
            family_size = sum(demographics.values())
            if family_size >= 4 and burner_count[0] >= 3:
                reasons.append(f"Premium {burner_count[0]}-burner configuration - perfect for family meals")
            else:
                reasons.append(f"Versatile {burner_count[0]}-burner setup - ideal for everyday cooking")
                
        if any('glass' in feature.lower() for feature in product.get('features', [])):
            reasons.append("Premium glass top - easy to clean and maintain")
        elif any('stainless steel' in feature.lower() for feature in product.get('features', [])):
            reasons.append("Durable stainless steel - long-lasting performance")
            
        if required_features and required_features.get('type'):
            preferred_type = required_features.get('type').lower()
            if preferred_type in ' '.join(product.get('features', [])).lower():
                reasons.append(f"Premium {preferred_type} style - matches your kitchen design")
                
        if any('auto' in feature.lower() and 'ignition' in feature.lower() for feature in product.get('features', [])):
            reasons.append("Advanced auto-ignition - safe and convenient operation")

    return " • " + "\n • ".join(reasons)

# Function to create a styled PDF
def create_styled_pdf(filename, user_data, recommendations, required_features: Dict[str, str] = None):
    doc = SimpleDocTemplate(filename, pagesize=letter)
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import Image
    from reportlab.lib.colors import HexColor
    
    # First check if the script is being run through Flask
    is_web_app = os.environ.get('BETTERHOME_WEB_APP') == 'true'
    
    # Check if logo exists in multiple possible locations
    possible_logo_paths = [
        "web_app/better_home_logo.png",  # Relative to script execution directory
        "./web_app/better_home_logo.png",  # Explicit relative path
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "better_home_logo.png"),  # Same directory as script
        "better_home_logo.png",  # Check in current directory as well
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
    
    # Try to register DejaVuSans font if available, otherwise use default fonts
    try:
        pdfmetrics.registerFont(TTFont('DejaVuSans', './DejaVuSans.ttf'))
        default_font = 'DejaVuSans'
    except:
        print("Using default fonts - DejaVuSans.ttf not found")
        default_font = 'Helvetica'
    
    styles = getSampleStyleSheet()
    for style_name in styles.byName:
        styles[style_name].fontName = default_font
    
    # Create custom styles for a more professional look
    title_style = ParagraphStyle(
        'Title',
        parent=styles['Title'],
        fontSize=24,
        leading=28,
        textColor=HexColor('#2c3e50'),
        spaceAfter=12,
        fontName=default_font
    )
    
    heading1_style = ParagraphStyle(
        'Heading1',
        parent=styles['Heading1'],
        fontSize=18,
        leading=22,
        textColor=HexColor('#2980b9'),
        spaceAfter=10,
        spaceBefore=15,
        fontName=default_font
    )
    
    heading2_style = ParagraphStyle(
        'Heading2',
        parent=styles['Heading2'],
        fontSize=14,
        leading=18,
        textColor=HexColor('#3498db'),
        spaceAfter=8,
        spaceBefore=12,
        fontName=default_font
    )
    
    normal_style = ParagraphStyle(
        'Normal',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        textColor=HexColor('#333333'),
        spaceAfter=8
    )
    
    info_label_style = ParagraphStyle(
        'InfoLabel',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        textColor=HexColor('#7f8c8d'),
        fontName=default_font
    )
    
    info_value_style = ParagraphStyle(
        'InfoValue',
        parent=styles['Normal'],
        fontSize=12,
        leading=16,
        textColor=HexColor('#2c3e50')
    )
    
    price_style = ParagraphStyle(
        'Price',
        parent=styles['Normal'],
        fontSize=14,
        leading=18,
        textColor=HexColor('#e74c3c'),
        fontName=default_font
    )
    
    savings_style = ParagraphStyle(
        'Savings',
        parent=styles['Normal'],
        fontSize=11,
        leading=15,
        textColor=HexColor('#27ae60'),
        fontName=default_font
    )
    
    bestseller_style = ParagraphStyle(
        'Bestseller',
        parent=styles['Normal'],
        fontSize=12,
        leading=15,
        textColor=HexColor('#ffffff'),
        fontName=default_font,
        backColor=HexColor('#ff6b00')
    )
    
    story = []

    # Add logo if it exists
    if logo_exists:
        logo = Image(logo_path, width=200, height=60)  # Adjust size as needed
        logo.hAlign = 'CENTER'
        story.append(logo)
        story.append(Spacer(1, 20))

    # Title and date
    story.append(Paragraph("BetterHome Recommendations", title_style))
    story.append(Paragraph(f"Generated on {datetime.now().strftime('%B %d, %Y')}", normal_style))
    story.append(Spacer(1, 15))

    # Create a horizontal line
    story.append(HRFlowable(
        color=HexColor('#3498db'),
        width="100%",
        thickness=2,
        spaceBefore=5,
        spaceAfter=15
    ))

    # Client Information Section
    story.append(Paragraph("Client Information", heading1_style))
    
    # Create a table for client info
    data = []
    data.append([Paragraph("<b>Name:</b>", info_label_style), 
                 Paragraph(user_data.get('name', 'Not provided'), info_value_style)])
    data.append([Paragraph("<b>Email:</b>", info_label_style), 
                 Paragraph(user_data.get('email', 'Not provided'), info_value_style)])
    data.append([Paragraph("<b>Phone:</b>", info_label_style), 
                 Paragraph(user_data.get('phone', 'Not provided'), info_value_style)])
    data.append([Paragraph("<b>Address:</b>", info_label_style), 
                 Paragraph(user_data.get('address', 'Not provided'), info_value_style)])
    
    if 'demographics' in user_data:
        data.append([Paragraph("<b>Number of Bedrooms:</b>", info_label_style), 
                     Paragraph(str(user_data['demographics'].get('bedrooms', 'Not provided')), info_value_style)])
        data.append([Paragraph("<b>Number of People:</b>", info_label_style), 
                     Paragraph(str(user_data['demographics'].get('num_people', 'Not provided')), info_value_style)])
    
    client_table = Table(data, colWidths=[150, 350])
    client_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), HexColor('#f8f9fa')),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#dddddd')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('PADDING', (0, 0), (-1, -1), 8),
        ('ROWBACKGROUNDS', (0, 0), (-1, -1), [HexColor('#ffffff'), HexColor('#f4f6f9')])
    ]))
    
    story.append(client_table)
    story.append(Spacer(1, 15))
    
    # Budget Information
    story.append(Paragraph("Budget Information", heading1_style))
    
    # Create a table for budget info
    budget_data = []
    budget_data.append([Paragraph("<b>Total Budget:</b>", info_label_style), 
                       Paragraph(f"₹{user_data.get('total_budget', 0):,.2f}", price_style)])
    
    if 'used_budget' in user_data:
        budget_data.append([Paragraph("<b>Used Budget:</b>", info_label_style), 
                           Paragraph(f"₹{user_data.get('used_budget', 0):,.2f}", info_value_style)])
        budget_data.append([Paragraph("<b>Remaining Budget:</b>", info_label_style), 
                           Paragraph(f"₹{user_data.get('total_budget', 0) - user_data.get('used_budget', 0):,.2f}", 
                                    savings_style if user_data.get('total_budget', 0) >= user_data.get('used_budget', 0) else price_style)])
    
    budget_table = Table(budget_data, colWidths=[150, 350])
    budget_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), HexColor('#f1f9ff')),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#b3d7ff')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('PADDING', (0, 0), (-1, -1), 8),
        ('ROWBACKGROUNDS', (0, 0), (-1, -1), [HexColor('#e6f7ff'), HexColor('#f1f9ff')])
    ]))
    
    story.append(budget_table)
    story.append(Spacer(1, 15))

    # Add room by room recommendations
    if 'rooms' in user_data:
        story.append(Paragraph("Room by Room Recommendations", heading1_style))
        story.append(Spacer(1, 10))
        rooms_iter = user_data['rooms'].items()
    else:
        # Fallback: use top-level room keys
        story.append(Paragraph("Room by Room Recommendations", heading1_style))
        story.append(Spacer(1, 10))
        # Only include known room keys
        room_keys = ['hall', 'kitchen', 'master_bedroom', 'bedroom_2', 'laundry', 'dining']
        rooms_iter = ((room, user_data.get(room, {})) for room in room_keys if room in user_data)
    for room, room_data in rooms_iter:
        room_title = Paragraph(f"{room.replace('_', ' ').title()} Room", heading2_style)
        story.append(room_title)
        # Add room description in a styled box
        room_description = room_data.get('description', '') if isinstance(room_data, dict) else ''
        if not room_description:
            try:
                room_description = get_room_description(room, user_data)
            except Exception:
                room_description = ''
        if room_description:
            story.append(Paragraph(room_description, normal_style))
            story.append(Spacer(1, 10))
        # Add appliance recommendations for this room
        if room in recommendations:
            for appliance_type, appliances in recommendations[room].items():
                appliance_title = appliance_type.replace('_', ' ').title()
                story.append(Paragraph(f"{appliance_title} Recommendations", heading2_style))
                for item in appliances:
                    if not isinstance(item, dict):
                        continue
                    brand = item.get('brand', 'Unknown Brand')
                    model = item.get('model', 'Unknown Model')
                    price = float(item.get('better_home_price', item.get('price', 0)))
                    retail_price = float(item.get('retail_price', price * 1.2))
                    # Add bestseller badge if applicable
                    if item.get('is_bestseller', False):
                        bestseller_text = Paragraph("<font color='white' backColor='#ff6b00'><b>BESTSELLER</b></font>", bestseller_style)
                        story.append(bestseller_text)
                        story.append(Spacer(1, 5))
                    story.append(Paragraph(f"{brand} {model}", heading2_style))
                    price_text = f"Price: ₹{price:,.2f}"
                    if retail_price > price:
                        savings = retail_price - price
                        price_text += f" (Retail: ₹{retail_price:,.2f}, Save: ₹{savings:,.2f})"
                    story.append(Paragraph(price_text, normal_style))
                    features = item.get('features', [])
                    if features:
                        story.append(Paragraph("Key Features:", heading2_style))
                        if isinstance(features, dict):
                            features_iter = list(features.values())[:5]
                        elif isinstance(features, list):
                            features_iter = features[:5]
                        else:
                            features_iter = []
                        features_list = [f"• {feature}" for feature in features_iter]
                        for feature in features_list:
                            story.append(Paragraph(feature, normal_style))
                        if len(features_iter) > 5:
                            story.append(Paragraph("• ...", normal_style))
                    # Always call get_product_recommendation_reason for the reason
                    reason = ""
                    try:
                        reason = get_product_recommendation_reason(
                            item, 
                            appliance_type, 
                            room, 
                            user_data.get('demographics', {}),
                            user_data.get('total_budget', 0),
                            {}  # required_features
                        )
                    except Exception as e:
                        # print(f"[DEBUG] Error getting recommendation reason: {e}")
                        reason = ""
                    if reason:
                        story.append(Paragraph("Why We Recommend This:", heading2_style))
                        reasons = [r.strip() for r in reason.split('•') if r.strip()]
                        for r in reasons[:3]:
                            story.append(Paragraph(f"• {r}", normal_style))
                        if len(reasons) > 3:
                            story.append(Paragraph("• ...", normal_style))
                    story.append(Spacer(1, 10))
                    story.append(HRFlowable(color=HexColor('#dddddd'), width="90%", thickness=1))
                    story.append(Spacer(1, 10))
        story.append(Spacer(1, 20))

    # Add a footer
    story.append(HRFlowable(color=HexColor('#3498db'), width="100%", thickness=1))
    story.append(Spacer(1, 10))
    footer_text = "Thank you for choosing BetterHome! For any questions, please contact support@betterhome.com"
    story.append(Paragraph(footer_text, normal_style))
    
    # Build the PDF
    doc.build(story)

def generate_text_file(user_data: Dict[str, Any], final_list: Dict[str, Any], txt_filename: str) -> None:
    """Generate a text file with user information and product recommendations"""
    with open(txt_filename, 'w') as f:
        # Write user information
        f.write("USER INFORMATION\n")
        f.write("================\n")
        f.write(f"Name: {user_data['name']}\n")
        f.write(f"Mobile: {user_data['mobile']}\n")
        f.write(f"Email: {user_data['email']}\n")
        f.write(f"Address: {user_data['address']}\n\n")
        
        f.write("BUDGET AND FAMILY SIZE\n")
        f.write("=====================\n")
        f.write(f"Total Budget: INR{user_data['total_budget']:,.2f}\n")
        f.write(f"Family Size: {sum(user_data['demographics'].values())} members\n\n")
        
        # Write room-wise recommendations
        f.write("ROOM-WISE RECOMMENDATIONS\n")
        f.write("========================\n\n")

        # Hall
        f.write("HALL\n")
        f.write("----\n")
        for fan in final_list['hall']['fans']:
            f.write(f"Ceiling Fan: {fan['brand']} {fan['model']}\n")
            f.write(f"Price: {format_currency(fan['price'])} (Retail: {format_currency(fan['retail_price'])})\n")
            f.write(f"Features: {', '.join(fan['features'])}\n")
            if fan.get('color_match', False):
                f.write(f"Color Options: {', '.join(fan.get('color_options', []))} - Matches your room's color theme!\n")
            reason = get_product_recommendation_reason(fan, 'ceiling_fan', 'hall', user_data['demographics'], final_list['summary']['total_budget'])
            f.write(f"Why we recommend this:\n{reason}\n\n")

        # Master Bedroom
        f.write("MASTER BEDROOM\n")
        f.write("--------------\n")
        for ac in final_list['master_bedroom']['ac']:
            f.write(f"Air Conditioner: {ac['brand']} {ac['model']}\n")
            f.write(f"Price: {format_currency(ac['price'])} (Retail: {format_currency(ac['retail_price'])})\n")
            f.write(f"Features: {', '.join(ac['features'])}\n")
            reason = get_product_recommendation_reason(ac, 'ac', 'master_bedroom', user_data['demographics'], final_list['summary']['total_budget'])
            f.write(f"Why we recommend this:\n{reason}\n\n")
        for fan in final_list['master_bedroom']['fans']:
            f.write(f"Ceiling Fan: {fan['brand']} {fan['model']}\n")
            f.write(f"Price: {format_currency(fan['price'])} (Retail: {format_currency(fan['retail_price'])})\n")
            f.write(f"Features: {', '.join(fan['features'])}\n")
            reason = get_product_recommendation_reason(fan, 'ceiling_fan', 'master_bedroom', user_data['demographics'], final_list['summary']['total_budget'])
            f.write(f"Why we recommend this:\n{reason}\n\n")
        # Master Bedroom Bathroom
        f.write("Master Bathroom:\n")
        for geyser in final_list['master_bedroom']['bathroom']['water_heater']:
            f.write(f"Water Heater: {geyser['brand']} {geyser['model']}\n")
            f.write(f"Price: {format_currency(geyser['price'])} (Retail: {format_currency(geyser['retail_price'])})\n")
            f.write(f"Features: {', '.join(geyser['features'])}\n")
            reason = get_product_recommendation_reason(geyser, 'geyser', 'master_bedroom', user_data['demographics'], final_list['summary']['total_budget'])
            f.write(f"Why we recommend this:\n{reason}\n\n")
        for exhaust in final_list['master_bedroom']['bathroom']['exhaust_fan']:
            f.write(f"Exhaust Fan: {exhaust['brand']} {exhaust['model']}\n")
            f.write(f"Price: {format_currency(exhaust['price'])} (Retail: {format_currency(exhaust['retail_price'])})\n")
            f.write(f"Features: {', '.join(exhaust['features'])}\n")
            reason = get_product_recommendation_reason(exhaust, 'bathroom_exhaust', 'master_bedroom', user_data['demographics'], final_list['summary']['total_budget'])
            f.write(f"Why we recommend this:\n{reason}\n\n")

        # Bedroom 2
        f.write("BEDROOM 2\n")
        f.write("---------\n")
        for ac in final_list['bedroom_2']['ac']:
            f.write(f"Air Conditioner: {ac['brand']} {ac['model']}\n")
            f.write(f"Price: {format_currency(ac['price'])} (Retail: {format_currency(ac['retail_price'])})\n")
            f.write(f"Features: {', '.join(ac['features'])}\n")
            reason = get_product_recommendation_reason(ac, 'ac', 'bedroom_2', user_data['demographics'], final_list['summary']['total_budget'])
            f.write(f"Why we recommend this:\n{reason}\n\n")
        for fan in final_list['bedroom_2']['fans']:
            f.write(f"Ceiling Fan: {fan['brand']} {fan['model']}\n")
            f.write(f"Price: {format_currency(fan['price'])} (Retail: {format_currency(fan['retail_price'])})\n")
            f.write(f"Features: {', '.join(fan['features'])}\n")
            reason = get_product_recommendation_reason(fan, 'ceiling_fan', 'bedroom_2', user_data['demographics'], final_list['summary']['total_budget'])
            f.write(f"Why we recommend this:\n{reason}\n\n")
        # Bedroom 2 Bathroom
        f.write("Bedroom 2 Bathroom:\n")
        for geyser in final_list['bedroom_2']['bathroom']['water_heater']:
            f.write(f"Water Heater: {geyser['brand']} {geyser['model']}\n")
            f.write(f"Price: {format_currency(geyser['price'])} (Retail: {format_currency(geyser['retail_price'])})\n")
            f.write(f"Features: {', '.join(geyser['features'])}\n")
            reason = get_product_recommendation_reason(geyser, 'geyser', 'bedroom_2', user_data['demographics'], final_list['summary']['total_budget'])
            f.write(f"Why we recommend this:\n{reason}\n\n")
        for exhaust in final_list['bedroom_2']['bathroom']['exhaust_fan']:
            f.write(f"Exhaust Fan: {exhaust['brand']} {exhaust['model']}\n")
            f.write(f"Price: {format_currency(exhaust['price'])} (Retail: {format_currency(exhaust['retail_price'])})\n")
            f.write(f"Features: {', '.join(exhaust['features'])}\n")
            reason = get_product_recommendation_reason(exhaust, 'bathroom_exhaust', 'bedroom_2', user_data['demographics'], final_list['summary']['total_budget'])
            f.write(f"Why we recommend this:\n{reason}\n\n")

        # Dining Room
        f.write("DINING ROOM\n")
        f.write("-----------\n")
        for fan in final_list['dining']['fans']:
            f.write(f"Ceiling Fan: {fan['brand']} {fan['model']}\n")
            f.write(f"Price: {format_currency(fan['price'])} (Retail: {format_currency(fan['retail_price'])})\n")
            f.write(f"Features: {', '.join(fan['features'])}\n")
            if fan.get('color_match', False):
                f.write(f"Color Options: {', '.join(fan.get('color_options', []))} - Matches your room's color theme!\n")
            reason = get_product_recommendation_reason(fan, 'ceiling_fan', 'dining', user_data['demographics'], final_list['summary']['total_budget'])
            f.write(f"Why we recommend this:\n{reason}\n\n")

        for ac in final_list['dining']['ac']:
            f.write(f"Air Conditioner: {ac['brand']} {ac['model']}\n")
            f.write(f"Price: {format_currency(ac['price'])} (Retail: {format_currency(ac['retail_price'])})\n")
            f.write(f"Features: {', '.join(ac['features'])}\n")
            if ac.get('color_match', False):
                f.write(f"Color Options: {', '.join(ac.get('color_options', []))} - Matches your room's color theme!\n")
            reason = get_product_recommendation_reason(ac, 'ac', 'dining', user_data['demographics'], final_list['summary']['total_budget'])
            f.write(f"Why we recommend this:\n{reason}\n\n")

        # Kitchen
        if user_data['kitchen'].get('chimney_width'):
            budget_category = get_budget_category(user_data['total_budget'], 'chimney')
            required_features = {'dimensions': user_data['kitchen'].get('chimney_width')}
            recommendations = get_specific_product_recommendations('chimney', budget_category, user_data['demographics'], user_data['kitchen'].get('color_theme'), user_data, required_features)
            final_list['kitchen']['chimney'] = recommendations
        
        if user_data['kitchen'].get('refrigerator_capacity'):
            budget_category = get_budget_category(user_data['total_budget'], 'refrigerator')
            required_features = {'capacity': user_data['kitchen'].get('refrigerator_capacity')}
            recommendations = get_specific_product_recommendations('refrigerator', budget_category, user_data['demographics'], user_data['kitchen'].get('color_theme'), user_data, required_features)
            final_list['kitchen']['refrigerator'] = recommendations
        
        if user_data['kitchen'].get('gas_stove_type'):
            # Debug: Print the gas stove type from user data
            print(f"[DEBUG GAS STOVE TYPE] User data gas stove type: {user_data['kitchen'].get('gas_stove_type')}")
            # Check for specific gas stove type
            gas_stove_type = user_data['kitchen'].get('gas_stove_type', '').strip()
            print(f"[DEBUG GAS STOVE] Processing gas stove type: '{gas_stove_type}'")
            
            # Case 1: If gas stove type is "Hob (built-in)", recommend a Hob Top
            if "hob (built-in)" in gas_stove_type.lower():
                print(f"[DEBUG GAS STOVE] Recommending Hob Top for built-in requirement: '{gas_stove_type}'")
                budget_category = get_budget_category(user_data['total_budget'], 'gas_stove')
                required_features = {'type': gas_stove_type, 'burners': user_data['kitchen'].get('num_burners', 4)}
                print(f"[DEBUG GAS STOVE] Budget category: {budget_category}, Features: {required_features}")
                # Debug: Print user data and required features before calling recommendation function
                print(f"[DEBUG HOB TOP INPUT] User data: {user_data}")
                print(f"[DEBUG HOB TOP INPUT] Required features: {required_features}")
                recommendations = get_specific_product_recommendations('hob_top', budget_category, user_data['demographics'], 
                                                                     user_data['kitchen'].get('color_theme'), user_data, required_features)
                print(f"[DEBUG GAS STOVE] Got {len(recommendations)} hob_top recommendations")
                #final_list['kitchen']['hob_top'] = recommendations
                print("[DEBUG FINAL_LIST ASSIGN] hob_top after assignment:", [h.get('title', 'No title') for h in final_list['kitchen']['hob_top']])
                # Clear gas_stove as we're using hob_top instead
                final_list['kitchen']['gas_stove'] = []
            elif "not needed" in gas_stove_type.lower():
                print(f"[DEBUG GAS STOVE] Skipping gas stove recommendation for: '{gas_stove_type}'")
                final_list['kitchen']['gas_stove'] = []
                final_list['kitchen']['hob_top'] = []
            else:
                print(f"[DEBUG GAS STOVE] Recommending regular gas stove for: '{gas_stove_type}'")
                budget_category = get_budget_category(user_data['total_budget'], 'gas_stove')
                required_features = {'type': gas_stove_type, 'burners': user_data['kitchen'].get('num_burners', 4)}
                recommendations = get_specific_product_recommendations('gas_stove', budget_category, user_data['demographics'], 
                                                                     user_data['kitchen'].get('color_theme'), user_data, required_features)
                print(f"[DEBUG GAS STOVE] Got {len(recommendations)} gas_stove recommendations")
                final_list['kitchen']['gas_stove'] = recommendations
        
        if user_data['kitchen'].get('small_fan', False):
            budget_category = get_budget_category(user_data['total_budget'], 'small_fan')
            recommendations = get_specific_product_recommendations('small_fan', budget_category, user_data['demographics'], user_data['kitchen'].get('color_theme'), user_data)
            final_list['kitchen']['small_fan'] = recommendations
        
        # Process laundry requirements
        user_selected_wm_type = str(user_data['laundry'].get('washing_machine_type', '')).strip()

        # Check if a specific type of washing machine is selected by the user
        # The values from HTML form/Excel are expected to be "Front-Load", "Top-Load", "Semi-Automatic"
        if user_selected_wm_type and user_selected_wm_type.lower() not in ['yes', 'no', 'none', '', 'not applicable', 'na']:
            budget_category = get_budget_category(user_data['total_budget'], 'washing_machine')
            
            required_features = {
                'capacity': estimate_washing_machine_capacity(user_data['demographics']),
                'type': user_selected_wm_type  # Pass the user's selection
            }
            
            recommendations = get_specific_product_recommendations(
                'washing_machine', 
                budget_category, 
                user_data['demographics'], 
                user_data['laundry'].get('color_theme'), 
                user_data, 
                required_features
            )
            final_list['laundry']['washing_machine'] = recommendations
        elif user_selected_wm_type.lower() == 'yes': # Fallback for old 'yes' logic if it's still possible
            budget_category = get_budget_category(user_data['total_budget'], 'washing_machine')
            required_features = {'capacity': estimate_washing_machine_capacity(user_data['demographics'])}
            recommendations = get_specific_product_recommendations('washing_machine', budget_category, user_data['demographics'], user_data['laundry'].get('color_theme'), user_data, required_features)
            final_list['laundry']['washing_machine'] = recommendations
        else: # User selected "No", "None", or it's empty, so no washing machine
            final_list['laundry']['washing_machine'] = []
        
        
        if user_data['laundry'].get('dryer_type', '').lower() == 'yes':
            #print(f"[DEBUG- inside dryer] User data: {user_data}")
            budget_category = get_budget_category(user_data['total_budget'], 'dryer')
            # Consistently use 'dryer' as appliance type
            recommendations = get_specific_product_recommendations(
                'dryer', 
                budget_category, 
                user_data['demographics'], 
                user_data['laundry'].get('color_theme'), 
                user_data
            )
            # Apply deduplication
            final_list['laundry']['dryer'] = deduplicate_recommendations(recommendations)
        
        # Add budget summary
        f.write("\nBUDGET SUMMARY\n")
        f.write("=============\n")
        f.write(f"Total Cost of Recommended Products: INR{calculate_total_cost(final_list):,.2f}\n")
        f.write(f"Your Budget: INR{user_data['total_budget']:,.2f}\n")
        budget_utilization = (calculate_total_cost(final_list) / user_data['total_budget']) * 100
        f.write(f"Budget Utilization: {budget_utilization:.1f}%\n")
        if budget_utilization <= 100:
            f.write("Your selected products fit within your budget!\n")
        else:
            f.write("Note: The total cost exceeds your budget. You may want to consider alternative options.\n")

# Function to download an image from a URL
def download_image(image_url: str, save_dir: str) -> str:
    """Download an image from a URL and save it to the specified directory."""

# Function to generate final product list
def generate_final_product_list(user_data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a final list of preferred products with specific recommendations"""
    # Debug: Print the entire user data to verify gas stove type
    # print(f"[DEBUG USER DATA] {user_data}")

    # Initialize the final list with empty dictionaries for each room
    final_list = {
        'hall': {'fans': [], 'ac': []},
        'kitchen': {'chimney': [], 'refrigerator': [], 'gas_stove': [], 'hob_top': [], 'small_fan': [], 'dishwasher': []},
        'master_bedroom': {
            'ac': [], 
            'fans': [], 
            'bathroom': {
                'water_heater': [],
                'exhaust_fan': [],
                'led_mirror': []  # Add LED mirror list
            }
        },
        'bedroom_2': {
            'ac': [], 
            'fans': [], 
            'bathroom': {
                'water_heater': [],
                'exhaust_fan': [],
                'led_mirror': []  # Add LED mirror list
            }
        },
        'bedroom_3': {
            'ac': [], 
            'fans': [], 
            'bathroom': {
                'water_heater': [],
                'exhaust_fan': [],
                'led_mirror': []  # Add LED mirror list
            }
        },
        'laundry': {'washing_machine': [], 'dryer': []},
        'dining': {'fans': [], 'ac': []},
        'summary': {'total_budget': user_data['total_budget'], 'used_budget': 0}
    }

    # Process hall requirements
    if user_data['hall'].get('ac', False):
        # print(f"[DEBUG] Hall AC requirement: {user_data['hall'].get('ac', False)}") # DEBUG
        budget_category = get_budget_category(user_data['total_budget'], 'ac')
        # Override the hall size to 200 sq ft for proper AC sizing, regardless of what was entered
        hall_size = 200.0  # Force hall size to 200 sq ft
        recommended_tonnage = determine_ac_tonnage(hall_size, 'hall')  # Pass 'hall' as room type for special handling
        # print(f"[DEBUG] Hall AC tonnage recommendation: {recommended_tonnage} Ton for {hall_size} sq ft (always using 2 Ton for hall)")
        required_features = {'tonnage': f"{recommended_tonnage} Ton"}
        recommendations = get_specific_product_recommendations('ac', budget_category, user_data['demographics'], user_data['hall'].get('color_theme'), user_data, required_features)
        final_list['hall']['ac'] = recommendations
    
    if user_data['hall'].get('fans', 0) > 0:
        budget_category = get_budget_category(user_data['total_budget'], 'ceiling_fan', quantity=user_data['hall']['fans'])
        recommendations = get_specific_product_recommendations('ceiling_fan', budget_category, user_data['demographics'], user_data['hall'].get('color_theme'), user_data)
        final_list['hall']['fans'] = recommendations
    
    # Process kitchen requirements
    if user_data['kitchen'].get('chimney_width'):
        budget_category = get_budget_category(user_data['total_budget'], 'chimney')
        required_features = {'dimensions': user_data['kitchen'].get('chimney_width')}
        recommendations = get_specific_product_recommendations('chimney', budget_category, user_data['demographics'], user_data['kitchen'].get('color_theme'), user_data, required_features)
        final_list['kitchen']['chimney'] = recommendations
    
    if user_data['kitchen'].get('refrigerator_capacity'):
        print("\n=== Refrigerator Debug ===")
        print(f"Required capacity: {user_data['kitchen'].get('refrigerator_capacity')}")
        catalog = load_product_catalog()
        refrigerators = [p for p in catalog.get('products', []) if p.get('product_type', '').lower() == 'refrigerator']
        print(f"Found {len(refrigerators)} refrigerators")
        if refrigerators:
            print(f"Sample: {refrigerators[0].get('brand')} - {refrigerators[0].get('title')}")
        final_list['kitchen']['refrigerator'] = refrigerators[:3]
        print(f"Added {len(final_list['kitchen']['refrigerator'])} refrigerators to recommendations")
        print("=== End Debug ===\n")
    
    if user_data['kitchen'].get('gas_stove_type'):
        gas_stove_type = user_data['kitchen'].get('gas_stove_type', '').strip()
        if "hob (built-in)" in gas_stove_type.lower():
            budget_category = get_budget_category(user_data['total_budget'], 'gas_stove')
            required_features = {'type': gas_stove_type, 'burners': user_data['kitchen'].get('num_burners', 4)}
            recommendations = get_specific_product_recommendations(
                'hob_top', budget_category, user_data['demographics'],
                user_data['kitchen'].get('color_theme'), user_data, required_features
            )
            # Deduplicate
            seen = set()
            deduped = []
            for p in recommendations:
                key = (p.get('brand', ''), p.get('title', ''))
                if key not in seen:
                    deduped.append(p)
                    seen.add(key)
            final_list['kitchen']['hob_top'] = deduped
            final_list['kitchen']['gas_stove'] = []
        elif "not needed" in gas_stove_type.lower():
            final_list['kitchen']['gas_stove'] = []
            final_list['kitchen']['hob_top'] = []
        else:
            budget_category = get_budget_category(user_data['total_budget'], 'gas_stove')
            required_features = {'type': gas_stove_type, 'burners': user_data['kitchen'].get('num_burners', 4)}
            recommendations = get_specific_product_recommendations(
                'gas_stove', budget_category, user_data['demographics'],
                user_data['kitchen'].get('color_theme'), user_data, required_features
            )
            final_list['kitchen']['gas_stove'] = recommendations
            final_list['kitchen']['hob_top'] = []
    
    if user_data['kitchen'].get('small_fan', False):
        # Recommend ceiling fans with blade_length of 60 cm or 90 cm
        def is_small_fan(product):
            if product.get('product_type', '').lower() != 'ceiling fan':
                return False
            features = product.get('features', {})
            # Check numeric_features
            numeric = features.get('numeric_features', {}) if isinstance(features, dict) else {}
            blade = numeric.get('blade_length') or numeric.get('blade length')
            if blade and isinstance(blade, dict):
                val = blade.get('value')
                unit = (blade.get('unit') or '').lower()
                if val is not None:
                    val_cm = val
                    if unit == 'mm':
                        val_cm = val / 10.0
                    elif unit == 'm':
                        val_cm = val * 100.0
                    if abs(val_cm - 60.0) <= 0.5 or abs(val_cm - 90.0) <= 0.5:
                        return True
            # Fallback: check parsed_features
            parsed = features.get('parsed_features', {}) if isinstance(features, dict) else {}
            blade_str = parsed.get('blade_length')
            if blade_str:
                match = re.match(r'(\d+(?:\.\d+)?)\s*cm', blade_str.lower())
                if match:
                    val_cm = float(match.group(1))
                    if abs(val_cm - 60.0) <= 0.5 or abs(val_cm - 90.0) <= 0.5:
                        return True
            return False
        small_fan_recommendations = [p for p in load_product_catalog().get('products', []) if is_small_fan(p)]
        final_list['kitchen']['small_fan'] = small_fan_recommendations[:3]  # Top 3 matches
    
    # Process master bedroom requirements
    if user_data['master_bedroom'].get('ac', False):
        # print(f"[DEBUG] Master Bedroom AC requirement: {user_data['master_bedroom'].get('ac', False)}") # DEBUG
        budget_category = get_budget_category(user_data['total_budget'], 'ac')
        # Calculate the recommended AC tonnage based on room size
        master_size = user_data['master_bedroom'].get('size_sqft', 140.0)  # Default to 140 sq ft if not specified
        recommended_tonnage = determine_ac_tonnage(master_size, 'master_bedroom')
        # print(f"[DEBUG] Master bedroom AC tonnage recommendation: {recommended_tonnage} Ton for {master_size} sq ft")
        required_features = {'tonnage': f"{recommended_tonnage} Ton"}
        recommendations = get_specific_product_recommendations('ac', budget_category, user_data['demographics'], user_data['master_bedroom'].get('color_theme'), user_data, required_features)
        final_list['master_bedroom']['ac'] = recommendations
    
    budget_category = get_budget_category(user_data['total_budget'], 'ceiling_fan')
    recommendations = get_specific_product_recommendations('ceiling_fan', budget_category, user_data['demographics'], user_data['master_bedroom'].get('color_theme'), user_data)
    final_list['master_bedroom']['fans'] = recommendations
    
    # Process master bedroom bathroom requirements
    master_bath = user_data['master_bedroom'].get('bathroom', {})
    master_bath_type = master_bath.get('water_heater_type', '')
    master_bath_ceiling = master_bath.get('water_heater_ceiling', '')
    geyser_recommendations = []
    if str(master_bath_type).strip().lower() == 'yes':
        budget_category = get_budget_category(user_data['total_budget'], 'geyser')
        # print(f"[DEBUG][Geyser] Calling get_specific_product_recommendations with appliance_type='geyser', budget_category={budget_category}, required_features={{}} (ceiling: {master_bath_ceiling})")
        geyser_recommendations = get_specific_product_recommendations('geyser', budget_category, user_data['demographics'], None, user_data, {}, 'master_bedroom_bathroom')
        # print(f"[DEBUG][Geyser] Number of geyser products returned: {len(geyser_recommendations)}")
    if str(master_bath_ceiling).strip().lower() == 'yes':
        budget_category = get_budget_category(user_data['total_budget'], 'geyser')
        # print(f"[DEBUG][Geyser] (Ceiling) Calling get_specific_product_recommendations with appliance_type='geyser', budget_category={budget_category}, required_features={{'type': 'horizontal'}}")
        geyser_recommendations = get_specific_product_recommendations('geyser', budget_category, user_data['demographics'], None, user_data, {'type': 'horizontal'}, 'master_bedroom_bathroom')
        # print(f"[DEBUG][Geyser] (Ceiling) Number of geyser products returned: {len(geyser_recommendations)}")
    final_list['master_bedroom']['bathroom']['water_heater'] = geyser_recommendations

    # Process exhaust fan for master (updated to use color)
    if user_data['master_bedroom'].get('bathroom') and user_data['master_bedroom']['bathroom'].get('exhaust_fan_size'):
        print("\n=== Master Bedroom Exhaust Fan Debug ===")
        print(f"Exhaust fan size: {user_data['master_bedroom']['bathroom'].get('exhaust_fan_size')}")
        print(f"Exhaust fan color: {user_data['master_bedroom']['bathroom'].get('exhaust_fan_color')}")
        budget_category = get_budget_category(user_data['total_budget'], 'bathroom_exhaust')
        required_features = {
            'dimensions': user_data['master_bedroom']['bathroom'].get('exhaust_fan_size'),
            'color': user_data['master_bedroom']['bathroom'].get('exhaust_fan_color')
        }
        recommendations = get_specific_product_recommendations('bathroom_exhaust', budget_category, user_data['demographics'], user_data['master_bedroom'].get('color_theme'), user_data, required_features)
        print(f"Found {len(recommendations)} recommendations")
        final_list['master_bedroom']['bathroom']['exhaust_fan'] = recommendations
        print("=== End Master Bedroom Exhaust Fan Debug ===\n")

    # Process bedroom 2 requirements
    if user_data['bedroom_2'].get('ac', False):
        # print(f"[DEBUG] Bedroom 2 AC requirement: {user_data['bedroom_2'].get('ac', False)}") # DEBUG
        budget_category = get_budget_category(user_data['total_budget'], 'ac')
        # Calculate the recommended AC tonnage based on room size
        bedroom2_size = user_data['bedroom_2'].get('size_sqft', 120.0)  # Default to 120 sq ft if not specified
        recommended_tonnage = determine_ac_tonnage(bedroom2_size, 'bedroom_2')
        # print(f"[DEBUG] Bedroom 2 AC tonnage recommendation: {recommended_tonnage} Ton for {bedroom2_size} sq ft")
        required_features = {'tonnage': f"{recommended_tonnage} Ton"}
        recommendations = get_specific_product_recommendations('ac', budget_category, user_data['demographics'], user_data['bedroom_2'].get('color_theme'), user_data, required_features)
        final_list['bedroom_2']['ac'] = recommendations
    
    budget_category = get_budget_category(user_data['total_budget'], 'ceiling_fan')
    recommendations = get_specific_product_recommendations('ceiling_fan', budget_category, user_data['demographics'], user_data['bedroom_2'].get('color_theme'), user_data)
    final_list['bedroom_2']['fans'] = recommendations
    
    # Process bedroom 2 bathroom requirements
    bedroom2_bath = user_data['bedroom_2'].get('bathroom', {})
    bedroom2_bath_type = bedroom2_bath.get('water_heater_type', '')
    bedroom2_bath_ceiling = bedroom2_bath.get('water_heater_ceiling', '')
    geyser_recommendations_2 = []
    if str(bedroom2_bath_type).strip().lower() == 'yes':
        budget_category = get_budget_category(user_data['total_budget'], 'geyser')
        # If ceiling is also yes, prefer horizontal
        if str(bedroom2_bath_ceiling).strip().lower() == 'yes':
            # First, try to get only horizontal water heaters
            all_geysers = get_specific_product_recommendations('geyser', budget_category, user_data['demographics'], user_data['bedroom_2'].get('color_theme'), user_data)
            horizontal_geysers = [g for g in all_geysers if is_horizontal_water_heater(g)]
            if horizontal_geysers:
                geyser_recommendations_2 = horizontal_geysers
            else:
                geyser_recommendations_2 = all_geysers
        else:
            geyser_recommendations_2 = get_specific_product_recommendations('geyser', budget_category, user_data['demographics'], user_data['bedroom_2'].get('color_theme'), user_data)
        final_list['bedroom_2']['bathroom']['water_heater'] = geyser_recommendations_2

    # Process exhaust fan for bedroom 2 (updated to use color)
    if user_data['bedroom_2'].get('bathroom') and user_data['bedroom_2']['bathroom'].get('exhaust_fan_size'):
        print("\n=== Bedroom 2 Exhaust Fan Debug ===")
        print(f"Exhaust fan size: {user_data['bedroom_2']['bathroom'].get('exhaust_fan_size')}")
        print(f"Exhaust fan color: {user_data['bedroom_2']['bathroom'].get('exhaust_fan_color')}")
        budget_category = get_budget_category(user_data['total_budget'], 'bathroom_exhaust')
        required_features = {
            'dimensions': user_data['bedroom_2']['bathroom'].get('exhaust_fan_size'),
            'color': user_data['bedroom_2']['bathroom'].get('exhaust_fan_color')
        }
        recommendations = get_specific_product_recommendations('bathroom_exhaust', budget_category, user_data['demographics'], user_data['bedroom_2'].get('color_theme'), user_data, required_features)
        print(f"Found {len(recommendations)} recommendations")
        final_list['bedroom_2']['bathroom']['exhaust_fan'] = recommendations
        print("=== End Bedroom 2 Exhaust Fan Debug ===\n")

######
    # Process bedroom 3 requirements
    num_bedrooms = int(user_data.get('num_bedrooms', user_data.get('Number of bedrooms', 2)))
    if num_bedrooms == 3:
        if user_data['bedroom_3'].get('ac', False):
            # print(f"[DEBUG] Bedroom 2 AC requirement: {user_data['bedroom_2'].get('ac', False)}") # DEBUG
            budget_category = get_budget_category(user_data['total_budget'], 'ac')
            # Calculate the recommended AC tonnage based on room size
            bedroom3_size = user_data['bedroom_3'].get('size_sqft', 120.0)  # Default to 120 sq ft if not specified
            recommended_tonnage = determine_ac_tonnage(bedroom3_size, 'bedroom_3')
            # print(f"[DEBUG] Bedroom 2 AC tonnage recommendation: {recommended_tonnage} Ton for {bedroom2_size} sq ft")
            required_features = {'tonnage': f"{recommended_tonnage} Ton"}
            recommendations = get_specific_product_recommendations('ac', budget_category, user_data['demographics'], user_data['bedroom_3'].get('color_theme'), user_data, required_features)
            final_list['bedroom_3']['ac'] = recommendations
    
        budget_category = get_budget_category(user_data['total_budget'], 'ceiling_fan')
        recommendations = get_specific_product_recommendations('ceiling_fan', budget_category, user_data['demographics'], user_data['bedroom_3'].get('color_theme'), user_data)
        final_list['bedroom_3']['fans'] = recommendations
        
        # Process bedroom 3 bathroom requirements
        bedroom3_bath = user_data['bedroom_3'].get('bathroom', {})
        bedroom3_bath_type = bedroom3_bath.get('water_heater_type', '')
        bedroom3_bath_ceiling = bedroom3_bath.get('water_heater_ceiling', '')

        geyser_recommendations_3 = []
        if str(bedroom3_bath_type).strip().lower() == 'yes':
            budget_category = get_budget_category(user_data['total_budget'], 'geyser')
            # Only recommend horizontal water heaters if ceiling is explicitly "Yes"
            ceiling_condition = str(bedroom3_bath_ceiling).strip().lower()
            
            if ceiling_condition == 'yes':
                # First, try to get only horizontal water heaters
                all_geysers = get_specific_product_recommendations('geyser', budget_category, user_data['demographics'], user_data['bedroom_3'].get('color_theme'), user_data)
                
                horizontal_geysers = [g for g in all_geysers if is_horizontal_water_heater(g)]
                
                if horizontal_geysers:
                    geyser_recommendations_3 = horizontal_geysers
                else:
                    # If no horizontal water heaters found, try to find slim or RHS water heaters
                    slim_geysers = [g for g in all_geysers if 'slim' in g.get('title', '').lower() or 'rhs' in g.get('title', '').lower()]
                    geyser_recommendations_3 = slim_geysers if slim_geysers else all_geysers
            else:
                # For non-ceiling installations, get all water heaters and filter out horizontal ones
                all_geysers = get_specific_product_recommendations('geyser', budget_category, user_data['demographics'], user_data['bedroom_3'].get('color_theme'), user_data)
                
                geyser_recommendations_3 = [g for g in all_geysers if not is_horizontal_water_heater(g)]
            
            final_list['bedroom_3']['bathroom']['water_heater'] = geyser_recommendations_3

        # Process exhaust fan for bedroom 3 (updated to use color)
        if user_data['bedroom_3'].get('bathroom') and user_data['bedroom_3']['bathroom'].get('exhaust_fan_size'):
            print("\n=== Bedroom 3 Exhaust Fan Debug ===")
            print(f"Exhaust fan size: {user_data['bedroom_3']['bathroom'].get('exhaust_fan_size')}")
            print(f"Exhaust fan color: {user_data['bedroom_3']['bathroom'].get('exhaust_fan_color')}")
            budget_category = get_budget_category(user_data['total_budget'], 'bathroom_exhaust')
            required_features = {
                'dimensions': user_data['bedroom_3']['bathroom'].get('exhaust_fan_size'),
                'color': user_data['bedroom_3']['bathroom'].get('exhaust_fan_color')
            }
            recommendations = get_specific_product_recommendations('bathroom_exhaust', budget_category, user_data['demographics'], user_data['bedroom_3'].get('color_theme'), user_data, required_features)
            print(f"Found {len(recommendations)} recommendations")
            final_list['bedroom_3']['bathroom']['exhaust_fan'] = recommendations
            print("=== End Bedroom 3 Exhaust Fan Debug ===\n")
 
######
    # Process laundry requirements
    user_selected_wm_type = str(user_data['laundry'].get('washing_machine_type', '')).strip()

    # Check if a specific type of washing machine is selected by the user
    # The values from HTML form/Excel are expected to be "Front-Load", "Top-Load", "Semi-Automatic"
    if user_selected_wm_type and user_selected_wm_type.lower() not in ['yes', 'no', 'none', '', 'not applicable', 'na']:
        budget_category = get_budget_category(user_data['total_budget'], 'washing_machine')
        
        required_features = {
            'capacity': estimate_washing_machine_capacity(user_data['demographics']),
            'type': user_selected_wm_type  # Pass the user's selection
        }
        
        recommendations = get_specific_product_recommendations(
            'washing_machine', 
            budget_category, 
            user_data['demographics'], 
            user_data['laundry'].get('color_theme'), 
            user_data, 
            required_features
        )
        final_list['laundry']['washing_machine'] = recommendations
    elif user_selected_wm_type.lower() == 'yes': # Fallback for old 'yes' logic if it's still possible
        budget_category = get_budget_category(user_data['total_budget'], 'washing_machine')
        required_features = {'capacity': estimate_washing_machine_capacity(user_data['demographics'])}
        recommendations = get_specific_product_recommendations('washing_machine', budget_category, user_data['demographics'], user_data['laundry'].get('color_theme'), user_data, required_features)
        final_list['laundry']['washing_machine'] = recommendations
    else: # User selected "No", "None", or it's empty, so no washing machine
        final_list['laundry']['washing_machine'] = []
    # DEBUG print statement left for troubleshooting - Dryer recommendations are now handled in one place below
    # print(f"[DEBUG DRYER_TYPE] Current dryer_type value: '{user_data['laundry'].get('dryer_type')}', Lowercased check value: '{str(user_data['laundry'].get('dryer_type', '')).strip().lower()}'")
    # Dryer recommendations will be processed in a single location further below
  
    # Process dining room requirements
    if user_data['dining'].get('fan_size'):
        fan_size_str = str(user_data['dining'].get('fan_size', '')).strip().upper()
        size_cm = None
        # Handle different unit formats
        if 'CM' in fan_size_str:
            size_cm = float(fan_size_str.replace('CM', '').strip())
        elif 'MM' in fan_size_str:
            size_cm = float(fan_size_str.replace('MM', '').strip()) / 10.0
        elif fan_size_str.isdigit():
            size_cm = float(fan_size_str)
        # For dining rooms, if no specific size is given, default to 120cm (1200mm)
        if not size_cm:
            size_cm = 120.0
        required_features = {
            'fan_size_cm': size_cm
        }
        print(f"[DEBUG DINING FAN] fan_size_str={fan_size_str}, size_cm={size_cm}, required_features={required_features}")
        budget_category = get_budget_category(user_data['total_budget'], 'ceiling_fan')
        def matches_blade_length_precise(product):
            features = product.get('features', {})
            # Prefer numeric_features if available
            numeric = features.get('numeric_features', {}) if isinstance(features, dict) else {}
            blade = numeric.get('blade_length') or numeric.get('blade length')
            if blade and isinstance(blade, dict):
                val = blade.get('value')
                unit = (blade.get('unit') or '').lower()
                if val is not None:
                    # Convert to cm if needed
                    val_cm = val
                    if unit == 'mm':
                        val_cm = val / 10.0
                    elif unit == 'm':
                        val_cm = val * 100.0
                    # Allow small tolerance (±0.5cm)
                    if abs(val_cm - size_cm) <= 0.5:
                        return True
            # Fallback: check parsed_features
            parsed = features.get('parsed_features', {}) if isinstance(features, dict) else {}
            blade_str = parsed.get('blade_length')
            if blade_str:
                match = re.match(r'(\d+(?:\.\d+)?)\s*cm', blade_str.lower())
                if match:
                    val_cm = float(match.group(1))
                    if abs(val_cm - size_cm) <= 0.5:
                        return True
            return False
        recommendations = [p for p in load_product_catalog().get('products', []) if p.get('product_type', '').lower() == 'ceiling fan' and matches_blade_length_precise(p)]
        # Optionally, sort or limit recommendations as before
        final_list['dining']['fans'] = recommendations[:3]  # Top 3 matches

    if user_data['dining'].get('ac'):
        budget_category = get_budget_category(user_data['total_budget'], 'ac')
        recommendations = get_specific_product_recommendations('ac', budget_category, user_data['demographics'], user_data['dining'].get('color_theme'))
        final_list['dining']['ac'] = recommendations

    # Add logic to recommend dishwasher
    if user_data['kitchen'].get('dishwasher_capacity'):
        budget_category = get_budget_category(user_data['total_budget'], 'dishwasher')
        required_features = {'capacity': user_data['kitchen'].get('dishwasher_capacity')}
        recommendations = get_specific_product_recommendations('dishwasher', budget_category, user_data['demographics'], user_data['kitchen'].get('color_theme'), user_data, required_features)
        final_list['kitchen']['dishwasher'] = recommendations

    # Add dryer recommendations if needed - standardized logic
    if user_data['laundry'].get('dryer_type', '').lower() == 'yes': # Standardized to use dryer_type
        print("\n=== Dryer Debug ===")
        print(f"Looking for dryer products. Dryer type value: '{user_data['laundry'].get('dryer_type')}'")
        
        budget_category = get_budget_category(user_data['total_budget'], 'dryer')
        # Standardized to use 'dryer' appliance type
        recommendations = get_specific_product_recommendations(
            'dryer', 
            budget_category, 
            user_data['demographics'], 
            user_data['laundry'].get('color_theme'), 
            user_data
        )
        print(f"Found {len(recommendations)} dryers before deduplication")
        
        # Check for identical products in recommendations
        dryer_ids = {}
        for dryer in recommendations:
            dryer_id = f"{dryer.get('brand', 'Unknown')}_{dryer.get('title', 'Unknown')}"
            if dryer_id in dryer_ids:
                print(f"  WARNING: Duplicate dryer found in recommendations: {dryer_id}")
                dryer_ids[dryer_id] += 1
            else:
                dryer_ids[dryer_id] = 1
        
        # Create a debug list of dryer products before deduplication
        for i, dryer in enumerate(recommendations):
            print(f"  Dryer {i+1}: {dryer.get('brand', 'Unknown')} - {dryer.get('title', 'Unknown')}")
        
        # Use standard deduplicate_recommendations function
        final_list['laundry']['dryer'] = deduplicate_recommendations(recommendations)
        print(f"Added {len(final_list['laundry']['dryer'])} unique dryers to recommendations")
        
        # Debug the final deduped list
        for i, dryer in enumerate(final_list['laundry']['dryer']):
            print(f"  Final Dryer {i+1}: {dryer.get('brand', 'Unknown')} - {dryer.get('title', 'Unknown')}")
            
        print("=== End Dryer Debug ===\n")

 

    # Master Bedroom Bathroom
    if user_data['master_bedroom'].get('bathroom', {}).get('water_heater_type'):
        # print("\n=== Water Heater Debug ===")
        # print(f"Master Bedroom Bathroom - False Ceiling: {user_data['master_bedroom']['bathroom'].get('water_heater_ceiling', False)}")
        budget_category = get_budget_category(user_data['total_budget'], 'geyser')
        master_bath_ceiling = str(user_data['master_bedroom']['bathroom'].get('water_heater_ceiling', '')).strip().lower()
        all_geysers = get_specific_product_recommendations(
            'geyser', budget_category, user_data['demographics'],
            user_data['master_bedroom'].get('color_theme'), user_data,
            {'water_heater_ceiling': master_bath_ceiling}, 'master_bedroom'
        )
        if master_bath_ceiling == 'yes':
            # For ceiling installations, prefer horizontal water heaters
            horizontal_geysers = [g for g in all_geysers if is_horizontal_water_heater(g)]
            if horizontal_geysers:
                recommendations = horizontal_geysers
            else:
                # If no horizontal water heaters found, try to find slim or RHS water heaters
                slim_geysers = [g for g in all_geysers if 'slim' in g.get('title', '').lower() or 'rhs' in g.get('title', '').lower()]
                recommendations = slim_geysers if slim_geysers else all_geysers
        else:
            # For non-ceiling installations, filter out horizontal water heaters
            recommendations = [g for g in all_geysers if not is_horizontal_water_heater(g)]
        final_list['master_bedroom']['bathroom']['water_heater'] = deduplicate_recommendations(recommendations)
        # print(f"Added {len(final_list['master_bedroom']['bathroom']['water_heater'])} unique water heaters to recommendations")
        # print("=== End Water Heater Debug ===\n")

    # Add LED mirror recommendations for master bedroom bathroom
    if user_data['master_bedroom'].get('bathroom', {}).get('led_mirror', False):
        # print("\n=== LED Mirror Debug ===")
        # print("Master Bedroom Bathroom - LED Mirror requested")
        budget_category = get_budget_category(user_data['total_budget'], 'led_mirror')
        recommendations = get_specific_product_recommendations(
            'led_mirror', budget_category, user_data['demographics'],
            user_data['master_bedroom'].get('color_theme'), user_data, None, 'master_bedroom'
        )
        # print(f"Found {len(recommendations)} LED mirrors")
        final_list['master_bedroom']['bathroom']['led_mirror'] = deduplicate_recommendations(recommendations)
        # print(f"Added {len(final_list['master_bedroom']['bathroom']['led_mirror'])} unique LED mirrors to recommendations")
        # print("=== End LED Mirror Debug ===\n")

    # Bedroom 2 Bathroom
    if user_data['bedroom_2'].get('bathroom', {}).get('water_heater_type'):
        # print("\n=== Water Heater Debug ===")
        # print(f"Bedroom 2 Bathroom - False Ceiling: {user_data['bedroom_2']['bathroom'].get('water_heater_ceiling', False)}")
        budget_category = get_budget_category(user_data['total_budget'], 'geyser')
        bedroom2_bath_ceiling = str(user_data['bedroom_2']['bathroom'].get('water_heater_ceiling', '')).strip().lower()
        
        # Get all geysers first
        all_geysers = get_specific_product_recommendations(
            'geyser', budget_category, user_data['demographics'],
            user_data['bedroom_2'].get('color_theme'), user_data,
            {'water_heater_ceiling': bedroom2_bath_ceiling}, 'bedroom_2'
        )
        # print(f"[DEBUG] Found {len(all_geysers)} total geysers")
        
        if bedroom2_bath_ceiling == 'yes':
            # For ceiling installations, prefer horizontal water heaters
            horizontal_geysers = [g for g in all_geysers if is_horizontal_water_heater(g)]
            # print(f"[DEBUG] Found {len(horizontal_geysers)} horizontal geysers")
            recommendations = horizontal_geysers if horizontal_geysers else all_geysers
        else:
            # For non-ceiling installations, filter out horizontal water heaters
            vertical_geysers = [g for g in all_geysers if not is_horizontal_water_heater(g)]
            # print(f"[DEBUG] Found {len(vertical_geysers)} vertical geysers")
            recommendations = vertical_geysers if vertical_geysers else all_geysers
        
        final_list['bedroom_2']['bathroom']['water_heater'] = deduplicate_recommendations(recommendations)
        # print(f"Added {len(final_list['bedroom_2']['bathroom']['water_heater'])} unique water heaters to recommendations")
        # print("=== End Water Heater Debug ===\n")

    # Add LED mirror recommendations for bedroom 2 bathroom
    if user_data['bedroom_2'].get('bathroom', {}).get('led_mirror', False):
        # print("\n=== LED Mirror Debug ===")
        # print("Bedroom 2 Bathroom - LED Mirror requested")
        budget_category = get_budget_category(user_data['total_budget'], 'led_mirror')
        recommendations = get_specific_product_recommendations(
            'led_mirror', budget_category, user_data['demographics'],
            user_data['bedroom_2'].get('color_theme'), user_data, None, 'bedroom_2'
        )
        # print(f"Found {len(recommendations)} LED mirrors")
        final_list['bedroom_2']['bathroom']['led_mirror'] = deduplicate_recommendations(recommendations)
        # print(f"Added {len(final_list['bedroom_2']['bathroom']['led_mirror'])} unique LED mirrors to recommendations")
        # print("=== End LED Mirror Debug ===\n")
#####
   # Bedroom 3 Bathroom
    if user_data['bedroom_3'].get('bathroom', {}).get('water_heater_type'):
        # print("\n=== Water Heater Debug ===")
        # print(f"Bedroom 3 Bathroom - False Ceiling: {user_data['bedroom_3']['bathroom'].get('water_heater_ceiling', False)}")
        budget_category = get_budget_category(user_data['total_budget'], 'geyser')
        bedroom3_bath_ceiling = str(user_data['bedroom_3']['bathroom'].get('water_heater_ceiling', '')).strip().lower()
        
        # Get all geysers first
        all_geysers = get_specific_product_recommendations(
            'geyser', budget_category, user_data['demographics'],
            user_data['bedroom_3'].get('color_theme'), user_data,
            {'water_heater_ceiling': bedroom3_bath_ceiling}, 'bedroom_3'
        )
        # print(f"[DEBUG] Found {len(all_geysers)} total geysers")
        
        if bedroom3_bath_ceiling == 'yes':
            # For ceiling installations, prefer horizontal water heaters
            horizontal_geysers = [g for g in all_geysers if is_horizontal_water_heater(g)]
            # print(f"[DEBUG] Found {len(horizontal_geysers)} horizontal geysers")
            recommendations = horizontal_geysers if horizontal_geysers else all_geysers
        else:
            # For non-ceiling installations, filter out horizontal water heaters
            vertical_geysers = [g for g in all_geysers if not is_horizontal_water_heater(g)]
            # print(f"[DEBUG] Found {len(vertical_geysers)} vertical geysers")
            recommendations = vertical_geysers if vertical_geysers else all_geysers
        
        final_list['bedroom_3']['bathroom']['water_heater'] = deduplicate_recommendations(recommendations)
        # print(f"Added {len(final_list['bedroom_3']['bathroom']['water_heater'])} unique water heaters to recommendations")
        # print("=== End Water Heater Debug ===\n")

    # Add LED mirror recommendations for bedroom 3 bathroom
    if user_data['bedroom_3'].get('bathroom', {}).get('led_mirror', False):
        # print("\n=== LED Mirror Debug ===")
        # print("Bedroom 3 Bathroom - LED Mirror requested")
        budget_category = get_budget_category(user_data['total_budget'], 'led_mirror')
        recommendations = get_specific_product_recommendations(
            'led_mirror', budget_category, user_data['demographics'],
            user_data['bedroom_3'].get('color_theme'), user_data, None, 'bedroom_3'
        )
        #  print(f"Found {len(recommendations)} LED mirrors")
        final_list['bedroom_3']['bathroom']['led_mirror'] = deduplicate_recommendations(recommendations)
        # print(f"Added {len(final_list['bedroom_3']['bathroom']['led_mirror'])} unique LED mirrors to recommendations")
        # print("=== End LED Mirror Debug ===\n")

#####
    if num_bedrooms == 2:
        del final_list['bedroom_3']
    return final_list


# Function to generate an HTML file with recommendations
def generate_html_file(user_data: Dict[str, Any], final_list: Dict[str, Any], html_filename: str) -> None:
    #import pprint
    #pprint.pprint(final_list)
    # print("[DEBUG HTML] final_list keys:", list(final_list.keys()))
    """Generate an HTML file with user information and product recommendations."""
    
    # First check if the script is being run through Flask
    is_web_app = os.environ.get('BETTERHOME_WEB_APP') == 'true'
    
    # Check if logo exists in multiple possible locations
    possible_logo_paths = [
        "web_app/better_home_logo.png",  # Relative to script execution directory
        "./web_app/better_home_logo.png",  # Explicit relative path
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "better_home_logo.png"),  # Same directory as script
        "better_home_logo.png",  # Check in current directory as well
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
    
    # For web app, use a URL path rather than a file path
    logo_html = ""
    if logo_exists:
        if is_web_app:
            # Use a relative URL path that will be handled by Flask
            logo_html = '<img src="/static/better_home_logo.png" alt="BetterHome Logo" class="logo">'
        else:
            # Use the file path for direct HTML viewing
            logo_html = f'<img src="{logo_path}" alt="BetterHome Logo" class="logo">'
    
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
                grid-template-columns: repeat(2, 1fr);
                gap: 15px;
            }
            
            @media (min-width: 768px) {
                .client-info {
                    grid-template-columns: repeat(3, 1fr);
                }
            }
            
            .client-info-item {
                margin-bottom: 0;
            }
            
            .client-info-label {
                font-weight: 500;
                color: #666;
                margin-bottom: 4px;
            }
            
            .client-info-value {
                font-weight: 600;
                color: #333;
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
                    grid-template-columns: repeat(3, 1fr);  // Change to 3 columns
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
            }

            .product-card.best-product {
                border: 2px solid #3498db;  // Emphasize the best product
                transform: scale(1.05);  // Slightly enlarge the best product
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
            
            /* Bestseller badge styling */
            .bestseller-badge {
                position: absolute;
                top: 10px;
                right: 10px;
                background-color: #ff6b00;
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
            
            .bestseller-badge i {
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
                    grid-template-columns: repeat(3, 1fr);
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
            
            /* Footer styling */
            footer {
                margin-top: 50px;
                padding-top: 20px;
                border-top: 1px solid #eaeaea;
                text-align: center;
                color: #7f8c8d;
                font-size: 14px;
            }
        </style>
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
                    <div class="client-info-value">₹{user_data['total_budget']:,.2f}</div>
                </div>
                
                <div class="client-info-item">
                    <div class="client-info-label">Family Size</div>
                    <div class="client-info-value">{sum(user_data['demographics'].values())} members</div>
                </div>
            </div>
    """
    html_content += client_info_section

    # Add budget summary
    total_cost = calculate_total_cost(final_list)
    budget_utilization = (total_cost / user_data['total_budget']) * 100
    
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

    # Add logic to ensure at least three recommendations are displayed
    def ensure_three_recommendations(products, allow_duplicates=False):
        """
        Ensure that there are at least three product recommendations.
        
        If allow_duplicates is True and there are fewer than 3 products, 
        duplicate existing products to reach 3 total.
        Otherwise, return the original list without duplication.
        """
        if allow_duplicates and len(products) < 3 and len(products) > 0:
            # Only duplicate if explicitly allowed and there's at least one product
            products = products.copy()  # Work on a copy to avoid modifying the original
            products.extend(products[:3 - len(products)])  # Duplicate some products if less than 3
        return products

    # Process each room
    for room, appliances in final_list.items():
        if room == 'summary':
            continue
        
        # Debug: Print what's in the kitchen section
        if room == 'kitchen':
            # print(f"[DEBUG HTML] Kitchen appliances keys: {list(appliances.keys())}")
            # for key, value in appliances.items():
            #     print(f"[DEBUG HTML] {key}: {len(value)} items")
            #     if key == 'hob_top':
            #         for item in value:
            #             print(f"[DEBUG HTML] Hob top item: {item.get('brand', 'Unknown')} - {item.get('model', item.get('title', 'Unknown'))}")
            pass
        
        # Check if the room has any products before creating the section
        has_products = False
        for appliance_type, products in appliances.items():
            if isinstance(products, list) and products:
                has_products = True
                break
        
        if not has_products:
            continue
            
        room_title = room.replace('_', ' ').title()
        html_content += f"""
            <div class="room-section">
                <h2>{room_title}</h2>
        """
        
        # Add room description
        room_desc = get_room_description(room, user_data)
        if room_desc:
            html_content += f'                <div class="room-description">{room_desc}</div>\n'
        
        html_content += """
                <div class="products-grid">
        """
        
        for appliance_type, products in appliances.items():
            # Ensure products is a list
            if not isinstance(products, list):
                continue
                
            # For dryers, don't duplicate products; for other types, it's fine
            allow_duplication = appliance_type != 'dryer'
            products = ensure_three_recommendations(products, allow_duplicates=allow_duplication)
                
            # Check if products list is not empty
            if not products:
                continue
            # Sort products to find the best one
            products.sort(key=lambda x: -x.get('feature_match_score', 0))
            best_product = products[0]  # Assume the first one is the best after sorting

            # Reorder products to place the best product in the middle
            if len(products) >= 3:
                products = [products[1], best_product, products[2]]

            for idx, product in enumerate(products):
                if not isinstance(product, dict):
                    continue
                
                # Ensure required data is available
                brand = product.get('brand', 'Unknown Brand')
                model = product.get('model', product.get('title', 'Unknown Model'))
                image_src = product.get('image_src', 'https://via.placeholder.com/300x300?text=No+Image+Available')
                description = product.get('description', 'No description available')
                
                # Use the correct pricing fields - better_home_price as the current price and retail_price as the original price
                better_home_price = float(product.get('better_home_price', 0.0))
                retail_price = float(product.get('retail_price', 0.0))
                
                # If better_home_price is missing or 0, use price as fallback
                if better_home_price <= 0:
                    better_home_price = float(product.get('price', retail_price * 0.8))  # Estimate if missing
                
                # If retail_price is missing or 0, estimate from better_home_price
                if retail_price <= 0:
                    retail_price = better_home_price * 1.25  # Estimate a 25% markup if missing
                
                # Ensure retail price is higher than better home price for proper display
                if retail_price <= better_home_price:
                    retail_price = better_home_price * 1.25  # Ensure a reasonable markup
                
                savings = retail_price - better_home_price
                warranty = product.get('warranty', 'Standard warranty applies')
                delivery_time = product.get('delivery_time', 'Contact store for details')
                purchase_url = product.get('url', '#')
                
                # Get formatted product type for display
                product_type_title = appliance_type.replace('_', ' ').title()
                
                # Get recommendation reason
                reason_text = get_product_recommendation_reason(
                    product, 
                    appliance_type, 
                    room, 
                    user_data['demographics'],
                    user_data['total_budget'],
                    {}  # Required features
                )
                
                # Parse reasons into a list for better display
                reasons = [r.strip() for r in reason_text.split('•') if r.strip()]
                
                # Check if product is a bestseller
                bestseller_badge = ""
                if product.get('is_bestseller', False):
                    bestseller_badge = '<div class="bestseller-badge"><i class="fa fa-star"></i> BESTSELLER</div>'

                # Format prices for display
                better_home_price_num = float(better_home_price)
                retail_price_num = float(retail_price)

                better_home_price = f"₹{better_home_price_num:,.2f}"
                retail_price = f"₹{retail_price_num:,.2f}"
                savings = f"₹{retail_price_num - better_home_price_num:,.2f}"

                # Calculate savings percentage
                savings_pct = 0
                if retail_price_num > 0:
                    savings_pct = ((retail_price_num - better_home_price_num) / retail_price_num) * 100

                # Add an icon for the reason text
                reasons_with_icons = []
                for reason in reasons:
                    icon = "check-circle"  # Default icon

                    # Choose different icons based on keywords in the reason
                    if any(keyword in reason.lower() for keyword in ["save", "budget", "price"]):
                        icon = "money-bill-wave"
                    elif any(keyword in reason.lower() for keyword in ["energy", "efficient", "power", "consumption"]):
                        icon = "leaf"
                    elif any(keyword in reason.lower() for keyword in ["feature", "advanced", "smart"]):
                        icon = "cogs"
                    elif any(keyword in reason.lower() for keyword in ["quality", "durable", "reliable"]):
                        icon = "medal"
                    elif any(keyword in reason.lower() for keyword in ["popular", "bestseller", "best-selling"]):
                        icon = "star"

                    reasons_with_icons.append(f'<li><i class="fas fa-{icon}"></i> {reason}</li>')

                # Determine if this is the best product and apply special styling
                best_class = " best-product" if product == best_product else ""

                # Create the HTML for the product card
                product_html = f'''
                    <div class="product-card{best_class}">
                        <div class="product-image-container">
                        <img class="product-image" src="{image_src}" alt="{brand} {model}">
                        {bestseller_badge}
                        </div>
                        <div class="product-details">
                        <span class="product-type">{appliance_type.replace('_', ' ').upper()}</span>
                        <h3 class="product-title">{brand} {model}</h3>
                            <div class="price-container">
                            <span class="current-price">{better_home_price}</span>
                            <span class="retail-price">{retail_price}</span>
                            <span class="savings">Save {savings} ({savings_pct:.0f}%)</span>
                            </div>
                            <div class="product-info-item">
                                <span class="product-info-label">Description:</span> {product.get('concise_description', product.get('description', 'No description available'))}
                            </div>
                            <div class="product-info-item">
                            <span class="product-info-label">Warranty:</span> {product.get('warranty', 'Standard warranty')}
                            </div>
                            <h4>Why We Recommend This:</h4>
                            <ul class="reasons-list">
                            {"".join(reasons_with_icons)}
                            </ul>
                        <a href="{product.get('url', '#')}" class="buy-button" target="_blank">View Details</a>
                        </div>
                    </div>
                '''
                html_content += product_html
        
        html_content += """
                </div>
            </div>
        """

        # NEW: Render bathroom appliances for bedrooms
        if room in ['master_bedroom', 'bedroom_2', 'bedroom_3'] and 'bathroom' in appliances:
            for bath_appliance_type, bath_products in appliances['bathroom'].items():
                # print(f"[DEBUG HTML] {room} bathroom {bath_appliance_type}: {bath_products}")
                if not isinstance(bath_products, list) or not bath_products:
                    continue
                bath_title = bath_appliance_type.replace('_', ' ').title()
                html_content += f"<h3>{bath_title} Recommendations</h3>\n<div class='products-grid'>"
                
                # Don't duplicate products for specific categories
                allow_bath_duplication = bath_appliance_type not in ['dryer', 'water_heater', 'exhaust_fan']
                bath_products = ensure_three_recommendations(bath_products, allow_duplicates=allow_bath_duplication)
                
                bath_products.sort(key=lambda x: -x.get('feature_match_score', 0))
                best_product = bath_products[0]
                if len(bath_products) >= 3:
                    bath_products = [bath_products[1], best_product, bath_products[2]]
                for idx, product in enumerate(bath_products):
                    if not isinstance(product, dict):
                        continue
                    brand = product.get('brand', 'Unknown Brand')
                    model = product.get('model', product.get('title', 'Unknown Model'))
                    image_src = product.get('image_src', 'https://via.placeholder.com/300x300?text=No+Image+Available')
                    description = product.get('description', 'No description available')
                    better_home_price = float(product.get('better_home_price', 0.0))
                    retail_price = float(product.get('retail_price', 0.0))
                    if better_home_price <= 0:
                        better_home_price = float(product.get('price', retail_price * 0.8))
                    if retail_price <= 0:
                        retail_price = better_home_price * 1.25
                    if retail_price <= better_home_price:
                        retail_price = better_home_price * 1.25
                    savings = retail_price - better_home_price
                    warranty = product.get('warranty', 'Standard warranty applies')
                    delivery_time = product.get('delivery_time', 'Contact store for details')
                    purchase_url = product.get('url', '#')
                    product_type_title = bath_appliance_type.replace('_', ' ').title()

                    # Matched color display
                    matched_color_html = ""
                    if product.get('matched_color'):
                        matched_color_html = f'''
                        <div class="product-info-item">
                            <span class="product-info-label">Matched Color:</span> {product['matched_color']}
                        </div>'''

                    reason_text = get_product_recommendation_reason(
                        product,
                        bath_appliance_type,
                        room,
                        user_data['demographics'],
                        user_data['total_budget'],
                        {}
                    )
                    reasons = [r.strip() for r in reason_text.split('•') if r.strip()]
                    bestseller_badge = ""
                    if product.get('is_bestseller', False):
                        bestseller_badge = '<div class="bestseller-badge"><i class="fa fa-star"></i> BESTSELLER</div>'
                    better_home_price_num = float(better_home_price)
                    retail_price_num = float(retail_price)
                    better_home_price = f"₹{better_home_price_num:,.2f}"
                    retail_price = f"₹{retail_price_num:,.2f}"
                    savings = f"₹{retail_price_num - better_home_price_num:,.2f}"
                    savings_pct = 0
                    if retail_price_num > 0:
                        savings_pct = ((retail_price_num - better_home_price_num) / retail_price_num) * 100
                    reasons_with_icons = []
                    for reason in reasons:
                        icon = "check-circle"
                        if any(keyword in reason.lower() for keyword in ["save", "budget", "price"]):
                            icon = "money-bill-wave"
                        elif any(keyword in reason.lower() for keyword in ["energy", "efficient", "power", "consumption"]):
                            icon = "leaf"
                        elif any(keyword in reason.lower() for keyword in ["feature", "advanced", "smart"]):
                            icon = "cogs"
                        elif any(keyword in reason.lower() for keyword in ["quality", "durable", "reliable"]):
                            icon = "medal"
                        elif any(keyword in reason.lower() for keyword in ["popular", "bestseller", "best-selling"]):
                            icon = "star"
                        reasons_with_icons.append(f'<li><i class="fas fa-{icon}"></i> {reason}</li>')
                    best_class = " best-product" if product == best_product else ""
                    product_html = f'''
                        <div class="product-card{best_class}">
                            <div class="product-image-container">
                            <img class="product-image" src="{image_src}" alt="{brand} {model}">
                            {bestseller_badge}
                            </div>
                            <div class="product-details">
                            <span class="product-type">{bath_appliance_type.replace('_', ' ').upper()}</span>
                            <h3 class="product-title">{brand} {model}</h3>
                                <div class="price-container">
                                <span class="current-price">{better_home_price}</span>
                                <span class="retail-price">{retail_price}</span>
                                <span class="savings">Save {savings} ({savings_pct:.0f}%)</span>
                                </div>
                                <div class="product-info-item">
                                    <span class="product-info-label">Description:</span> {product.get('concise_description', product.get('description', 'No description available'))}
                                </div>
                                <div class="product-info-item">
                                <span class="product-info-label">Warranty:</span> {product.get('warranty', 'Standard warranty')}
                                </div>
                                {matched_color_html}
                                <h4>Why We Recommend This:</h4>
                                <ul class="reasons-list">
                                {"".join(reasons_with_icons)}
                                </ul>
                            <a href="{product.get('url', '#')}" class="buy-button" target="_blank">View Details</a>
                            </div>
                        </div>
                    '''
                    html_content += product_html
                html_content += "</div>"

    # Add footer
    current_date = pd.Timestamp.now().strftime("%Y-%m-%d")
    html_content += f"""
            <footer>
                <p>This product recommendation brochure was created for {user_data['name']} on {current_date}</p>
                <p>© {pd.Timestamp.now().year} BetterHome. All recommendations are personalized based on your specific requirements.</p>
            </footer>
        </div>
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

# Function to process features and store them in JSON
def process_features(features_str: str) -> List[str]:
    """Split features string by semicolon and return as a list."""
    return [feature.strip() for feature in features_str.split(';') if feature.strip()]

# Update the function that processes the CSV to JSON conversion
def convert_csv_to_json(csv_file: str, json_file: str):
    df = pd.read_csv(csv_file)
    products = []
    for _, row in df.iterrows():
        product = {
            'handle': row['handle'],
            'title': row['title'],
            'product_type': row['Product Type'],
            'category': row['Category'],
            'tags': row['tags'],
            'sku': row['SKU'],
            'weight': row['Weight'],
            'better_home_price': row['Better Home Price'],
            'retail_price': row['Retail Price'],
            'description': row['Description'],
            'brand': row['Brand'],
            'features': process_features(row['Features']),  # Process features
            'returns_policy': row['Returns Policy'],
            'warranty': row['Warranty'],
            'url': row['url'],
            'image_src': row['Image Src'],
            'image_alt_text': row['Image Alt Text']
        }
        products.append(product)
    with open(json_file, 'w') as f:
        json.dump({'products': products}, f, indent=4)
    print(f"Data from {csv_file} converted and saved to {json_file}")

def estimate_washing_machine_capacity(demographics: Dict[str, int]) -> float:
    """Estimate the required washing machine capacity based on household demographics."""
    adults_capacity = demographics.get('adults', 0) * 1.5
    kids_capacity = demographics.get('kids', 0) * 1.0
    elders_capacity = demographics.get('elders', 0) * 1.2
    total_capacity = adults_capacity + kids_capacity + elders_capacity
    # print(f"[DEBUG] Estimated washing machine capacity: {total_capacity} kg for demographics: {demographics}")
    return total_capacity

# Function to determine AC tonnage with special handling for hall
def determine_ac_tonnage(square_feet: float, room_type: str = None) -> float:
    """Determine the appropriate AC tonnage based on room size in square feet.
    
    Tonnage guidelines:
    - 0.75 Ton: Ideal for rooms up to 90 sq ft.
    - 1 Ton: Suitable for rooms of 91-130 sq ft.
    - 1.5 Ton: Best for rooms between 131-190 sq ft.
    - 2 Ton: Recommended for rooms ranging from 191-250 sq ft.
    - 2.5 Ton: Suitable for larger spaces, potentially up to 300-350 sq ft.
    """
    if square_feet <= 90:
        return 0.75
    elif square_feet <= 130:
        return 1.0
    elif square_feet <= 190:
        return 1.5
    elif square_feet <= 250:
        return 2.0
    else:
        return 2.5

def deduplicate_recommendations(recommendations: List[Dict[str, Any]], max_recommendations: int = 3) -> List[Dict[str, Any]]:
    """Deduplicate recommendations based on brand and title, keeping only the first occurrence."""
    seen = set()
    deduped = []
    for p in recommendations:
        key = (p.get('brand', ''), p.get('title', ''))
        if key not in seen:
            deduped.append(p)
            seen.add(key)
    return deduped[:max_recommendations]

# Main function
if __name__ == "__main__":
    # Check if the Excel filename is provided as an argument
    if len(sys.argv) < 2:
        print("Error: Excel filename must be provided as an argument.")
        sys.exit(1)

    # Get the Excel filename from the command line arguments
    excel_filename = sys.argv[1]

    # Get user information
    user_data = analyze_user_requirements(excel_filename)
    
    # Generate initial recommendations
    final_list = generate_final_product_list(user_data)
    
    # Generate output files with the correct suffixes
    output_base_path = excel_filename.replace('.xlsx', '')
    pdf_filename = f"{output_base_path}.pdf"
    txt_filename = f"{output_base_path}.txt"
    html_filename = f"{output_base_path}.html"
    required_features = {}  # Initialize as an empty dictionary or populate as needed
    # create_styled_pdf(pdf_filename, user_data, final_list, required_features)
    #generate_text_file(user_data, final_list, txt_filename)
    generate_html_file(user_data, final_list, html_filename)
    
    print("\nProduct recommendations have been generated!")
    print(f"Check {pdf_filename}, {txt_filename}, and {html_filename} for details.")

