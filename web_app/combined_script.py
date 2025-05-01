import pandas as pd
import json
import yaml
from typing import Dict, Any, List
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
import sys
from reportlab.pdfgen import canvas
import requests
import os
from urllib.parse import urlparse
import re # Add import for regex

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
            print("\n[DEBUG] Loaded product_catalog.json. Top-level keys:", list(catalog.keys()))
            if isinstance(catalog, dict):
                for k in catalog:
                    print(f"[DEBUG] Catalog key: {k}, #items: {len(catalog[k]) if isinstance(catalog[k], list) else 'N/A'}")
                    # Debug statement to print image_src for each product
                    if k == 'products':
                        for product in catalog[k]:
                            print(f"[DEBUG] Product image_src: {product.get('image_src', 'No image_src available')}")
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
        
        # Debug: Print column names to identify discrepancies
        print("Debug: Excel file columns:", df.columns.tolist())
        
        # Clean up column names by removing newlines and extra spaces
        df.columns = [col.split('\n')[0].strip() for col in df.columns]
        row = df.iloc[0]
        
        # Convert DataFrame to dictionary
        user_data = {
            'name': df.iloc[0]['Name'],
            'mobile': df.iloc[0]['Mobile Number (Preferably on WhatsApp)'],
            'email': df.iloc[0]['E-mail'],
            'address': df.iloc[0]['Apartment Address (building, floor, and what feeling does this Chennai location bring you?)'],
            'total_budget': float(df.iloc[0]['What is your overall budget for home appliances?']),
            'num_bedrooms': int(df.iloc[0]['Number of bedrooms']),
            'num_bathrooms': int(df.iloc[0]['Number of bathrooms']),
            'demographics': {
                'adults': int(df.iloc[0]['Adults (between the age 18 to 50)']),
                'elders': int(df.iloc[0]['Elders (above the age 60)']),
                'kids': int(df.iloc[0]['Kids (below the age 18)'])
            }
        }
        
        # Extract room requirements
        requirements = {
            'hall': {
                'fans': int(df.iloc[0]["Hall: Fan(s)?"]),
                'ac': df.iloc[0]["Hall: Air Conditioner (AC)?"] == 'Yes',
                'color_theme': df.iloc[0]['Hall: Colour theme?']
            },
            'kitchen': {
                'chimney_width': df.iloc[0]['Kitchen: Chimney width?'],
                'stove_type': df.iloc[0]['Kitchen: Gas stove type?'],
                'num_burners': int(df.iloc[0]['Kitchen: Number of burners?']),
                'small_fan': df.iloc[0]['Kitchen: Do you need a small fan?'] == 'Yes',
                'color_theme': None  # No color theme specified for kitchen
            },
            'master_bedroom': {
                'ac': df.iloc[0]['Master: Air Conditioner (AC)?'] == 'Yes',
                'bathroom': {
                    'water_heater_type': df.iloc[0]['Master: How do you bath with the hot & cold water?'],
                    'exhaust_fan_size': df.iloc[0]['Master: Exhaust fan size?']
                },
                'color_theme': df.iloc[0]['Master: What is the colour theme?']
            },
            'bedroom_2': {
                'ac': df.iloc[0]['Bedroom 2: Air Conditioner (AC)?'] == 'Yes',
                'bathroom': {
                    'water_heater_type': df.iloc[0]['Bedroom 2: How do you bath with the hot & cold water?'],
                    'exhaust_fan_size': df.iloc[0]['Bedroom 2: Exhaust fan size?']
                },
                'color_theme': df.iloc[0]['Bedroom 2: What is the colour theme?']
            },
            'laundry': {
                'washing_machine_type': df.iloc[0]['Laundry: Washing Machine?'],
                'dryer_type': df.iloc[0]['Laundry: Dryer?'],
                'color_theme': None  # No color theme specified for laundry
            }
        }
        
        # Merge requirements into user_data
        user_data.update(requirements)
        
        print("\nDebug: Processed user data:", user_data)
        print("\nDebug: Processed requirements:", requirements)
        
        return user_data
        
    except Exception as e:
        print(f"Error reading Excel file: {str(e)}")
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
                    print(f"[DEBUG] Calculating total cost for {product_type}: prices={prices}")
                    if prices:
                        max_price = max(prices)
                        total_cost += max_price
                    processed_types.add(product_type)
                except (ValueError, TypeError):
                    continue
    
    return total_cost 

# Function to get budget category
def get_budget_category(total_budget: float, appliance_type: str) -> str:
    """Determine budget category based on total budget and appliance type"""
    # Define priority and allocation for each appliance type
    appliance_priorities = {
        'ac': {'priority': 1, 'allocation': 0.25},  # Essential for Chennai climate
        'refrigerator': {'priority': 1, 'allocation': 0.30},  # Essential appliance
        'washing_machine': {'priority': 1, 'allocation': 0.20},  # Essential appliance
        'chimney': {'priority': 2, 'allocation': 0.15},  # Important but secondary
        'ceiling_fan': {'priority': 2, 'allocation': 0.05},  # Important for air circulation
        'gas_stove': {'priority': 2, 'allocation': 0.10},  # Important for cooking
        'geyser': {'priority': 3, 'allocation': 0.08},  # Comfort appliance
        'bathroom_exhaust': {'priority': 3, 'allocation': 0.02},  # Comfort appliance
        'small_fan': {'priority': 3, 'allocation': 0.02},  # Optional appliance
    }

    # Get appliance priority and allocation
    appliance_info = appliance_priorities.get(appliance_type, {'priority': 3, 'allocation': 0.10})
    allocation = appliance_info['allocation']
    priority = appliance_info['priority']

    # Calculate allocated budget
    allocated_budget = total_budget * allocation

    # For high priority appliances (AC, refrigerator, washing machine)
    if priority == 1:
        if total_budget >= 500000:  # High total budget
            return "premium"
        elif total_budget >= 300000:  # Medium total budget
            if allocated_budget >= 40000:
                return "premium"
            else:
                return "mid"
        else:  # Lower total budget (<=300000)
            if allocated_budget >= 25000:  # More conservative threshold
                return "mid"
            else:
                return "budget"

    # For medium priority appliances (chimney, ceiling fan, gas stove)
    elif priority == 2:
        if total_budget >= 400000:  # High total budget
            return "premium"
        elif total_budget >= 200000:  # Medium total budget
            if allocated_budget >= 30000:
                return "premium"
            else:
                return "mid"
        else:  # Lower total budget
            if allocated_budget >= 15000:  # More conservative threshold
                return "mid"
            else:
                return "budget"

    # For lower priority appliances
    else:
        if total_budget >= 300000:  # High total budget
            if allocated_budget >= 20000:
                return "premium"
            else:
                return "mid"
        else:  # Lower total budget
            if allocated_budget >= 10000:  # More conservative threshold
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
        'ac': {'budget': 35000, 'mid': 45000, 'premium': 60000},  # Updated AC ranges to match product prices
        'dishwasher': {'budget': 30000, 'mid': 50000},
        'dryer': {'budget': 25000, 'mid': 45000},
        'shower_system': {'budget': 30000, 'mid': 50000}
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

        if unit_match and abs(req_val - prod_value_num) < 1e-6: # Tolerance for float comparison
             print(f"[DEBUG] Feature Match: Req='{req_value_str}', Prod='{prod_feature['raw_value']}' -> NUMERICAL MATCH")
             return True
             
    # Fallback to case-insensitive raw string comparison (handles non-numeric values)
    match = req_value_str.strip().lower() == prod_feature.get('raw_value', '').lower()
    if match:
        print(f"[DEBUG] Feature Match: Req='{req_value_str}', Prod='{prod_feature['raw_value']}' -> STRING MATCH")
    return match

# Function to get specific product recommendations
def get_specific_product_recommendations(
    appliance_type: str, 
    target_budget_category: str, 
    demographics: Dict[str, int], 
    room_color_theme: str = None, 
    user_data: Dict[str, Any] = None,
    required_features: Dict[str, str] = None  # Added parameter
) -> List[Dict[str, Any]]:
    """Get specific product recommendations based on appliance type, budget category, demographics, color theme, and specific features."""
    required_features = required_features or {} # Ensure it's a dict
    catalog = load_product_catalog()
    print(f"\n[DEBUG] Looking for recommendations for appliance_type='{appliance_type}' with budget_category='{target_budget_category}' and required_features={required_features}")
    recommendations = []
    product_groups = {}  # Dictionary to group products by model
    
    # Define budget ranges for each appliance type
    budget_ranges = {
        'refrigerator': {'budget': 60000, 'mid': 85000},
        'washing_machine': {'budget': 35000, 'mid': 40000},
        'chimney': {'budget': 25000, 'mid': 35000},
        'geyser': {'budget': 10000, 'mid': 20000},
        'ceiling_fan': {'budget': 4000, 'mid': 6000},
        'bathroom_exhaust': {'budget': 2000, 'mid': 4000},
        'ac': {'budget': 35000, 'mid': 45000, 'premium': 60000},
        'dishwasher': {'budget': 30000, 'mid': 50000},
        'dryer': {'budget': 25000, 'mid': 45000},
        'shower_system': {'budget': 30000, 'mid': 50000},
        'gas_stove': {'budget': 15000, 'mid': 25000},
        'small_fan': {'budget': 2000, 'mid': 4000}
    }
    
    # Default ranges if appliance type is not in the budget_ranges
    default_ranges = {'budget': 20000, 'mid': 40000}
    ranges = budget_ranges.get(appliance_type, default_ranges)
    
    # Process available products
    if catalog and "products" in catalog:
        norm_type = appliance_type.lower().replace('_', ' ')
        filtered_products = [
            p for p in catalog["products"]
            if isinstance(p.get("product_type", ""), str) and p.get("product_type", "").lower().replace('_', ' ') == norm_type
        ]
        print(f"[DEBUG] Found {len(filtered_products)} products for type '{appliance_type}' in catalog (filtered by product_type).")
        
        matching_products_data = [] # Store product data along with scores
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
                print(f"[DEBUG] Skipping product (price={price}, budget={target_budget_category}): {product.get('brand', '')} {product.get('title', '')}")
                continue

            # Calculate feature match score
            feature_match_score = 0
            product_features_list = product.get('features', []) # Already a list from json
            if required_features and product_features_list:
                print(f"[DEBUG] Checking features for {product.get('title')}: Required={required_features}, Product Features={product_features_list}")
                for req_key, req_value in required_features.items():
                    req_key_norm = req_key.strip().lower()
                    found_match_for_req = False
                    for feature_str in product_features_list:
                        parsed_prod_feature = parse_product_feature(feature_str)
                        if parsed_prod_feature.get('key') == req_key_norm:
                             if compare_features(req_value, parsed_prod_feature):
                                 feature_match_score += 5 # Significant boost for matching feature
                                 found_match_for_req = True
                                 break # Stop checking this product's features for this required key
                    if not found_match_for_req:
                         print(f"[DEBUG] No matching feature found for required key '{req_key_norm}' with value '{req_value}'")

            # Calculate relevance score (existing logic)
            relevance_score = 0
            product_data = product.copy() # Work on a copy
            product_data['color_match'] = False # Initialize color match flag
            
            if product_data.get('color_options'):
                product_data['color_options'] = list(set(product_data.get('color_options', [])))
                if room_color_theme:
                    room_colors = room_color_theme.lower().split()
                    product_colors = [c.lower() for c in product_data.get('color_options', [])]
                    for room_color in room_colors:
                        if any(room_color in pc for pc in product_colors):
                            product_data['color_match'] = True
                            relevance_score += 2
                            break
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

            # Store product with scores
            product_data['feature_match_score'] = feature_match_score
            product_data['relevance_score'] = relevance_score
            # Ensure price fields are present and numeric before adding
            product_data['price'] = price # Store the calculated price used for budget check
            product_data['retail_price'] = float(product.get('retail_price', price * 1.2)) # Estimate retail if missing
            product_data['better_home_price'] = float(product.get('better_home_price', price / 1.2 if price > 0 else 0)) # Estimate BH if missing
            
            matching_products_data.append(product_data)

        print(f"[DEBUG] {len(matching_products_data)} products matched budget for '{appliance_type}'. Now sorting...")

        # Sort by feature match score, then relevance score, then price (descending)
        matching_products_data.sort(key=lambda x: (
            -x.get('feature_match_score', 0), 
            -x.get('relevance_score', 0), 
            -float(x.get('price', 0)) # Sort by the price used for budget matching
        ))

        # Group top products by model (using existing logic, but apply to sorted list)
        # Take the top N unique models after sorting
        final_recommendations = []
        seen_models = set()
        limit = 2 # Number of unique models to recommend
        
        for product_data in matching_products_data:
            # Create a unique key for the product model (use title as fallback)
            model_key = f"{product_data.get('brand', 'UnknownBrand')}_{product_data.get('title', 'UnknownModel')}"
            
            if model_key not in seen_models:
                # Format the recommendation dict as needed by downstream functions
                recommendation = {
                    'brand': product_data.get('brand', 'UnknownBrand'),
                    'model': product_data.get('title', 'UnknownModel'), # Use title for model name
                    'price': product_data.get('price', 0.0),
                    'retail_price': product_data.get('retail_price', 0.0),
                    'better_home_price': product_data.get('better_home_price', 0.0),
                    'features': product_data.get('features', []),
                    'description': f"{product_data.get('type', '')} {product_data.get('capacity', '')}",
                    'color_options': product_data.get('color_options', []),
                    'color_match': product_data.get('color_match', False),
                    'warranty': product_data.get('warranty', 'Standard warranty applies'),
                    'in_stock': product_data.get('in_stock', True),
                    'delivery_time': product_data.get('delivery_time', 'Contact store for details'),
                    'url': product_data.get('url', 'https://betterhomeapp.com'),
                    'relevance_score': product_data.get('relevance_score', 0),
                    'feature_match_score': product_data.get('feature_match_score', 0),
                    'energy_rating': product_data.get('energy_rating', None),
                    'capacity': product_data.get('capacity', ''),
                    'type': product_data.get('type', ''),
                    'suction_power': product_data.get('suction_power', ''),
                    'image_src': product_data.get('image_src', 'https://via.placeholder.com/300x300?text=No+Image+Available'),
                }
                
                final_recommendations.append(recommendation)
                seen_models.add(model_key)
                if len(final_recommendations) >= limit:
                    break
                    
        return final_recommendations

    else: # No catalog loaded or no products key
      print("[DEBUG] Product catalog not loaded or empty.")
      return []

# Function to generate final product list
def generate_final_product_list(user_data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a final list of preferred products with specific recommendations"""
    # Initialize room-specific recommendations
    final_list = {
        'hall': {
            'ceiling_fans': [],
            'ac': []
        },
        'kitchen': {
            'chimney': [],
            'refrigerator': [],
            'gas_stove': [],
            'small_fan': []
        },
        'master_bedroom': {
            'ac': [],
            'fans': [],
            'bathroom': {
                'water_heater': [],
                'exhaust_fan': [],
                'shower': []
            }
        },
        'bedroom_2': {
            'ac': [],
            'fans': [],
            'bathroom': {
                'water_heater': [],
                'exhaust_fan': [],
                'shower': []
            }
        },
        'laundry': {
            'washing_machine': [],
            'dryer': []
        },
        'summary': {
            'total_budget': user_data['total_budget'],
            'family_size': sum(user_data['demographics'].values()),
            'location_considerations': ['Chennai climate considered for appliance selection'],
            'budget_allocation': {},
            'lifestyle_factors': ['Family size and composition considered']
        }
    }

    # Add debug statements to verify image_src
    for room, appliances in final_list.items():
        for appliance_type, products in appliances.items():
            if isinstance(products, list):  # Ensure products is a list
                for product in products:
                    if isinstance(product, dict):  # Ensure product is a dictionary
                        image_src = product.get('image_src', '')
                        if not image_src or not image_src.startswith(('http://', 'https://')):
                            print(f"[DEBUG] Invalid image URL: {image_src}")
                            image_src = 'https://via.placeholder.com/300x300?text=No+Image+Available'  # Use a placeholder image
                        else:
                            print(f"[DEBUG] Valid image URL: {image_src}")
                        # Add buy button
                        purchase_url = product.get('url', '#')
                        html_content = f"""
                        <div class='product'>
                            <img src='{image_src}' alt='Product Image'>
                            <div class='product-details'>
                                <div class='product-title'>{product.get('brand', 'Unknown Brand')} {product.get('model', 'Unknown Model')}</div>
                                <div class='product-price'>Price: ₹{product.get('price', 0):,.2f}</div>
                                <div class='product-description'>Description: {product.get('description', 'No description available')}</div>
                                <div class='product-description'>Retail Price: ₹{product.get('retail_price', 0):,.2f}</div>
                                <div class='product-description'>You Save: ₹{product.get('retail_price', 0) - product.get('price', 0):,.2f}</div>
                                <div class='product-description'>Warranty: {product.get('warranty', 'Standard warranty applies')}</div>
                                <div class='product-description'>Delivery: {product.get('delivery_time', 'Contact store for details')}</div>
                                <div class='product-description'>Reason: {get_product_recommendation_reason(product, appliance_type, room, user_data['demographics'], user_data['total_budget'])}</div>
                                <a href='{purchase_url}' target='_blank'>Buy Now</a>
                            </div>
                        </div>
                        """
                        print(f"[DEBUG] Generated HTML content for product: {product.get('brand', 'Unknown Brand')} {product.get('model', 'Unknown Model')}\n{html_content}")
                    else:
                        print(f"[DEBUG] Unexpected product type: {type(product)}")
            else:
                print(f"[DEBUG] Unexpected products type: {type(products)}")

    # Process hall requirements
    if user_data['hall'].get('ac', False):
        budget_category = get_budget_category(user_data['total_budget'], 'ac')
        recommendations = get_specific_product_recommendations('ac', budget_category, user_data['demographics'], user_data['hall'].get('color_theme'), user_data)
        final_list['hall']['ac'] = recommendations
    
    if user_data['hall'].get('fans'):
        budget_category = get_budget_category(user_data['total_budget'], 'ceiling_fan')
        recommendations = get_specific_product_recommendations('ceiling_fan', budget_category, user_data['demographics'], user_data['hall'].get('color_theme'), user_data)
        final_list['hall']['ceiling_fans'] = recommendations
    
    # Process kitchen requirements
    if user_data['kitchen'].get('chimney_width'):
        budget_category = get_budget_category(user_data['total_budget'], 'chimney')
        recommendations = get_specific_product_recommendations('chimney', budget_category, user_data['demographics'], user_data['kitchen'].get('color_theme'), user_data)
        final_list['kitchen']['chimney'] = recommendations
    
    if user_data['kitchen'].get('refrigerator_capacity'):
        budget_category = get_budget_category(user_data['total_budget'], 'refrigerator')
        recommendations = get_specific_product_recommendations('refrigerator', budget_category, user_data['demographics'], user_data['kitchen'].get('color_theme'), user_data)
        final_list['kitchen']['refrigerator'] = recommendations
    
    if user_data['kitchen'].get('gas_stove_type'):
        budget_category = get_budget_category(user_data['total_budget'], 'gas_stove')
        recommendations = get_specific_product_recommendations('gas_stove', budget_category, user_data['demographics'], user_data['kitchen'].get('color_theme'), user_data)
        final_list['kitchen']['gas_stove'] = recommendations
    
    if user_data['kitchen'].get('small_fan', False):
        budget_category = get_budget_category(user_data['total_budget'], 'small_fan')
        recommendations = get_specific_product_recommendations('small_fan', budget_category, user_data['demographics'], user_data['kitchen'].get('color_theme'), user_data)
        final_list['kitchen']['small_fan'] = recommendations
    
    # Process master bedroom requirements
    if user_data['master_bedroom'].get('ac', False):
        budget_category = get_budget_category(user_data['total_budget'], 'ac')
        recommendations = get_specific_product_recommendations('ac', budget_category, user_data['demographics'], user_data['master_bedroom'].get('color_theme'), user_data)
        final_list['master_bedroom']['ac'] = recommendations
    
    budget_category = get_budget_category(user_data['total_budget'], 'ceiling_fan')
    recommendations = get_specific_product_recommendations('ceiling_fan', budget_category, user_data['demographics'], user_data['master_bedroom'].get('color_theme'), user_data)
    final_list['master_bedroom']['fans'] = recommendations
    
    # Process master bedroom bathroom requirements
    if user_data['master_bedroom'].get('bathroom') and user_data['master_bedroom']['bathroom'].get('water_heater_type'):
        budget_category = get_budget_category(user_data['total_budget'], 'geyser')
        recommendations = get_specific_product_recommendations('geyser', budget_category, user_data['demographics'], user_data['master_bedroom'].get('color_theme'), user_data)
        final_list['master_bedroom']['bathroom']['water_heater'] = recommendations
    
    if user_data['master_bedroom'].get('bathroom') and user_data['master_bedroom']['bathroom'].get('exhaust_fan_size'):
        budget_category = get_budget_category(user_data['total_budget'], 'bathroom_exhaust')
        recommendations = get_specific_product_recommendations('bathroom_exhaust', budget_category, user_data['demographics'], user_data['master_bedroom'].get('color_theme'), user_data)
        final_list['master_bedroom']['bathroom']['exhaust_fan'] = recommendations
    
    # Process bedroom 2 requirements
    if user_data['bedroom_2'].get('ac', False):
        budget_category = get_budget_category(user_data['total_budget'], 'ac')
        recommendations = get_specific_product_recommendations('ac', budget_category, user_data['demographics'], user_data['bedroom_2'].get('color_theme'), user_data)
        final_list['bedroom_2']['ac'] = recommendations
    
    budget_category = get_budget_category(user_data['total_budget'], 'ceiling_fan')
    recommendations = get_specific_product_recommendations('ceiling_fan', budget_category, user_data['demographics'], user_data['bedroom_2'].get('color_theme'), user_data)
    final_list['bedroom_2']['fans'] = recommendations
    
    # Process bedroom 2 bathroom requirements
    if user_data['bedroom_2'].get('bathroom') and user_data['bedroom_2']['bathroom'].get('water_heater_type'):
        budget_category = get_budget_category(user_data['total_budget'], 'geyser')
        recommendations = get_specific_product_recommendations('geyser', budget_category, user_data['demographics'], user_data['bedroom_2'].get('color_theme'), user_data)
        final_list['bedroom_2']['bathroom']['water_heater'] = recommendations
    
    if user_data['bedroom_2'].get('bathroom') and user_data['bedroom_2']['bathroom'].get('exhaust_fan_size'):
        budget_category = get_budget_category(user_data['total_budget'], 'bathroom_exhaust')
        recommendations = get_specific_product_recommendations('bathroom_exhaust', budget_category, user_data['demographics'], user_data['bedroom_2'].get('color_theme'), user_data)
        final_list['bedroom_2']['bathroom']['exhaust_fan'] = recommendations
    
    # Process laundry requirements
    print("\nDebug: Laundry data:", user_data['laundry'])
    if str(user_data['laundry'].get('washing_machine_type', '')).strip().lower() == 'yes':
        print("\nDebug: Found washing machine type:", user_data['laundry']['washing_machine_type'])
        budget_category = get_budget_category(user_data['total_budget'], 'washing_machine')
        print("\nDebug: Budget category:", budget_category)
        recommendations = get_specific_product_recommendations('washing_machine', budget_category, user_data['demographics'], user_data['laundry'].get('color_theme'), user_data)
        print("\nDebug: Washing machine recommendations:", recommendations)
        final_list['laundry']['washing_machine'] = recommendations
    
    if user_data['laundry'].get('dryer_type', '').lower() == 'yes':
        budget_category = get_budget_category(user_data['total_budget'], 'dryer')
        recommendations = get_specific_product_recommendations('dryer', budget_category, user_data['demographics'], user_data['laundry'].get('color_theme'), user_data)
        final_list['laundry']['dryer'] = recommendations
    
    return final_list

def get_room_description(room: str, user_data: Dict[str, Any]) -> str:
    """Generate a description for each room based on user requirements"""
    if room == 'hall':
        return f"A welcoming space with {user_data['hall'].get('fans', 'no')} fan(s) and {'an AC' if user_data['hall'].get('ac', False) else 'no AC'}, " \
               f"complemented by a {user_data['hall'].get('color_theme', 'neutral')} color theme."
    
    elif room == 'kitchen':
        return f"A functional kitchen with a {user_data['kitchen'].get('chimney_width', 'standard')} chimney, " \
               f"{user_data['kitchen'].get('stove_type', 'standard')} with {user_data['kitchen'].get('num_burners', '4')} burners, " \
               f"and {'a small fan' if user_data['kitchen'].get('small_fan', False) else 'no fan'}."
    
    elif room == 'master_bedroom':
        return f"Master bedroom with {user_data['master_bedroom'].get('color_theme', 'neutral')} theme, " \
               f"{'an AC' if user_data['master_bedroom'].get('ac', False) else 'no AC'}, " \
               f"and a bathroom equipped with {user_data['master_bedroom'].get('bathroom', {}).get('water_heater_type', 'standard')} water heating."
    
    elif room == 'bedroom_2':
        return f"Second bedroom with {user_data['bedroom_2'].get('color_theme', 'neutral')} theme, " \
               f"{'an AC' if user_data['bedroom_2'].get('ac', False) else 'no AC'}, " \
               f"and a bathroom equipped with {user_data['bedroom_2'].get('bathroom', {}).get('water_heater_type', 'standard')} water heating."
    
    elif room == 'laundry':
        return f"Laundry area equipped with a {user_data['laundry'].get('washing_machine_type', 'standard')} washing machine" \
               f"{' and a dryer' if user_data['laundry'].get('dryer_type', '').lower() == 'yes' else ''}."
    
    return ""

def get_user_information(excel_filename: str) -> Dict[str, Any]:
    """Read user information from the Excel file"""
    try:
        # Read the Excel file
        df = pd.read_excel(excel_filename)
        
        # Debug: Print column names to identify discrepancies
        print("Debug: Excel file columns:", df.columns.tolist())
        
        # Clean up column names by removing newlines and extra spaces
        df.columns = [col.split('\n')[0].strip() for col in df.columns]
        row = df.iloc[0]
        
        # Convert DataFrame to dictionary
        user_data = {
            'name': df.iloc[0]['Name'],
            'mobile': df.iloc[0]['Mobile Number (Preferably on WhatsApp)'],
            'email': df.iloc[0]['E-mail'],
            'address': df.iloc[0]['Apartment Address (building, floor, and what feeling does this Chennai location bring you?)'],
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
def get_product_recommendation_reason(product: Dict[str, Any], appliance_type: str, room: str, demographics: Dict[str, int], total_budget: float) -> str:
    """Generate a personalized recommendation reason for a product"""
    reasons = []
    
    # Budget consideration
    budget_saved = product.get('retail_price', 0) - product.get('price', 0)
    if budget_saved > 0:
        reasons.append(f"Offers excellent value with savings of {format_currency(budget_saved)} compared to retail price")

    # Color matching
    if product.get('color_match', False):
        reasons.append(f"Color options ({', '.join(product.get('color_options', []))}) complement your room's color theme")

    # Energy efficiency
    if product.get('energy_rating') in ['5 Star', '4 Star']:
        reasons.append(f"High energy efficiency ({product['energy_rating']}) helps reduce electricity bills")

    # Room and appliance specific reasons
    if appliance_type == 'ceiling_fan':
        if 'BLDC Motor' in product.get('features', []):
            reasons.append("BLDC motor technology ensures high energy efficiency and silent operation - perfect for Chennai's climate")
        if room == 'hall':
            reasons.append("Ideal for your hall, providing effective air circulation in the common area")
    
    elif appliance_type == 'bathroom_exhaust':
        if demographics.get('elders', 0) > 0 and 'humidity sensor' in [f.lower() for f in product.get('features', [])]:
            reasons.append("Automatic humidity sensing is beneficial for elder care, preventing bathroom dampness")
        reasons.append("Essential for Chennai's humid climate to prevent mold and maintain bathroom freshness")
    
    elif appliance_type == 'geyser':
        if demographics.get('elders', 0) > 0:
            if 'Temperature Control' in product.get('features', []):
                reasons.append("Temperature control feature ensures safety for elderly family members")
        if product.get('capacity', '').lower().endswith('l'):
            capacity = int(product.get('capacity', '0L')[:-1])
            family_size = sum(demographics.values())
            if capacity >= family_size * 5:
                reasons.append(f"Capacity of {product['capacity']} is suitable for your family size of {family_size} members")
    
    elif appliance_type == 'refrigerator':
        if product.get('capacity', '').lower().endswith('l'):
            capacity = int(product.get('capacity', '0L')[:-1])
            family_size = sum(demographics.values())
            if capacity >= family_size * 100:
                reasons.append(f"Capacity of {product['capacity']} is ideal for your family size of {family_size} members")
        if demographics.get('kids', 0) > 0 and 'Child lock' in product.get('features', []):
            reasons.append("Child lock feature provides additional safety for homes with children")
    
    elif appliance_type == 'washing_machine':
        family_size = sum(demographics.values())
        if product.get('capacity', '').lower().endswith('kg'):
            capacity = float(product.get('capacity', '0kg')[:-2])
            if capacity >= family_size * 1.5:
                reasons.append(f"Capacity of {product['capacity']} is perfect for your family size")
        if demographics.get('kids', 0) > 0:
            if 'Anti-allergen' in product.get('features', []):
                reasons.append("Anti-allergen feature is beneficial for families with children")
    
    elif appliance_type == 'chimney':
        if 'auto-clean' in product.get('type', '').lower():
            reasons.append("Auto-clean feature reduces maintenance effort, ideal for busy families")
        if product.get('suction_power', '').lower().endswith('m³/hr'):
            power = int(product.get('suction_power', '0 m³/hr').split()[0])
            if power >= 1200:
                reasons.append("Strong suction power effectively handles Indian cooking needs")

    # Add a general note about warranty if available
    if product.get('warranty'):
        reasons.append(f"Comes with {product['warranty']} for peace of mind")

    return " • " + "\n • ".join(reasons)

# Function to create a styled PDF
def create_styled_pdf(filename, user_data, recommendations):
    doc = SimpleDocTemplate(filename, pagesize=letter)
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    pdfmetrics.registerFont(TTFont('DejaVuSans', './DejaVuSans.ttf'))
    styles = getSampleStyleSheet()
    for style_name in styles.byName:
        styles[style_name].fontName = 'DejaVuSans'
    story = []

    # Add custom styles
    styles.add(ParagraphStyle(fontName='DejaVuSans', 
        name='RoomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        spaceAfter=20
    ))
    styles.add(ParagraphStyle(fontName='DejaVuSans', 
        name='ProductTitle',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=10
    ))
    styles.add(ParagraphStyle(fontName='DejaVuSans', 
        name='Description',
        parent=styles['Normal'],
        fontSize=12,
        spaceAfter=15
    ))

    # Cover page with user information
    story.append(Paragraph("BetterHome Product Recommendations", styles['Title']))
    story.append(Spacer(1, 20))
    story.append(Paragraph(f"Name: {user_data['name']}", styles['Normal']))
    story.append(Paragraph(f"Mobile: {user_data['mobile']}", styles['Normal']))
    story.append(Paragraph(f"Email: {user_data['email']}", styles['Normal']))
    story.append(Paragraph(f"Address: {user_data['address']}", styles['Normal']))
    story.append(Paragraph(f"Total Budget: ₹{user_data['total_budget']:,.2f}", styles['Normal']))
    story.append(Paragraph(f"Family Size: {sum(user_data['demographics'].values())} members", styles['Normal']))
    story.append(PageBreak())

    total_cost = calculate_total_cost(recommendations)
    
    # Process each room
    for room in ['hall', 'kitchen', 'master_bedroom', 'bedroom_2', 'laundry']:
        if room in recommendations and recommendations[room]:
            story.append(Paragraph(room.replace('_', ' ').title(), styles['RoomTitle']))
            
            # Add room description
            room_desc = get_room_description(room, user_data)
            story.append(Paragraph(room_desc, styles['Description']))
            
            # Add products for the room
            for appliance_type, items in recommendations[room].items():
                if isinstance(items, list) and items:  # Check if items is a non-empty list
                    for item in items:
                        if isinstance(item, dict):  # Check if item is a dictionary
                            # Get product details
                            brand = item.get('brand', 'Unknown Brand')
                            model = item.get('model', 'Unknown Model')
                            price = float(item.get('retail_price', item.get('price', item.get('better_home_price', 0))))
                            if item.get('retail_price') is not None:
                                print(f"[DEBUG] Using retail_price for product: {brand} {model} -> {item.get('retail_price')}")
                            elif item.get('price') is not None:
                                print(f"[DEBUG] Using price for product: {brand} {model} -> {item.get('price')}")
                            else:
                                print(f"[DEBUG] Using better_home_price for product: {brand} {model} -> {item.get('better_home_price')}")
                            description = item.get('description', 'No description available')
                            retail_price = price * 1.2  # 20% markup for retail price
                            savings = retail_price - price
                            
                            # Correctly access the nested URL field
                            purchase_url = item.get('url', 'https://betterhomeapp.com')
                            print(f"Debug: Using purchase link for {brand} {model}: {purchase_url}")
                            product_name = f"<link href='{purchase_url}'>{brand} {model}</link>"
                            story.append(Paragraph(product_name, styles['ProductTitle']))
                            
                            # Get recommendation reason with total budget
                            reason = get_product_recommendation_reason(
                                item, 
                                appliance_type, 
                                room, 
                                user_data['demographics'],
                                user_data['total_budget']
                            )
                            
                            details = f"""
                            Price: ₹{price:,.2f}<br/>
                            Retail Price: ₹{retail_price:,.2f}<br/>
                            You Save: ₹{savings:,.2f}<br/>
                            Description: {description}<br/>
                            Reason: {reason}<br/>
                            """
                            story.append(Paragraph(details, styles['Normal']))
                            story.append(Spacer(1, 15))
            
            story.append(PageBreak())
    
    # Add budget summary
    budget_utilization = (total_cost / user_data['total_budget']) * 100
    summary = f"""
    Total Cost of Recommended Products: ₹{total_cost:,.2f}<br/>
    Your Budget: ₹{user_data['total_budget']:,.2f}<br/>
    Budget Utilization: {budget_utilization:.1f}%<br/>
    """
    if budget_utilization <= 100:
        summary += "Your selected products fit within your budget!"
    else:
        summary += "Note: The total cost exceeds your budget. You may want to consider alternative options."
    
    story.append(Paragraph("Budget Summary", styles['RoomTitle']))
    story.append(Paragraph(summary, styles['Normal']))
    
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
        f.write(f"Total Budget: ₹{user_data['total_budget']:,.2f}\n")
        f.write(f"Family Size: {sum(user_data['demographics'].values())} members\n\n")
        
        f.write("ROOM-WISE RECOMMENDATIONS\n")
        f.write("========================\n\n")
        
        total_cost = calculate_total_cost(final_list)
        
        # Process each room
        for room in ['hall', 'kitchen', 'master_bedroom', 'bedroom_2', 'laundry']:
            if room in final_list and final_list[room]:
                f.write(f"{room.replace('_', ' ').upper()}\n")
                f.write("-" * len(room) + "\n")
                
                # Add room description
                room_desc = get_room_description(room, user_data)
                f.write(f"{room_desc}\n\n")
                
                # Add products for the room
                for appliance_type, items in final_list[room].items():
                    if isinstance(items, list) and items:
                        for item in items:
                            if isinstance(item, dict):
                                # Get product details
                                brand = item.get('brand', 'Unknown Brand')
                                model = item.get('model', 'Unknown Model')
                                price = float(item.get('retail_price', item.get('price', item.get('better_home_price', 0))))
                                if item.get('retail_price') is not None:
                                    print(f"[DEBUG] Using retail_price for product: {brand} {model} -> {item.get('retail_price')}")
                                elif item.get('price') is not None:
                                    print(f"[DEBUG] Using price for product: {brand} {model} -> {item.get('price')}")
                                else:
                                    print(f"[DEBUG] Using better_home_price for product: {brand} {model} -> {item.get('better_home_price')}")
                                description = item.get('description', 'No description available')
                                retail_price = float(item.get('retail_price', price * 1.2))
                                savings = retail_price - price
                                warranty = item.get('warranty', 'Standard warranty applies')
                                delivery_time = item.get('delivery_time', 'Contact store for details')
                                
                                f.write(f"{appliance_type.replace('_', ' ').title()}: {brand} {model}\n")
                                f.write(f"Price: ₹{price:,.2f} (Retail: ₹{retail_price:,.2f})\n")
                                f.write(f"Description: {description}\n")
                                
                                if item.get('color_options'):
                                    f.write(f"Color Options: {', '.join(item['color_options'])}")
                                    if item.get('color_match'):
                                        f.write(" - Matches your room's color theme!")
                                    f.write("\n")
                                
                                f.write("Why we recommend this:\n")
                                if savings > 0:
                                    f.write(f" • Offers excellent value with savings of ₹{savings:,.2f} compared to retail price\n")
                                
                                if item.get('color_match'):
                                    f.write(" • Color options complement your room's color theme\n")
                                
                                # Add specific features based on appliance type
                                if appliance_type == 'ceiling_fan':
                                    f.write(" • BLDC motor technology ensures high energy efficiency and silent operation - perfect for Chennai's climate\n")
                                elif appliance_type == 'bathroom_exhaust':
                                    f.write(" • Essential for Chennai's humid climate to prevent mold and maintain bathroom freshness\n")
                                elif appliance_type == 'refrigerator':
                                    f.write(" • Energy-efficient design helps reduce electricity bills\n")
                                elif appliance_type == 'washing_machine':
                                    f.write(" • Advanced washing technology ensures thorough cleaning while being gentle on clothes\n")
                                
                                f.write(f"Warranty: {warranty}\n")
                                f.write(f"Delivery: {delivery_time}\n\n")
            
        # Add budget summary
        f.write("\nBUDGET SUMMARY\n")
        f.write("=============\n")
        f.write(f"Total Cost of Recommended Products: ₹{total_cost:,.2f}\n")
        f.write(f"Your Budget: ₹{user_data['total_budget']:,.2f}\n")
        budget_utilization = (total_cost / user_data['total_budget']) * 100
        f.write(f"Budget Utilization: {budget_utilization:.1f}%\n")
        if budget_utilization <= 100:
            f.write("Your selected products fit within your budget!\n")
        else:
            f.write("Note: The total cost exceeds your budget. You may want to consider alternative options.\n")

# Function to download an image from a URL
def download_image(image_url: str, save_dir: str) -> str:
    """Download an image from a URL and save it to the specified directory."""
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        # Extract the image filename from the URL
        parsed_url = urlparse(image_url)
        image_filename = os.path.basename(parsed_url.path)
        image_path = os.path.join(save_dir, image_filename)
        # Save the image to the specified directory
        with open(image_path, 'wb') as f:
            f.write(response.content)
        return image_path
    except Exception as e:
        print(f"[DEBUG] Error downloading image from {image_url}: {e}")
        return ""

# Function to generate an HTML file with recommendations
def generate_html_file(user_data: Dict[str, Any], final_list: Dict[str, Any], html_filename: str) -> None:
    """Generate an HTML file with user information and product recommendations."""
    html_content = """
    <html>
    <head>
        <title>BetterHome Product Recommendations</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2 {{ color: #2c3e50; }}
            .product {{ border: 1px solid #ddd; padding: 10px; margin-bottom: 20px; }}
            .product img {{ max-width: 100px; float: left; margin-right: 10px; }}
            .product-details {{ overflow: hidden; }}
            .product-title {{ font-size: 18px; font-weight: bold; }}
            .product-price {{ color: #e74c3c; }}
            .product-description {{ font-size: 14px; }}
        </style>
    </head>
    <body>
        <h1>BetterHome Product Recommendations</h1>
        <p><strong>Name:</strong> {name}</p>
        <p><strong>Mobile:</strong> {mobile}</p>
        <p><strong>Email:</strong> {email}</p>
        <p><strong>Address:</strong> {address}</p>
        <p><strong>Total Budget:</strong> ₹{total_budget:,.2f}</p>
        <p><strong>Family Size:</strong> {family_size} members</p>
        <hr>
    """.format(
        name=user_data['name'],
        mobile=user_data['mobile'],
        email=user_data['email'],
        address=user_data['address'],
        total_budget=user_data['total_budget'],
        family_size=sum(user_data['demographics'].values())
    )

    # Add budget summary
    total_cost = calculate_total_cost(final_list)
    budget_utilization = (total_cost / user_data['total_budget']) * 100
    summary = f"""
    <h2>Budget Summary</h2>
    <p>Total Cost of Recommended Products: ₹{total_cost:,.2f}</p>
    <p>Your Budget: ₹{user_data['total_budget']:,.2f}</p>
    <p>Budget Utilization: {budget_utilization:.1f}%</p>
    """
    if budget_utilization <= 100:
        summary += "<p>Your selected products fit within your budget!</p>"
    else:
        summary += "<p>Note: The total cost exceeds your budget. You may want to consider alternative options.</p>"
    html_content += summary

    # Process each room
    for room, appliances in final_list.items():
        if room == 'summary':
            continue
        html_content += f"<h2>{room.replace('_', ' ').title()}</h2>"
        for appliance_type, products in appliances.items():
            for product in products:
                if not isinstance(product, dict):
                    print(f"[DEBUG] Unexpected product type: {type(product)}")
                    continue
                image_src = product.get('image_src', 'https://via.placeholder.com/300x300?text=No+Image+Available')
                description = product.get('description', 'No description available')
                # Debug statements for image_src and description
                print(f"[DEBUG] Processing image_src: {image_src}")
                print(f"[DEBUG] Product description: {description}")

                # Extract brand and model from the product dictionary
                brand = product.get('brand', 'Unknown Brand')
                model = product.get('model', product.get('title', 'Unknown Model'))

                # Debug statements to check values
                print(f"[DEBUG] Processing product: {brand} {model}")
                print(f"[DEBUG] Image src: {image_src}")
                print(f"[DEBUG] Price: {product.get('retail_price', product.get('price', product.get('better_home_price', 0)))}")

                # Include all relevant information
                price = float(product.get('retail_price', product.get('price', product.get('better_home_price', 0))))
                retail_price = price * 1.2  # 20% markup for retail price
                savings = retail_price - price
                warranty = product.get('warranty', 'Standard warranty applies')
                delivery_time = product.get('delivery_time', 'Contact store for details')
                reason = get_product_recommendation_reason(
                    product, 
                    appliance_type, 
                    room, 
                    user_data['demographics'],
                    user_data['total_budget']
                )
                purchase_url = product.get('url', '#')  # Default to '#' if URL is not available
                html_content += f"""
                <div class='product'>
                    <img src='{image_src}' alt='Product Image'>
                    <div class='product-details'>
                        <div class='product-title'>{brand} {model}</div>
                        <div class='product-price'>Price: ₹{price:,.2f}</div>
                        <div class='product-description'>Description: {description}</div>
                        <div class='product-description'>Retail Price: ₹{retail_price:,.2f}</div>
                        <div class='product-description'>You Save: ₹{savings:,.2f}</div>
                        <div class='product-description'>Warranty: {warranty}</div>
                        <div class='product-description'>Delivery: {delivery_time}</div>
                        <div class='product-description'>Reason: {reason}</div>
                        <a href='{purchase_url}' target='_blank'>Buy Now</a>
                    </div>
                </div>
                """

    html_content += """
    </body>
    </html>
    """

    # Write the HTML content to a file
    with open(html_filename, 'w') as f:
        f.write(html_content)

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
    print("\n[DEBUG] Final recommendations structure:")
    import pprint
    pprint.pprint(final_list)
    
    # Generate output files with the correct suffixes
    output_base_path = excel_filename.replace('.xlsx', '')
    pdf_filename = f"{output_base_path}.pdf"
    txt_filename = f"{output_base_path}.txt"
    html_filename = f"{output_base_path}.html"
    create_styled_pdf(pdf_filename, user_data, final_list)
    generate_text_file(user_data, final_list, txt_filename)
    generate_html_file(user_data, final_list, html_filename)
    
    print("\nProduct recommendations have been generated!")
    print(f"Check {pdf_filename}, {txt_filename}, and {html_filename} for details.")

