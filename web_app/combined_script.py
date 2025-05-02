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
                'color_theme': df.iloc[0]['Hall: Colour theme?'],
                'size_sqft': float(df.iloc[0].get('Hall: What is the square feet ?', 150.0))  # Updated column name
            },
            'kitchen': {
                'chimney_width': df.iloc[0]['Kitchen: Chimney width?'],
                'stove_type': df.iloc[0]['Kitchen: Gas stove type?'],
                'num_burners': int(df.iloc[0]['Kitchen: Number of burners?']),
                'small_fan': df.iloc[0]['Kitchen: Do you need a small fan?'] == 'Yes',
                'color_theme': None,  # No color theme specified for kitchen
                'refrigerator_type': df.iloc[0].get('Kitchen: Refrigerator type?', None), # Add refrigerator type
                'refrigerator_capacity': df.iloc[0].get('Kitchen: Refrigerator capacity?', None), # Add refrigerator capacity
                'size_sqft': float(df.iloc[0].get('Kitchen: Size (square feet)', 100.0))  # Default to 100 sq ft if not specified
            },
            'master_bedroom': {
                'ac': df.iloc[0]['Master: Air Conditioner (AC)?'] == 'Yes',
                'bathroom': {
                    'water_heater_type': df.iloc[0]['Master: How do you bath with the hot & cold water?'],
                    'exhaust_fan_size': df.iloc[0]['Master: Exhaust fan size?']
                },
                'color_theme': df.iloc[0]['Master: What is the colour theme?'],
                'size_sqft': float(df.iloc[0].get('Master: What is the area of the bedroom in square feet?', 140.0))  # Updated column name
            },
            'bedroom_2': {
                'ac': df.iloc[0]['Bedroom 2: Air Conditioner (AC)?'] == 'Yes',
                'bathroom': {
                    'water_heater_type': df.iloc[0]['Bedroom 2: How do you bath with the hot & cold water?'],
                    'exhaust_fan_size': df.iloc[0]['Bedroom 2: Exhaust fan size?']
                },
                'color_theme': df.iloc[0]['Bedroom 2: What is the colour theme?'],
                'size_sqft': float(df.iloc[0].get('Bedroom 2: What is the area of the bedroom in square feet?', 120.0))  # Updated column name
            },
            'laundry': {
                'washing_machine_type': df.iloc[0]['Laundry: Washing Machine?'],
                'dryer_type': df.iloc[0]['Laundry: Dryer?'],
                'color_theme': None,  # No color theme specified for laundry
                'size_sqft': float(df.iloc[0].get('Laundry: Size (square feet)', 50.0))  # Default to 50 sq ft if not specified
            }
        }
        
        # Merge requirements into user_data
        user_data.update(requirements)
        
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
        'ac': {'budget': 75000, 'mid': 100000, 'premium': 150000},  # Increased thresholds to prioritize proper tonnage
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

        # Use a small tolerance for float comparison
        TOLERANCE = 1e-6
        if unit_match and abs(req_val - prod_value_num) < TOLERANCE:
             return True
             
    # Fallback to case-insensitive raw string comparison (handles non-numeric values)
    match = req_value_str.strip().lower() == prod_feature.get('raw_value', '').lower()
    return match

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
    budget_ranges = {
        'refrigerator': {'budget': 60000, 'mid': 85000},
        'washing_machine': {'budget': 35000, 'mid': 40000},
        'chimney': {'budget': 25000, 'mid': 35000},
        'geyser': {'budget': 10000, 'mid': 20000},
        'ceiling_fan': {'budget': 4000, 'mid': 6000},
        'bathroom_exhaust': {'budget': 2000, 'mid': 4000},
        'ac': {'budget': 75000, 'mid': 100000, 'premium': 150000},  # Increased thresholds to prioritize proper tonnage
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
        # Adjust filtering logic to handle 'ac' vs 'Air Conditioner'
        filtered_products = []
        for p in catalog["products"]:
            if isinstance(p.get("product_type", ""), str):
                product_type_norm = p.get("product_type", "").lower().replace('_', ' ')
                matches = False
                if norm_type == 'ac':
                    # Check for both 'ac' and 'air conditioner' if requested type is 'ac'
                    matches = product_type_norm == 'ac' or product_type_norm == 'air conditioner'
                else:
                    # Standard matching for other types
                    matches = product_type_norm == norm_type
                
                if matches:
                    filtered_products.append(p)

        # Simple debug print after filtering, specific types
        if appliance_type == 'refrigerator':
            print(f"[DEBUG] Filtered products count immediately after filtering for '{appliance_type}': {len(filtered_products)}")
            if not filtered_products:
                 all_product_types = [p.get('product_type', 'No type') for p in catalog.get('products', [])]
                 print(f"[DEBUG] All product types found: {all_product_types}") # Check if 'Refrigerator' is even there
        elif appliance_type == 'ac': # Add specific check for AC
            print(f"[DEBUG AC FILTERING] Required features: {required_features}") # Print required features for AC
            print(f"[DEBUG AC FILTERING] Filtered products count immediately after filtering for '{appliance_type}': {len(filtered_products)}")
            if not filtered_products:
                 all_product_types = [p.get('product_type', 'No type') for p in catalog.get('products', [])]
                 print(f"[DEBUG AC FILTERING] All product types found (checking for AC): {all_product_types}")

        matching_products_data = [] # Store product data along with scores
        
        # DEBUG: Print initial list of ACs after type filter
        if appliance_type == 'ac':
            print("[DEBUG AC DETAILS] Initial AC candidates before budget/feature check:")
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
                                    
                    print(f"  {idx+1}. {p.get('brand', 'N/A')} - {p.get('title', 'N/A')} - Tonnage: {p_tonnage} - Price: {p.get('retail_price', p.get('price', p.get('better_home_price', 0)))}")
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
                            print(f"[TEMP DEBUG] Tonnage Mismatch! Product: {product.get('title', 'N/A')[:30]}... Penalty Applied. New FeatScore: {feature_match_score}, New TonnageScore: {tonnage_feature_score}")
                            # --- End Temporary Debug Print ---
                            
                        # Even if no match, we handled the tonnage requirement check
                        continue # Move to the next required feature

                    # --- Handle Other Feature Matching (using existing logic) ---
                    if product_features_list: # Only iterate if features list exists
                        for feature_str in product_features_list:
                            parsed_prod_feature = parse_product_feature(feature_str)

                            # Extract product tonnage for debug print (only, not for matching logic)
                            if appliance_type == 'ac' and parsed_prod_feature.get('key') == 'tonnage':
                                product_tonnage_value_for_debug = parsed_prod_feature.get('raw_value', 'N/A')

                            if parsed_prod_feature.get('key') == req_key_norm:
                                 if compare_features(req_value, parsed_prod_feature):
                                     feature_match_score += 5 # Significant boost for matching feature
                                     found_match_for_req = True
                                     break # Stop checking this product's features for this required key
                    # No need for the 'if not found_match_for_req: pass' anymore

            # DEBUG: Print tonnage match result (using the specific debug variable)
            if appliance_type == 'ac':
                req_tonnage = required_features.get('tonnage', 'Not Specified') if required_features else 'Not Specified'
                # Use the actual tonnage if found, otherwise fallback to the debug value from features
                display_tonnage = f"{actual_product_tonnage} Ton" if actual_product_tonnage is not None else product_tonnage_value_for_debug
                print(f"[DEBUG AC TONNAGE MATCH] Product: {product.get('brand', 'N/A')} {product.get('title', 'N/A')} - Required: {req_tonnage}, Product has: {display_tonnage}, Tonnage Feature Score: {tonnage_feature_score}")

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
            
            matching_products_data.append(product_data)

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
        if appliance_type == 'ac':
             print("[DEBUG AC SORTING] Top 5 AC candidates after sorting:")
             for idx, p_data in enumerate(matching_products_data[:5]):
                 print(f"  {idx+1}. {p_data.get('brand', 'N/A')} - {p_data.get('model', 'N/A')} "
                       f"(Price: {p_data.get('price', 0):.2f}, FeatScore: {p_data.get('feature_match_score', 0)}, RelScore: {p_data.get('relevance_score', 0)})")

        # Group top products by model (using existing logic, but apply to sorted list)
        # Take the top N unique models after sorting
        final_recommendations = []
        seen_models = set()
        limit = 2 # Number of unique models to recommend
        
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
                    'model': product_data.get('title', 'UnknownModel'), # Use title for model name
                    'price': float(product_data.get('price', 0.0)),
                    'retail_price': float(product_data.get('retail_price', 0.0)),
                    'better_home_price': float(product_data.get('better_home_price', 0.0)),
                    'features': product_data.get('features', []),
                    'description': f"{product_data.get('type', '')} {product_data.get('capacity', '')}", # Capacity might still be missing here
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

        return final_recommendations

    else: # No catalog loaded or no products key
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

    # Process hall requirements
    if user_data['hall'].get('ac', False):
        print(f"[DEBUG] Hall AC requirement: {user_data['hall'].get('ac', False)}") # DEBUG
        budget_category = get_budget_category(user_data['total_budget'], 'ac')
        # Override the hall size to 200 sq ft for proper AC sizing, regardless of what was entered
        hall_size = 200.0  # Force hall size to 200 sq ft
        recommended_tonnage = determine_ac_tonnage(hall_size, 'hall')  # Pass 'hall' as room type for special handling
        print(f"[DEBUG] Hall AC tonnage recommendation: {recommended_tonnage} Ton for {hall_size} sq ft (always using 2 Ton for hall)")
        required_features = {'tonnage': f"{recommended_tonnage} Ton"}
        recommendations = get_specific_product_recommendations('ac', budget_category, user_data['demographics'], user_data['hall'].get('color_theme'), user_data, required_features)
        final_list['hall']['ac'] = recommendations
    
    if user_data['hall'].get('fans'):
        budget_category = get_budget_category(user_data['total_budget'], 'ceiling_fan')
        recommendations = get_specific_product_recommendations('ceiling_fan', budget_category, user_data['demographics'], user_data['hall'].get('color_theme'), user_data)
        final_list['hall']['ceiling_fans'] = recommendations
    
    # Process kitchen requirements
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
        budget_category = get_budget_category(user_data['total_budget'], 'gas_stove')
        required_features = {'type': user_data['kitchen'].get('gas_stove_type')}
        recommendations = get_specific_product_recommendations('gas_stove', budget_category, user_data['demographics'], user_data['kitchen'].get('color_theme'), user_data, required_features)
        final_list['kitchen']['gas_stove'] = recommendations
    
    if user_data['kitchen'].get('small_fan', False):
        budget_category = get_budget_category(user_data['total_budget'], 'small_fan')
        recommendations = get_specific_product_recommendations('small_fan', budget_category, user_data['demographics'], user_data['kitchen'].get('color_theme'), user_data)
        final_list['kitchen']['small_fan'] = recommendations
    
    # Process master bedroom requirements
    if user_data['master_bedroom'].get('ac', False):
        print(f"[DEBUG] Master Bedroom AC requirement: {user_data['master_bedroom'].get('ac', False)}") # DEBUG
        budget_category = get_budget_category(user_data['total_budget'], 'ac')
        # Calculate the recommended AC tonnage based on room size
        master_size = user_data['master_bedroom'].get('size_sqft', 140.0)  # Default to 140 sq ft if not specified
        recommended_tonnage = determine_ac_tonnage(master_size, 'master_bedroom')
        print(f"[DEBUG] Master bedroom AC tonnage recommendation: {recommended_tonnage} Ton for {master_size} sq ft")
        required_features = {'tonnage': f"{recommended_tonnage} Ton"}
        recommendations = get_specific_product_recommendations('ac', budget_category, user_data['demographics'], user_data['master_bedroom'].get('color_theme'), user_data, required_features)
        final_list['master_bedroom']['ac'] = recommendations
    
    budget_category = get_budget_category(user_data['total_budget'], 'ceiling_fan')
    recommendations = get_specific_product_recommendations('ceiling_fan', budget_category, user_data['demographics'], user_data['master_bedroom'].get('color_theme'), user_data)
    final_list['master_bedroom']['fans'] = recommendations
    
    # Process master bedroom bathroom requirements
    if user_data['master_bedroom'].get('bathroom') and user_data['master_bedroom']['bathroom'].get('water_heater_type'):
        budget_category = get_budget_category(user_data['total_budget'], 'geyser')
        required_features = {'type': user_data['master_bedroom']['bathroom'].get('water_heater_type')}
        recommendations = get_specific_product_recommendations('geyser', budget_category, user_data['demographics'], user_data['master_bedroom'].get('color_theme'), user_data, required_features)
        final_list['master_bedroom']['bathroom']['water_heater'] = recommendations
    
    if user_data['master_bedroom'].get('bathroom') and user_data['master_bedroom']['bathroom'].get('exhaust_fan_size'):
        budget_category = get_budget_category(user_data['total_budget'], 'bathroom_exhaust')
        required_features = {'dimensions': user_data['master_bedroom']['bathroom'].get('exhaust_fan_size')}
        recommendations = get_specific_product_recommendations('bathroom_exhaust', budget_category, user_data['demographics'], user_data['master_bedroom'].get('color_theme'), user_data, required_features)
        final_list['master_bedroom']['bathroom']['exhaust_fan'] = recommendations
    
    # Process bedroom 2 requirements
    if user_data['bedroom_2'].get('ac', False):
        print(f"[DEBUG] Bedroom 2 AC requirement: {user_data['bedroom_2'].get('ac', False)}") # DEBUG
        budget_category = get_budget_category(user_data['total_budget'], 'ac')
        # Calculate the recommended AC tonnage based on room size
        bedroom2_size = user_data['bedroom_2'].get('size_sqft', 120.0)  # Default to 120 sq ft if not specified
        recommended_tonnage = determine_ac_tonnage(bedroom2_size, 'bedroom_2')
        print(f"[DEBUG] Bedroom 2 AC tonnage recommendation: {recommended_tonnage} Ton for {bedroom2_size} sq ft")
        required_features = {'tonnage': f"{recommended_tonnage} Ton"}
        recommendations = get_specific_product_recommendations('ac', budget_category, user_data['demographics'], user_data['bedroom_2'].get('color_theme'), user_data, required_features)
        final_list['bedroom_2']['ac'] = recommendations
    
    budget_category = get_budget_category(user_data['total_budget'], 'ceiling_fan')
    recommendations = get_specific_product_recommendations('ceiling_fan', budget_category, user_data['demographics'], user_data['bedroom_2'].get('color_theme'), user_data)
    final_list['bedroom_2']['fans'] = recommendations
    
    # Process bedroom 2 bathroom requirements
    if user_data['bedroom_2'].get('bathroom') and user_data['bedroom_2']['bathroom'].get('water_heater_type'):
        budget_category = get_budget_category(user_data['total_budget'], 'geyser')
        required_features = {'type': user_data['bedroom_2']['bathroom'].get('water_heater_type')}
        recommendations = get_specific_product_recommendations('geyser', budget_category, user_data['demographics'], user_data['bedroom_2'].get('color_theme'), user_data, required_features)
        final_list['bedroom_2']['bathroom']['water_heater'] = recommendations
    
    if user_data['bedroom_2'].get('bathroom') and user_data['bedroom_2']['bathroom'].get('exhaust_fan_size'):
        budget_category = get_budget_category(user_data['total_budget'], 'bathroom_exhaust')
        required_features = {'dimensions': user_data['bedroom_2']['bathroom'].get('exhaust_fan_size')}
        recommendations = get_specific_product_recommendations('bathroom_exhaust', budget_category, user_data['demographics'], user_data['bedroom_2'].get('color_theme'), user_data, required_features)
        final_list['bedroom_2']['bathroom']['exhaust_fan'] = recommendations
    
    # Process laundry requirements
    if str(user_data['laundry'].get('washing_machine_type', '')).strip().lower() == 'yes':
        budget_category = get_budget_category(user_data['total_budget'], 'washing_machine')
        required_features = {'capacity': estimate_washing_machine_capacity(user_data['demographics'])}
        recommendations = get_specific_product_recommendations('washing_machine', budget_category, user_data['demographics'], user_data['laundry'].get('color_theme'), user_data, required_features)
        final_list['laundry']['washing_machine'] = recommendations
    
    if user_data['laundry'].get('dryer_type', '').lower() == 'yes':
        budget_category = get_budget_category(user_data['total_budget'], 'dryer')
        recommendations = get_specific_product_recommendations('dryer', budget_category, user_data['demographics'], user_data['laundry'].get('color_theme'), user_data)
        final_list['laundry']['dryer'] = recommendations
    
    return final_list

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
def get_product_recommendation_reason(product: Dict[str, Any], appliance_type: str, room: str, demographics: Dict[str, int], total_budget: float, required_features: Dict[str, str] = None) -> str:
    """Generate a personalized recommendation reason for a product, highlighting matching features."""
    reasons = []
    required_features = required_features or {}
    
    # Check if product is a bestseller
    if product.get('is_bestseller', False):
        reasons.append("One of our most popular bestsellers - frequently chosen by other customers")
        
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
        reasons.append("Key features that match your requirements: " + ", ".join(matching_features))

    # Room and appliance specific reasons
    if appliance_type == 'ceiling_fan':
        if 'BLDC Motor' in product.get('features', []):
            reasons.append("BLDC motor technology ensures high energy efficiency and silent operation - perfect for Chennai's climate")
        if room == 'hall':
            reasons.append("Ideal for your hall, providing effective air circulation in the common area")
    
    elif appliance_type == 'ac':
        # Add AC tonnage recommendation based on room size
        room_size = 0
        room_type = None
        if room == 'hall':
            room_size = user_data['hall'].get('size_sqft', 150.0)
            room_type = 'hall'
        elif room == 'master_bedroom':
            room_size = user_data['master_bedroom'].get('size_sqft', 140.0)
            room_type = 'master_bedroom'
        elif room == 'bedroom_2':
            room_size = user_data['bedroom_2'].get('size_sqft', 120.0)
            room_type = 'bedroom_2'
        
        if room_size > 0:
            recommended_tonnage = determine_ac_tonnage(room_size, room_type)
            if room_type == 'hall':
                reasons.append(f"For your hall of {room_size} sq ft, we recommend at least a {recommended_tonnage} Ton AC for effective cooling")
            else:
                reasons.append(f"Based on your {room} size of {room_size} sq ft, a {recommended_tonnage} Ton AC is recommended for optimal cooling")
        
        # Check if the AC tonnage matches the recommended tonnage
        product_tonnage = None
        for feature in product.get('features', []):
            if 'ton' in feature.lower():
                # Try to extract tonnage from feature
                tonnage_match = re.search(r'(\d+\.?\d*)\s*ton', feature.lower())
                if tonnage_match:
                    product_tonnage = float(tonnage_match.group(1))
                    break
        
        if product_tonnage is not None and room_type == 'hall' and product_tonnage >= 1.5:
            reasons.append(f"This {product_tonnage} Ton AC meets our minimum recommendation of 1.5 Ton for hall areas")
        elif product_tonnage is not None and room_size > 0:
            if abs(product_tonnage - recommended_tonnage) <= 0.25:  # Within 0.25 tons of recommendation
                reasons.append(f"This {product_tonnage} Ton AC closely matches our recommendation for your {room} size")
            elif product_tonnage > recommended_tonnage:
                reasons.append(f"This {product_tonnage} Ton AC provides extra cooling capacity for your {room} size")
            else:
                reasons.append(f"This AC's tonnage is slightly below our recommendation, but may be adequate for energy efficiency")
        
        # Add inverter technology benefit if mentioned in features
        if any('inverter' in feature.lower() for feature in product.get('features', [])):
            reasons.append("Inverter technology provides energy efficiency and consistent cooling")
        
        # Add star rating benefit if available
        energy_rating = product.get('energy_rating')
        if energy_rating:
            reasons.append(f"{energy_rating} energy rating helps reduce electricity consumption")
    
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

        for room, room_data in user_data['rooms'].items():
            # Room header with blue background
            room_title = Paragraph(f"{room.replace('_', ' ').title()} Room", heading2_style)
            story.append(room_title)
            
            # Add room description in a styled box
            room_description = room_data.get('description', '')
            if room_description:
                story.append(Paragraph(room_description, normal_style))
                story.append(Spacer(1, 10))
            
            # Add appliance recommendations for this room
            if room in recommendations:
                for appliance_type, appliances in recommendations[room].items():
                    appliance_title = appliance_type.replace('_', ' ').title()
                    story.append(Paragraph(f"{appliance_title} Recommendations", heading2_style))
                    
                    # Add each recommended appliance
                    for item in appliances:
                        if not isinstance(item, dict):
                            continue
                            
                        # Get product details
                        brand = item.get('brand', 'Unknown Brand')
                        model = item.get('model', 'Unknown Model')
                        price = float(item.get('better_home_price', item.get('price', 0)))
                        retail_price = float(item.get('retail_price', price * 1.2))
                        
                        # Add bestseller badge if applicable
                        if item.get('is_bestseller', False):
                            bestseller_text = Paragraph("<font color='white' backColor='#ff6b00'><b>BESTSELLER</b></font>", bestseller_style)
                            story.append(bestseller_text)
                            story.append(Spacer(1, 5))
                        
                        # Add product title
                        story.append(Paragraph(f"{brand} {model}", heading2_style))
                        
                        # Add product price with formatting
                        price_text = f"Price: ₹{price:,.2f}"
                        if retail_price > price:
                            savings = retail_price - price
                            price_text += f" (Retail: ₹{retail_price:,.2f}, Save: ₹{savings:,.2f})"
                        story.append(Paragraph(price_text, normal_style))
                        
                        # Add key features with bullet points
                        features = item.get('features', [])
                        if features:
                            story.append(Paragraph("Key Features:", heading2_style))
                            features_list = [f"• {feature}" for feature in features[:5]]  # Limit to 5 features
                            for feature in features_list:
                                story.append(Paragraph(feature, normal_style))
                            if len(features) > 5:
                                story.append(Paragraph("• ...", normal_style))
                        
                        # Add recommendation reasons
                        reason = item.get('reason', '')
                        if reason:
                            story.append(Paragraph("Why We Recommend This:", heading2_style))
                            reasons = [r.strip() for r in reason.split('•') if r.strip()]
                            for r in reasons[:3]:  # Limit to 3 reasons
                                story.append(Paragraph(f"• {r}", normal_style))
                            if len(reasons) > 3:
                                story.append(Paragraph("• ...", normal_style))
                        
                        # Add a divider between products
                        story.append(Spacer(1, 10))
                        story.append(HRFlowable(color=HexColor('#dddddd'), width="90%", thickness=1))
                        story.append(Spacer(1, 10))
            
            # Add space between rooms
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
                                description = item.get('description', 'No description available')
                                retail_price = float(item.get('retail_price', price * 1.2))
                                savings = retail_price - price
                                warranty = item.get('warranty', 'Standard warranty applies')
                                delivery_time = item.get('delivery_time', 'Contact store for details')
                                
                                # Add bestseller badge if applicable
                                bestseller_badge = "🔥 BESTSELLER: " if item.get('is_bestseller', False) else ""
                                f.write(f"{appliance_type.replace('_', ' ').title()}: {bestseller_badge}{brand} {model}\n")
                                f.write(f"Price: INR{price:,.2f} (Retail: INR{retail_price:,.2f})\n")
                                f.write(f"Description: {description}\n")
                                
                                if item.get('color_options'):
                                    f.write(f"Color Options: {', '.join(item['color_options'])}")
                                    if item.get('color_match'):
                                        f.write(" - Matches your room's color theme!")
                                    f.write("\n")
                                
                                f.write("Why we recommend this:\n")
                                if savings > 0:
                                    f.write(f" • Offers excellent value with savings of INR{savings:,.2f} compared to retail price\n")
                                
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
        f.write(f"Total Cost of Recommended Products: INR{total_cost:,.2f}\n")
        f.write(f"Your Budget: INR{user_data['total_budget']:,.2f}\n")
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
    html_content += header_section
    
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

    # Process each room
    for room, appliances in final_list.items():
        if room == 'summary':
            continue
        
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
            for product in products:
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
                
                # Create the HTML for the product card
                product_html = f'''
                    <div class="product-card">
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
                                <span class="product-info-label">Description:</span> {description}
                            </div>
                            <div class="product-info-item">
                            <span class="product-info-label">Delivery:</span> {product.get('delivery_time', 'Contact for details')}
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
    create_styled_pdf(pdf_filename, user_data, final_list, required_features)
    generate_text_file(user_data, final_list, txt_filename)
    generate_html_file(user_data, final_list, html_filename)
    
    print("\nProduct recommendations have been generated!")
    print(f"Check {pdf_filename}, {txt_filename}, and {html_filename} for details.")

