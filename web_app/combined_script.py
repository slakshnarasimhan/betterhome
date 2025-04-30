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
from bs4 import BeautifulSoup
import re
from urllib.parse import urlparse
import os
from datetime import datetime

# Function to format currency
def format_currency(amount: float) -> str:
    """Format amount in Indian Rupees"""
    return f"₹{amount:,.2f}"

# Function to load product catalog
def load_product_catalog() -> Dict[str, Any]:
    """Load product catalog from JSON file"""
    try:
        with open('product_catalog.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("Product catalog file not found")
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
                    prices = [float(option.get('price', 0)) for option in options if isinstance(option, dict)]
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

# Function to get specific product recommendations
def get_specific_product_recommendations(appliance_type: str, target_budget_category: str, demographics: Dict[str, int], room_color_theme: str = None, user_data: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """Get specific product recommendations based on appliance type, budget category, demographics, and room color theme"""
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
    if catalog and appliance_type in catalog:
        matching_products = []
        for product in catalog[appliance_type]:
            price = float(product.get('price', 0))
            
            # Determine if product matches budget category and type requirements
            product_matches = False
            if appliance_type == 'washing_machine':
                user_type = user_data['laundry'].get('washing_machine_type', '').lower()
                if user_type in ['yes', '']:
                    type_matches = True  # Accept all
                else:
                    product_type = product.get('type', '').lower()
                    type_matches = (product_type == user_type)

                
                if type_matches:
                    if target_budget_category == 'premium':
                        product_matches = True
                    elif target_budget_category == 'mid' and price <= ranges['mid']:
                        product_matches = True
                    elif target_budget_category == 'budget' and price <= ranges['budget']:
                        product_matches = True
            else:
                # For other appliances, match based on budget category
                if target_budget_category == 'premium':
                    product_matches = True  # Accept all prices for premium category
                elif target_budget_category == 'mid':
                    product_matches = (price <= ranges.get('mid', float('inf')))
                else:  # budget category
                    product_matches = (price <= ranges.get('budget', float('inf')))
            
            # Add matching products to list
            if product_matches:
                matching_products.append(product)
        
        # Sort matching products by price in descending order
        matching_products.sort(key=lambda x: float(x.get('price', 0)), reverse=True)
        
        # Take the top 2 products
        for product in matching_products[:2]:
            # Create a unique key for the product group
            product_key = f"{product['brand']}_{product['model']}"
            
            if product_key not in product_groups:
                # Initialize product group
                product_groups[product_key] = {
                    'brand': product['brand'],
                    'model': product['model'],
                    'price': product['price'],
                    'features': product.get('features', []),
                    'retail_price': product.get('retail_price', product['price'] * 1.2),
                    'description': f"{product.get('type', '')} {product.get('capacity', '')}",
                    'color_options': set(),
                    'color_match': False,
                    'warranty': product.get('warranty', 'Standard warranty applies'),
                    'in_stock': product.get('in_stock', True),
                    'delivery_time': product.get('delivery_time', 'Contact store for details'),
                    'url': product.get('url', 'https://betterhomeapp.com'),
                    'relevance_score': 0
                }
            
            # Add color options and check color match
            if product.get('color_options'):
                product_groups[product_key]['color_options'].update(product['color_options'])
                
                if room_color_theme:
                    room_colors = room_color_theme.lower().split()
                    product_colors = [c.lower() for c in product.get('color_options', [])]
                    
                    for room_color in room_colors:
                        if any(room_color in pc for pc in product_colors):
                            product_groups[product_key]['color_match'] = True
                            product_groups[product_key]['relevance_score'] += 2
                            break
            
            # Add points for premium features
            if 'BLDC' in str(product.get('features', '')).upper():
                product_groups[product_key]['relevance_score'] += 1
            if 'remote' in str(product.get('features', '')).lower():
                product_groups[product_key]['relevance_score'] += 1
            if target_budget_category == 'premium':
                product_groups[product_key]['relevance_score'] += 2  # Prefer premium products
    
    # Convert product groups to list
    for product in product_groups.values():
        product['color_options'] = list(product['color_options'])
        recommendations.append(product)
    
    # Sort by relevance score and price (preferring higher-priced options)
    recommendations.sort(key=lambda x: (-x['relevance_score'], -float(x.get('price', 0))))
    
    return recommendations[:2]

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
                            price = float(item.get('price', 0))
                            features = item.get('features', [])
                            retail_price = float(item.get('retail_price', price * 1.2))
                            savings = retail_price - price
                            warranty = item.get('warranty', 'Standard warranty applies')
                            delivery_time = item.get('delivery_time', 'Contact store for details')
                            purchase_link = item.get('url', 'https://betterhomeapp.com')
                            
                            # Get product image and Amazon rating
                            image_url = get_product_image(item)
                            amazon_data = get_amazon_rating(purchase_link)
                            
                            story.append(Paragraph(f"{brand} {model}", styles['ProductTitle']))
                            
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
                            Features: {', '.join(features)}<br/>
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

def get_product_image(product: Dict[str, Any]) -> str:
    """Get product image URL from product data or fall back to scraping"""
    try:
        # First try to use the image_src field from the product catalog
        if product.get('image_src') and product['image_src'] != 'Not Available':
            print(f"Debug: Using image_src from product catalog: {product['image_src']}")
            return product['image_src']
            
        # If no image_src, try to scrape from the product URL
        url = product.get('url', '')
        if not url:
            print("Debug: No product URL available")
            return "https://via.placeholder.com/300x300?text=No+Image+Available"
            
        print(f"Debug: Attempting to get product image from URL: {url}")
        
        # If the URL is already an image URL, return it directly
        if url.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
            print(f"Debug: URL is already an image URL: {url}")
            return url
            
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
        }
        
        # Add timeout and verify SSL
        response = requests.get(url, headers=headers, timeout=10, verify=True)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Try different selectors for product image
        image_selectors = [
            '#landingImage',  # Amazon main product image
            '#imgTagWrapperId img',  # Alternative Amazon selector
            '.product-image img',  # Generic product image
            'img[src*="product"]',  # Generic product image
            'img[src*="image"]',    # Generic image
            'img[src*="photo"]'     # Generic photo
        ]
        
        for selector in image_selectors:
            img = soup.select_one(selector)
            if img and img.get('src'):
                image_url = img['src']
                # Ensure the URL is absolute
                if not image_url.startswith(('http://', 'https://')):
                    from urllib.parse import urljoin
                    image_url = urljoin(url, image_url)
                print(f"Debug: Found image with selector {selector}: {image_url}")
                return image_url
        
        print("Debug: No image found with any selector")
        return "https://via.placeholder.com/300x300?text=No+Image+Available"
    except requests.exceptions.RequestException as e:
        print(f"Error getting product image (RequestException): {str(e)}")
        return "https://via.placeholder.com/300x300?text=Error+Loading+Image"
    except Exception as e:
        print(f"Error getting product image: {str(e)}")
        return "https://via.placeholder.com/300x300?text=Error+Loading+Image"

def get_amazon_rating(url: str) -> Dict[str, Any]:
    """Extract Amazon rating and review count from the product page"""
    try:
        print(f"Debug: Attempting to get ratings from URL: {url}")
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
        }
        
        # Add timeout and verify SSL
        response = requests.get(url, headers=headers, timeout=10, verify=True)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        rating = None
        review_count = None
        
        # Try multiple selectors for rating
        rating_selectors = [
            '#acrPopover',
            '.a-icon-star',
            '[data-hook="rating-out-of-text"]',
            '.review-rating'
        ]
        
        for selector in rating_selectors:
            rating_element = soup.select_one(selector)
            if rating_element:
                rating = rating_element.get('title', '').split()[0]
                if rating:
                    print(f"Debug: Found rating with selector {selector}: {rating}")
                    break
        
        # Try multiple selectors for review count
        review_selectors = [
            '#acrCustomerReviewText',
            '[data-hook="total-review-count"]',
            '.totalReviewCount',
            '.review-count'
        ]
        
        for selector in review_selectors:
            review_element = soup.select_one(selector)
            if review_element:
                review_text = review_element.text
                print(f"Debug: Review text found with selector {selector}: {review_text}")
                review_count = re.search(r'(\d+,?\d*)', review_text)
                if review_count:
                    review_count = review_count.group(1)
                    print(f"Debug: Found review count: {review_count}")
                    break
        
        if not rating or not review_count:
            print("Debug: Could not find rating or review count")
        
        return {
            'rating': rating or 'Not available',
            'review_count': review_count or 'Not available'
        }
    except requests.exceptions.RequestException as e:
        print(f"Error getting Amazon rating (RequestException): {str(e)}")
        return {'rating': 'Not available', 'review_count': 'Not available'}
    except Exception as e:
        print(f"Error getting Amazon rating: {str(e)}")
        return {'rating': 'Not available', 'review_count': 'Not available'}

def generate_html_file(user_data: Dict[str, Any], final_list: Dict[str, Any], html_filename: str) -> None:
    """Generate an HTML file with user information and product recommendations"""
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>BetterHome Product Recommendations</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
                color: #333;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
            }}
            .header {{
                background-color: #f8f9fa;
                padding: 20px;
                border-radius: 5px;
                margin-bottom: 20px;
            }}
            .room-section {{
                margin-bottom: 30px;
                padding: 20px;
                border: 1px solid #ddd;
                border-radius: 5px;
            }}
            .room-title {{
                color: #2c3e50;
                border-bottom: 2px solid #3498db;
                padding-bottom: 10px;
            }}
            .product-card {{
                display: flex;
                margin: 20px 0;
                padding: 20px;
                border: 1px solid #eee;
                border-radius: 5px;
                background-color: #fff;
            }}
            .product-image {{
                flex: 0 0 300px;
                margin-right: 20px;
            }}
            .product-image img {{
                width: 100%;
                height: auto;
                border-radius: 5px;
            }}
            .product-details {{
                flex: 1;
            }}
            .product-title {{
                font-size: 1.2em;
                margin-bottom: 10px;
            }}
            .product-price {{
                color: #e74c3c;
                font-size: 1.1em;
                margin: 10px 0;
            }}
            .amazon-rating {{
                display: flex;
                align-items: center;
                margin: 10px 0;
            }}
            .rating-stars {{
                color: #f39c12;
                margin-right: 10px;
            }}
            .features-list {{
                list-style-type: none;
                padding: 0;
            }}
            .features-list li {{
                margin: 5px 0;
                padding-left: 20px;
                position: relative;
            }}
            .features-list li:before {{
                content: "•";
                color: #3498db;
                position: absolute;
                left: 0;
            }}
            .budget-summary {{
                background-color: #f8f9fa;
                padding: 20px;
                border-radius: 5px;
                margin-top: 20px;
            }}
            .purchase-link {{
                display: inline-block;
                background-color: #3498db;
                color: white;
                padding: 10px 20px;
                text-decoration: none;
                border-radius: 5px;
                margin-top: 10px;
            }}
            .purchase-link:hover {{
                background-color: #2980b9;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>BetterHome Product Recommendations</h1>
                <p><strong>Generated on:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Name:</strong> {user_data['name']}</p>
                <p><strong>Mobile:</strong> {user_data['mobile']}</p>
                <p><strong>Email:</strong> {user_data['email']}</p>
                <p><strong>Address:</strong> {user_data['address']}</p>
                <p><strong>Total Budget:</strong> ₹{user_data['total_budget']:,.2f}</p>
                <p><strong>Family Size:</strong> {sum(user_data['demographics'].values())} members</p>
            </div>
    """
    
    total_cost = calculate_total_cost(final_list)
    
    # Process each room
    for room in ['hall', 'kitchen', 'master_bedroom', 'bedroom_2', 'laundry']:
        if room in final_list and final_list[room]:
            room_title = room.replace('_', ' ').title()
            room_desc = get_room_description(room, user_data)
            
            html_content += f"""
            <div class="room-section">
                <h2 class="room-title">{room_title}</h2>
                <p>{room_desc}</p>
            """
            
            # Add products for the room
            for appliance_type, items in final_list[room].items():
                if isinstance(items, list) and items:
                    for item in items:
                        if isinstance(item, dict):
                            # Get product details
                            brand = item.get('brand', 'Unknown Brand')
                            model = item.get('model', 'Unknown Model')
                            price = float(item.get('price', 0))
                            features = item.get('features', [])
                            retail_price = float(item.get('retail_price', price * 1.2))
                            savings = retail_price - price
                            warranty = item.get('warranty', 'Standard warranty applies')
                            delivery_time = item.get('delivery_time', 'Contact store for details')
                            purchase_link = item.get('url', 'https://betterhomeapp.com')
                            
                            # Get product image and Amazon rating
                            image_url = get_product_image(item)
                            amazon_data = get_amazon_rating(purchase_link)
                            
                            html_content += f"""
                            <div class="product-card">
                                <div class="product-image">
                                    <img src="{image_url}" alt="{brand} {model}">
                                </div>
                                <div class="product-details">
                                    <h3 class="product-title">{brand} {model}</h3>
                                    <div class="product-price">
                                        Price: ₹{price:,.2f} (Retail: ₹{retail_price:,.2f})
                                        <br>You Save: ₹{savings:,.2f}
                                    </div>
                                    <div class="amazon-rating">
                                        <span class="rating-stars">{"★" * int(float(amazon_data['rating']) if amazon_data['rating'].replace('.', '').isdigit() else 0)}</span>
                                        <span>{amazon_data['rating']} ({amazon_data['review_count']} reviews)</span>
                                    </div>
                                    <ul class="features-list">
                            """
                            
                            # Add features
                            for feature in features:
                                html_content += f"<li>{feature}</li>"
                            
                            # Add color options if available
                            if item.get('color_options'):
                                color_text = f"Color Options: {', '.join(item['color_options'])}"
                                if item.get('color_match'):
                                    color_text += " - Matches your room's color theme!"
                                html_content += f"<li>{color_text}</li>"
                            
                            html_content += f"""
                                    </ul>
                                    <p><strong>Warranty:</strong> {warranty}</p>
                                    <p><strong>Delivery:</strong> {delivery_time}</p>
                                    <a href="{purchase_link}" class="purchase-link" target="_blank">View on BetterHome</a>
                                </div>
                            </div>
                            """
            
            html_content += "</div>"
    
    # Add budget summary
    budget_utilization = (total_cost / user_data['total_budget']) * 100
    html_content += f"""
            <div class="budget-summary">
                <h2>Budget Summary</h2>
                <p><strong>Total Cost of Recommended Products:</strong> ₹{total_cost:,.2f}</p>
                <p><strong>Your Budget:</strong> ₹{user_data['total_budget']:,.2f}</p>
                <p><strong>Budget Utilization:</strong> {budget_utilization:.1f}%</p>
                <p>{'Your selected products fit within your budget!' if budget_utilization <= 100 else 'Note: The total cost exceeds your budget. You may want to consider alternative options.'}</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    with open(html_filename, 'w', encoding='utf-8') as f:
        f.write(html_content)

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
    html_filename = f"{output_base_path}.html"
    create_styled_pdf(pdf_filename, user_data, final_list)
    generate_html_file(user_data, final_list, html_filename)
    
    print("\nProduct recommendations have been generated!")
    print(f"Check {pdf_filename} and {html_filename} for details.")

