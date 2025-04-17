import pandas as pd
import json
import yaml
from typing import Dict, Any, List
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
import sys
import os
import subprocess
from datetime import datetime
from flask import request, render_template

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
        df = pd.read_excel(excel_file)
        print("\nDebug: Excel file columns:", df.columns.tolist())
        print("\nDebug: First row of data:", df.iloc[0].to_dict())
        
        # Extract user information
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
                'kids': int(df.iloc[0]['Kids (below the age 18)'])}
        }
        
        # Extract room requirements
        requirements = {
            'hall': {
                'fans': int(df.iloc[0]["Hall: Fan(s)?\n(A gentle breeze, a comforting presence during Chennai's warm evenings) "]),
                'ac': df.iloc[0]["Hall: Air Conditioner (AC)?\n(A cool, refreshing escape from Chennai's warmth, creating a space for relaxation)"] == 'Yes',
                'color_theme': df.iloc[0]['Hall: Colour theme?']
            },
            'kitchen': {
                'chimney_width': df.iloc[0]['Kitchen: Chimney width?'],
                'stove_type': df.iloc[0]['Kitchen: Gas stove type?\n(Where the warmth of home-cooked meals brings comfort and connection)'],
                'num_burners': int(df.iloc[0]['Kitchen: Number of burners?']),
                'stove_width': df.iloc[0]['Kitchen: Stove width?'],
                'small_fan': df.iloc[0]['Kitchen: Do you need a small fan?'] == 'Yes',
                'dishwasher_capacity': df.iloc[0]['Kitchen: Dishwasher capacity?'],
                'refrigerator_type': df.iloc[0]['Kitchen: Refrigerator type?'],
                'refrigerator_capacity': df.iloc[0]['Kitchen: Refrigerator capacity?'],
                'color_theme': None  # No color theme specified for kitchen
            },
            'master_bedroom': {
                'ac': df.iloc[0]['Master: Air Conditioner (AC)?'] == 'Yes',
                'exhaust_fan': {
                    'water_heater_type': df.iloc[0]['Master: How do you bath with the hot & cold water?'],
                    'exhaust_fan_size': df.iloc[0]['Master: Exhaust fan size?']
                },
                'color_theme': df.iloc[0]['Master: What is the colour theme?']
            },
            'bedroom_2': {
                'ac': df.iloc[0]['Bedroom 2: Air Conditioner (AC)?'] == 'Yes',
                'exhaust_fan': {
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
        return None

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
                product_type = product.get('type', '').lower()
                user_type = user_data['laundry'].get('washing_machine_type', '').lower().replace('-', ' ')
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
                    'relevance_score': 0,
                    'is_bestseller': product.get('is_bestseller', False)  # Add bestseller status
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
            
            # Add extra points for bestsellers
            if product.get('is_bestseller', False):
                product_groups[product_key]['relevance_score'] += 5  # Give significant boost to bestsellers
    
    # Convert product groups to list
    for product in product_groups.values():
        product['color_options'] = list(product['color_options'])
        recommendations.append(product)
    
    # Sort by relevance score (including bestseller boost) and price
    recommendations.sort(key=lambda x: (-x['relevance_score'], -float(x.get('price', 0))))
    
    return recommendations[:2]

# Function to generate final product list
def generate_final_product_list(user_data):
    """Generate the final product list based on user requirements"""
    return {
        'hall': {
            'ceiling_fans': [],
            'ac': []
        },
        'kitchen': {
            'chimney': [],
            'refrigerator': [],
            'gas_stove': []
        },
        'master_bedroom': {
            'ac': [],
            'fans': [],
            'bathroom': {
                'water_heater': [],
                'exhaust_fan': []
            }
        },
        'bedroom_2': {
            'ac': [],
            'fans': [],
            'bathroom': {
                'water_heater': [],
                'exhaust_fan': []
            }
        },
        'laundry': {
            'washing_machine': [],
            'dryer': []
        },
        'total_budget': float(user_data['What is your overall budget for home appliances?']),
        'demographics': {
            'adults': int(user_data['Adults (between the age 18 to 50)']),
            'elders': int(user_data['Elders (above the age 60)']),
            'kids': int(user_data['Kids (below the age 18)'])
        }
    }

# Function to get product recommendation reason
def get_product_recommendation_reason(product: Dict[str, Any], appliance_type: str, room: str, user_data: Dict[str, Any], total_budget: float) -> str:
    """Generate a personalized recommendation reason for a product"""
    reasons = []
    
    # Calculate price savings
    retail_price = float(product.get('retail_price', 0))
    actual_price = float(product.get('price', 0))
    savings = retail_price - actual_price
    
    if savings > 0:
        reasons.append(f"Offers excellent value with savings of ₹{savings:,.2f} compared to retail price")
    
    # Add bestseller status if applicable
    if product.get('is_bestseller', False):
        reasons.append("One of our best-selling products with proven customer satisfaction")
    
    # Add color match reason if applicable
    if product.get('color_match', False):
        reasons.append("Color options complement your room's color theme")
    
    # Calculate family size
    family_size = int(user_data['Adults (between the age 18 to 50)']) + int(user_data['Elders (above the age 60)']) + int(user_data['Kids (below the age 18)'])
    
    # Add appliance-specific reasons
    if appliance_type == 'refrigerator':
        if 'inverter' in str(product.get('features', '')).lower():
            reasons.append("Energy-efficient design helps reduce electricity bills")
        if family_size >= 4 and float(product.get('capacity', 0)) >= 300:
            reasons.append("Capacity well-suited for your family size")
    
    elif appliance_type == 'washing_machine':
        if 'front load' in str(product.get('features', '')).lower():
            reasons.append("Advanced washing technology ensures thorough cleaning while being gentle on clothes")
        if family_size >= 4 and float(product.get('capacity', 0)) >= 7:
            reasons.append("Capacity suitable for your family's laundry needs")
    
    elif appliance_type == 'ac':
        if 'inverter' in str(product.get('features', '')).lower():
            reasons.append("Energy-efficient inverter technology for lower electricity bills")
        if family_size >= 4 and float(product.get('capacity', 0)) >= 1.5:
            reasons.append("Cooling capacity appropriate for your room size")
    
    elif appliance_type == 'ceiling_fan':
        if 'BLDC' in str(product.get('features', '')).upper():
            reasons.append("Energy-efficient BLDC motor technology")
        if 'remote' in str(product.get('features', '')).lower():
            reasons.append("Convenient remote control operation")
    
    elif appliance_type == 'chimney':
        if 'auto clean' in str(product.get('features', '')).lower():
            reasons.append("Auto-clean feature reduces maintenance effort")
        if family_size >= 4:
            reasons.append("Suction capacity suitable for regular family cooking")
    
    return "\n • " + "\n • ".join(reasons)

# Function to generate PDF
def generate_pdf(user_data: Dict[str, Any], final_list: Dict[str, Any], excel_filename: str) -> None:
    """Generate a beautiful PDF with user information and product recommendations"""
    # Extract base name from Excel filename
    base_name = os.path.splitext(os.path.basename(excel_filename))[0]
    pdf_filename = f"uploads/{base_name}.pdf"
    
    doc = SimpleDocTemplate(pdf_filename, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Add custom styles
    styles.add(ParagraphStyle(
        name='RoomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        spaceAfter=20
    ))
    styles.add(ParagraphStyle(
        name='ProductTitle',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=10
    ))
    styles.add(ParagraphStyle(
        name='Description',
        parent=styles['Normal'],
        fontSize=12,
        spaceAfter=15
    ))

    # Cover page with user information
    story.append(Paragraph("BetterHome Product Recommendations", styles['Title']))
    story.append(Spacer(1, 20))
    story.append(Paragraph(f"Name: {user_data['Name']}", styles['Normal']))
    story.append(Paragraph(f"Mobile: {user_data['Mobile Number (Preferably on WhatsApp)']}", styles['Normal']))
    story.append(Paragraph(f"Email: {user_data['E-mail']}", styles['Normal']))
    story.append(Paragraph(f"Address: {user_data['Apartment Address (building, floor, and what feeling does this Chennai location bring you?)']}", styles['Normal']))
    story.append(Paragraph(f"Total Budget: ₹{float(user_data['What is your overall budget for home appliances?']):,.2f}", styles['Normal']))
    story.append(Paragraph(f"Family Size: {int(user_data['Adults (between the age 18 to 50)']) + int(user_data['Elders (above the age 60)']) + int(user_data['Kids (below the age 18)'])} members", styles['Normal']))
    story.append(PageBreak())

    total_cost = calculate_total_cost(final_list)
    
    # Process each room
    for room in ['hall', 'kitchen', 'master_bedroom', 'bedroom_2', 'laundry']:
        if room in final_list and final_list[room]:
            story.append(Paragraph(room.replace('_', ' ').title(), styles['RoomTitle']))
            
            # Add room description
            room_desc = get_room_description(room, user_data)
            story.append(Paragraph(room_desc, styles['Description']))
            
            # Add products for the room
            for appliance_type, items in final_list[room].items():
                if isinstance(items, list) and items:  # Check if items is a non-empty list
                    for item in items:
                        if isinstance(item, dict):  # Check if item is a dictionary
                            # Get product details
                            brand = item.get('brand', 'Unknown Brand')
                            model = item.get('model', 'Unknown Model')
                            price = float(item.get('price', 0))
                            features = item.get('features', [])
                            retail_price = price * 1.2  # 20% markup for retail price
                            savings = retail_price - price
                            
                            story.append(Paragraph(f"{brand} {model}", styles['ProductTitle']))
                            
                            # Get recommendation reason with total budget
                            reason = get_product_recommendation_reason(
                                item, 
                                appliance_type, 
                                room, 
                                user_data,
                                float(user_data['What is your overall budget for home appliances?'])
                            )
                            
                            details = f"""
                            Price: ₹{price:,.2f}<br/>
                            Retail Price: ₹{retail_price:,.2f}<br/>
                            You Save: ₹{savings:,.2f}<br/>
                            Features: {', '.join(features)}<br/>
                            Reason: {reason}<br/>
                            Purchase Link: <link href="{item.get('purchase_link', 'https://betterhome.co.in')}">Buy from BetterHome</link>
                            """
                            story.append(Paragraph(details, styles['Normal']))
                            story.append(Spacer(1, 15))
            
            story.append(PageBreak())
    
    # Add budget summary
    budget_utilization = (total_cost / float(user_data['What is your overall budget for home appliances?'])) * 100
    summary = f"""
    Total Cost of Recommended Products: ₹{total_cost:,.2f}<br/>
    Your Budget: ₹{float(user_data['What is your overall budget for home appliances?']):,.2f}<br/>
    Budget Utilization: {budget_utilization:.1f}%<br/>
    """
    if budget_utilization <= 100:
        summary += "Your selected products fit within your budget!"
    else:
        summary += "Note: The total cost exceeds your budget. You may want to consider alternative options."
    
    story.append(Paragraph("Budget Summary", styles['RoomTitle']))
    story.append(Paragraph(summary, styles['Normal']))
    
    doc.build(story)

def generate_text_file(user_data: Dict[str, Any], final_list: Dict[str, Any], excel_filename: str) -> None:
    """Generate a text file with user information and product recommendations"""
    # Extract base name from Excel filename
    base_name = os.path.splitext(os.path.basename(excel_filename))[0]
    txt_filename = f"uploads/{base_name}.txt"
    
    with open(txt_filename, 'w') as f:
        # Write user information
        f.write("USER INFORMATION\n")
        f.write("================\n")
        f.write(f"Name: {user_data['Name']}\n")
        f.write(f"Mobile: {user_data['Mobile Number (Preferably on WhatsApp)']}\n")
        f.write(f"Email: {user_data['E-mail']}\n")
        f.write(f"Address: {user_data['Apartment Address (building, floor, and what feeling does this Chennai location bring you?)']}\n\n")
        
        f.write("BUDGET AND FAMILY SIZE\n")
        f.write("=====================\n")
        f.write(f"Total Budget: ₹{float(user_data['What is your overall budget for home appliances?']):,.2f}\n")
        f.write(f"Family Size: {int(user_data['Adults (between the age 18 to 50)']) + int(user_data['Elders (above the age 60)']) + int(user_data['Kids (below the age 18)'])} members\n\n")
        
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
                                price = float(item.get('price', 0))
                                features = item.get('features', [])
                                retail_price = float(item.get('retail_price', price * 1.2))
                                savings = retail_price - price
                                warranty = item.get('warranty', 'Standard warranty applies')
                                delivery_time = item.get('delivery_time', 'Contact store for details')
                                
                                f.write(f"{appliance_type.replace('_', ' ').title()}: {brand} {model}\n")
                                f.write(f"Price: ₹{price:,.2f} (Retail: ₹{retail_price:,.2f})\n")
                                f.write(f"Features: {', '.join(features)}\n")
                                
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
        f.write(f"Your Budget: ₹{float(user_data['What is your overall budget for home appliances?']):,.2f}\n")
        budget_utilization = (total_cost / float(user_data['What is your overall budget for home appliances?'])) * 100
        f.write(f"Budget Utilization: {budget_utilization:.1f}%\n")
        if budget_utilization <= 100:
            f.write("Your selected products fit within your budget!\n")
        else:
            f.write("Note: The total cost exceeds your budget. You may want to consider alternative options.\n")

# Main function
def main():
    """Main function to process user requirements and generate recommendations"""
    # Get the Excel file path from command line arguments
    if len(sys.argv) < 2:
        print("Error: Please provide the Excel file path as an argument")
        sys.exit(1)
    
    excel_file = sys.argv[1]
    excel_filename = os.path.basename(excel_file)
    base_name = os.path.splitext(excel_filename)[0]  # Define base_name here
    
    # Read the Excel file
    df = pd.read_excel(excel_file)
    
    # Get the first row of data
    user_data = df.iloc[0].to_dict()
    
    # Generate initial recommendations
    final_list = generate_final_product_list(user_data)
    
    # Process hall requirements
    if user_data.get('Hall AC', '').lower() == 'yes':
        budget_category = get_budget_category(user_data['What is your overall budget for home appliances?'], 'ac')
        recommendations = get_specific_product_recommendations('ac', budget_category, user_data['Adults (between the age 18 to 50)'], user_data.get('Hall Colour Theme', ''), user_data)
        final_list['hall']['ac'] = recommendations
    
    if user_data.get('Hall Fan', '').lower() == 'yes':
        budget_category = get_budget_category(user_data['What is your overall budget for home appliances?'], 'ceiling_fan')
        recommendations = get_specific_product_recommendations('ceiling_fan', budget_category, user_data['Adults (between the age 18 to 50)'], user_data.get('Hall Colour Theme', ''), user_data)
        final_list['hall']['ceiling_fans'] = recommendations
    
    # Process kitchen requirements
    if user_data.get('Kitchen Chimney', '').lower() == 'yes':
        budget_category = get_budget_category(user_data['What is your overall budget for home appliances?'], 'chimney')
        recommendations = get_specific_product_recommendations('chimney', budget_category, user_data['Adults (between the age 18 to 50)'], None, user_data)
        final_list['kitchen']['chimney'] = recommendations
    
    if user_data.get('Kitchen Hob', '').lower() == 'yes':
        budget_category = get_budget_category(user_data['What is your overall budget for home appliances?'], 'gas_stove')
        recommendations = get_specific_product_recommendations('gas_stove', budget_category, user_data['Adults (between the age 18 to 50)'], None, user_data)
        final_list['kitchen']['gas_stove'] = recommendations
    
    # Process master bedroom requirements
    if user_data.get('Master Bedroom AC', '').lower() == 'yes':
        budget_category = get_budget_category(user_data['What is your overall budget for home appliances?'], 'ac')
        recommendations = get_specific_product_recommendations('ac', budget_category, user_data['Adults (between the age 18 to 50)'], user_data.get('Master Bedroom Colour Theme', ''), user_data)
        final_list['master_bedroom']['ac'] = recommendations
    
    if user_data.get('Master Bedroom Fan', '').lower() == 'yes':
        budget_category = get_budget_category(user_data['What is your overall budget for home appliances?'], 'ceiling_fan')
        recommendations = get_specific_product_recommendations('ceiling_fan', budget_category, user_data['Adults (between the age 18 to 50)'], user_data.get('Master Bedroom Colour Theme', ''), user_data)
        final_list['master_bedroom']['fans'] = recommendations
    
    # Process bedroom 2 requirements
    if user_data.get('Bedroom 2 AC', '').lower() == 'yes':
        budget_category = get_budget_category(user_data['What is your overall budget for home appliances?'], 'ac')
        recommendations = get_specific_product_recommendations('ac', budget_category, user_data['Adults (between the age 18 to 50)'], user_data.get('Bedroom 2 Colour Theme', ''), user_data)
        final_list['bedroom_2']['ac'] = recommendations
    
    if user_data.get('Bedroom 2 Fan', '').lower() == 'yes':
        budget_category = get_budget_category(user_data['What is your overall budget for home appliances?'], 'ceiling_fan')
        recommendations = get_specific_product_recommendations('ceiling_fan', budget_category, user_data['Adults (between the age 18 to 50)'], user_data.get('Bedroom 2 Colour Theme', ''), user_data)
        final_list['bedroom_2']['fans'] = recommendations
    
    # Process laundry requirements
    if user_data.get('Laundry Washing Machine', '').lower() == 'yes':
        budget_category = get_budget_category(user_data['What is your overall budget for home appliances?'], 'washing_machine')
        recommendations = get_specific_product_recommendations('washing_machine', budget_category, user_data['Adults (between the age 18 to 50)'], None, user_data)
        final_list['laundry']['washing_machine'] = recommendations
    
    # Generate output files
    generate_pdf(user_data, final_list, excel_filename)
    generate_text_file(user_data, final_list, excel_filename)
    
    print("\nProduct recommendations have been generated!")
    print(f"Check uploads/{base_name}.pdf and uploads/{base_name}.txt for details.")

def get_room_description(room: str, user_data: Dict[str, Any]) -> str:
    """Generate a description for each room based on user requirements"""
    if room == 'hall':
        return f"A welcoming space with {'a fan' if user_data.get('Hall Fan', '').lower() == 'yes' else 'no fan'} and {'an AC' if user_data.get('Hall AC', '').lower() == 'yes' else 'no AC'}, " \
               f"complemented by a {user_data.get('Hall Colour Theme', 'neutral')} color theme."
    
    elif room == 'kitchen':
        return f"A functional kitchen with {'a chimney' if user_data.get('Kitchen Chimney', '').lower() == 'yes' else 'no chimney'}, " \
               f"{'a hob' if user_data.get('Kitchen Hob', '').lower() == 'yes' else 'no hob'}, " \
               f"complemented by a {user_data.get('Kitchen Colour Theme', 'neutral')} color theme."
    
    elif room == 'master_bedroom':
        return f"Master bedroom with {user_data.get('Master Bedroom Colour Theme', 'neutral')} theme, " \
               f"{'an AC' if user_data.get('Master Bedroom AC', '').lower() == 'yes' else 'no AC'}, " \
               f"and {'a fan' if user_data.get('Master Bedroom Fan', '').lower() == 'yes' else 'no fan'}."
    
    elif room == 'bedroom_2':
        return f"Second bedroom with {user_data.get('Bedroom 2 Colour Theme', 'neutral')} theme, " \
               f"{'an AC' if user_data.get('Bedroom 2 AC', '').lower() == 'yes' else 'no AC'}, " \
               f"and {'a fan' if user_data.get('Bedroom 2 Fan', '').lower() == 'yes' else 'no fan'}."
    
    elif room == 'laundry':
        return f"Laundry area equipped with a {user_data.get('Laundry Washing Machine', 'standard')} washing machine."
    
    return ""

def get_user_information() -> Dict[str, Any]:
    """Read user information from the Excel file"""
    try:
        # Read the Excel file
        df = pd.read_excel('betterhome-order-form.xlsx')
        
        # Clean up column names by removing newlines and extra spaces
        df.columns = [col.split('\n')[0].strip() for col in df.columns]
        row = df.iloc[0]
        
        # Convert DataFrame to dictionary
        user_data = {
            'name': str(row['Name']),
            'mobile': str(row['Mobile Number (Preferably on WhatsApp)']),
            'email': str(row['E-mail']),
            'address': str(row['Apartment Address (building, floor, and what feeling does this Chennai location bring you?)']),
            'total_budget': float(str(row['What is your overall budget for home appliances?']).replace(',', '')),
            'demographics': {
                'adults': int(row['Adults (between the age 18 to 50)']),
                'children': int(row['Kids (below the age 18)']),
                'seniors': int(row['Elders (above the age 60)'])
            },
            'hall': {
                'fans': str(row['Hall: Fan(s)?']),
                'ac': str(row['Hall: Air Conditioner (AC)?']),
                'color_theme': str(row['Hall: Colour theme?'])
            },
            'kitchen': {
                'chimney_width': str(row['Kitchen: Chimney width?']),
                'chimney_type': None,  # Not in Excel
                'refrigerator_type': str(row['Kitchen: Refrigerator type?']),
                'refrigerator_capacity': str(row['Kitchen: Refrigerator capacity?']),
                'gas_stove_type': str(row['Kitchen: Gas stove type?']),
                'gas_stove_burners': str(row['Kitchen: Number of burners?']),
                'small_fan': str(row['Kitchen: Do you need a small fan?']),
                'color_theme': None  # Not in Excel
            },
            'master_bedroom': {
                'ac': str(row['Master: Air Conditioner (AC)?']),
                'ac_type': None,  # Not in Excel
                'color_theme': str(row['Master: What is the colour theme?']),
                'bathroom': {
                    'water_heater_type': str(row['Master: How do you bath with the hot & cold water?']),
                    'exhaust_fan_size': str(row['Master: Exhaust fan size?']),
                    'shower_type': None  # Not in Excel
                }
            },
            'bedroom_2': {
                'ac': str(row['Bedroom 2: Air Conditioner (AC)?']),
                'ac_type': None,  # Not in Excel
                'color_theme': str(row['Bedroom 2: What is the colour theme?']),
                'bathroom': {
                    'water_heater_type': str(row['Bedroom 2: How do you bath with the hot & cold water?']),
                    'exhaust_fan_size': str(row['Bedroom 2: Exhaust fan size?']),
                    'shower_type': None  # Not in Excel
                }
            },
            'laundry': {
                'washing_machine_type': str(row['Laundry: Washing Machine?']),
                'washing_machine_capacity': None,  # Not in Excel
                'dryer_type': str(row['Laundry: Dryer?']),
                'dryer_capacity': None,  # Not in Excel
                'color_theme': None  # Not in Excel
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
        
        # Override budget for testing
        user_data['total_budget'] = 100000.0
        
        return user_data
    except Exception as e:
        print(f"Error reading user information: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def calculate_total_cost(recommendations):
    """Calculate total cost of recommendations, taking only the highest-priced option for each product type"""
    total_cost = 0
    processed_types = set()  # Track which product types we've already counted
    
    for room, products in recommendations.items():
        if not isinstance(products, dict):
            continue
            
        for product_type, options in products.items():
            if not options:  # Skip if no options available
                continue
                
            # Only count the highest-priced option for each product type
            if product_type not in processed_types:
                try:
                    if isinstance(options, list):
                        prices = []
                        for option in options:
                            if isinstance(option, dict):
                                price = float(option.get('price', 0))
                            elif isinstance(option, str):
                                # Try to extract price from string format
                                try:
                                    price = float(''.join(filter(str.isdigit, option)))
                                except ValueError:
                                    price = 0
                            else:
                                price = 0
                            prices.append(price)
                        max_price = max(prices) if prices else 0
                    else:
                        max_price = 0
                    
                    total_cost += max_price
                    processed_types.add(product_type)
                except Exception as e:
                    print(f"Error processing {product_type} in {room}: {str(e)}")
                    continue
    
    return total_cost

def generate_recommendation_text(product, color_match=False):
    """Generate formatted text for a product recommendation"""
    text = []
    
    # Product name and price
    text.append(f"{product['brand']} {product['model']}")
    text.append(f"Price: {format_currency(product['price'])} (Retail: {format_currency(product['retail_price'])})")
    
    # Features
    if product.get('features'):
        text.append("Features: " + ", ".join(product['features']))
    
    # Color options
    if product.get('color_options'):
        color_text = f"Color Options: {', '.join(product['color_options'])}"
        if color_match:
            color_text += " - Matches your room's color theme!"
        text.append(color_text)
    
    # Recommendation reasons
    text.append("Why we recommend this:")
    # Add price savings reason
    savings = product['retail_price'] - product['price']
    text.append(f" • Offers excellent value with savings of {format_currency(savings)} compared to retail price")
    
    # Add bestseller status if applicable
    if product.get('is_bestseller', False):
        text.append(" • One of our best-selling products with proven customer satisfaction")
    
    # Add color match reason if applicable
    if color_match:
        text.append(" • Color options complement your room's color theme")
    
    # Add specific reasons based on product type
    if product.get('type') == 'refrigerator':
        text.append(" • Energy-efficient design helps reduce electricity bills")
    elif product.get('type') == 'washing_machine':
        text.append(" • Advanced washing technology ensures thorough cleaning while being gentle on clothes")
    elif product.get('type') == 'ac':
        text.append(" • Energy-efficient cooling with advanced features for comfort")
    
    # Add warranty and delivery info
    if product.get('warranty'):
        text.append(f"Warranty: {product['warranty']}")
    if product.get('delivery_time'):
        text.append(f"Delivery: {product['delivery_time']}")
    
    return "\n".join(text)

if __name__ == "__main__":
    main() 