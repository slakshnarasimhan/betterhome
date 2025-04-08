import pandas as pd
import json
import yaml
from typing import Dict, Any, List
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

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
                'ac': df.iloc[0]["Hall: Air Conditioner (AC)?\n(A cool, refreshing escape from Chennai's warmth, creating a space for relaxation)"] == 'Yes'
            },
            'kitchen': {
                'chimney_width': df.iloc[0]['Kitchen: Chimney width?'],
                'stove_type': df.iloc[0]['Kitchen: Gas stove type?\n(Where the warmth of home-cooked meals brings comfort and connection)'],
                'num_burners': int(df.iloc[0]['Kitchen: Number of burners?']),
                'stove_width': df.iloc[0]['Kitchen: Stove width?'],
                'small_fan': df.iloc[0]['Kitchen: Do you need a small fan?'] == 'Yes',
                'dishwasher_capacity': df.iloc[0]['Kitchen: Dishwasher capacity?'],
                'refrigerator_type': df.iloc[0]['Kitchen: Refrigerator type?'],
                'refrigerator_capacity': df.iloc[0]['Kitchen: Refrigerator capacity?']
            },
            'master_bedroom': {
                'ac': df.iloc[0]['Master: Air Conditioner (AC)?'] == 'Yes',
                'exhaust_fan': {
                    'color': df.iloc[0]['Master: Exhaust fan colour?'],
                    'size': df.iloc[0]['Master: Exhaust fan size?']
                }
            },
            'bedroom_2': {
                'ac': df.iloc[0]['Bedroom 2: Air Conditioner (AC)?'] == 'Yes',
                'exhaust_fan': {
                    'color': df.iloc[0]['Bedroom 2: Exhaust fan colour?'],
                    'size': df.iloc[0]['Bedroom 2: Exhaust fan size?']
                }
            },
            'laundry': {
                'washing_machine': df.iloc[0]['Laundry: Washing Machine?\n(Making daily chores easier, leaving more time for moments of connection and joy)'],
                'dryer': df.iloc[0]['Laundry: Dryer?\n(For hygiene: no pollen or dust on your fragrant clothes); Plus quick and convenient.'] == 'Yes'
            }
        }
        
        print("\nDebug: Processed user data:", user_data)
        print("\nDebug: Processed requirements:", requirements)
        
        return user_data, requirements
        
    except Exception as e:
        print(f"Error reading Excel file: {str(e)}")
        return None, None

# Function to get budget category
def get_budget_category(total_budget: float, appliance_type: str) -> str:
    """Determine budget category based on total budget and appliance type"""
    # Define budget allocation percentages for each appliance type
    budget_allocations = {
        'refrigerator': 0.30,  # 30% of total budget
        'washing_machine': 0.20,  # 20% of total budget
        'chimney': 0.15,  # 15% of total budget
        'ac': 0.25,  # 25% of total budget
        'geyser': 0.08,  # 8% of total budget
        'ceiling_fan': 0.03,  # 3% of total budget
        'bathroom_exhaust': 0.02,  # 2% of total budget
        'dishwasher': 0.20,  # 20% of total budget
        'dryer': 0.15,  # 15% of total budget
        'shower_system': 0.10  # 10% of total budget
    }

    # Get the allocation percentage for the appliance type (default to 10% if not specified)
    allocation = budget_allocations.get(appliance_type, 0.10)
    
    # Calculate the allocated budget for this appliance type
    allocated_budget = total_budget * allocation

    # Define budget category thresholds based on allocated budget
    # For higher total budgets, we'll prioritize premium products
    if total_budget >= 500000:  # For very high budgets (5L+), prioritize premium
        if allocated_budget >= 30000:
            return "premium"
        elif allocated_budget >= 15000:
            return "mid"
        else:
            return "budget"
    elif total_budget >= 300000:  # For high budgets (3L+), more likely to recommend premium
        if allocated_budget >= 40000:
            return "premium"
        elif allocated_budget >= 20000:
            return "mid"
        else:
            return "budget"
    else:  # For lower budgets, standard thresholds
        if allocated_budget >= 50000:
            return "premium"
        elif allocated_budget >= 25000:
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
        'ac': {'budget': 35000, 'mid': 60000},
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
def get_specific_product_recommendations(appliance_type: str, target_budget_category: str, demographics: Dict[str, int], room_color_theme: str = None) -> List[Dict[str, Any]]:
    """Get specific product recommendations based on appliance type, budget category, demographics, and room color theme"""
    catalog = load_product_catalog()
    recommendations = []
    
    # Define budget ranges for each appliance type
    budget_ranges = {
        'refrigerator': {'budget': 60000, 'mid': 85000},  # Above 85000 is premium
        'washing_machine': {'budget': 35000, 'mid': 45000},  # Adjusted to match catalog
        'chimney': {'budget': 25000, 'mid': 35000},
        'geyser': {'budget': 10000, 'mid': 20000},
        'ceiling_fan': {'budget': 4000, 'mid': 6000},
        'bathroom_exhaust': {'budget': 2000, 'mid': 4000},
        'ac': {'budget': 35000, 'mid': 60000},
        'dishwasher': {'budget': 30000, 'mid': 50000},
        'dryer': {'budget': 25000, 'mid': 45000},
        'shower_system': {'budget': 30000, 'mid': 50000}
    }
    
    # Default ranges if appliance type is not in the budget_ranges
    default_ranges = {'budget': 20000, 'mid': 40000}
    ranges = budget_ranges.get(appliance_type, default_ranges)
    
    if catalog and appliance_type in catalog:
        for product in catalog[appliance_type]:
            price = product.get('price', 0)
            
            # Determine product's budget category based on its price
            if price <= ranges['budget']:
                product_budget_category = 'budget'
            elif price <= ranges['mid']:
                product_budget_category = 'mid'
            else:
                product_budget_category = 'premium'
            
            # Only add products that match the target budget category
            if product_budget_category == target_budget_category:
                # Check if product color matches room color theme
                color_match = False
                if room_color_theme and product.get('color_options'):
                    # Simple color matching logic - can be enhanced
                    room_colors = room_color_theme.lower().split()
                    product_colors = [c.lower() for c in product.get('color_options', [])]
                    
                    for room_color in room_colors:
                        if any(room_color in pc for pc in product_colors):
                            color_match = True
                            break
                
                recommendations.append({
                    'brand': product['brand'],
                    'model': product['model'],
                    'price': product['price'],
                    'features': product.get('features', []),
                    'retail_price': product.get('price') * 1.2,  # Assuming retail price is 20% higher
                    'description': f"{product.get('type', '')} {product.get('capacity', '')}",
                    'color_options': product.get('color_options', []),
                    'color_match': color_match
                })
    
    # Sort recommendations - prioritize color matches and then by price (higher price first for premium category)
    if target_budget_category == 'premium':
        recommendations.sort(key=lambda x: (-x['color_match'], -x['price']))
    else:
        recommendations.sort(key=lambda x: (-x['color_match'], x['price']))
    
    return recommendations

# Function to generate final product list
def generate_final_product_list(user_data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a final list of preferred products with specific recommendations"""
    total_budget = user_data['total_budget']
    demographics = user_data['demographics']
    
    # Initialize room-specific recommendations
    final_list = {
        'hall': {
            'ceiling_fans': [],
            'ac': []
        },
        'kitchen': {
            'chimney': [],
            'refrigerator': [],
            'dishwasher': [],
            'small_appliances': []
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
            'total_budget': total_budget,
            'family_size': sum(demographics.values()),
            'location_considerations': ['Chennai climate considered for appliance selection'],
            'budget_allocation': {},
            'lifestyle_factors': ['Family size and composition considered']
        }
    }

    # Process hall requirements
    if user_data['hall']['fans'] and user_data['hall']['ac']:
        budget_category = get_budget_category(total_budget, 'ac')
        recommendations = get_specific_product_recommendations('ac', budget_category, demographics, user_data['hall']['color_theme'])
        final_list['hall']['ac'] = recommendations

    # Process kitchen requirements
    if user_data['kitchen']['chimney_width'] and str(user_data['kitchen']['chimney_width']).strip():  # Check if chimney is needed
        budget_category = get_budget_category(total_budget, 'chimney')
        recommendations = get_specific_product_recommendations('chimney', budget_category, demographics, user_data['kitchen']['color_theme'])
        final_list['kitchen']['chimney'] = recommendations

    if user_data['kitchen']['refrigerator_type'] and str(user_data['kitchen']['refrigerator_type']).strip():  # Check if refrigerator is needed
        budget_category = get_budget_category(total_budget, 'refrigerator')
        recommendations = get_specific_product_recommendations('refrigerator', budget_category, demographics, user_data['kitchen']['color_theme'])
        final_list['kitchen']['refrigerator'] = recommendations

    if user_data['kitchen']['dishwasher_capacity'] and str(user_data['kitchen']['dishwasher_capacity']).strip():  # Check if dishwasher is needed
        budget_category = get_budget_category(total_budget, 'dishwasher')
        recommendations = get_specific_product_recommendations('dishwasher', budget_category, demographics, user_data['kitchen']['color_theme'])
        final_list['kitchen']['dishwasher'] = recommendations

    # Process master bedroom requirements
    if user_data['master_bedroom']['ac'] and str(user_data['master_bedroom']['ac']).strip():  # Check if AC is needed
        budget_category = get_budget_category(total_budget, 'ac')
        recommendations = get_specific_product_recommendations('ac', budget_category, demographics, user_data['master_bedroom']['color_theme'])
        final_list['master_bedroom']['ac'] = recommendations

    # Add fans to master bedroom (default)
    budget_category = get_budget_category(total_budget, 'ceiling_fan')
    recommendations = get_specific_product_recommendations('ceiling_fan', budget_category, demographics, user_data['master_bedroom']['color_theme'])
    final_list['master_bedroom']['fans'] = recommendations

    # Process master bedroom bathroom requirements
    if user_data['master_bedroom']['exhaust_fan']['water_heater_type'] and str(user_data['master_bedroom']['exhaust_fan']['water_heater_type']).strip():  # Check if water heater is needed
        budget_category = get_budget_category(total_budget, 'geyser')
        recommendations = get_specific_product_recommendations('geyser', budget_category, demographics, user_data['master_bedroom']['color_theme'])
        final_list['master_bedroom']['bathroom']['water_heater'] = recommendations

    if user_data['master_bedroom']['exhaust_fan']['exhaust_fan_size'] and str(user_data['master_bedroom']['exhaust_fan']['exhaust_fan_size']).strip():  # Check if exhaust fan is needed
        budget_category = get_budget_category(total_budget, 'bathroom_exhaust')
        recommendations = get_specific_product_recommendations('bathroom_exhaust', budget_category, demographics, user_data['master_bedroom']['color_theme'])
        final_list['master_bedroom']['bathroom']['exhaust_fan'] = recommendations

    # Process bedroom 2 requirements
    if user_data['bedroom_2']['ac'] and str(user_data['bedroom_2']['ac']).strip():  # Check if AC is needed
        budget_category = get_budget_category(total_budget, 'ac')
        recommendations = get_specific_product_recommendations('ac', budget_category, demographics, user_data['bedroom_2']['color_theme'])
        final_list['bedroom_2']['ac'] = recommendations

    # Add fans to bedroom 2 (default)
    budget_category = get_budget_category(total_budget, 'ceiling_fan')
    recommendations = get_specific_product_recommendations('ceiling_fan', budget_category, demographics, user_data['bedroom_2']['color_theme'])
    final_list['bedroom_2']['fans'] = recommendations

    # Process bedroom 2 bathroom requirements
    if user_data['bedroom_2']['exhaust_fan']['water_heater_type'] and str(user_data['bedroom_2']['exhaust_fan']['water_heater_type']).strip():  # Check if water heater is needed
        budget_category = get_budget_category(total_budget, 'geyser')
        recommendations = get_specific_product_recommendations('geyser', budget_category, demographics, user_data['bedroom_2']['color_theme'])
        final_list['bedroom_2']['bathroom']['water_heater'] = recommendations

    if user_data['bedroom_2']['exhaust_fan']['exhaust_fan_size'] and str(user_data['bedroom_2']['exhaust_fan']['exhaust_fan_size']).strip():  # Check if exhaust fan is needed
        budget_category = get_budget_category(total_budget, 'bathroom_exhaust')
        recommendations = get_specific_product_recommendations('bathroom_exhaust', budget_category, demographics, user_data['bedroom_2']['color_theme'])
        final_list['bedroom_2']['bathroom']['exhaust_fan'] = recommendations

    # Process laundry requirements
    if user_data['laundry']['washing_machine'] and str(user_data['laundry']['washing_machine']).strip():  # Check if washing machine is needed
        budget_category = get_budget_category(total_budget, 'washing_machine')
        recommendations = get_specific_product_recommendations('washing_machine', budget_category, demographics, user_data['laundry']['color_theme'])
        final_list['laundry']['washing_machine'] = recommendations

    if user_data['laundry']['dryer'] and str(user_data['laundry']['dryer']).strip():  # Check if dryer is needed
        budget_category = get_budget_category(total_budget, 'dryer')
        recommendations = get_specific_product_recommendations('dryer', budget_category, demographics, user_data['laundry']['color_theme'])
        final_list['laundry']['dryer'] = recommendations

    return final_list

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

# Function to generate PDF
def generate_pdf(user_data: Dict[str, Any], final_list: Dict[str, Any]) -> None:
    """Generate a beautiful PDF with user information and product recommendations"""
    doc = SimpleDocTemplate("product_recommendations.pdf", pagesize=letter)
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
    story.append(Paragraph(f"Name: {user_data['name']}", styles['Normal']))
    story.append(Paragraph(f"Mobile: {user_data['mobile']}", styles['Normal']))
    story.append(Paragraph(f"Email: {user_data['email']}", styles['Normal']))
    story.append(Paragraph(f"Address: {user_data['address']}", styles['Normal']))
    story.append(Paragraph(f"Total Budget: ₹{user_data['total_budget']:,.2f}", styles['Normal']))
    story.append(Paragraph(f"Family Size: {sum(user_data['demographics'].values())} members", styles['Normal']))
    story.append(PageBreak())

    total_cost = 0
    
    # Process each room
    for room in ['hall', 'kitchen', 'master_bedroom', 'bedroom_2', 'laundry']:
        if room in final_list and final_list[room]:
            story.append(Paragraph(room.replace('_', ' ').title(), styles['RoomTitle']))
            
            # Add room description
            room_desc = get_room_description(room, user_data)
            story.append(Paragraph(room_desc, styles['Description']))
            
            # Add products for the room
            for appliance_type, items in final_list[room].items():
                for item in items:
                    story.append(Paragraph(f"{item['name']}", styles['ProductTitle']))
                    
                    # Get recommendation reason with total budget
                    reason = get_product_recommendation_reason(
                        item, 
                        appliance_type, 
                        room, 
                        user_data[room.replace('_', '')]['color_theme'] if 'color_theme' in user_data[room.replace('_', '')] else None,
                        user_data['total_budget']
                    )
                    
                    price = float(item['price'])
                    retail_price = price * 1.2  # 20% markup for retail price
                    savings = retail_price - price
                    
                    details = f"""
                    Price: ₹{price:,.2f}<br/>
                    Retail Price: ₹{retail_price:,.2f}<br/>
                    You Save: ₹{savings:,.2f}<br/>
                    Features: {item['features']}<br/>
                    Reason: {reason}<br/>
                    Purchase Link: <link href="{item.get('purchase_link', 'https://betterhome.co.in')}">Buy from BetterHome</link>
                    """
                    story.append(Paragraph(details, styles['Normal']))
                    story.append(Spacer(1, 15))
                    
                    total_cost += price
            
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

def generate_text_file(user_data: Dict[str, Any], final_list: Dict[str, Any]) -> None:
    """Generate a text file with user information and product recommendations"""
    with open("product_recommendations.txt", "w") as f:
        # Write user information
        f.write("USER INFORMATION\n")
        f.write("================\n")
        f.write(f"Name: {user_data['name']}\n")
        f.write(f"Mobile: {user_data['mobile']}\n")
        f.write(f"Email: {user_data['email']}\n")
        f.write(f"Address: {user_data['address']}\n\n")

        # Write budget and family size
        f.write("BUDGET AND FAMILY SIZE\n")
        f.write("=====================\n")
        f.write(f"Total Budget: {format_currency(final_list['summary']['total_budget'])}\n")
        f.write(f"Family Size: {final_list['summary']['family_size']} members\n\n")

        # Write room-wise recommendations
        f.write("ROOM-WISE RECOMMENDATIONS\n")
        f.write("========================\n\n")

        # Hall
        f.write("HALL\n")
        f.write("----\n")
        for fan in final_list['hall']['ceiling_fans']:
            f.write(f"Ceiling Fan: {fan['brand']} {fan['model']}\n")
            f.write(f"Price: {format_currency(fan['price'])} (Retail: {format_currency(fan['retail_price'])})\n")
            f.write(f"Features: {', '.join(fan['features'])}\n")
            if fan.get('color_match', False):
                f.write(f"Color Options: {', '.join(fan.get('color_options', []))} - Matches your room's color theme!\n")
            reason = get_product_recommendation_reason(fan, 'ceiling_fan', 'hall', user_data['demographics'], final_list['summary']['total_budget'])
            f.write(f"Why we recommend this:\n{reason}\n\n")

        # Kitchen
        f.write("KITCHEN\n")
        f.write("-------\n")
        for appliance_type in ['chimney', 'refrigerator', 'dishwasher']:
            if final_list['kitchen'][appliance_type]:
                for item in final_list['kitchen'][appliance_type]:
                    f.write(f"{appliance_type.title()}: {item['brand']} {item['model']}\n")
                    f.write(f"Price: {format_currency(item['price'])} (Retail: {format_currency(item['retail_price'])})\n")
                    f.write(f"Features: {', '.join(item['features'])}\n")
                    if item.get('color_match', False):
                        f.write(f"Color Options: {', '.join(item.get('color_options', []))} - Matches your room's color theme!\n")
                    reason = get_product_recommendation_reason(item, appliance_type, 'kitchen', user_data['kitchen']['color_theme'])
                    f.write(f"Why we recommend this:\n{reason}\n\n")

        # Bedrooms
        for bedroom in ['master_bedroom', 'bedroom_2']:
            f.write(f"{bedroom.replace('_', ' ').upper()}\n")
            f.write("-" * len(bedroom) + "\n")
            # Bedroom appliances
            for fan in final_list[bedroom]['fans']:
                f.write(f"Ceiling Fan: {fan['brand']} {fan['model']}\n")
                f.write(f"Price: {format_currency(fan['price'])} (Retail: {format_currency(fan['retail_price'])})\n")
                f.write(f"Features: {', '.join(fan['features'])}\n")
                if fan.get('color_match', False):
                    f.write(f"Color Options: {', '.join(fan.get('color_options', []))} - Matches your room's color theme!\n")
                reason = get_product_recommendation_reason(fan, 'ceiling_fan', bedroom, user_data['demographics'], final_list['summary']['total_budget'])
                f.write(f"Why we recommend this:\n{reason}\n\n")
            
            # Bathroom appliances
            f.write("Bathroom\n")
            for exhaust_fan in final_list[bedroom]['bathroom']['exhaust_fan']:
                f.write(f"Exhaust Fan: {exhaust_fan['brand']} {exhaust_fan['model']}\n")
                f.write(f"Price: {format_currency(exhaust_fan['price'])} (Retail: {format_currency(exhaust_fan['retail_price'])})\n")
                f.write(f"Features: {', '.join(exhaust_fan['features'])}\n")
                if exhaust_fan.get('color_match', False):
                    f.write(f"Color Options: {', '.join(exhaust_fan.get('color_options', []))} - Matches your room's color theme!\n")
                reason = get_product_recommendation_reason(exhaust_fan, 'bathroom_exhaust', f"{bedroom}.bathroom", user_data['demographics'], final_list['summary']['total_budget'])
                f.write(f"Why we recommend this:\n{reason}\n\n")

        # Laundry
        f.write("LAUNDRY\n")
        f.write("-------\n")
        for appliance_type in ['washing_machine', 'dryer']:
            if final_list['laundry'][appliance_type]:
                for item in final_list['laundry'][appliance_type]:
                    f.write(f"{appliance_type.replace('_', ' ').title()}: {item['brand']} {item['model']}\n")
                    f.write(f"Price: {format_currency(item['price'])} (Retail: {format_currency(item['retail_price'])})\n")
                    f.write(f"Features: {', '.join(item['features'])}\n")
                    if item.get('color_match', False):
                        f.write(f"Color Options: {', '.join(item.get('color_options', []))} - Matches your room's color theme!\n")
                    reason = get_product_recommendation_reason(item, appliance_type, 'laundry', user_data['demographics'], final_list['summary']['total_budget'])
                    f.write(f"Why we recommend this:\n{reason}\n\n")

# Main function
def main():
    user_data, requirements = analyze_user_requirements('betterhome-order-form.xlsx')
    if user_data:
        final_list = generate_final_product_list(user_data)
        generate_pdf(user_data, final_list)
        generate_text_file(user_data, final_list)
        print("PDF and text file generated successfully:")
        print("- product_recommendations.pdf")
        print("- product_recommendations.txt")
    else:
        print("Failed to analyze user requirements.")

if __name__ == "__main__":
    main() 