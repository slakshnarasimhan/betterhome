import pandas as pd
import json
import yaml
from typing import Dict, Any, List
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
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
def analyze_user_requirements(excel_file: str) -> Dict[str, Any]:
    """Analyze user requirements from Excel file"""
    try:
        df = pd.read_excel(excel_file)
        print("Columns found in Excel file:", df.columns.tolist())  # Debug line to print columns
        user_data = {
            'name': df['Name'].iloc[0],
            'mobile': df['Mobile Number (Preferably on WhatsApp)'].iloc[0],
            'email': df['E-mail'].iloc[0],
            'address': df['Apartment Address (building, floor, and what feeling does this Chennai location bring you?)'].iloc[0],
            'requirements': {
                'bedrooms': df['Number of bedrooms'].iloc[0],
                'bathrooms': df['Number of bathrooms'].iloc[0],
                'budget': df['What is your overall budget for home appliances?'].iloc[0],
                'total_family_size': (
                    df['Adults (between the age 18 to 50)'].iloc[0] +
                    df['Elders (above the age 60)'].iloc[0] +
                    df['Kids (below the age 18)'].iloc[0]
                )
            },
            'demographics': {
                'adults': df['Adults (between the age 18 to 50)'].iloc[0],
                'elders': df['Elders (above the age 60)'].iloc[0],
                'kids': df['Kids (below the age 18)'].iloc[0]
            }
        }
        return user_data
    except Exception as e:
        print(f"Error processing Excel file: {e}")
        print("Available columns:", df.columns.tolist())  # Print available columns if there's an error
        return {}

# Function to get budget category
def get_budget_category(total_budget: float, appliance_type: str) -> str:
    """Determine budget category based on total budget and appliance type"""
    if total_budget >= 1000000:
        return "premium"
    elif total_budget >= 500000:
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
        'ceiling_fan': {'budget': 3000, 'mid': 6000},
        'bathroom_exhaust': {'budget': 2000, 'mid': 4000},  # Updated from 'exhaust_fan'
        'ac': {'budget': 35000, 'mid': 60000},
        'dishwasher': {'budget': 30000, 'mid': 50000},
        'dryer': {'budget': 25000, 'mid': 45000},
        'shower_system': {'budget': 30000, 'mid': 50000}  # Added shower systems
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
def get_specific_product_recommendations(appliance_type: str, target_budget_category: str, demographics: Dict[str, int]) -> List[Dict[str, Any]]:
    """Get specific product recommendations based on appliance type, budget category, and demographics"""
    catalog = load_product_catalog()
    recommendations = []
    
    if catalog and appliance_type in catalog:
        for product in catalog[appliance_type]:
            product_budget_category = get_budget_category_for_product(product['price'], appliance_type)
            if product_budget_category == target_budget_category:
                recommendations.append({
                    'brand': product['brand'],
                    'model': product['model'],
                    'price': product['price'],
                    'features': product.get('features', []),
                    'retail_price': product.get('price') * 1.2,  # Assuming retail price is 20% higher
                    'description': f"{product.get('type', '')} {product.get('capacity', '')}"
                })
    
    return recommendations

# Function to generate final product list
def generate_final_product_list(user_data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a final list of preferred products with specific recommendations"""
    total_budget = user_data['requirements']['budget']
    demographics = user_data['demographics']
    
    # Initialize room-specific recommendations
    final_list = {
        'hall': {
            'fans': [],
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
            'family_size': user_data['requirements']['total_family_size'],
            'location_considerations': ['Chennai climate considered for appliance selection'],
            'budget_allocation': {},
            'lifestyle_factors': ['Family size and composition considered']
        }
    }

    # Get recommendations for each appliance type
    appliance_types = {
        'ceiling_fan': ['hall', 'master_bedroom', 'bedroom_2'],
        'ac': ['hall', 'master_bedroom', 'bedroom_2'],
        'bathroom_exhaust': ['master_bedroom.bathroom', 'bedroom_2.bathroom'],  # Updated from 'exhaust_fan'
        'chimney': ['kitchen'],
        'refrigerator': ['kitchen'],
        'dishwasher': ['kitchen'],
        'geyser': ['master_bedroom.bathroom', 'bedroom_2.bathroom'],  # Updated from 'water_heater'
        'washing_machine': ['laundry'],
        'dryer': ['laundry'],
        'shower_system': ['master_bedroom.bathroom', 'bedroom_2.bathroom']  # Added shower systems
    }

    for appliance, rooms in appliance_types.items():
        budget_category = get_budget_category(total_budget, appliance)
        recommendations = get_specific_product_recommendations(appliance, budget_category, demographics)
        
        for room in rooms:
            if '.' in room:  # Handle nested structure (e.g., master_bedroom.bathroom)
                main_room, sub_room = room.split('.')
                if appliance == 'bathroom_exhaust':
                    final_list[main_room][sub_room]['exhaust_fan'] = recommendations
                elif appliance == 'geyser':
                    final_list[main_room][sub_room]['water_heater'] = recommendations
                elif appliance == 'shower_system':
                    final_list[main_room][sub_room]['shower'] = recommendations
            else:
                if appliance == 'ceiling_fan':
                    final_list[room]['fans'] = recommendations
                elif appliance == 'ac':
                    final_list[room]['ac'] = recommendations
                elif appliance in ['chimney', 'refrigerator', 'dishwasher']:
                    final_list['kitchen'][appliance] = recommendations
                elif appliance in ['washing_machine', 'dryer']:
                    final_list['laundry'][appliance] = recommendations

    return final_list

# Function to generate PDF
def generate_pdf(user_data: Dict[str, Any], final_list: Dict[str, Any]) -> None:
    """Generate a PDF with user information and product recommendations"""
    doc = SimpleDocTemplate("product_recommendations.pdf", pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Add user information
    story.append(Paragraph("User Information", styles['Heading1']))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Name: {user_data['name']}", styles['Normal']))
    story.append(Paragraph(f"Mobile: {user_data['mobile']}", styles['Normal']))
    story.append(Paragraph(f"Email: {user_data['email']}", styles['Normal']))
    story.append(Paragraph(f"Address: {user_data['address']}", styles['Normal']))
    story.append(Spacer(1, 12))

    # Add budget and family size
    story.append(Paragraph("Budget and Family Size", styles['Heading1']))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Total Budget: {format_currency(final_list['summary']['total_budget'])}", styles['Normal']))
    story.append(Paragraph(f"Family Size: {final_list['summary']['family_size']} members", styles['Normal']))
    story.append(Spacer(1, 12))

    # Add room-wise recommendations
    story.append(Paragraph("Room-wise Recommendations", styles['Heading1']))
    story.append(Spacer(1, 12))

    # Hall
    story.append(Paragraph("Hall", styles['Heading2']))
    for fan in final_list['hall']['fans']:
        story.append(Paragraph(f"Ceiling Fan: {fan['brand']} {fan['model']}", styles['Normal']))
        story.append(Paragraph(f"Price: {format_currency(fan['price'])} (Retail: {format_currency(fan['retail_price'])})", styles['Normal']))
        story.append(Paragraph(f"Features: {', '.join(fan['features'])}", styles['Normal']))
        story.append(Spacer(1, 12))

    # Kitchen
    story.append(Paragraph("Kitchen", styles['Heading2']))
    for appliance_type in ['chimney', 'refrigerator', 'dishwasher']:
        if final_list['kitchen'][appliance_type]:
            for item in final_list['kitchen'][appliance_type]:
                story.append(Paragraph(f"{appliance_type.title()}: {item['brand']} {item['model']}", styles['Normal']))
                story.append(Paragraph(f"Price: {format_currency(item['price'])} (Retail: {format_currency(item['retail_price'])})", styles['Normal']))
                story.append(Paragraph(f"Features: {', '.join(item['features'])}", styles['Normal']))
                story.append(Spacer(1, 12))

    # Bedrooms
    for bedroom in ['master_bedroom', 'bedroom_2']:
        story.append(Paragraph(bedroom.replace('_', ' ').title(), styles['Heading2']))
        # Bedroom appliances
        for fan in final_list[bedroom]['fans']:
            story.append(Paragraph(f"Ceiling Fan: {fan['brand']} {fan['model']}", styles['Normal']))
            story.append(Paragraph(f"Price: {format_currency(fan['price'])} (Retail: {format_currency(fan['retail_price'])})", styles['Normal']))
            story.append(Paragraph(f"Features: {', '.join(fan['features'])}", styles['Normal']))
            story.append(Spacer(1, 12))
        
        # Bathroom appliances
        story.append(Paragraph("Bathroom", styles['Heading3']))
        for exhaust_fan in final_list[bedroom]['bathroom']['exhaust_fan']:
            story.append(Paragraph(f"Exhaust Fan: {exhaust_fan['brand']} {exhaust_fan['model']}", styles['Normal']))
            story.append(Paragraph(f"Price: {format_currency(exhaust_fan['price'])} (Retail: {format_currency(exhaust_fan['retail_price'])})", styles['Normal']))
            story.append(Paragraph(f"Features: {', '.join(exhaust_fan['features'])}", styles['Normal']))
            story.append(Spacer(1, 12))

    # Laundry
    story.append(Paragraph("Laundry", styles['Heading2']))
    for appliance_type in ['washing_machine', 'dryer']:
        if final_list['laundry'][appliance_type]:
            for item in final_list['laundry'][appliance_type]:
                story.append(Paragraph(f"{appliance_type.replace('_', ' ').title()}: {item['brand']} {item['model']}", styles['Normal']))
                story.append(Paragraph(f"Price: {format_currency(item['price'])} (Retail: {format_currency(item['retail_price'])})", styles['Normal']))
                story.append(Paragraph(f"Features: {', '.join(item['features'])}", styles['Normal']))
                story.append(Spacer(1, 12))

    doc.build(story)

# Main function
def main():
    user_data = analyze_user_requirements('betterhome-order-form.xlsx')
    if user_data:
        final_list = generate_final_product_list(user_data)
        generate_pdf(user_data, final_list)
        print("PDF generated successfully: product_recommendations.pdf")
    else:
        print("Failed to analyze user requirements.")

if __name__ == "__main__":
    main() 