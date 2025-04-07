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

# Function to get specific product recommendations
def get_specific_product_recommendations(appliance_type: str, budget_category: str, demographics: Dict[str, int]) -> Dict[str, Any]:
    """Get specific product recommendations based on appliance type, budget category, and demographics"""
    catalog = load_product_catalog()
    recommendations = {
        'catalog_matches': [],
        'maintenance_tips': []
    }
    # Logic to match products from catalog
    return recommendations

# Function to generate final product list
def generate_final_product_list(user_data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a final list of preferred products with specific recommendations"""
    final_list = {
        'kitchen_appliances': {},
        'bathroom_appliances': {},
        'comfort_appliances': {},
        'summary': {
            'total_budget': user_data['requirements']['budget'],
            'family_size': user_data['requirements']['total_family_size'],
            'location_considerations': [],
            'budget_allocation': {},
            'lifestyle_factors': []
        }
    }
    # Logic to populate final_list
    return final_list

# Function to generate PDF
def generate_pdf(user_data: Dict[str, Any], final_list: Dict[str, Any]) -> None:
    """Generate a PDF with user information and product recommendations"""
    doc = SimpleDocTemplate("product_recommendations.pdf", pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Add user information
    story.append(Paragraph(f"User Information", styles['Heading1']))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Name: {user_data['name']}", styles['Normal']))
    story.append(Paragraph(f"Mobile: {user_data['mobile']}", styles['Normal']))
    story.append(Paragraph(f"Email: {user_data['email']}", styles['Normal']))
    story.append(Paragraph(f"Address: {user_data['address']}", styles['Normal']))
    story.append(Spacer(1, 12))

    # Add budget and family size
    story.append(Paragraph(f"Budget and Family Size", styles['Heading1']))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Total Budget: {format_currency(final_list['summary']['total_budget'])}", styles['Normal']))
    story.append(Paragraph(f"Family Size: {final_list['summary']['family_size']} members", styles['Normal']))
    story.append(Spacer(1, 12))

    # Add product recommendations
    story.append(Paragraph(f"Product Recommendations", styles['Heading1']))
    story.append(Spacer(1, 12))
    for category, products in final_list.items():
        if category != 'summary':
            story.append(Paragraph(f"{category.replace('_', ' ').title()}", styles['Heading2']))
            story.append(Spacer(1, 12))
            for product in products:
                story.append(Paragraph(f"{product['brand']} {product['model']}", styles['Normal']))
                story.append(Paragraph(f"Price: {format_currency(product['price'])}", styles['Normal']))
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