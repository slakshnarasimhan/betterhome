import json
import pandas as pd
from typing import Dict, Any, List
import yaml

def format_currency(amount: float) -> str:
    """Format amount in Indian Rupees format"""
    return f"₹{amount:,.2f}"

def print_product_details(product: Dict[str, Any]) -> None:
    """Print detailed product information in a formatted way"""
    print(f"\n    {product['brand']} {product['model']}")
    print(f"    {'='*50}")
    print(f"    Price: {format_currency(product['price'])}")
    print(f"    Type: {product.get('type', 'Not specified')}")
    
    if 'capacity' in product:
        print(f"    Capacity: {product['capacity']}")
    if 'energy_rating' in product:
        print(f"    Energy Rating: {product['energy_rating']}")
    
    print(f"    Features:")
    for feature in product.get('features', []):
        print(f"      • {feature}")
    
    print(f"    Warranty: {product.get('warranty', 'Standard warranty applies')}")
    
    if 'dimensions' in product:
        dim = product['dimensions']
        print(f"    Dimensions: {dim['width']}W x {dim['depth']}D x {dim['height']}H cm")
    
    if 'color_options' in product:
        print(f"    Colors Available: {', '.join(product['color_options'])}")
    elif 'finish' in product:
        print(f"    Finish: {product['finish']}")
        
    if 'airflow_rate' in product:
        print(f"    Airflow Rate: {product['airflow_rate']}")
    if 'flow_rate' in product:
        print(f"    Flow Rate: {product['flow_rate']}")
    if 'power_consumption' in product:
        print(f"    Power Consumption: {product['power_consumption']}")
    if 'wattage' in product:
        print(f"    Wattage: {product['wattage']}")
    if 'pressure_rating' in product:
        print(f"    Pressure Rating: {product['pressure_rating']}")
    
    print(f"    Availability: {'In Stock' if product.get('in_stock', False) else 'Out of Stock'}")
    print(f"    Estimated Delivery: {product.get('delivery_time', 'Contact store for details')}")

def get_budget_category(total_budget: float) -> str:
    """Determine budget category based on total budget"""
    if total_budget >= 500000:
        return "premium"
    elif total_budget >= 300000:
        return "mid"
    else:
        return "budget"

def load_product_catalog() -> Dict[str, List[Dict[str, Any]]]:
    """Load product catalog from JSON file"""
    try:
        with open('product_catalog.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("Product catalog file not found.")
        return {}

def load_config() -> Dict[str, Any]:
    """Load configuration from YAML file"""
    try:
        with open('home_config.yaml', 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print("Configuration file not found.")
        return {}

def analyze_user_requirements(excel_file: str = 'betterhome-order-form.xlsx') -> Dict[str, Any]:
    """Analyze user requirements from Excel file"""
    try:
        df = pd.read_excel(excel_file)
        print(f"Successfully read {len(df)} entries from Excel file")
        print("\nColumns found:")
        for col in df.columns:
            print(f"- {col}")
        
        # Process first entry
        entry = df.iloc[0]
        
        # Extract user information with safe access
        def safe_get(key: str, default: str = "") -> str:
            try:
                return str(entry[key]) if pd.notna(entry[key]) else default
            except:
                return default

        def safe_get_int(key: str, default: int = 0) -> int:
            try:
                return int(float(entry[key])) if pd.notna(entry[key]) else default
            except:
                return default
        
        user_info = {
            'name': safe_get('Name'),
            'mobile': safe_get('Mobile Number (Preferably on WhatsApp)'),
            'email': safe_get('E-mail'),
            'address': safe_get('Apartment Address (building, floor, and what feeling does this Chennai location bring you?)')
        }
        
        # Calculate total family size
        adults = safe_get_int('Adults (between the age 18 to 50)')
        elders = safe_get_int('Elders (above the age 60)')
        kids = safe_get_int('Kids (below the age 18)')
        total_family_size = adults + elders + kids
        
        # Extract home requirements
        bedrooms = safe_get_int('Number of bedrooms')
        bathrooms = safe_get_int('Number of bathrooms')
        budget = safe_get_int('What is your overall budget for home appliances?', 500000)  # Default budget
        
        requirements = {
            'total_family_size': total_family_size,
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'budget': budget,
            'hall': {
                'color_theme': safe_get('Hall: Colour theme?'),
                'square_feet': safe_get('Hall: What is the square feet ?'),
                'fan_requirement': safe_get('Hall: Fan(s)?'),
                'ac_requirement': safe_get('Hall: Air Conditioner (AC)?')
            },
            'kitchen': {
                'chimney_width': safe_get('Kitchen: Chimney width?'),
                'stove_type': safe_get('Kitchen: Gas stove type?'),
                'burners': safe_get('Kitchen: Number of burners?'),
                'stove_width': safe_get('Kitchen: Stove width?'),
                'small_fan': safe_get('Kitchen: Do you need a small fan?'),
                'dishwasher': safe_get('Kitchen: Dishwasher capacity?'),
                'refrigerator_type': safe_get('Kitchen: Refrigerator type?'),
                'refrigerator_capacity': safe_get('Kitchen: Refrigerator capacity?')
            }
        }
        
        # Demographics information
        demographics = {
            'age_groups': {
                'adults': adults,
                'elders': elders,
                'kids': kids
            },
            'location_factors': {
                'climate': {
                    'type': 'tropical',  # Chennai is tropical
                    'humidity': 'high'
                },
                'coastal_influence': True  # Chennai is a coastal city
            }
        }
        
        return {
            'user_info': user_info,
            'requirements': requirements,
            'demographics': demographics
        }
        
    except Exception as e:
        print(f"Error processing Excel file: {str(e)}")
        # Return default values for testing
        return {
            'user_info': {
                'name': 'Test User',
                'mobile': '9876543210',
                'email': 'test@example.com',
                'address': 'Chennai'
            },
            'requirements': {
                'total_family_size': 5,
                'bedrooms': 3,
                'bathrooms': 2,
                'budget': 500000,
                'hall': {
                    'color_theme': 'White',
                    'square_feet': '300',
                    'fan_requirement': 'Yes',
                    'ac_requirement': 'Yes'
                },
                'kitchen': {
                    'chimney_width': '90',
                    'stove_type': '4 burner',
                    'burners': '4',
                    'stove_width': '60',
                    'small_fan': 'Yes',
                    'dishwasher': 'Standard',
                    'refrigerator_type': 'Double door',
                    'refrigerator_capacity': '500L'
                }
            },
            'demographics': {
                'age_groups': {
                    'adults': 2,
                    'elders': 2,
                    'kids': 1
                },
                'location_factors': {
                    'climate': {
                        'type': 'tropical',
                        'humidity': 'high'
                    },
                    'coastal_influence': True
                }
            }
        }

def get_specific_product_recommendations(appliance_type: str, budget_category: str, demographics: Dict[str, Any], room_info: Dict[str, Any] = None) -> Dict[str, Any]:
    """Get specific product recommendations with model numbers and brands"""
    recommendations = {
        'primary_recommendation': {},
        'alternative_options': [],
        'maintenance_tips': [],
        'installation_requirements': {},
        'energy_efficiency': {},
        'warranty_info': {},
        'catalog_matches': []
    }
    
    # Load product catalog
    catalog = load_product_catalog()
    
    # Find matching products from catalog
    if appliance_type in catalog:
        for product in catalog[appliance_type]:
            if budget_category == 'premium' and product['price'] >= 30000:
                recommendations['catalog_matches'].append(product)
            elif budget_category == 'mid' and 15000 <= product['price'] < 30000:
                recommendations['catalog_matches'].append(product)
            elif budget_category == 'budget' and product['price'] < 15000:
                recommendations['catalog_matches'].append(product)
    
    # Sort matches by price (descending for premium, ascending for budget)
    recommendations['catalog_matches'].sort(
        key=lambda x: x['price'],
        reverse=(budget_category == 'premium')
    )
    
    # Add maintenance tips based on location factors
    if demographics.get('location_factors', {}).get('climate', {}).get('type') == 'tropical':
        recommendations['maintenance_tips'].extend([
            'Regular cleaning of filters/components',
            'Anti-corrosion spray application every 6 months',
            'Annual professional maintenance recommended',
            'Check seals and gaskets quarterly',
            'Use dehumidifier bags if needed'
        ])
    
    if demographics.get('location_factors', {}).get('coastal_influence'):
        recommendations['maintenance_tips'].extend([
            'Monthly inspection for rust spots',
            'Apply marine-grade protective coating',
            'Increased frequency of maintenance checks',
            'Use rust-prevention sprays',
            'Additional warranty coverage recommended'
        ])
    
    return recommendations

def generate_final_product_list(user_data: Dict[str, Any], suggestions: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a final list of preferred products with specific recommendations"""
    final_list = {
        'kitchen_appliances': {},
        'bathroom_appliances': {},
        'comfort_appliances': {},
        'summary': {
            'total_budget': user_data['requirements']['budget'],
            'family_size': user_data['requirements']['total_family_size'],
            'location_considerations': [],
            'budget_allocation': {}
        }
    }

    # Calculate budget allocation
    total_budget = float(user_data['requirements']['budget'])
    budget_category = get_budget_category(total_budget)
    
    print("\n" + "="*80)
    print("FINAL PRODUCT RECOMMENDATIONS")
    print("="*80)
    
    print("\nBudget Summary:")
    print("--------------")
    print(f"Total Budget: {format_currency(total_budget)}")
    print(f"Budget Category: {budget_category.upper()}")
    print(f"Family Size: {user_data['requirements']['total_family_size']} members")
    
    # Kitchen Appliances
    print("\nKitchen Appliances:")
    print("-----------------")
    kitchen_budget = total_budget * 0.4
    print(f"Allocated Budget: {format_currency(kitchen_budget)}")
    
    for appliance in ['chimney', 'refrigerator']:
        specific_recommendation = get_specific_product_recommendations(
            appliance, budget_category, user_data['demographics']
        )
        print(f"\n{appliance.title()} Recommendations:")
        if specific_recommendation['catalog_matches']:
            for product in specific_recommendation['catalog_matches']:
                print_product_details(product)
        else:
            print("    No matching products found in catalog")
    
    # Bathroom Appliances
    print("\nBathroom Appliances:")
    print("------------------")
    bathroom_budget = total_budget * 0.3
    print(f"Allocated Budget: {format_currency(bathroom_budget)}")
    
    for appliance in ['geyser', 'shower_system', 'bathroom_exhaust']:
        specific_recommendation = get_specific_product_recommendations(
            appliance, budget_category, user_data['demographics']
        )
        print(f"\n{appliance.replace('_', ' ').title()} Recommendations:")
        if specific_recommendation['catalog_matches']:
            for product in specific_recommendation['catalog_matches']:
                print_product_details(product)
        else:
            print("    No matching products found in catalog")
    
    # Comfort Appliances
    print("\nComfort Appliances:")
    print("-----------------")
    comfort_budget = total_budget * 0.3
    print(f"Allocated Budget: {format_currency(comfort_budget)}")
    
    for appliance in ['washing_machine', 'ceiling_fan']:
        specific_recommendation = get_specific_product_recommendations(
            appliance, budget_category, user_data['demographics']
        )
        print(f"\n{appliance.replace('_', ' ').title()} Recommendations:")
        if specific_recommendation['catalog_matches']:
            for product in specific_recommendation['catalog_matches']:
                print_product_details(product)
        else:
            print("    No matching products found in catalog")
    
    # Add location-specific considerations
    if user_data.get('location_factors', {}).get('climate', {}).get('type') == 'tropical':
        print("\nLocation-Specific Considerations:")
        print("------------------------------")
        print("• High humidity environment - recommended anti-rust and moisture-resistant models")
        print("• Regular maintenance schedule advised")
        print("• Additional warranty coverage recommended for sensitive electronics")
    
    return final_list

if __name__ == "__main__":
    # Analyze user requirements
    user_data = analyze_user_requirements()
    
    if user_data:
        # Generate and print recommendations
        suggestions = {}  # This would be populated with initial suggestions
        final_list = generate_final_product_list(user_data, suggestions) 