"""
Shared business logic utilities for the BetterHome project.
Contains functions for budget categorization, user requirements analysis, and product recommendations.
"""

import pandas as pd
from typing import Dict, Any, List, Optional, Union
from .data_utils import safe_int, safe_float, safe_str


def get_budget_category(total_budget: float, appliance_type: str) -> str:
    """
    Determine budget category based on total budget and appliance type.
    
    Args:
        total_budget: Total budget amount
        appliance_type: Type of appliance
    
    Returns:
        Budget category: "budget", "mid", or "premium"
    """
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


def get_budget_category_for_product(price: float, appliance_type: str) -> str:
    """
    Determine budget category based on product price and appliance type.
    
    Args:
        price: Product price
        appliance_type: Type of appliance
    
    Returns:
        Budget category: "budget", "mid", or "premium"
    """
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


def is_appliance_needed(value) -> bool:
    """
    Check if an appliance is needed based on the input value.
    
    Args:
        value: The value to check (can be string, boolean, number, or None)
        
    Returns:
        bool: True if appliance is needed, False otherwise
    """
    if value is None:
        return False
    
    # Convert to string for consistent checking
    value_str = str(value).strip().lower()
    
    # Values that indicate appliance is NOT needed
    not_needed_values = [
        'no', 'none', 'not needed', 'not applicable', 'na', 'n/a', 
        'false', '0', '', 'not required', 'skip', 'omit'
    ]
    
    # Values that indicate appliance IS needed
    needed_values = [
        'yes', 'true', '1', 'required', 'needed', 'must have'
    ]
    
    # Check if value indicates not needed
    if value_str in not_needed_values:
        return False
    
    # Check if value indicates needed
    if value_str in needed_values:
        return True
    
    # For numeric values, check if > 0
    try:
        num_value = float(value)
        return num_value > 0
    except (ValueError, TypeError):
        pass
    
    # For boolean values
    if isinstance(value, bool):
        return value
    
    # For string values that might contain specific requirements (like dimensions, types, etc.)
    # If it's not explicitly "not needed" and contains meaningful content, assume it's needed
    if value_str and value_str not in not_needed_values:
        return True
    
    # Default to False if we can't determine
    return False


def get_user_data_value(user_data: Dict[str, Any], key: str, default: Any = None) -> Any:
    """
    Helper function to safely access user_data with either lowercase or capitalized keys.
    This helps handle both form_data (direct from web forms) and processed user_data.

    Args:
        user_data: Dictionary containing user information
        key: The key to look for (will try both lowercase and capitalized versions)
        default: The default value to return if the key is not found

    Returns:
        The value from user_data if found, otherwise the default
    """
    # Try lowercase first (processed data format)
    if key in user_data:
        return user_data[key]

    # Try capitalized (form data format)
    capitalized_key = key.capitalize()
    if capitalized_key in user_data:
        return user_data[capitalized_key]

    # Try all uppercase (form data might use all caps for some fields)
    uppercase_key = key.upper()
    if uppercase_key in user_data:
        return user_data[uppercase_key]

    # Return default if not found in any format
    return default


def analyze_user_requirements_from_excel(excel_file: str) -> Optional[Dict[str, Any]]:
    """
    Extract and analyze user requirements from Excel file.
    
    Args:
        excel_file: Path to Excel file containing user requirements
    
    Returns:
        Dictionary containing analyzed user data or None if error
    """
    try:
        # Read the Excel file
        print(f"Reading Excel file: {excel_file}")
        df = pd.read_excel(excel_file)

        print(f"Excel file loaded, columns: {df.columns.tolist()}")

        # Clean up column names by removing newlines and extra spaces
        df.columns = [col.split('\\n')[0].strip() for col in df.columns]

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

        # Extract user information with safe conversions
        user_data = {
            'name': safe_str(row.get('Name')),
            'mobile': safe_str(row.get('Mobile Number (Preferably on WhatsApp)')),
            'email': safe_str(row.get('E-mail')),
            'address': safe_str(row.get('Apartment Address')),
            'total_budget': safe_float(row.get('What is your overall budget for home appliances?')),
            'num_bedrooms': safe_int(row.get('Number of bedrooms')),
            'num_bathrooms': safe_int(row.get('Number of bathrooms')),
            'demographics': {
                'adults': safe_int(row.get('Adults (between the age 18 to 50)')),
                'elders': safe_int(row.get('Elders (above the age 60)')),
                'kids': safe_int(row.get('Kids (below the age 18)'))
            }
        }
        
        # Extract room requirements with more flexible column matching
        requirements = {}
        
        # Hall requirements
        hall_columns = {
            'fans': "Hall: Fan(s)?",
            'ac': "Hall: Air Conditioner (AC)?",
            'color_theme': 'Hall: Colour theme?'
        }
        
        # Kitchen requirements
        kitchen_columns = {
            'chimney_width': 'Kitchen: Chimney width?',
            'stove_type': 'Kitchen: Gas stove type?',
            'num_burners': 'Kitchen: Number of burners?',
            'stove_width': 'Kitchen: Stove width?',
            'small_fan': 'Kitchen: Do you need a small fan?',
            'dishwasher_capacity': 'Kitchen: Dishwasher capacity?',
            'refrigerator_type': 'Kitchen: Refrigerator type?',
            'refrigerator_capacity': 'Kitchen: Refrigerator capacity?'
        }
        
        # Extract hall requirements
        requirements['hall'] = {}
        for key, col_pattern in hall_columns.items():
            # Find column that starts with the pattern
            matching_col = None
            for col in df.columns:
                if col.startswith(col_pattern):
                    matching_col = col
                    break
            
            if matching_col:
                value = row.get(matching_col)
                if key == 'fans':
                    requirements['hall'][key] = safe_int(value)
                elif key == 'ac':
                    requirements['hall'][key] = safe_str(value) == 'Yes'
                else:
                    requirements['hall'][key] = safe_str(value)
        
        # Extract kitchen requirements
        requirements['kitchen'] = {}
        for key, col_pattern in kitchen_columns.items():
            # Find column that starts with the pattern
            matching_col = None
            for col in df.columns:
                if col.startswith(col_pattern):
                    matching_col = col
                    break
            
            if matching_col:
                value = row.get(matching_col)
                if key == 'num_burners':
                    requirements['kitchen'][key] = safe_int(value)
                elif key == 'small_fan':
                    requirements['kitchen'][key] = safe_str(value) == 'Yes'
                else:
                    requirements['kitchen'][key] = safe_str(value)
        
        # Extract bedroom requirements (master and bedroom 2)
        for bedroom_name in ['Master', 'Bedroom 2', 'Bedroom 3']:
            bedroom_key = bedroom_name.lower().replace(' ', '_')
            requirements[bedroom_key] = {}
            
            # Common bedroom columns
            bedroom_columns = {
                'ac': f'{bedroom_name}: Air Conditioner (AC)?',
                'water_heater_type': f'{bedroom_name}: How do you bath with the hot & cold water?',
                'exhaust_fan_size': f'{bedroom_name}: Exhaust fan size?',
                'color_theme': f'{bedroom_name}: What is the colour theme?'
            }
            
            for key, col_pattern in bedroom_columns.items():
                # Find column that starts with the pattern
                matching_col = None
                for col in df.columns:
                    if col.startswith(col_pattern):
                        matching_col = col
                        break
                
                if matching_col:
                    value = row.get(matching_col)
                    if key == 'ac':
                        requirements[bedroom_key][key] = safe_str(value) == 'Yes'
                    else:
                        requirements[bedroom_key][key] = safe_str(value)
        
        # Extract laundry requirements
        requirements['laundry'] = {
            'washing_machine_type': safe_str(row.get('Laundry: Washing Machine?')),
            'dryer_type': safe_str(row.get('Laundry: Dryer?'))
        }
        
        # Extract dining requirements
        dining_columns = {
            'fan': 'Dining: Fan',
            'ac': 'Dining: Air Conditioner (AC)?',
            'color_theme': 'Dining: Colour theme?',
            'square_feet': 'Dining: What is the square feet?'
        }
        
        requirements['dining'] = {}
        for key, col_pattern in dining_columns.items():
            # Find column that starts with the pattern
            matching_col = None
            for col in df.columns:
                if col.startswith(col_pattern):
                    matching_col = col
                    break
            
            if matching_col:
                value = row.get(matching_col)
                if key == 'ac':
                    requirements['dining'][key] = safe_str(value) == 'Yes'
                else:
                    requirements['dining'][key] = safe_str(value)
        
        # Merge requirements into user_data
        user_data.update(requirements)
        
        print(f"\\nDebug: Processed user data: {user_data}")
        print(f"\\nDebug: Processed requirements: {requirements}")
        
        return user_data
        
    except Exception as e:
        print(f"Error reading Excel file: {str(e)}")
        return None


def find_product_type(query: str, product_terms: Dict[str, Any]) -> Optional[str]:
    """
    Find the most relevant product type for a given query using the product terms dictionary.
    
    Args:
        query: Query string to analyze
        product_terms: Product terms dictionary
    
    Returns:
        Product type if found, None otherwise
    """
    query_lower = query.lower()
    
    for product_type, info in product_terms.items():
        # Check standard name
        if product_type.lower() in query_lower:
            return product_type
        
        # Check alternatives
        for alternative in info.get('alternatives', []):
            if alternative.lower() in query_lower:
                return product_type
        
        # Check categories
        for category in info.get('categories', []):
            if category.lower() in query_lower:
                return product_type
    
    return None


def determine_query_type(query: str, product_terms: Dict[str, Any]) -> tuple:
    """
    Determine the type of query and find relevant product type if applicable.
    
    Args:
        query: Query string to analyze
        product_terms: Product terms dictionary
    
    Returns:
        Tuple of (query_type, product_type)
    """
    query_lower = query.lower()

    # Price-related keywords
    price_keywords = ['price', 'cost', 'expensive', 'cheap', 'cheaper', 'discount', 'savings']

    # Product type keywords
    type_keywords = ['type', 'category', 'kind', 'variety']

    # Brand keywords
    brand_keywords = ['brand', 'make', 'manufacturer', 'company']

    # First check if we can identify a specific product type
    product_type = find_product_type(query, product_terms)
    if product_type:
        return 'product_type', product_type

    # If no specific product type found, check other query types
    if any(keyword in query_lower for keyword in price_keywords):
        return 'price', None
    elif any(keyword in query_lower for keyword in type_keywords):
        return 'product_type', None
    elif any(keyword in query_lower for keyword in brand_keywords):
        return 'brand', None
    else:
        return 'general', None