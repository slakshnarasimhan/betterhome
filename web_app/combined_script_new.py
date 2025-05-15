import os
import sys
import json
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional

# Path to the Excel file
if len(sys.argv) != 2:
    print("Usage: python combined_script.py <excel_file>")
    sys.exit(1)

excel_file = sys.argv[1]

# Load user information
def get_user_information(excel_filename: str) -> Dict[str, Any]:
    """Extract user information from the Excel file."""
    try:
        df = pd.read_excel(excel_filename)
        print(f"Excel file loaded. Columns: {df.columns.tolist()}")
        
        # Extract user info based on actual column names
        user_data = {
            'name': df['Name'].iloc[0] if 'Name' in df.columns else 'Unknown',
            'email': df['E-mail'].iloc[0] if 'E-mail' in df.columns else 'Unknown',
            'mobile': df['Mobile Number (Preferably on WhatsApp)'].iloc[0] if 'Mobile Number (Preferably on WhatsApp)' in df.columns else 'Unknown',
            'address': df['Apartment Address'].iloc[0] if 'Apartment Address' in df.columns else 'Unknown',
            'total_budget': float(df['What is your overall budget for home appliances?'].iloc[0]) if 'What is your overall budget for home appliances?' in df.columns else 0.0,
            'demographics': {},
            'rooms': {}
        }
        
        # Parse demographics
        if 'Adults (between the age 18 to 50)' in df.columns:
            adults_val = df['Adults (between the age 18 to 50)'].iloc[0]
            user_data['demographics']['adults'] = int(adults_val) if pd.notna(adults_val) else 0
        
        if 'Elders (above the age 60)' in df.columns:
            elders_val = df['Elders (above the age 60)'].iloc[0]
            user_data['demographics']['elders'] = int(elders_val) if pd.notna(elders_val) else 0
        
        if 'Kids (below the age 18)' in df.columns:
            kids_val = df['Kids (below the age 18)'].iloc[0]
            user_data['demographics']['kids'] = int(kids_val) if pd.notna(kids_val) else 0
        
        return user_data
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        sys.exit(1)

# Get user information from Excel
try:
    user_data = get_user_information(excel_file)
    
    print(f"User data loaded for: {user_data['name']}")
    print(f"Total budget: ₹{user_data['total_budget']:,.2f}")
    print(f"Demographics: {user_data['demographics']}")
    
    # Create a simpler output without complex JavaScript
    def generate_simple_output(user_data: Dict[str, Any]) -> None:
        """Generate a simple text output with user information."""
        output_file = os.path.splitext(excel_file)[0] + "_output.txt"
        
        with open(output_file, 'w') as f:
            f.write(f"Better Home Recommendations for {user_data['name']}\n")
            f.write("="*50 + "\n\n")
            
            f.write("User Information:\n")
            f.write(f"Name: {user_data['name']}\n")
            f.write(f"Email: {user_data['email']}\n")
            f.write(f"Mobile: {user_data['mobile']}\n")
            f.write(f"Address: {user_data['address']}\n")
            f.write(f"Total Budget: ₹{user_data['total_budget']:,.2f}\n")
            
            f.write("\nDemographics:\n")
            for key, value in user_data['demographics'].items():
                f.write(f"  {key.title()}: {value}\n")
            
            f.write("\nRecommendations would be generated here based on user data.\n")
        
        print(f"Simple output generated: {output_file}")
    
    # Generate simple output to avoid JavaScript issues
    generate_simple_output(user_data)
    
    print("Program completed successfully.")
except Exception as e:
    print(f"Error processing data: {e}")
    sys.exit(1) 