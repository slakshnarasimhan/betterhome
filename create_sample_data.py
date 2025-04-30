import pandas as pd
import datetime

# Create sample data
data = {
    'Name': ['John Doe'],
    'Mobile Number (Preferably on WhatsApp)': ['1234567890'],
    'E-mail': ['john@example.com'],
    'Apartment Address (building, floor, and what feeling does this Chennai location bring you?)': ['123 Main St, Chennai'],
    'What is your overall budget for home appliances?': ['500000'],
    'Adults (between the age 18 to 50)': ['2'],
    'Elders (above the age 60)': ['1'],
    'Kids (below the age 18)': ['1'],
    'Hall AC': ['Yes'],
    'Hall Fan': ['Yes'],
    'Hall Colour Theme': ['White'],
    'Kitchen Chimney': ['Yes'],
    'Kitchen Hob': ['Yes'],
    'Kitchen Colour Theme': ['Grey'],
    'Master Bedroom AC': ['Yes'],
    'Master Bedroom Fan': ['Yes'],
    'Master Bedroom Colour Theme': ['Blue'],
    'Bedroom 2 AC': ['Yes'],
    'Bedroom 2 Fan': ['Yes'],
    'Bedroom 2 Colour Theme': ['Green'],
    'Laundry Washing Machine': ['Front-Load']
}

# Create DataFrame
df = pd.DataFrame(data)

# Generate filename with timestamp
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
filename = f'uploads/customer_data_{timestamp}.xlsx'

# Save to Excel
df.to_excel(filename, index=False)
print(f"Created sample Excel file: {filename}") 