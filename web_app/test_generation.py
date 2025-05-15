#!/usr/bin/env python3
import os
import sys
from subprocess import run

print('Testing PDF and HTML generation...')

# Create test directory
os.makedirs('uploads', exist_ok=True)

# Create a test Excel file
excel_file = 'uploads/test_user_20250515_000000.xlsx'
run(['touch', excel_file])

# Import functions from combined_script
from combined_script import create_styled_pdf, generate_text_file, generate_html_file

# Define output paths
pdf_file = excel_file.replace('.xlsx', '.pdf')
txt_file = excel_file.replace('.xlsx', '.txt')
html_file = excel_file.replace('.xlsx', '.html')

# Create minimal test data
user_data = {
    'name': 'Test User',
    'email': 'test@example.com',
    'mobile': '1234567890',
    'address': '123 Test St',
    'total_budget': 100000,
    'demographics': {'adults': 2, 'elders': 1, 'kids': 1, 'bedrooms': 2},
    'kitchen': {},
    'laundry': {},
    'master_bedroom': {},
    'bedroom_2': {}
}

final_list = {
    'summary': {'total_budget': 100000},
    'hall': {'fans': []},
    'master_bedroom': {'ac': [], 'fans': [], 'bathroom': {'water_heater': [], 'exhaust_fan': []}},
    'bedroom_2': {'ac': [], 'fans': [], 'bathroom': {'water_heater': [], 'exhaust_fan': []}},
    'kitchen': {'chimney': [], 'refrigerator': [], 'gas_stove': [], 'hob_top': [], 'small_fan': []},
    'dining': {'fans': [], 'ac': []},
    'laundry': {'washing_machine': [], 'dryer': []}
}

# Test HTML generation
print('Generating HTML file...')
try:
    generate_html_file(user_data, final_list, html_file)
    print(f'HTML file generated: {os.path.exists(html_file)}')
except Exception as e:
    print(f'Error generating HTML: {e}')
    import traceback
    traceback.print_exc()

# Test TXT generation
print('Generating TXT file...')
try:
    generate_text_file(user_data, final_list, txt_file)
    print(f'TXT file generated: {os.path.exists(txt_file)}')
except Exception as e:
    print(f'Error generating TXT: {e}')
    import traceback
    traceback.print_exc()

# Test PDF generation
print('Generating PDF file...')
try:
    required_features = {}
    create_styled_pdf(pdf_file, user_data, final_list, required_features)
    print(f'PDF file generated: {os.path.exists(pdf_file)}')
except Exception as e:
    print(f'Error generating PDF: {e}')
    import traceback
    traceback.print_exc()

print('Test completed.') 