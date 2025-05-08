import os
from flask import Flask, render_template, request, redirect, url_for, send_file, send_from_directory
from markupsafe import Markup
import pandas as pd
import subprocess
from datetime import datetime
import shutil

app = Flask(__name__)

# Ensure uploads directory exists
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Ensure static directory exists for images
STATIC_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
if not os.path.exists(STATIC_FOLDER):
    os.makedirs(STATIC_FOLDER)

# Copy the logo to the static directory if it doesn't exist there
logo_source_paths = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'better_home_logo.png'),
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'better_home_logo.png'),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'web_app', 'better_home_logo.png'),
]
logo_dest_path = os.path.join(STATIC_FOLDER, 'better_home_logo.png')

if not os.path.exists(logo_dest_path):
    for source_path in logo_source_paths:
        if os.path.exists(source_path):
            shutil.copy(source_path, logo_dest_path)
            print(f"Copied logo from {source_path} to {logo_dest_path}")
            break

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    # Assuming the form data is collected in a dictionary called form_data
    form_data = {
        'Name': request.form.get('name'),
        'Mobile Number (Preferably on WhatsApp)': request.form.get('mobile'),
        'E-mail': request.form.get('email'),
        'Apartment Address': request.form.get('address'),
        'What is your overall budget for home appliances?': request.form.get('budget'),
        'Adults (between the age 18 to 50)': request.form.get('adults'),
        'Elders (above the age 60)': request.form.get('elders'),
        'Kids (below the age 18)': request.form.get('kids'),
        'Number of bedrooms': request.form.get('bedrooms'),
        'Number of bathrooms': request.form.get('bathrooms'),
        'Hall: Fan(s)?': request.form.get('hall_fans'),
        'Hall: Air Conditioner (AC)?': request.form.get('hall_ac'),
        'Hall: Colour theme?': request.form.get('hall_color'),
        'Hall: What is the square feet?': request.form.get('hall_square_feet'),
        'Hall: Any other information?': request.form.get('hall_other_info'),
        'Kitchen: Chimney width?': request.form.get('kitchen_chimney'),
        'Kitchen: Gas stove type?': request.form.get('kitchen_stove'),
        'Kitchen: Number of burners?': request.form.get('kitchen_burners'),
        'Kitchen: Stove width?': request.form.get('kitchen_stove_width'),
        'Kitchen: Do you need a small fan?': request.form.get('kitchen_fan'),
        'Kitchen: Dishwasher capacity?': request.form.get('kitchen_dishwasher_capacity'),
        'Kitchen: Refrigerator type?': request.form.get('kitchen_refrigerator_type'),
        'Kitchen: Refrigerator capacity?': request.form.get('kitchen_refrigerator_capacity'),
        'Kitchen: Do you need any other appliances or do you have any other information?': request.form.get('kitchen_other_info'),
        'Master: Air Conditioner (AC)?': request.form.get('master_ac'),
        'Master: How do you bath with the hot & cold water?': request.form.get('master_water'),
        'Master: Exhaust fan size?': request.form.get('master_exhaust_size'),
        'Master: What is the colour theme?': request.form.get('master_color'),
        'Master: What is the area of the bedroom in square feet?': request.form.get('master_area'),
        'Master: Is this bathroom for elders (above the age 60)?': request.form.get('master_bathroom_for_elders'),
        'Master: Is the water heater going to be inside the false ceiling in the bathroom?': request.form.get('master_water_heater_ceiling'),
        'Master: Exhaust fan colour?': request.form.get('master_exhaust_color'),
        'Master: Would you like to have a LED Mirror?': request.form.get('master_led_mirror'),
        'Master: Any other information?': request.form.get('master_other_info'),
        'Bedroom 2: Air Conditioner (AC)?': request.form.get('bedroom2_ac'),
        'Bedroom 2: How do you bath with the hot & cold water?': request.form.get('bedroom2_water'),
        'Bedroom 2: Exhaust fan size?': request.form.get('bedroom2_exhaust_size'),
        'Bedroom 2: What is the colour theme?': request.form.get('bedroom2_color'),
        'Bedroom 2: What is the area of the bedroom in square feet?': request.form.get('bedroom2_area'),
        'Bedroom 2: Is this for kids above': request.form.get('bedroom2_for_kids'),
        'Bedroom 2: Is the water heater going to be inside the false ceiling in the bathroom?': request.form.get('bedroom2_water_heater_ceiling'),
        'Bedroom 2: Exhaust fan colour?': request.form.get('bedroom2_exhaust_color'),
        'Bedroom 2: Would you like to have a LED Mirror?': request.form.get('bedroom2_led_mirror'),
        'Bedroom 2: Any other information?': request.form.get('bedroom2_other_info'),
        'Laundry: Washing Machine?': request.form.get('laundry_washing'),
        'Laundry: Dryer?': request.form.get('laundry_dryer'),
        'Any other information?': request.form.get('other_info'),
        'Questions and comments': request.form.get('questions_comments'),
        'Dining: Fan': request.form.get('dining_fan'),
        'Dining: Air Conditioner (AC)?': request.form.get('dining_ac'),
        'Dining: Colour theme?': request.form.get('dining_color'),
        'Dining: What is the square feet?': request.form.get('dining_square_feet'),
    }

    # Create a DataFrame from the form data
    df = pd.DataFrame([form_data])
    
    # Capture the timestamp once for consistent file naming
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Generate a unique filename for the Excel file
    excel_filename = f"uploads/{form_data['Name'].replace(' ', '_')}_{timestamp}.xlsx"
    
    # Save the DataFrame to an Excel file
    with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    
    # Make sure the logo is in the static folder before running the script
    source_logo_paths = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'better_home_logo.png'),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'better_home_logo.png')
    ]
    
    static_logo_path = os.path.join(STATIC_FOLDER, 'better_home_logo.png')
    
    if not os.path.exists(static_logo_path):
        for source_path in source_logo_paths:
            if os.path.exists(source_path):
                shutil.copy(source_path, static_logo_path)
                print(f"Copied logo from {source_path} to {static_logo_path}")
                break
    
    # Run the combined script with the Excel filename
    env = os.environ.copy()
    env['FLASK_APP'] = 'app.py'  # Set Flask environment variable
    env['BETTERHOME_WEB_APP'] = 'true'  # Custom environment variable to indicate web app mode
    subprocess.run(['python3', 'combined_script.py', excel_filename], env=env)
    
    # Check if the recommendation files were created
    pdf_filename = excel_filename.replace('.xlsx', '.pdf')
    html_filename = excel_filename.replace('.xlsx', '.html')
    
    if os.path.exists(pdf_filename) and os.path.exists(html_filename):
        # Get the basename for the PDF download link
        pdf_basename = os.path.basename(pdf_filename)
            
        # For the HTML content, we'll simply redirect to a route that serves the HTML file directly
        html_basename = os.path.basename(html_filename)
        
        return render_template('results.html', 
                             html_file=html_basename,
                             pdf_path=url_for('download_file', filename=pdf_basename))
    else:
        return "Error generating recommendations. Please try again."

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(UPLOAD_FOLDER, filename), as_attachment=True)

@app.route('/uploads/<path:filename>')
def serve_uploads(filename):
    """Serve files from the uploads directory (for images in HTML)"""
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/view_html/<filename>')
def view_html(filename):
    """Serve the HTML file with proper content type"""
    # Read the HTML file
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
            
        # Check if the HTML content contains unprocessed template variables
        if '{logo_html}' in html_content or "{user_data['name']}" in html_content:
            # Handle the case where variables weren't substituted
            return "Error: Template variables not correctly processed. Please check the logs and try again."
            
        return send_from_directory(UPLOAD_FOLDER, filename, mimetype='text/html')
    except Exception as e:
        print(f"Error serving HTML file: {e}")
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002) 