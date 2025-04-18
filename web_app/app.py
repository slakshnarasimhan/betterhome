import os
from flask import Flask, render_template, request, redirect, url_for, send_file
import pandas as pd
import subprocess
from datetime import datetime

app = Flask(__name__)

# Ensure uploads directory exists
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

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
        'Apartment Address (building, floor, and what feeling does this Chennai location bring you?)': request.form.get('address'),
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
        'Master: Exhaust fan size?': request.form.get('master_exhaust'),
        'Master: What is the colour theme?': request.form.get('master_color'),
        'Master: What is the area of the bedroom in square feet?': request.form.get('master_area'),
        'Master: Is this bathroom for elders (above the age 60)?': request.form.get('master_bathroom_for_elders'),
        'Master: Is the water heater going to be inside the false ceiling in the bathroom?': request.form.get('master_water_heater_ceiling'),
        'Master: Exhaust fan colour?': request.form.get('master_exhaust_color'),
        'Master: Would you like to have a LED Mirror?': request.form.get('master_led_mirror'),
        'Master: Any other information?': request.form.get('master_other_info'),
        'Bedroom 2: Air Conditioner (AC)?': request.form.get('bedroom2_ac'),
        'Bedroom 2: How do you bath with the hot & cold water?': request.form.get('bedroom2_water'),
        'Bedroom 2: Exhaust fan size?': request.form.get('bedroom2_exhaust'),
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
        'Questions and comments': request.form.get('questions_comments')
    }

    # Create a DataFrame from the form data
    df = pd.DataFrame([form_data])
    
    # Capture the timestamp once for consistent file naming
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Generate a unique filename for the Excel file
    filename = f"uploads/{form_data['Name'].replace(' ', '_')}_{timestamp}.xlsx"
    
    # Save the DataFrame to an Excel file
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    
    # Run the combined script with the Excel filename
    subprocess.run(['python3', 'combined_script.py', filename])
    
    # Check if the recommendation files were created
    pdf_filename = f"uploads/{form_data['Name'].replace(' ', '_')}_{timestamp}.pdf"
    txt_filename = f"uploads/{form_data['Name'].replace(' ', '_')}_{timestamp}.txt"
    if os.path.exists(pdf_filename) and os.path.exists(txt_filename):
        # Read the text file content
        with open(txt_filename, 'r') as f:
            txt_content = f.read()
        
        return render_template('results.html', 
                             pdf_path=url_for('download_file', filename=os.path.basename(pdf_filename)),
                             txt_content=txt_content)
    else:
        # Debugging output to check filenames and their existence
        print(f"Checking for PDF file: {pdf_filename}")
        print(f"Checking for TXT file: {txt_filename}")
        print(f"PDF exists: {os.path.exists(pdf_filename)}")
        print(f"TXT exists: {os.path.exists(txt_filename)}")
        return "Error generating recommendations. Please try again."

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(UPLOAD_FOLDER, filename), as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002) 