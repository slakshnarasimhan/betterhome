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
    # Create a DataFrame from the form data
    df = pd.DataFrame({
        'Name': [request.form['name']],
        'Mobile Number (Preferably on WhatsApp)': [request.form['mobile']],
        'E-mail': [request.form['email']],
        'Apartment Address (building, floor, and what feeling does this Chennai location bring you?)': [request.form['address']],
        'What is your overall budget for home appliances?': [request.form['budget']],
        'Number of bedrooms': [request.form['bedrooms']],
        'Number of bathrooms': [request.form['bathrooms']],
        'Adults (between the age 18 to 50)': [request.form['adults']],
        'Elders (above the age 60)': [request.form['elders']],
        'Kids (below the age 18)': [request.form['kids']],
        'Hall: Colour theme?': [request.form['hall_color']],
        'Hall: Fan(s)?': [request.form['hall_fans']],
        'Hall: Air Conditioner (AC)?': [request.form['hall_ac']],
        'Kitchen: Chimney width?': [request.form['kitchen_chimney']],
        'Kitchen: Gas stove type?': [request.form['kitchen_stove']],
        'Kitchen: Number of burners?': [request.form['kitchen_burners']],
        'Kitchen: Do you need a small fan?': [request.form['kitchen_fan']],
        'Master: What is the colour theme?': [request.form['master_color']],
        'Master: Air Conditioner (AC)?': [request.form['master_ac']],
        'Master: How do you bath with the hot & cold water?': [request.form['master_water']],
        'Master: Exhaust fan size?': [request.form['master_exhaust']],
        'Bedroom 2: What is the colour theme?': [request.form['bedroom2_color']],
        'Bedroom 2: Air Conditioner (AC)?': [request.form['bedroom2_ac']],
        'Bedroom 2: How do you bath with the hot & cold water?': [request.form['bedroom2_water']],
        'Bedroom 2: Exhaust fan size?': [request.form['bedroom2_exhaust']],
        'Laundry: Washing Machine?': [request.form['laundry_washing']],
        'Laundry: Dryer?': [request.form['laundry_dryer']]
    })
    
    # Generate a unique filename using the customer's name and timestamp
    customer_name = request.form['name'].replace(' ', '_')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    excel_filename = f'uploads/{customer_name}_{timestamp}.xlsx'
    
    # Save the DataFrame to Excel
    df.to_excel(excel_filename, index=False)
    
    # Run the combined script with the Excel filename
    subprocess.run(['python3', 'combined_script.py', excel_filename])
    
    # Check if the recommendation files were created
    pdf_filename = f'uploads/{customer_name}_{timestamp}.pdf'
    txt_filename = f'uploads/{customer_name}_{timestamp}.txt'
    if os.path.exists(pdf_filename) and os.path.exists(txt_filename):
        # Read the text file content
        with open(txt_filename, 'r') as f:
            txt_content = f.read()
        
        return render_template('results.html', 
                             pdf_file=pdf_filename,
                             txt_content=txt_content)
    else:
        return "Error generating recommendations. Please try again."

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(UPLOAD_FOLDER, filename), as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002) 