import os
from flask import Flask, render_template, request, redirect, url_for, send_file, send_from_directory, make_response
from markupsafe import Markup
import pandas as pd
import subprocess
from datetime import datetime
import shutil
from werkzeug.utils import secure_filename

betterhome = Flask(__name__)
betterhome.config['DEBUG'] = True
betterhome.config['TEMPLATES_AUTO_RELOAD'] = True

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

@betterhome.route('/')
def index():
    response = make_response(render_template('index.html'))
    # Add cache control headers to prevent caching
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@betterhome.route('/submit', methods=['POST'])
def submit():
    print("\n=== Form Submission Started ===")
    print("Submit route accessed")
    print("Request method:", request.method)
    print("Form data received:", request.form)
    
    try:
        # Get number of bedrooms first
        num_bedrooms = request.form.get('bedrooms')
        print(f"Number of bedrooms selected: {num_bedrooms}")
        
        # Capture the timestamp once for consistent file naming
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create a subfolder for this user's recommendations
        user_folder = f"uploads/{request.form.get('name').replace(' ', '_')}_{timestamp}"
        if not os.path.exists(user_folder):
            os.makedirs(user_folder)
            
        # Create room_images subfolder
        room_images_folder = os.path.join(user_folder, 'room_images')
        if not os.path.exists(room_images_folder):
            os.makedirs(room_images_folder)
            
        # Handle file uploads
        if 'room_images' in request.files:
            files = request.files.getlist('room_images')
            for file in files:
                if file.filename:  # Check if file was selected
                    # Secure the filename
                    filename = secure_filename(file.filename)
                    # Save the file
                    file.save(os.path.join(room_images_folder, filename))
                    print(f"Saved file: {filename}")

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
            'Number of bedrooms': num_bedrooms,
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
            'Master: Do you want a Glass Partition in the bathroom?': request.form.get('master_bathroom_for_elders'),
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
            'Bedroom 2: Do you want a Glass Partition in the bathroom?': request.form.get('bedroom2_bathroom_for_elders'),
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
            'Additional Information: Any other information?': request.form.get('other_info'),
            'Additional Information: Questions and comments': request.form.get('questions_comments')
        }
        
        # Only add Bedroom 3 data if 3 bedrooms are selected
        if num_bedrooms == '3':
            form_data.update({
                'Bedroom 3: Air Conditioner (AC)?': request.form.get('bedroom3_ac'),
                'Bedroom 3: How do you bath with the hot & cold water?': request.form.get('bedroom3_water'),
                'Bedroom 3: Exhaust fan size?': request.form.get('bedroom3_exhaust_size'),
                'Bedroom 3: What is the colour theme?': request.form.get('bedroom3_color'),
                'Bedroom 3: What is the area of the bedroom in square feet?': request.form.get('bedroom3_area'),
                'Bedroom 3: Is this for kids above': request.form.get('bedroom3_for_kids'),
                'Bedroom 3: Is the water heater going to be inside the false ceiling in the bathroom?': request.form.get('bedroom3_water_heater_ceiling'),
                'Bedroom 3: Do you want a Glass Partition in the bathroom?': request.form.get('bedroom3_bathroom_for_elders'),
                'Bedroom 3: Exhaust fan colour?': request.form.get('bedroom3_exhaust_color'),
                'Bedroom 3: Would you like to have a LED Mirror?': request.form.get('bedroom3_led_mirror'),
                'Bedroom 3: Any other information?': request.form.get('bedroom3_other_info'),
            })
        else:
            # Add empty values for Bedroom 3 to maintain consistent columns
            form_data.update({
                'Bedroom 3: Air Conditioner (AC)?': '',
                'Bedroom 3: How do you bath with the hot & cold water?': '',
                'Bedroom 3: Exhaust fan size?': '',
                'Bedroom 3: What is the colour theme?': '',
                'Bedroom 3: What is the area of the bedroom in square feet?': '',
                'Bedroom 3: Is this for kids above': '',
                'Bedroom 3: Is the water heater going to be inside the false ceiling in the bathroom?': '',
                'Bedroom 3: Do you want a Glass Partition in the bathroom?': '',
                'Bedroom 3: Exhaust fan colour?': '',
                'Bedroom 3: Would you like to have a LED Mirror?': '',
                'Bedroom 3: Any other information?': '',
            })

        print("Processed form data:", form_data)

        # Create a DataFrame from the form data
        df = pd.DataFrame([form_data])
        
        # Debug: Print DataFrame to verify Bedroom 3 data
        print("DataFrame:")
        print(df)
        
        # Generate a unique filename for the Excel file
        excel_filename = f"{user_folder}/{form_data['Name'].replace(' ', '_')}_{timestamp}.xlsx"
        
        print(f"Creating Excel file at: {excel_filename}")
        
        # Save the DataFrame to an Excel file
        with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
            df.to_excel(writer, index=False)
        
        print(f"Excel file created successfully")
        
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
        
        try:
            # Get the absolute path to the web_app directory
            web_app_dir = os.path.dirname(os.path.abspath(__file__))
            script_path = os.path.join(web_app_dir, 'combined_script.py')
            
            print(f"Web app directory: {web_app_dir}")
            print(f"Script path: {script_path}")
            print(f"Excel filename: {excel_filename}")
            
            if not os.path.exists(script_path):
                print(f"Error: combined_script.py not found at {script_path}")
                return "Error: Recommendation script not found. Please check the installation."
            
            print("Running combined_script.py...")
            
            # Run the script with the correct working directory
            result = subprocess.run(
                ['python3', script_path, excel_filename],
                env=env,
                cwd=web_app_dir,
                capture_output=True,
                text=True
            )
            
            # Print the full output for debugging
            print("Script stdout:", result.stdout)
            print("Script stderr:", result.stderr)
            print("Script return code:", result.returncode)
            
            # Check if the script ran successfully
            if result.returncode != 0:
                print(f"Error running combined_script.py: {result.stderr}")
                return "Error generating recommendations. Please try again."
                
            # Check if the recommendation files were created
            pdf_filename = excel_filename.replace('.xlsx', '.pdf')
            html_filename = excel_filename.replace('.xlsx', '.html')
            
            print(f"Checking for files:")
            print(f"PDF file exists: {os.path.exists(pdf_filename)}")
            print(f"HTML file exists: {os.path.exists(html_filename)}")
            
            if os.path.exists(html_filename):  # Only check for HTML file since PDF is optional
                # Get the basename for the files
                pdf_basename = os.path.basename(pdf_filename) if os.path.exists(pdf_filename) else None
                html_basename = os.path.basename(html_filename)
                
                # Get the relative path from uploads directory
                html_relative_path = os.path.relpath(html_filename, UPLOAD_FOLDER)
                pdf_relative_path = os.path.relpath(pdf_filename, UPLOAD_FOLDER) if pdf_basename else None
                
                # Create the URLs for the files
                html_url = url_for('view_html', filename=html_relative_path)
                pdf_url = url_for('download_file', filename=pdf_relative_path) if pdf_relative_path else None
                
                print(f"HTML URL: {html_url}")
                print(f"PDF URL: {pdf_url}")
                
                return render_template('results.html', 
                                     html_file=html_relative_path,
                                     pdf_path=pdf_url)
            else:
                print(f"Recommendation files not found. HTML: {os.path.exists(html_filename)}")
                return "Error generating recommendations. Please try again."
                
        except Exception as e:
            print(f"Error in recommendation generation: {str(e)}")
            import traceback
            print("Full traceback:")
            print(traceback.format_exc())
            return "Error generating recommendations. Please try again."
            
    except Exception as e:
        print(f"Error processing form data: {str(e)}")
        import traceback
        print("Full traceback:")
        print(traceback.format_exc())
        return "Error processing form data. Please try again."

@betterhome.route('/view_html/<path:filename>')
def view_html(filename):
    """Serve the HTML file with proper content type"""
    try:
        # Get the full path to the HTML file
        # The filename will now include the subfolder name
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        print(f"Attempting to serve HTML file: {file_path}")
        
        if not os.path.exists(file_path):
            print(f"HTML file not found at: {file_path}")
            return "Error: HTML file not found", 404
            
        # Read the HTML content
        with open(file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
            
        # Check if the HTML content contains unprocessed template variables
        if '{logo_html}' in html_content or "{user_data['name']}" in html_content:
            print("Warning: Template variables not correctly processed")
            return "Error: Template variables not correctly processed. Please check the logs and try again."
            
        # Serve the file with the correct content type
        return send_file(
            file_path,
            mimetype='text/html',
            as_attachment=False
        )
    except Exception as e:
        print(f"Error serving HTML file: {str(e)}")
        import traceback
        print("Full traceback:")
        print(traceback.format_exc())
        return f"Error serving HTML file: {str(e)}", 500

@betterhome.route('/download/<path:filename>')
def download_file(filename):
    """Serve files from the uploads directory with proper path handling"""
    try:
        # The filename will now include the subfolder name
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        if not os.path.exists(file_path):
            return "File not found", 404
        return send_file(file_path, as_attachment=True)
    except Exception as e:
        print(f"Error downloading file: {str(e)}")
        return f"Error downloading file: {str(e)}", 500

@betterhome.route('/uploads/<path:filename>')
def serve_uploads(filename):
    """Serve files from the uploads directory (for images in HTML)"""
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    betterhome.run(debug=True, host='0.0.0.0', port=5002) 
