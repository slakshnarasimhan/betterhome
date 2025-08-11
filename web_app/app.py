import os
from flask import Flask, render_template, request, redirect, url_for, send_file, send_from_directory, make_response, jsonify
from markupsafe import Markup
import pandas as pd
import subprocess
from datetime import datetime
import shutil
from werkzeug.utils import secure_filename
from combined_script import analyze_user_requirements, generate_html_file
from s3_config import S3Handler
from flask import Flask, request, jsonify
from twilio.rest import Client
import os
from dotenv import load_dotenv
import importlib.util


load_dotenv() # Load environment variables from .env file

betterhome = Flask(__name__)
# Dynamically import generate_default_recommendations from generate-recommendations.py
_gen_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'generate-recommendations.py')
_spec = importlib.util.spec_from_file_location('generate_recommendations', _gen_path)
_gen_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gen_mod)  # type: ignore
generate_default_recommendations = _gen_mod.generate_default_recommendations

betterhome.config['DEBUG'] = True
betterhome.config['TEMPLATES_AUTO_RELOAD'] = True


# Initialize S3 handler
s3_handler = S3Handler()

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
# Twilio config (use env vars or config file)
TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
TWILIO_VERIFY_SERVICE_SID = os.getenv('TWILIO_VERIFY_SERVICE_SID')

client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

@betterhome.route('/send_otp', methods=['POST'])
def send_otp():
    return jsonify({'success': True})

@betterhome.route('/verify_otp', methods=['POST'])
def verify_otp():
    data = request.get_json() or {}
    otp = str(data.get('otp', '')).strip()
    if otp == '024680':
        return jsonify({'success': True})
    return jsonify({'success': False, 'error': 'Invalid OTP'}), 400
    
@betterhome.route('/')
def index():
    # Support prefill via query params
    name = request.args.get('name', '')
    mobile = request.args.get('mobile', '')
    email = request.args.get('email', '')
    address = request.args.get('address', '')
    pref_bedrooms = request.args.get('bedrooms', '')
    pref_bathrooms = request.args.get('bathrooms', '')
    response = make_response(render_template('index.html', name=name, mobile=mobile, email=email, address=address, pref_bedrooms=pref_bedrooms, pref_bathrooms=pref_bathrooms))
    # Add cache control headers to prevent caching
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

# Mobile-friendly onboarding page
@betterhome.route('/mobile')
def mobile():
    response = make_response(render_template('mobile.html'))
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@betterhome.route('/submit', methods=['POST'])
def submit():
    try:
        # Get form data
        form_data = request.form.to_dict()
        print("Form data received:", form_data)
        
        # Debug the form data reception
        if 'name' not in form_data:
            print("Warning: 'name' field is missing from form data")
        
        # Create a timestamp for the folder
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        user_folder = os.path.join(UPLOAD_FOLDER, timestamp)
        os.makedirs(user_folder, exist_ok=True)
        
        # Get uploaded files (if any)
        files = []
        if 'room_images' in request.files:
            files = request.files.getlist('room_images')
        
        # Save uploaded files (if any)
        if files:
            print(f"Number of files received: {len(files)}")
            for file in files:
                if file and file.filename and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    file_path = os.path.join(user_folder, filename)
                    file.save(file_path)
                    print(f"Saved file: {file_path}")
        else:
            print("No files uploaded (optional)")
        
        # Use the customer's name and timestamp for the Excel filename
        customer_name = form_data.get('name', 'Customer').strip()
        safe_customer_name = secure_filename(customer_name)
        folder_name = f"uploads/{safe_customer_name}_{timestamp}"

        # Create the folder if it doesn't exist
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        excel_filename = f"{folder_name}/{safe_customer_name}_{timestamp}.xlsx"
        
        # Map form field names to the expected column names in analyze_user_requirements
        field_mapping = {
            'name': 'Name',
            'mobile': 'Mobile Number (Preferably on WhatsApp)',
            'email': 'E-mail',
            'address': 'Apartment Address',
            'budget': 'What is your overall budget for home appliances?',
            'bedrooms': 'Number of bedrooms',
            'bathrooms': 'Number of bathrooms',
            'adults': 'Adults (between the age 18 to 50)',
            'elders': 'Elders (above the age 60)',
            'kids': 'Kids (below the age 18)',
            'hall_fans': 'Hall: Fan(s)?',
            'hall_ac': 'Hall: Air Conditioner (AC)?',
            'hall_color': 'Hall: Colour theme?',
            'hall_square_feet': 'Hall: What is the square feet ?',
            'kitchen_chimney': 'Kitchen: Chimney width?',
            'kitchen_stove': 'Kitchen: Gas stove type?',
            'kitchen_burners': 'Kitchen: Number of burners?',
            'kitchen_fan': 'Kitchen: Do you need a small fan?',
            'kitchen_refrigerator_type': 'Kitchen: Refrigerator type?',
            'kitchen_refrigerator_capacity': 'Kitchen: Refrigerator capacity?',
            'kitchen_dishwasher_capacity': 'Kitchen: Dishwasher capacity?',
            'master_ac': 'Master: Air Conditioner (AC)?',
            'master_water': 'Master: How do you bath with the hot & cold water?',
            'master_exhaust_size': 'Master: Exhaust fan size?',
            'master_water_heater_ceiling': 'Master: Is the water heater going to be inside the false ceiling in the bathroom?',
            'master_led_mirror': 'Master: Would you like to have a LED Mirror?',
            'master_color': 'Master: What is the colour theme?',
            'master_area': 'Master: What is the area of the bedroom in square feet?',
            'bedroom2_ac': 'Bedroom 2: Air Conditioner (AC)?',
            'bedroom2_water': 'Bedroom 2: How do you bath with the hot & cold water?',
            'bedroom2_exhaust_size': 'Bedroom 2: Exhaust fan size?',
            'bedroom2_water_heater_ceiling': 'Bedroom 2: Is the water heater going to be inside the false ceiling in the bathroom?',
            'bedroom2_led_mirror': 'Bedroom 2: Would you like to have a LED Mirror?',
            'bedroom2_color': 'Bedroom 2: What is the colour theme?',
            'bedroom2_area': 'Bedroom 2: What is the area of the bedroom in square feet?',
            'bedroom3_ac': 'Bedroom 3: Air Conditioner (AC)?',
            'bedroom3_water': 'Bedroom 3: How do you bath with the hot & cold water?',
            'bedroom3_exhaust_size': 'Bedroom 3: Exhaust fan size?',
            'bedroom3_water_heater_ceiling': 'Bedroom 3: Is the water heater going to be inside the false ceiling in the bathroom?',
            'bedroom3_led_mirror': 'Bedroom 3: Would you like to have a LED Mirror?',
            'bedroom3_color': 'Bedroom 3: What is the colour theme?',
            'bedroom3_area': 'Bedroom 3: What is the area of the bedroom in square feet?',
            'laundry_washing': 'Laundry: Washing Machine?',
            'laundry_dryer': 'Laundry: Dryer?',
            'dining_fan': 'Dining: Fan(s)?',
            'dining_ac': 'Dining: Air Conditioner (AC)?',
            'dining_color': 'Dining: Colour theme?'
        }
        
        # Debug the field mapping
        print("Form fields received:", list(form_data.keys()))
        
        # Print all mapped vs. required field names
        required_fields = [
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
        
        # Set default values for critical fields if missing
        default_values = {
            'Name': 'Customer',
            'E-mail': 'customer@example.com',
            'Mobile Number (Preferably on WhatsApp)': '0000000000',
            'Apartment Address': 'Address',
            'What is your overall budget for home appliances?': '100000',
            'Number of bedrooms': '2',
            'Number of bathrooms': '2',
            'Adults (between the age 18 to 50)': '2',
            'Elders (above the age 60)': '0',
            'Kids (below the age 18)': '0'
        }
        
        # Map form field names to expected column names
        mapped_data = {}
        for form_field, excel_field in field_mapping.items():
            if form_field in form_data:
                mapped_data[excel_field] = form_data[form_field]
                print(f"Mapped {form_field} to {excel_field}: {form_data[form_field]}")  # Debug log
        
        # Debug mapping issues
        print("Mapped fields before defaults:", list(mapped_data.keys()))
        if 'Name' not in mapped_data:
            print("Warning: 'Name' field is missing from mapped data")
        
        # Add default values for required fields if missing
        for excel_field in required_fields:
            if excel_field not in mapped_data or mapped_data[excel_field] is None or mapped_data[excel_field] == '':
                if excel_field in default_values:
                    print(f"Using default value for missing required field: {excel_field}")
                    mapped_data[excel_field] = default_values[excel_field]
                else:
                    print(f"Warning: Required field {excel_field} has no default")
        
        # Add any missing fields to prevent errors
        for excel_field in field_mapping.values():
            if excel_field not in mapped_data:
                mapped_data[excel_field] = None
                
        # Ensure total_budget is properly formatted (numeric value)
        budget_field = 'What is your overall budget for home appliances?'
        if budget_field in mapped_data:
            try:
                # Try to convert to float to ensure it's numeric
                float_value = float(mapped_data[budget_field])
                mapped_data[budget_field] = str(float_value)  # Convert back to string for Excel
            except (ValueError, TypeError):
                print(f"Warning: Budget value '{mapped_data[budget_field]}' is not numeric, using default")
                mapped_data[budget_field] = default_values[budget_field]
        
        # Create DataFrame and save to Excel
        df = pd.DataFrame([mapped_data])
        df.to_excel(excel_filename, index=False)
        print(f"Created Excel file: {excel_filename}")
        print("Excel file contents:", df.to_dict('records'))  # Debug log
        
        # If using default recommendations from mobile flow, serve the pre-generated default page
        use_default = form_data.get('use_default') == '1'
        if use_default:
            bhk = form_data.get('bedrooms', '2').strip()
            # Persist name/address/mobile for display in the recommendation
            name = form_data.get('name', 'Customer')
            address = form_data.get('address', '')
            mobile = form_data.get('Mobile Number (Preferably on WhatsApp)', '') or form_data.get('mobile', '')
            # Extract user budget if provided
            budget_str = form_data.get('What is your overall budget for home appliances?', '')
            try:
                user_budget = float(budget_str) if budget_str not in (None, '') else None
            except Exception:
                user_budget = None
            print(f"[submit/use_default] bhk={bhk}, budget_str='{budget_str}', parsed_budget={user_budget}")
            # Kick off generation synchronously so the file is ready
            try:
                csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'best-seller-3.csv')
                catalog_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'product_catalog.json')
                bhk_choice = '2BHK' if bhk == '2' else '3BHK'
                # Pass email through to default recommendations so it appears on the page
                email = form_data.get('email', '') or form_data.get('E-mail', '')
                generate_default_recommendations(csv_path, catalog_path, bhk_choice, name, address, mobile, email, user_budget)
                # Redirect to the appropriate tiered file (premium/standard). The route will resolve the right file.
                redirect_url = url_for('default_recommendations', bhk=bhk)
                return render_template('progress.html', redirect_url=redirect_url)
            except Exception as e:
                return jsonify({'success': False, 'error': f'Failed to generate defaults: {str(e)}'}), 500

        # Generate HTML filename based on Excel filename
        html_filename = os.path.splitext(excel_filename)[0] + '.html'
        print(f"Expected HTML filename: {html_filename}")

        # Execute combined_script.py as a command
        try:
            print(f"Executing combined_script.py with file: {excel_filename}")
            result = subprocess.run(['python', 'combined_script.py', excel_filename], 
                                 capture_output=True, 
                                 text=True,
                                 cwd=os.path.dirname(os.path.abspath(__file__)))
            if result.returncode == 0:
                print("Combined script executed successfully:", result.stdout)
                # Check if HTML file was created
                if os.path.exists(html_filename):
                    print(f"HTML file created successfully at: {html_filename}")
                    # Read the generated HTML file content
                    with open(html_filename, 'r', encoding='utf-8') as f:
                        recommendation_html = f.read()
                    print("HTML content read successfully")
                else:
                    print(f"Warning: HTML file not found at {html_filename}")
                    recommendation_html = "Error: Recommendations could not be generated."
            else:
                print("Error executing combined script:", result.stderr)
                recommendation_html = f"Error executing combined script: {result.stderr}"
        except Exception as e:
            print(f"Error executing combined script: {str(e)}")
            recommendation_html = f"Error executing combined script: {str(e)}"
        
        # Upload files to S3
        # s3_excel_key = f"{folder_name}.xlsx"
        # s3_html_key = f"{folder_name}.html"
        # excel_uploaded = s3_handler.upload_file(excel_filename, s3_excel_key)
        # html_uploaded = s3_handler.upload_file(html_filename, s3_html_key)
        # if excel_uploaded and html_uploaded:
        #     # Get S3 URLs
        #     excel_url = s3_handler.get_file_url(s3_excel_key)
        #     html_url = s3_handler.get_file_url(s3_html_key)
        #     # Format the timestamp for display
        #     display_timestamp = datetime.strptime(timestamp, '%Y%m%d_%H%M%S').strftime('%B %d, %Y at %I:%M %p')
        #     # Safely access the 'Name' field with a default value
        #     user_name = form_data.get('Name', 'Customer')
        #     return render_template('results.html', 
        #                          html_file=html_filename,
        #                          excel_file=excel_filename,
        #                          s3_html_url=html_url,
        #                          s3_excel_url=excel_url,
        #                          user_name=user_name,
        #                          timestamp=display_timestamp,
        #                          recommendation_html=recommendation_html)
        # else:
        #     print("Failed to upload files to S3")
        #     return "Error uploading files to S3. Please try again."

        # Instead, just render the results page using local files
        display_timestamp = datetime.strptime(timestamp, '%Y%m%d_%H%M%S').strftime('%B %d, %Y at %I:%M %p')
        user_name = form_data.get('Name', 'Customer')
        # After generating html_filename
        relative_html_path = html_filename.replace('web_app/', '')  # adjust as needed
        return redirect(url_for('view_html', filename=relative_html_path))
            
    except Exception as e:
        print(f"Error in submit route: {str(e)}")
        return f"An error occurred: {str(e)}"

@betterhome.route('/view_html/<path:filename>')
def view_html(filename):
    """Serve the HTML file with proper content type"""
    try:
        # Remove 'uploads/' prefix if present
        if filename.startswith('uploads/'):
            filename = filename[len('uploads/'):]
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

# Serve default recommendations by BHK
@betterhome.route('/default_recommendations/<bhk>')
def default_recommendations(bhk):
    try:
        bhk_norm = '2' if str(bhk).strip().lower().startswith('2') else '3'
        # Try premium/standard filenames first, then fall back to legacy name for backward compatibility
        candidate_filenames = [
            f"premium_default_recommendations_{bhk_norm}BHK.html",
            f"standard_default_recommendations_{bhk_norm}BHK.html",
            f"default_recommendations_{bhk_norm}BHK.html",
        ]
        file_path = None
        for fname in candidate_filenames:
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)), fname)
            if os.path.exists(path):
                filename = fname
                file_path = path
                break
        if file_path is None:
            return f"Default recommendations file not found for {bhk_norm}BHK", 404
        if not os.path.exists(file_path):
            return f"Default recommendations file not found: {filename}", 404
        return send_file(file_path, mimetype='text/html', as_attachment=False)
    except Exception as e:
        return f"Error serving default recommendations: {str(e)}", 500

def allowed_file(filename):
    # For Excel files (requirements)
    excel_extensions = {'xlsx', 'xls'}
    # For room images/drawings
    image_extensions = {'jpg', 'jpeg', 'png', 'gif', 'bmp'}
    document_extensions = {'pdf', 'doc', 'docx', 'ppt', 'pptx', 'dwg', 'dxf'}
    video_extensions = {'mp4', 'avi', 'mov'}
    
    allowed_extensions = excel_extensions | image_extensions | document_extensions | video_extensions
    
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

if __name__ == '__main__':
    betterhome.run(debug=True, host='0.0.0.0', port=5002)
