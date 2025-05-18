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

betterhome = Flask(__name__)
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
    try:
        # Get form data
        form_data = request.form.to_dict()
        print("Form data received:", form_data)
        
        # Get uploaded files
        files = request.files.getlist('files[]')
        print(f"Number of files received: {len(files)}")
        
        # Create a timestamp for the folder
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        user_folder = os.path.join(UPLOAD_FOLDER, timestamp)
        os.makedirs(user_folder, exist_ok=True)
        
        # Save uploaded files
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(user_folder, filename)
                file.save(file_path)
                print(f"Saved file: {file_path}")
        
        # Create Excel file from form data
        excel_filename = os.path.join(user_folder, 'user_requirements.xlsx')
        df = pd.DataFrame([form_data])
        df.to_excel(excel_filename, index=False)
        print(f"Created Excel file: {excel_filename}")
        
        # Process the requirements
        final_list = analyze_user_requirements(excel_filename)
        
        # Generate HTML file
        html_filename = os.path.join(user_folder, 'recommendations.html')
        generate_html_file(form_data, final_list, html_filename)
        print(f"Generated HTML file: {html_filename}")
        
        # Upload files to S3
        s3_excel_key = f"recommendations/{timestamp}/user_requirements.xlsx"
        s3_html_key = f"recommendations/{timestamp}/recommendations.html"
        
        excel_uploaded = s3_handler.upload_file(excel_filename, s3_excel_key)
        html_uploaded = s3_handler.upload_file(html_filename, s3_html_key)
        
        if excel_uploaded and html_uploaded:
            # Get S3 URLs
            excel_url = s3_handler.get_file_url(s3_excel_key)
            html_url = s3_handler.get_file_url(s3_html_key)
            
            # Format the timestamp for display
            display_timestamp = datetime.strptime(timestamp, '%Y%m%d_%H%M%S').strftime('%B %d, %Y at %I:%M %p')
            
            return render_template('results.html', 
                                 html_file=html_filename,
                                 excel_file=excel_filename,
                                 s3_html_url=html_url,
                                 s3_excel_url=excel_url,
                                 user_name=form_data['Name'],
                                 timestamp=display_timestamp)
        else:
            print("Failed to upload files to S3")
            return "Error uploading files to S3. Please try again."
            
    except Exception as e:
        print(f"Error in submit route: {str(e)}")
        return f"An error occurred: {str(e)}"

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

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'xlsx', 'xls'}

if __name__ == '__main__':
    betterhome.run(debug=True, host='0.0.0.0', port=5002) 
