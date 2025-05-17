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

# S3 Configuration
S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME', 'betterhome-recommendation')
s3_handler = S3Handler(S3_BUCKET_NAME)

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
        generate_html_file(form_data, final_list, html_filename, is_web_app=True)
        print(f"Generated HTML file: {html_filename}")
        
        # Upload the entire folder to S3
        s3_prefix = f"recommendations/{timestamp}"
        if s3_handler.upload_folder(user_folder, s3_prefix):
            print(f"Successfully uploaded to S3: {s3_prefix}")
            # Get the S3 URL for the folder
            s3_url = s3_handler.get_folder_url(s3_prefix)
            print(f"S3 URL: {s3_url}")
        else:
            print("Failed to upload to S3")
        
        # Check if the recommendation files were created
        html_filename = excel_filename.replace('.xlsx', '.html')
        
        print(f"Checking for files:")
        print(f"HTML file exists: {os.path.exists(html_filename)}")
        
        if os.path.exists(html_filename):
            # Get the basename for the files
            html_basename = os.path.basename(html_filename)
            
            # Get the relative path from uploads directory
            html_relative_path = os.path.relpath(html_filename, UPLOAD_FOLDER)
            
            # Create the URL for the HTML file
            html_url = url_for('view_html', filename=html_relative_path)
            
            print(f"HTML URL: {html_url}")
            
            # Format the timestamp for display
            display_timestamp = datetime.strptime(timestamp, '%Y%m%d_%H%M%S').strftime('%B %d, %Y at %I:%M %p')
            
            return render_template('results.html', 
                                 html_file=html_relative_path,
                                 user_name=form_data['Name'],
                                 timestamp=display_timestamp,
                                 s3_url=s3_url)
        else:
            print(f"Recommendation files not found. HTML: {os.path.exists(html_filename)}")
            return "Error generating recommendations. Please try again."
            
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
