#!/bin/bash

# Create static directory if it doesn't exist
mkdir -p static

# Find and copy AppliancesBazaar logo to static directory
if [ -f "web_app/static/AB-Logo.jpg" ]; then
    echo "AppliancesBazaar logo already in web_app/static/AB-Logo.jpg"
elif [ -f "web_app/AB-Logo.jpg" ]; then
    cp web_app/AB-Logo.jpg web_app/static/
    echo "Copied logo from web_app/AB-Logo.jpg to web_app/static/"
else
    echo "Logo not found. Please place AB-Logo.jpg in web_app/static/."
fi

# Set Flask environment variables
export FLASK_APP=web_app/app.py
export FLASK_ENV=production

echo "Environment setup complete. Run 'flask run --host=0.0.0.0 --port=5002' to start the server." 