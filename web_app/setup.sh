#!/bin/bash

# Create static directory if it doesn't exist
mkdir -p static

# Find and copy logo to static directory
if [ -f "web_app/better_home_logo.png" ]; then
    cp web_app/better_home_logo.png web_app/static/
    echo "Copied logo from web_app/better_home_logo.png to web_app/static/"
elif [ -f "better_home_logo.png" ]; then
    cp better_home_logo.png web_app/static/
    echo "Copied logo from better_home_logo.png to web_app/static/"
else
    echo "Logo not found. Please place better_home_logo.png in the web_app directory or project root."
fi

# Set Flask environment variables
export FLASK_APP=web_app/app.py
export FLASK_ENV=production

echo "Environment setup complete. Run 'flask run --host=0.0.0.0 --port=5002' to start the server." 