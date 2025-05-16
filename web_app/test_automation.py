import os
import sys
import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import subprocess
import threading

def start_flask_app():
    """Start the Flask application in a separate thread"""
    def run_flask():
        subprocess.run(['python3', 'app.py'])
    
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.daemon = True
    flask_thread.start()
    time.sleep(2)  # Give Flask time to start

def fill_form(driver, form_data):
    """Fill the form with the provided data"""
    # Wait for the form to be fully loaded
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.NAME, 'name'))
    )
    
    # Basic information
    driver.find_element(By.NAME, 'name').send_keys(form_data['Name'])
    driver.find_element(By.NAME, 'mobile').send_keys(form_data['Mobile Number (Preferably on WhatsApp)'])
    driver.find_element(By.NAME, 'email').send_keys(form_data['E-mail'])
    driver.find_element(By.NAME, 'address').send_keys(form_data['Apartment Address'])
    driver.find_element(By.NAME, 'budget').send_keys(form_data['What is your overall budget for home appliances?'])
    
    # Demographics
    driver.find_element(By.NAME, 'adults').send_keys(form_data['Adults (between the age 18 to 50)'])
    driver.find_element(By.NAME, 'elders').send_keys(form_data['Elders (above the age 60)'])
    driver.find_element(By.NAME, 'kids').send_keys(form_data['Kids (below the age 18)'])
    
    # Property details
    driver.find_element(By.NAME, 'bedrooms').send_keys(form_data['Number of bedrooms'])
    driver.find_element(By.NAME, 'bathrooms').send_keys(form_data['Number of bathrooms'])
    
    # Hall details
    driver.find_element(By.NAME, 'hall_fans').send_keys(form_data['Hall: Fan(s)?'])
    driver.find_element(By.NAME, 'hall_ac').send_keys(form_data['Hall: Air Conditioner (AC)?'])
    driver.find_element(By.NAME, 'hall_color').send_keys(form_data['Hall: Colour theme?'])
    driver.find_element(By.NAME, 'hall_square_feet').send_keys(form_data['Hall: What is the square feet?'])
    driver.find_element(By.NAME, 'hall_other_info').send_keys(form_data['Hall: Any other information?'])
    
    # Kitchen details
    driver.find_element(By.NAME, 'kitchen_chimney').send_keys(form_data['Kitchen: Chimney width?'])
    driver.find_element(By.NAME, 'kitchen_stove').send_keys(form_data['Kitchen: Gas stove type?'])
    driver.find_element(By.NAME, 'kitchen_burners').send_keys(form_data['Kitchen: Number of burners?'])
    driver.find_element(By.NAME, 'kitchen_stove_width').send_keys(form_data['Kitchen: Stove width?'])
    driver.find_element(By.NAME, 'kitchen_fan').send_keys(form_data['Kitchen: Do you need a small fan?'])
    driver.find_element(By.NAME, 'kitchen_dishwasher_capacity').send_keys(form_data['Kitchen: Dishwasher capacity?'])
    driver.find_element(By.NAME, 'kitchen_refrigerator_type').send_keys(form_data['Kitchen: Refrigerator type?'])
    driver.find_element(By.NAME, 'kitchen_refrigerator_capacity').send_keys(form_data['Kitchen: Refrigerator capacity?'])
    driver.find_element(By.NAME, 'kitchen_other_info').send_keys(form_data['Kitchen: Do you need any other appliances or do you have any other information?'])
    
    # Master bedroom details
    driver.find_element(By.NAME, 'master_ac').send_keys(form_data['Master: Air Conditioner (AC)?'])
    driver.find_element(By.NAME, 'master_water').send_keys(form_data['Master: How do you bath with the hot & cold water?'])
    driver.find_element(By.NAME, 'master_exhaust_size').send_keys(form_data['Master: Exhaust fan size?'])
    driver.find_element(By.NAME, 'master_color').send_keys(form_data['Master: What is the colour theme?'])
    driver.find_element(By.NAME, 'master_area').send_keys(form_data['Master: What is the area of the bedroom in square feet?'])
    driver.find_element(By.NAME, 'master_bathroom_for_elders').send_keys(form_data['Master: Do you want a Glass Partition in the bathroom?'])
    driver.find_element(By.NAME, 'master_water_heater_ceiling').send_keys(form_data['Master: Is the water heater going to be inside the false ceiling in the bathroom?'])
    driver.find_element(By.NAME, 'master_exhaust_color').send_keys(form_data['Master: Exhaust fan colour?'])
    driver.find_element(By.NAME, 'master_led_mirror').send_keys(form_data['Master: Would you like to have a LED Mirror?'])
    driver.find_element(By.NAME, 'master_other_info').send_keys(form_data['Master: Any other information?'])
    
    # Bedroom 2 details
    driver.find_element(By.NAME, 'bedroom2_ac').send_keys(form_data['Bedroom 2: Air Conditioner (AC)?'])
    driver.find_element(By.NAME, 'bedroom2_water').send_keys(form_data['Bedroom 2: How do you bath with the hot & cold water?'])
    driver.find_element(By.NAME, 'bedroom2_exhaust_size').send_keys(form_data['Bedroom 2: Exhaust fan size?'])
    driver.find_element(By.NAME, 'bedroom2_color').send_keys(form_data['Bedroom 2: What is the colour theme?'])
    driver.find_element(By.NAME, 'bedroom2_area').send_keys(form_data['Bedroom 2: What is the area of the bedroom in square feet?'])
    driver.find_element(By.NAME, 'bedroom2_for_kids').send_keys(form_data['Bedroom 2: Is this for kids above'])
    driver.find_element(By.NAME, 'bedroom2_water_heater_ceiling').send_keys(form_data['Bedroom 2: Is the water heater going to be inside the false ceiling in the bathroom?'])
    driver.find_element(By.NAME, 'bedroom2_bathroom_for_elders').send_keys(form_data['Bedroom 2: Do you want a Glass Partition in the bathroom?'])
    driver.find_element(By.NAME, 'bedroom2_exhaust_color').send_keys(form_data['Bedroom 2: Exhaust fan colour?'])
    driver.find_element(By.NAME, 'bedroom2_led_mirror').send_keys(form_data['Bedroom 2: Would you like to have a LED Mirror?'])
    driver.find_element(By.NAME, 'bedroom2_other_info').send_keys(form_data['Bedroom 2: Any other information?'])
    
    # If 3 bedrooms, add Bedroom 3 details
    if form_data['Number of bedrooms'] == '3':
        driver.find_element(By.NAME, 'bedroom3_ac').send_keys(form_data['Bedroom 3: Air Conditioner (AC)?'])
        driver.find_element(By.NAME, 'bedroom3_water').send_keys(form_data['Bedroom 3: How do you bath with the hot & cold water?'])
        driver.find_element(By.NAME, 'bedroom3_exhaust_size').send_keys(form_data['Bedroom 3: Exhaust fan size?'])
        driver.find_element(By.NAME, 'bedroom3_color').send_keys(form_data['Bedroom 3: What is the colour theme?'])
        driver.find_element(By.NAME, 'bedroom3_area').send_keys(form_data['Bedroom 3: What is the area of the bedroom in square feet?'])
        driver.find_element(By.NAME, 'bedroom3_for_kids').send_keys(form_data['Bedroom 3: Is this for kids above'])
        driver.find_element(By.NAME, 'bedroom3_water_heater_ceiling').send_keys(form_data['Bedroom 3: Is the water heater going to be inside the false ceiling in the bathroom?'])
        driver.find_element(By.NAME, 'bedroom3_bathroom_for_elders').send_keys(form_data['Bedroom 3: Do you want a Glass Partition in the bathroom?'])
        driver.find_element(By.NAME, 'bedroom3_exhaust_color').send_keys(form_data['Bedroom 3: Exhaust fan colour?'])
        driver.find_element(By.NAME, 'bedroom3_led_mirror').send_keys(form_data['Bedroom 3: Would you like to have a LED Mirror?'])
        driver.find_element(By.NAME, 'bedroom3_other_info').send_keys(form_data['Bedroom 3: Any other information?'])
    
    # Laundry details
    driver.find_element(By.NAME, 'laundry_washing').send_keys(form_data['Laundry: Washing Machine?'])
    driver.find_element(By.NAME, 'laundry_dryer').send_keys(form_data['Laundry: Dryer?'])
    
    # Dining details
    driver.find_element(By.NAME, 'dining_fan').send_keys(form_data['Dining: Fan'])
    driver.find_element(By.NAME, 'dining_ac').send_keys(form_data['Dining: Air Conditioner (AC)?'])
    driver.find_element(By.NAME, 'dining_color').send_keys(form_data['Dining: Colour theme?'])
    driver.find_element(By.NAME, 'dining_square_feet').send_keys(form_data['Dining: What is the square feet?'])
    
    # Other information
    driver.find_element(By.NAME, 'other_info').send_keys(form_data['Any other information?'])
    driver.find_element(By.NAME, 'questions_comments').send_keys(form_data['Questions and comments'])

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 test_automation.py <input_excel_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found")
        sys.exit(1)
    
    # Read the input Excel file
    df = pd.read_excel(input_file)
    
    # Start the Flask application
    start_flask_app()
    
    # Set up Chrome options
    chrome_options = Options()
    chrome_options.add_argument('--headless')  # Run in headless mode
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--window-size=1920,1080')  # Set a specific window size
    
    # Initialize the Chrome driver
    driver = webdriver.Chrome(options=chrome_options)
    
    try:
        # Process each row in the Excel file
        for index, row in df.iterrows():
            print(f"\nProcessing row {index + 1}")
            
            # Navigate to the form
            driver.get('http://localhost:5003')
            
            # Fill the form with the current row's data
            fill_form(driver, row)
            
            # Wait for the submit button to be present and scroll to it
            submit_button = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 'button[type="submit"]'))
            )
            
            # Scroll to the button
            driver.execute_script("arguments[0].scrollIntoView(true);", submit_button)
            time.sleep(1)  # Give time for the scroll to complete
            
            # Click the submit button using JavaScript
            driver.execute_script("arguments[0].click();", submit_button)
            
            # Wait for the results page to load
            WebDriverWait(driver, 30).until(
                EC.presence_of_element_located((By.ID, "generate-final"))
            )
            
            # Wait for the generate final button to be clickable and scroll to it
            generate_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.ID, "generate-final"))
            )
            driver.execute_script("arguments[0].scrollIntoView(true);", generate_button)
            time.sleep(1)  # Give time for the scroll to complete
            
            # Click the generate final button using JavaScript
            driver.execute_script("arguments[0].click();", generate_button)
            
            # Wait for the final recommendation page to load
            WebDriverWait(driver, 30).until(
                EC.presence_of_element_located((By.CLASS_NAME, "final-recommendation"))
            )
            
            print(f"Successfully generated recommendations for row {index + 1}")
            
            # Wait a bit before processing the next row
            time.sleep(2)
    
    finally:
        driver.quit()

if __name__ == "__main__":
    main() 