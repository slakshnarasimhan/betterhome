from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
import time
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

# Set up the driver (use Chrome, or change to Firefox if you prefer)
driver = webdriver.Chrome()  # or webdriver.Firefox()

# Open the form page
driver.get("http://localhost:5002/")

def safe_click(driver, by, value, timeout=10):
    element = WebDriverWait(driver, timeout).until(
        EC.element_to_be_clickable((by, value))
    )
    # Try different scroll positions
    for block in ['center', 'end', 'start']:
        driver.execute_script(f"arguments[0].scrollIntoView({{block: '{block}'}});", element)
        time.sleep(0.2)
        try:
            element.click()
            return
        except Exception:
            pass
    # Fallback to JS click
    driver.execute_script("arguments[0].click();", element)


# --- Personal Information ---
driver.find_element(By.ID, "name").send_keys("Test User")
driver.find_element(By.ID, "mobile").send_keys("9876543210")
driver.find_element(By.ID, "email").send_keys("testuser@example.com")
driver.find_element(By.ID, "address").send_keys("123 Test Street, Test City")

# --- Budget and Family Information ---
driver.find_element(By.ID, "budget").send_keys("150000")
driver.find_element(By.ID, "adults").send_keys("2")
driver.find_element(By.ID, "elders").send_keys("1")
driver.find_element(By.ID, "kids").send_keys("1")

# --- Room Images/Drawings (optional, skip for now) ---

# --- Room Information ---
Select(driver.find_element(By.ID, "bedrooms")).select_by_value("3")
Select(driver.find_element(By.ID, "bathrooms")).select_by_value("3")

# --- Hall Requirements ---
Select(driver.find_element(By.ID, "hall_fans")).select_by_value("2")
Select(driver.find_element(By.ID, "hall_ac")).select_by_value("Yes")
Select(driver.find_element(By.ID, "hall_color")).select_by_value("Blue")
driver.find_element(By.ID, "hall_square_feet").send_keys("200")
driver.find_element(By.ID, "hall_other_info").send_keys("Spacious hall.")

# --- Kitchen Requirements ---
safe_click(driver, By.ID, "chimney_90")
safe_click(driver, By.ID, "stove_hob")
safe_click(driver, By.ID, "kitchen_burners_4")
driver.find_element(By.ID, "kitchen_fan").send_keys("Yes")
safe_click(driver, By.ID, "dishwasher_15")
safe_click(driver, By.ID, "fridge_side_by_side")
safe_click(driver, By.ID, "fridge_capacity_400_600")
driver.find_element(By.ID, "kitchen_other_info").send_keys("Need a microwave.")

# --- Master Bedroom Requirements ---
Select(driver.find_element(By.ID, "master_ac")).select_by_value("Yes")
safe_click(driver, By.ID, "master_water_shower")
safe_click(driver, By.ID, "master_exhaust_200")
safe_click(driver, By.ID, "master_exhaust_white")
Select(driver.find_element(By.ID, "master_color")).select_by_value("White")
driver.find_element(By.ID, "master_area").send_keys("150")
Select(driver.find_element(By.ID, "master_bathroom_for_elders")).select_by_value("No")
Select(driver.find_element(By.ID, "master_water_heater_ceiling")).select_by_value("No")
Select(driver.find_element(By.ID, "master_led_mirror")).select_by_value("Yes")
driver.find_element(By.ID, "master_other_info").send_keys("Master bedroom info.")

# --- Bedroom 2 Requirements ---
Select(driver.find_element(By.ID, "bedroom2_ac")).select_by_value("No")
safe_click(driver, By.ID, "bedroom2_water_bucket")
safe_click(driver, By.ID, "bedroom2_exhaust_150")
safe_click(driver, By.ID, "bedroom2_exhaust_black")
Select(driver.find_element(By.ID, "bedroom2_color")).select_by_value("Grey")
driver.find_element(By.ID, "bedroom2_area").send_keys("120")
Select(driver.find_element(By.ID, "bedroom2_for_kids")).select_by_value("No")
Select(driver.find_element(By.ID, "bedroom2_water_heater_ceiling")).select_by_value("No")
Select(driver.find_element(By.ID, "bedroom2_bathroom_for_elders")).select_by_value("No")
Select(driver.find_element(By.ID, "bedroom2_led_mirror")).select_by_value("No")
driver.find_element(By.ID, "bedroom2_other_info").send_keys("Bedroom 2 info.")

# --- Bedroom 3 Requirements (shown only if 3 bedrooms selected) ---
safe_click(driver, By.ID, "bedroom3_ac")
safe_click(driver, By.ID, "bedroom3_water_shower")
safe_click(driver, By.ID, "bedroom3_exhaust_200")
safe_click(driver, By.ID, "bedroom3_exhaust_white")
Select(driver.find_element(By.ID, "bedroom3_color")).select_by_value("Blue")
driver.find_element(By.ID, "bedroom3_area").send_keys("110")
Select(driver.find_element(By.ID, "bedroom3_for_kids")).select_by_value("Yes")
Select(driver.find_element(By.ID, "bedroom3_water_heater_ceiling")).select_by_value("Yes")
Select(driver.find_element(By.ID, "bedroom3_bathroom_for_elders")).select_by_value("No")
Select(driver.find_element(By.ID, "bedroom3_led_mirror")).select_by_value("Yes")
driver.find_element(By.ID, "bedroom3_other_info").send_keys("Bedroom 3 info.")

# --- Laundry Requirements ---
safe_click(driver, By.ID, "laundry_washing_front")
Select(driver.find_element(By.ID, "laundry_dryer")).select_by_value("Yes")

# --- Dining Room Requirements ---
safe_click(driver, By.ID, "dining_fan_large")
Select(driver.find_element(By.ID, "dining_ac")).select_by_value("Yes")
Select(driver.find_element(By.ID, "dining_color")).select_by_value("Green")

# --- Additional Information ---
driver.find_element(By.ID, "other_info").send_keys("No additional info.")
driver.find_element(By.ID, "questions_comments").send_keys("No questions.")

# --- Submit the form ---
time.sleep(1)  # Wait for any dynamic JS
# Optionally, take a screenshot for debugging
# driver.save_screenshot("submit_debug.png")
safe_click(driver, By.ID, "submitButton")

# Optionally, wait and close
#time.sleep(5)
#driver.quit()

# --- All form fields filled above ---

# Wait for manual inspection and submission
input("Form filled. Press Enter in this terminal to close the browser...")

driver.quit()

