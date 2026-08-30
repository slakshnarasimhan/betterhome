from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
wait = WebDriverWait(driver, 15)

driver.get("http://localhost:5002/")
wait.until(EC.presence_of_element_located((By.ID, "name")))


def dispatch_events(element):
    driver.execute_script(
        "arguments[0].dispatchEvent(new Event('input', {bubbles: true}));"
        "arguments[0].dispatchEvent(new Event('change', {bubbles: true}));"
        "arguments[0].dispatchEvent(new Event('keyup', {bubbles: true}));",
        element,
    )


def fill(field_id, value):
    el = wait.until(EC.presence_of_element_located((By.ID, field_id)))
    el.clear()
    el.send_keys(value)
    dispatch_events(el)


def select_value(field_id, value):
    el = wait.until(EC.presence_of_element_located((By.ID, field_id)))
    Select(el).select_by_value(value)
    dispatch_events(el)


def click_radio(field_id):
    el = wait.until(EC.presence_of_element_located((By.ID, field_id)))
    driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", el)
    time.sleep(0.1)
    try:
        el.click()
    except Exception:
        driver.execute_script("arguments[0].click();", el)
    dispatch_events(el)


def missing_required_fields():
    return driver.execute_script("""
        const missing = [];
        document.querySelectorAll('[required]').forEach(field => {
            if (field.type === 'radio') {
                const checked = document.querySelectorAll(
                    'input[name="' + field.name + '"]:checked'
                );
                if (checked.length === 0 && !missing.includes(field.name)) {
                    missing.push(field.name);
                }
            } else if (!field.value) {
                missing.push(field.id || field.name);
            }
        });
        return missing;
    """)


def refresh_submit_state():
    driver.execute_script("""
        document.querySelectorAll('input, select, textarea').forEach(function(el) {
            el.dispatchEvent(new Event('input', {bubbles: true}));
            el.dispatchEvent(new Event('change', {bubbles: true}));
        });
    """)


# --- Personal Information ---
fill("name", "Test User")
fill("mobile", "9876543210")
fill("email", "testuser@example.com")
fill("address", "123 Test Street, Test City")
fill("city", "Bengaluru")

# --- Budget and Family Information ---
fill("budget", "150000")
fill("adults", "2")
fill("elders", "1")
fill("kids", "1")

# --- Room Information (3 bedrooms makes Bedroom 3 required) ---
select_value("bedrooms", "3")
wait.until(EC.visibility_of_element_located((By.ID, "bedroom3-section")))
select_value("bathrooms", "3")

# --- Hall Requirements ---
select_value("hall_fans", "2")
select_value("hall_ac", "Yes")
select_value("hall_color", "Blue")
fill("hall_square_feet", "200")
fill("hall_other_info", "Spacious hall.")

# --- Kitchen Requirements ---
click_radio("chimney_90")
click_radio("stove_hob")
click_radio("kitchen_burners_4")
select_value("kitchen_fan", "Yes")
click_radio("dishwasher_15")
click_radio("fridge_side_by_side")
click_radio("fridge_capacity_400_600")
fill("kitchen_other_info", "Need a microwave.")

# --- Master Bedroom Requirements ---
select_value("master_ac", "Yes")
click_radio("master_water_shower")
click_radio("master_exhaust_200")
click_radio("master_exhaust_white")
select_value("master_color", "White")
fill("master_area", "150")
select_value("master_bathroom_for_elders", "No")
select_value("master_water_heater_ceiling", "No")
select_value("master_led_mirror", "Yes")
fill("master_other_info", "Master bedroom info.")

# --- Bedroom 2 Requirements ---
select_value("bedroom2_ac", "No")
click_radio("bedroom2_water_bucket")
click_radio("bedroom2_exhaust_150")
click_radio("bedroom2_exhaust_black")
select_value("bedroom2_color", "Grey")
fill("bedroom2_area", "120")
select_value("bedroom2_for_kids", "No")
select_value("bedroom2_water_heater_ceiling", "No")
select_value("bedroom2_bathroom_for_elders", "No")
select_value("bedroom2_led_mirror", "No")
fill("bedroom2_other_info", "Bedroom 2 info.")

# --- Bedroom 3 Requirements (required when bedrooms === 3) ---
select_value("bedroom3_ac", "Yes")
click_radio("bedroom3_water_shower")
click_radio("bedroom3_exhaust_200")
click_radio("bedroom3_exhaust_white")
select_value("bedroom3_color", "Blue")
fill("bedroom3_area", "110")
select_value("bedroom3_for_kids", "Yes")
select_value("bedroom3_water_heater_ceiling", "Yes")
select_value("bedroom3_bathroom_for_elders", "No")
select_value("bedroom3_led_mirror", "Yes")
fill("bedroom3_other_info", "Bedroom 3 info.")

# --- Laundry Requirements ---
click_radio("laundry_washing_front")
select_value("laundry_dryer", "Yes")

# --- Dining Room Requirements ---
click_radio("dining_fan_large")
select_value("dining_ac", "Yes")
select_value("dining_color", "Green")

# --- Additional Information ---
fill("other_info", "No additional info.")
fill("questions_comments", "No questions.")

refresh_submit_state()
missing = missing_required_fields()
if missing:
    raise RuntimeError("Required fields still empty: " + ", ".join(missing))

submit = wait.until(EC.element_to_be_clickable((By.ID, "submitButton")))
driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", submit)
time.sleep(0.2)
try:
    submit.click()
except Exception:
    driver.execute_script("arguments[0].click();", submit)

input("Form submitted. Press Enter in this terminal to close the browser...")
driver.quit()
