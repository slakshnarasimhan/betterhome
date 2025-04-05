
import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

CHROMEDRIVER_PATH = '/opt/homebrew/bin/chromedriver'
WHATSAPP_NUMBER = '+91 62064 85496'
OUTPUT_CSV = 'geyser_whatsapp_journey.csv'

# Launch browser
service = Service(CHROMEDRIVER_PATH)
driver = webdriver.Chrome(service=service)
driver.get('https://web.whatsapp.com')
input("Scan QR Code on WhatsApp Web, then press Enter here to continue...")

wait = WebDriverWait(driver, 30)
search_box = wait.until(EC.presence_of_element_located((By.XPATH, '//div[@contenteditable="true"][@data-tab="3"]')))
search_box.click()
search_box.send_keys(WHATSAPP_NUMBER)
time.sleep(3)
search_box.send_keys(Keys.ENTER)
time.sleep(2)

conversation = []

def send_and_get_response(message):
    print(f"Sending: {message}")
    message_box = driver.find_element(By.XPATH, '//div[@contenteditable="true"][@data-tab="10"]')
    message_box.send_keys(message)
    message_box.send_keys(Keys.ENTER)

    previous_msgs = driver.find_elements(By.XPATH, '//div[contains(@class,"message-in")]')
    previous_count = len(previous_msgs)

    timeout = 40
    interval = 2
    waited = 0
    while waited < timeout:
        current_msgs = driver.find_elements(By.XPATH, '//div[contains(@class,"message-in")]')
        if len(current_msgs) > previous_count:
            new_msg = current_msgs[-1].text
            print(f"Received: {new_msg}")
            return new_msg
        time.sleep(interval)
        waited += interval

    return "No response found"

# Conversation plan for geysers
initial_messages = [
    "Hi again! Now I’m looking for a geyser for our bathroom. It needs to be safe and energy-efficient — especially since we have elderly parents at home. Can you help?"
]

follow_ups = [
    "What type of geysers do you have — storage or instant?",
    "Which one would be better for a family of 4?",
    "Are there compact geysers that would fit in a small bathroom?",
    "Do any come with child safety features or auto cut-off?",
    "What’s the price range like? I’d like something below ₹7000 if possible.",
    "Which brands are most durable and reliable?",
    "How long is the warranty on these models?"
]

# Send initial message
for message in initial_messages:
    response = send_and_get_response(message)
    conversation.append({'User': message, 'Bot': response})
    time.sleep(3)

# Follow-up based on conversation
for message in follow_ups:
    response = send_and_get_response(message)
    conversation.append({'User': message, 'Bot': response})
    time.sleep(3)

# Save full conversation
df = pd.DataFrame(conversation)
df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved conversation to {OUTPUT_CSV}")

driver.quit()
