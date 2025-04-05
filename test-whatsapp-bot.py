import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

# Configs
CHROMEDRIVER_PATH = '/usr/local/bin/chromedriver'
WHATSAPP_NUMBER = '+91 62064 85496'
QUESTIONS_CSV = 'test_questions.csv'
OUTPUT_CSV = 'whatsapp_qa_results.csv'

# Load questions
df = pd.read_csv(QUESTIONS_CSV)
questions = df['question'].dropna().tolist()

# Launch browser
from selenium.webdriver.chrome.service import Service

service = Service(CHROMEDRIVER_PATH)
driver = webdriver.Chrome(service=service)
driver.get('https://web.whatsapp.com')
input("Scan QR Code on WhatsApp Web, then press Enter here to continue...")

# Search for number
search_box = driver.find_element(By.XPATH, '//div[@contenteditable="true"][@data-tab="3"]')
search_box.click()
search_box.send_keys(WHATSAPP_NUMBER)
time.sleep(3)
search_box.send_keys(Keys.ENTER)
time.sleep(2)

# Message sending and reading
qa_pairs = []

def send_and_get_response(question):
    # Send question
    message_box = driver.find_element(By.XPATH, '//div[@contenteditable="true"][@data-tab="10"]')
    message_box.send_keys(question)
    message_box.send_keys(Keys.ENTER)
    time.sleep(20)  # Wait for reply

    # Get latest response
    messages = driver.find_elements(By.XPATH, '//div[contains(@class,"message-in")]/div/div/div/div/span')
    if messages:
        return messages[-1].text
    else:
        return "No response found"

# Loop over questions
for question in questions:
    print(f"Asking: {question}")
    answer = send_and_get_response(question)
    qa_pairs.append({'Question': question, 'Answer': answer})
    time.sleep(3)

# Save responses
output_df = pd.DataFrame(qa_pairs)
output_df.to_csv(OUTPUT_CSV, index=False)
print(f"Done! Responses saved to {OUTPUT_CSV}")

driver.quit()

