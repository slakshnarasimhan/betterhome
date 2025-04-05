
import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Configs
CHROMEDRIVER_PATH = '/usr/local/bin/chromedriver'
WHATSAPP_NUMBER = '+91 62064 85496'
QUESTIONS_CSV = 'questions.csv'
OUTPUT_CSV = 'whatsapp_qa_results_diagnostic.csv'

# Load questions
df = pd.read_csv(QUESTIONS_CSV)
questions = df['question'].dropna().tolist()

# Launch browser
service = Service(CHROMEDRIVER_PATH)
driver = webdriver.Chrome(service=service)
driver.get('https://web.whatsapp.com')
input("Scan QR Code on WhatsApp Web, then press Enter here to continue...")

# Wait until search box appears
wait = WebDriverWait(driver, 30)
search_box = wait.until(EC.presence_of_element_located((By.XPATH, '//div[@contenteditable="true"][@data-tab="3"]')))
search_box.click()
search_box.send_keys(WHATSAPP_NUMBER)
time.sleep(3)
search_box.send_keys(Keys.ENTER)
time.sleep(2)

qa_pairs = []

def send_and_get_response(question):
    print(f"Sending question: {question}")
    message_box = driver.find_element(By.XPATH, '//div[@contenteditable="true"][@data-tab="10"]')
    message_box.send_keys(question)
    message_box.send_keys(Keys.ENTER)

    # Count existing messages
    previous_msgs = driver.find_elements(By.XPATH, '//div[contains(@class,"message-in")]')
    previous_count = len(previous_msgs)

    # Wait and check for new messages
    timeout = 30
    interval = 2
    waited = 0
    while waited < timeout:
        current_msgs = driver.find_elements(By.XPATH, '//div[contains(@class,"message-in")]')
        if len(current_msgs) > previous_count:
            print("New message received!")

            # Try multiple strategies
            print("Strategy 1: div[@dir='ltr']")
            messages1 = driver.find_elements(By.XPATH, '//div[contains(@class,"message-in")]//div[@dir="ltr"]')
            for i, msg in enumerate(messages1[-3:]):
                print(f"[div[@dir='ltr'] #{i}] {msg.text}")

            print("Strategy 2: span[@class]")
            messages2 = driver.find_elements(By.XPATH, '//div[contains(@class,"message-in")]//span[@class]')
            for i, msg in enumerate(messages2[-3:]):
                print(f"[span[@class] #{i}] {msg.text}")

            print("Strategy 3: JS innerText")
            js_texts = driver.execute_script("""
                return Array.from(document.querySelectorAll("div.message-in"))
                    .map(el => el.innerText)
                    .filter(text => text.trim().length > 0);
            """)
            for i, msg in enumerate(js_texts[-3:]):
                print(f"[JS innerText #{i}] {msg}")

            return js_texts[-1] if js_texts else "No response found"

        time.sleep(interval)
        waited += interval

    print("Timed out waiting for response.")
    return "No response found"

# Main loop
for question in questions:
    answer = send_and_get_response(question)
    qa_pairs.append({'Question': question, 'Answer': answer})
    time.sleep(3)

# Save to CSV
output_df = pd.DataFrame(qa_pairs)
output_df.to_csv(OUTPUT_CSV, index=False)
print(f"Done! Responses saved to {OUTPUT_CSV}")

driver.quit()
