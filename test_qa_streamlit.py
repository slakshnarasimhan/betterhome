from selenium import webdriver
from selenium.webdriver.common.by import By
import time

driver = webdriver.Chrome()
driver.get("http://13.232.247.253:8501/")

# Wait and find input box
time.sleep(3)
text_input = driver.find_element(By.TAG_NAME, "input")
text_input.send_keys("What is the most expensive UPVC Doors and Windows?")
text_input.send_keys(u'\ue007')  # Press Enter

# Wait for results to render
time.sleep(5)
page_text = driver.find_element(By.TAG_NAME, "body").text
print(page_text)

driver.quit()

