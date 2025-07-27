from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time

options = Options()
options.add_argument("--user-data-dir=/Users/yourname/Library/Application Support/Google/Chrome")
options.add_argument("--profile-directory=Default")  # or 'Profile 1', etc.
driver = webdriver.Chrome(options=options)

driver.get("https://your-kb-page.com")
time.sleep(10)  # let it render
print("✅ Page title:", driver.title)

# Optional: get text from main content
html = driver.page_source
driver.quit()

