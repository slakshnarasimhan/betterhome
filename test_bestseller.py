import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def scrape_infinite_scroll(url):
    # Set up Chrome options (remove '--headless' if you want to watch the browser)
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--no-sandbox')
    
    # Initialize the Chrome WebDriver
    driver = webdriver.Chrome(options=chrome_options)
    driver.get(url)
    
    # Wait for the main collection container to load
    try:
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.collection"))
        )
        print("Found collection container!")
    except Exception as e:
        print("Timeout waiting for collection container. Check your CSS selector!")
        driver.quit()
        return []
    
    # Perform infinite scroll until no new content loads.
    SCROLL_PAUSE_TIME = 2  # seconds to pause to allow content to load
    MAX_RETRIES = 3  # number of consecutive scroll attempts with no height increase
    
    last_height = driver.execute_script("return document.body.scrollHeight")
    retries = 0

    while retries < MAX_RETRIES:
        # Scroll down to the bottom
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(SCROLL_PAUSE_TIME)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            retries += 1
            print(f"No new content loaded. Retry {retries}/{MAX_RETRIES}")
        else:
            retries = 0
            last_height = new_height
            print("New content loaded; resetting retry counter.")
    
    # Wait briefly to ensure all content is fully loaded
    time.sleep(2)

    # Find all product elements inside the collection container.
    # We use a descendant selector here so that only products within div.collection are captured.
    product_elements = driver.find_elements(By.CSS_SELECTOR, "div.collection div.product-item.rfq-collection-item.rfq-collection-loaded")
    print("Number of product items found:", len(product_elements))
    
    products = []
    for elem in product_elements:
        try:
            # Try to locate an <a> tag within the product element
            a_tag = elem.find_element(By.TAG_NAME, "a")
            link = a_tag.get_attribute("href")
            title = a_tag.text.strip()
        except Exception as e:
            # Fallback: use the element's full text if no anchor is found
            link = "N/A"
            title = elem.text.strip()
            
        products.append({
            "title": title,
            "link": link
        })
    
    driver.quit()
    return products

if __name__ == "__main__":
    url = "https://betterhomeapp.com/collections/bestsellers/"
    products = scrape_infinite_scroll(url)
    print(f"\nFound {len(products)} products:")
    for product in products:
        print(product)
