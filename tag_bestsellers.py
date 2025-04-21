import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import json

def get_bestsellers():
    """Get all best-selling products from the Better Home website"""
    url = "https://betterhomeapp.com/collections/bestsellers/"
    
    # Set up Chrome options
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
    product_elements = driver.find_elements(By.CSS_SELECTOR, "div.collection div.product-item.rfq-collection-item.rfq-collection-loaded")
    print("Number of product items found:", len(product_elements))
    
    bestsellers = []
    for elem in product_elements:
        try:
            # Try to locate an <a> tag within the product element
            a_tag = elem.find_element(By.CSS_SELECTOR, "a.card-title")
            link = a_tag.get_attribute("href")
            title = a_tag.text.strip()
            
            if title:
                bestsellers.append({
                    "title": title,
                    "url": link
                })
                print(f"Found product: {title}")
        except Exception as e:
            print(f"Error processing product: {str(e)}")
            continue
    
    driver.quit()
    return bestsellers

def tag_bestsellers():
    """Tag best-selling products in the catalog"""
    # Get bestsellers from website
    bestsellers = get_bestsellers()
    print(f"\nTotal bestsellers found on website: {len(bestsellers)}")
    
    # Read the product catalog
    with open('product_catalog.json', 'r') as f:
        catalog = json.load(f)
    
    # Clean bestseller URLs by removing variant parameters
    cleaned_bestsellers = []
    for bs in bestsellers:
        url = bs.get('url', '').lower()
        # Remove variant parameter if present
        base_url = url.split('?variant=')[0].split('?')[0]
        cleaned_bestsellers.append({
            "title": bs.get('title', ''),
            "url": base_url,
            "original_url": url
        })
    
    print(f"Cleaned {len(cleaned_bestsellers)} bestseller URLs")
    
    # Tag products in catalog
    matches = 0
    matched_urls = set()
    
    for category, products in catalog.items():
        for product in products:
            # Get product URL without any parameters
            product_url = product.get('url', '').lower().split('?')[0]
            
            # Check if product URL matches any cleaned bestseller URL
            is_bestseller = any(bs['url'] == product_url for bs in cleaned_bestsellers)
            
            if is_bestseller:
                product['is_bestseller'] = True
                matches += 1
                matched_urls.add(product_url)
            else:
                product['is_bestseller'] = False
    
    print(f"\nTagged {matches} products as bestsellers in catalog")
    
    # Print unmatched bestsellers
    print("\nBestseller products not found in catalog:")
    unmatched_count = 0
    for bs in cleaned_bestsellers:
        if bs['url'] not in matched_urls:
            print(f"- {bs['title']}")
            print(f"  Cleaned URL: {bs['url']}")
            print(f"  Original URL: {bs['original_url']}")
            unmatched_count += 1
    
    print(f"\nTotal unmatched bestsellers: {unmatched_count}")
    
    # Save updated catalog
    with open('product_catalog.json', 'w') as f:
        json.dump(catalog, f, indent=4)
    
    print(f"Updated catalog saved to product_catalog.json")

if __name__ == "__main__":
    tag_bestsellers() 