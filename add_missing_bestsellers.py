import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import re

def get_product_details(url):
    """Scrape product details from a given URL"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract basic product information
        title = soup.select_one('h1.product-title')
        title = title.text.strip() if title else ''
        
        price = soup.select_one('span.price')
        price = price.text.strip().replace('₹', '').replace(',', '') if price else '0'
        
        brand = soup.select_one('span.brand-name')
        brand = brand.text.strip() if brand else ''
        
        description = soup.select_one('div.product-description')
        description = description.text.strip() if description else ''
        
        # Extract product type from breadcrumbs or title
        product_type = ''
        breadcrumbs = soup.select('nav.breadcrumb a')
        if breadcrumbs and len(breadcrumbs) > 1:
            product_type = breadcrumbs[-2].text.strip()
        
        if not product_type:
            # Try to infer from title
            common_types = ['Fan', 'Chimney', 'Water Heater', 'Washing Machine', 'Mixer Grinder']
            for type_ in common_types:
                if type_.lower() in title.lower():
                    product_type = type_
                    break
        
        # Extract color from description or title
        color_pattern = r'(?i)(?:color|colour):\s*(\w+)'
        color_match = re.search(color_pattern, description)
        color = color_match.group(1) if color_match else ''
        
        if not color:
            common_colors = ['Black', 'White', 'Silver', 'Brown']
            for c in common_colors:
                if c.lower() in title.lower() or c.lower() in description.lower():
                    color = c
                    break
        
        return {
            'handle': url.split('/')[-1],
            'title': title,
            'Product Type': product_type,
            'Category': 'Home Appliances',
            'Better Home Price': price,
            'Retail Price': str(float(price) * 1.2),  # Estimated retail price
            'Description': description,
            'Brand': brand,
            'Color': color,
            'url': url
        }
    except Exception as e:
        print(f"Error scraping {url}: {str(e)}")
        return None

def get_bestsellers():
    """Get bestseller products directly from the website"""
    url = "https://betterhomeapp.com/collections/bestsellers"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        print(f"Fetching bestsellers from {url}")
        response = requests.get(url, headers=headers)
        print(f"Response status code: {response.status_code}")
        
        soup = BeautifulSoup(response.text, 'html.parser')
        print(f"HTML content length: {len(response.text)}")
        
        # Find all product cards
        product_cards = soup.select('div.product-item')
        print(f"Found {len(product_cards)} product cards")
        
        # Debug: print the first 500 characters of HTML
        print("\nFirst 500 chars of HTML:")
        print(response.text[:500])
        
        bestsellers = []
        for card in product_cards:
            try:
                # Get product URL
                link = card.select_one('a.card-title')
                if link:
                    product_url = 'https://betterhomeapp.com' + link['href']
                    bestsellers.append({
                        'title': link.text.strip(),
                        'url': product_url
                    })
                    print(f"Found bestseller: {link.text.strip()}")
            except Exception as e:
                print(f"Error processing card: {str(e)}")
                continue
        
        return bestsellers
    except Exception as e:
        print(f"Error scraping bestsellers page: {str(e)}")
        return []

def main():
    # Read existing products
    df_existing = pd.read_csv('cleaned_products.csv')
    existing_urls = set(df_existing['url'].str.lower())
    print("\nExisting URLs in cleaned_products.csv:")
    for url in existing_urls:
        print(f"  {url}")
    
    # Get bestsellers directly from website
    bestsellers = get_bestsellers()
    print(f"\nFound {len(bestsellers)} bestseller products on website")
    
    print("\nBestseller URLs from website:")
    for product in bestsellers:
        print(f"  {product['url'].lower()}")
    
    # Find missing bestsellers
    missing_products = []
    for product in bestsellers:
        if product['url'].lower() not in existing_urls:
            print(f"\nFound missing bestseller: {product['title']}")
            print(f"URL: {product['url']}")
            details = get_product_details(product['url'])
            if details:
                missing_products.append(details)
                print(f"Successfully scraped details for: {details['title']}")
            time.sleep(1)  # Be nice to the server
    
    if missing_products:
        # Convert to DataFrame
        df_new = pd.DataFrame(missing_products)
        
        # Append to existing CSV
        df_new.to_csv('cleaned_products.csv', mode='a', header=False, index=False)
        print(f"\nAdded {len(missing_products)} missing bestseller products to cleaned_products.csv")
    else:
        print("\nNo missing bestseller products found")

if __name__ == "__main__":
    main() 