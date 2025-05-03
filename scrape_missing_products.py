import requests
from bs4 import BeautifulSoup
import csv
import json
import time
import re

def extract_url_from_line(line):
    """Extract URL from a line in the missing-products.txt file"""
    if 'Cleaned URL:' in line:
        return line.strip().replace('Cleaned URL: ', '')
    return None

def get_missing_urls():
    """Extract unique URLs from missing-products.txt"""
    urls = []
    with open('missing-products.txt', 'r') as f:
        for line in f:
            url = extract_url_from_line(line)
            if url and url not in urls:
                urls.append(url)
    return urls

def scrape_product_details(url):
    """Scrape product details from a given URL"""
    print(f"Scraping: {url}")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Initialize product data with defaults
        product_data = {
            'handle': '',
            'title': '',
            'Product Type': '',
            'Category': '',
            'tags': '',
            'SKU': '',
            'Weight': '',
            'Better Home Price': '',
            'Retail Price': '',
            'Description': '',
            'Brand': '',
            'Material': '',
            'Returns Policy': '',
            'Warranty': '',
            'url': url,
            'Color': '',
            'Finish': '',
            'Style': ''
        }
        
        # Extract JSON-LD structured data
        structured_data = None
        for script in soup.find_all('script', {'type': 'application/ld+json'}):
            try:
                data = json.loads(script.string)
                if '@type' in data and data['@type'] == 'Product':
                    structured_data = data
                    break
            except:
                continue
        
        # Extract product title
        if structured_data and 'name' in structured_data:
            product_data['title'] = structured_data['name']
        else:
            title_tag = soup.find('h1', class_='product-title')
            if title_tag:
                product_data['title'] = title_tag.text.strip()
        
        # Extract product brand
        if structured_data and 'brand' in structured_data and 'name' in structured_data['brand']:
            product_data['Brand'] = structured_data['brand']['name']
        else:
            brand_elem = soup.find('div', class_='vendor')
            if brand_elem:
                product_data['Brand'] = brand_elem.text.strip()
        
        # Extract product price
        if structured_data and 'offers' in structured_data:
            offers = structured_data['offers']
            if isinstance(offers, list) and len(offers) > 0:
                offers = offers[0]
            
            if 'price' in offers:
                price = offers['price']
                try:
                    product_data['Better Home Price'] = float(price)
                except:
                    pass
            
            if 'priceValidUntil' in offers:
                try:
                    # This could be a good indicator of retail price or original price
                    retail_price_elem = soup.find('span', class_='product-price-savings')
                    if retail_price_elem:
                        retail_price_text = retail_price_elem.text.strip()
                        retail_price_match = re.search(r'(?:₹|Rs\.?)\s*([0-9,]+)', retail_price_text)
                        if retail_price_match:
                            product_data['Retail Price'] = float(retail_price_match.group(1).replace(',', ''))
                except:
                    pass
        else:
            price_elem = soup.find('span', class_='product-price')
            if price_elem:
                price_text = price_elem.text.strip()
                price_match = re.search(r'(?:₹|Rs\.?)\s*([0-9,]+)', price_text)
                if price_match:
                    try:
                        product_data['Better Home Price'] = float(price_match.group(1).replace(',', ''))
                    except:
                        pass
        
        # Extract product description
        if structured_data and 'description' in structured_data:
            product_data['Description'] = structured_data['description']
        else:
            description_elem = soup.find('div', class_='product-description')
            if description_elem:
                product_data['Description'] = description_elem.text.strip()
        
        # Extract product type and category
        breadcrumbs = soup.find('ul', class_='breadcrumb')
        if breadcrumbs:
            links = breadcrumbs.find_all('a')
            if len(links) > 1:
                # The last breadcrumb is usually the product category
                product_data['Category'] = links[-1].text.strip()
                # Best guess for product type based on category
                product_data['Product Type'] = links[-1].text.strip()
        
        # Extract product SKU
        sku_elem = soup.find('span', class_='sku')
        if sku_elem:
            product_data['SKU'] = sku_elem.text.strip()
        
        # Extract color, finish, material from product description or details
        details_elem = soup.find('div', class_='description-area')
        if details_elem:
            details_text = details_elem.text.lower()
            
            # Extract color
            color_match = re.search(r'color:?\s*([a-zA-Z\s]+)', details_text)
            if color_match:
                product_data['Color'] = color_match.group(1).strip()
            
            # Extract material
            material_match = re.search(r'material:?\s*([a-zA-Z\s]+)', details_text)
            if material_match:
                product_data['Material'] = material_match.group(1).strip()
            
            # Extract finish
            finish_match = re.search(r'finish:?\s*([a-zA-Z\s]+)', details_text)
            if finish_match:
                product_data['Finish'] = finish_match.group(1).strip()
        
        # Extract warranty information
        warranty_match = re.search(r'warranty:?\s*([a-zA-Z0-9\s\+]+)', product_data['Description'].lower())
        if warranty_match:
            product_data['Warranty'] = warranty_match.group(1).strip()
        
        # Generate a handle from the title
        if product_data['title']:
            handle = product_data['title'].lower().replace(' ', '-')
            # Remove special characters
            handle = re.sub(r'[^a-z0-9\-]', '', handle)
            product_data['handle'] = handle
        
        return product_data
    
    except Exception as e:
        print(f"Error scraping {url}: {str(e)}")
        return None

def main():
    # Get missing product URLs
    missing_urls = get_missing_urls()
    print(f"Found {len(missing_urls)} missing product URLs")
    
    # Scrape product details
    products = []
    for url in missing_urls:
        product_data = scrape_product_details(url)
        if product_data:
            products.append(product_data)
        time.sleep(2)  # Be nice to the server
    
    # Save to CSV
    if products:
        output_file = 'missing_bestsellers.csv'
        csv_fields = [
            'handle', 'title', 'Product Type', 'Category', 'tags', 'SKU', 'Weight', 
            'Better Home Price', 'Retail Price', 'Description', 'Brand', 'Material',
            'Returns Policy', 'Warranty', 'url', 'Color', 'Finish', 'Style'
        ]
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writeheader()
            for product in products:
                writer.writerow(product)
        
        print(f"Saved {len(products)} products to {output_file}")

if __name__ == "__main__":
    main() 