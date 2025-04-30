import requests
from bs4 import BeautifulSoup
import re

# URL of the Atomberg blog page
url = "https://atomberg.com/blog"

# Fetch the page content
headers = {
    "User-Agent": "Mozilla/5.0"
}
response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.text, "html.parser")

# Regex pattern for date-like strings (e.g., Apr 22, 2024)
date_pattern = re.compile(r'\b\w{3} \d{1,2}, \d{4}\b')

# Extract blog links near date patterns
links = []
for element in soup.find_all(string=date_pattern):
    parent = element.find_parent()
    if parent:
        # Traverse upward if needed to find the clickable blog container
        for a_tag in parent.find_all("a", href=True):
            link = a_tag['href']
            if link.startswith("/"):
                link = f"https://atomberg.com{link}"
            if link not in links:
                links.append(link)

# Print the extracted links
for link in links:
    print(link)
