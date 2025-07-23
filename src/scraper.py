import requests
from bs4 import BeautifulSoup
import os
import time
import json
from urllib.parse import urljoin, urlparse

# The starting point for our scrape
BASE_URL = "https://docs.djangoproject.com/en/stable/"
# Directory to save the scraped data
OUTPUT_DIR = "../data/django_docs"

# Create the output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


# Finds all valid, unvisited links on a given page
def get_all_links(page_url, visited_urls):
    # Set only allows unique values
    links = set()
    try:
        response = requests.get(page_url, headers={'User-Agent': 'My RAG Project Scraper'})
        response.raise_for_status() # Raise an exception for bad status codes
        soup = BeautifulSoup(response.text, 'html.parser')

        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            # Create an absolute URL from a relative one
            full_url = urljoin(page_url, href)
            full_url = full_url.split('#')[0]

            if full_url.startswith(BASE_URL) and full_url not in visited_urls:
                links.add(full_url)

    except requests.exceptions.RequestException as e:
        print(f"Error fetching {page_url}: {e}")
    return links


# Scrapes the main content from a page and returns it with its URL
def scrape_page_content(url):
    try:
        response = requests.get(url, headers={'User-Agent': 'My RAG Project Scraper'})
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find the main content div
        content_div = soup.find('article', id='docs-content')

        if content_div:
            # Extract clean text from the main content
            text = content_div.get_text(separator='\n', strip=True)
            return {"url": url, "content": text}
        else:
            print(f"Could not find main content div on {url}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url} for scraping: {e}")
        return None


def main():
    urls_to_visit = [BASE_URL]
    visited_urls = set()

    while urls_to_visit:
        # Get a URL from our to-do list
        current_url = urls_to_visit.pop(0)
        
        if current_url in visited_urls:
            continue
            
        print(f"Visiting: {current_url}")
        visited_urls.add(current_url)

        # Scrape the content from the page
        page_data = scrape_page_content(current_url)
        if page_data:
            # Create a safe filename from the URL path
            path = current_url.replace(BASE_URL, "").replace('/', '_').strip('_')
            if not path:
                path = "index"
            filename = os.path.join(OUTPUT_DIR, f"{path}.json")

            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(page_data, f, ensure_ascii=False, indent=4)
            print(f"  -> Saved to {filename}")

        # Find all new links on the current page to add to our to-do list
        new_links = get_all_links(current_url, visited_urls)
        for link in new_links:
            if link not in visited_urls and link not in urls_to_visit:
                urls_to_visit.append(link)
        
        # Wait a second between requests to be a polite scraper
        time.sleep(1)

if __name__ == "__main__":
    main()