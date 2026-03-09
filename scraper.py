import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import time
import logging
from urllib.parse import urlparse
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import json

with open("config.json") as config:
    years = json.load(config)["years"]

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def setup_session():
    """Create a requests session with retries and connection pooling."""
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36'
    })
    return session

def get_conference_urls(start_year, end_year):
    """Generate URLs for all General Conferences from start_year to end_year."""
    base_url = 'https://www.churchofjesuschrist.org/study/general-conference/{year}/{month}?lang=eng'
    return [(base_url.format(year=year, month=month), str(year), month)
            for year in range(start_year, end_year + 1)
            for month in ['04', '10']]

def get_talk_urls(conference_url, year, month, session):
    """Fetch talk URLs from a conference page, excluding session videos."""
    try:
        response = session.get(conference_url, timeout=10)
        response.raise_for_status()
        response.encoding = 'utf-8'
    except requests.RequestException as e:
        logging.error(f"Error accessing {conference_url}: {e}")
        return []

    soup = BeautifulSoup(response.text, 'html.parser')
    talk_urls = []
    seen_urls = set()
    talk_counter = 1
    session_slugs = [
        'saturday-morning', 'saturday-afternoon', 'sunday-morning', 'sunday-afternoon',
        'general-womens-session', 'priesthood-session', 'women-session', 'womens-session',
        'general-conference', 'session', 'video', 'all-sessions', 'full-session'
    ]

    for link in soup.select('div.talk-list a[href*="/study/general-conference/"], article a[href*="/study/general-conference/"]'):
        href = link.get('href')
        if not href or 'lang=eng' not in href:
            continue

        canonical_url = urlparse('https://www.churchofjesuschrist.org' + href).geturl()
        if canonical_url in seen_urls:
            continue
        seen_urls.add(canonical_url)

        match = re.search(r'/(\d{4})/(\d{2})/(.+)', canonical_url)
        if not match:
            continue
        url_year, url_month, slug = match.groups()
        if any(session_slug in slug.lower() for session_slug in session_slugs):
            continue

        try:
            talk_response = session.get(canonical_url, timeout=5)
            talk_response.raise_for_status()
            talk_response.encoding = 'utf-8'
            talk_soup = BeautifulSoup(talk_response.text, 'html.parser')
            if not (talk_soup.find("p", {"class": "author-name"}) or talk_soup.find("div", {"class": "body-block"})):
                continue
        except requests.RequestException:
            continue

        talk_urls.append((canonical_url, str(talk_counter).zfill(2)))
        talk_counter += 1

    logging.info(f"Found {len(talk_urls)} talk URLs for {year}-{month}")
    return talk_urls

def scrape_talk(args):
    """Scrape metadata and transcript for a single talk."""
    talk_url, year, talk_number, session = args
    start_time = time.time()
    try:
        response = session.get(talk_url, timeout=10)
        response.raise_for_status()
        response.encoding = 'utf-8'
    except requests.RequestException as e:
        logging.error(f"Error accessing {talk_url}: {e}")
        return None, talk_number

    soup = BeautifulSoup(response.text, 'html.parser')

    def clean_text(text):
        if not text:
            return text
        text = text.replace('â\x80\x99', "'").replace('â\x80\x9c', '"').replace('â\x80\x9d', '"').replace('Â', ' ')
        return text.encode('ascii', 'ignore').decode('ascii').strip()

    def clean_author_name(text):
        if not text:
            return text
        text = text.replace('\u00a0', ' ').replace('Â', ' ')
        text = re.sub(r'([A-Za-z])\.([A-Za-z])', r'\1. \2', text)
        return clean_text(text)

    title = clean_text(soup.find("h1").text) if soup.find("h1") else "No Title Found"
    speaker = clean_author_name(soup.find("p", {"class": "author-name"}).text) if soup.find("p", {"class": "author-name"}) else "No Speaker Found"
    calling = clean_text(soup.find("p", {"class": "author-role"}).text) if soup.find("p", {"class": "author-role"}) else "No Calling Found"
    content_array = soup.find("div", {"class": "body-block"})
    content = "\n\n".join(clean_text(p.text) for p in content_array.find_all("p")) if content_array else "No Content Found"

    if speaker == "No Speaker Found" and content == "No Content Found":
        return None, talk_number

    year = re.search(r'/(\d{4})/', talk_url).group(1)
    season = "April" if "/04/" in talk_url else "October"

    elapsed_time = time.time() - start_time
    logging.debug(f"Processed {talk_url} in {elapsed_time:.2f} seconds")

    return {
        "title": title,
        "speaker": speaker,
        "calling": calling,
        "year": year,
        "season": season,
        "url": talk_url,
        "text": content,
    }, talk_number

def split_talks(talk):
    """Split the talk content into paragraphs."""
    paragraph_data = []
    paragraphs = talk["text"].split('\n\n')
    for i, paragraph in enumerate(paragraphs, 1):
        paragraph_data.append({
            "title": talk["title"],
            "speaker": talk["speaker"],
            "calling": talk["calling"],
            "year": talk["year"],
            "season": talk["season"],
            "url": talk["url"],
            "paragraph_number": i,
            "text": paragraph.strip(),
        })
    return paragraph_data

if __name__ == "__main__":
    print("Start Time:", datetime.now().strftime("%H:%M:%S"))

    talks_data = []
    paragraphs_data = []
    session = setup_session()

    # Step 1: Get conference URLs
    conference_urls = get_conference_urls(2025 - years, 2025)

    # Step 2: Get all talk URLs sequentially (or parallelize if needed)
    all_talk_urls = []
    for conf_url, year, month in conference_urls:
        talk_urls = get_talk_urls(conf_url, year, month, session)
        all_talk_urls.extend([(url, year, talk_number, session) for url, talk_number in talk_urls])

    # Step 3: Scrape talks concurrently
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_url = {executor.submit(scrape_talk, args): args[0] for args in all_talk_urls}
        for future in as_completed(future_to_url):
            talk, _ = future.result()
            if talk:
                talks_data.append(talk)
                paragraphs_data.extend(split_talks(talk))

    session.close()

    logging.info(f"Scraped {len(talks_data)} talks")

    # Save to CSV
    talks_df = pd.DataFrame(talks_data)
    paragraphs_df = pd.DataFrame(paragraphs_data)
    talks_df.to_csv('SCRAPED_TALKS.csv', index=False)
    paragraphs_df.to_csv('SCRAPED_PARAGRAPHS.csv', index=False)
    print("End Time:", datetime.now().strftime("%H:%M:%S"))