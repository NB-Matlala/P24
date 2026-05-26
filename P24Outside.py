import re
import csv
import gzip
import math
import time
import threading
import os
from queue import Queue, Empty
from datetime import datetime
from bs4 import BeautifulSoup
from requests_html import HTMLSession
from azure.storage.blob import BlobClient, BlobServiceClient

base_url = os.getenv("BASE_URL")
con_str = os.getenv("CON_STR") 
#################################################################################
NUM_THREADS = 5
RETRY_LIMIT = 5

page_queue   = Queue()
data_list    = []
results_lock = threading.Lock()
failed_pages = []
tile_urls_with_fake_class = []

session = HTMLSession()
session.headers.update({
    "User-Agent":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection":      "keep-alive",
})


url_to_property_type = {
    f'{base_url}'+'/houses-for-sale/advanced-search/results/p{page}?sp=pid%3d1%2c5%2c6%2c9%2c7%2c8%2c2%2c3%2c14%26so%3dNewest':              'houses',
    f'{base_url}'+'/apartments-for-sale/advanced-search/results/p{page}?sp=pid%3d1%2c5%2c6%2c9%2c7%2c8%2c2%2c3%2c14%26so%3dNewest':          'apartments',
    f'{base_url}'+'/townhouses-for-sale/advanced-search/results/p{page}?sp=pid%3d1%2c5%2c6%2c9%2c7%2c8%2c2%2c3%2c14%26so%3dNewest':          'townhouses',
    f'{base_url}'+'/vacant-land-for-sale/advanced-search/results/p{page}?sp=pid%3d1%2c5%2c6%2c9%2c7%2c8%2c2%2c3%2c14%26so%3dNewest':         'vacant land',
    f'{base_url}'+'/farms-for-sale/advanced-search/results/p{page}?sp=pid%3d1%2c5%2c6%2c9%2c7%2c8%2c2%2c3%2c14%26so%3dNewest':               'farms',
    f'{base_url}'+'/commercial-property-for-sale/advanced-search/results/p{page}?sp=pid%3d1%2c5%2c6%2c9%2c7%2c8%2c2%2c3%2c14%26so%3dNewest': 'commercial property',
    f'{base_url}'+'/industrial-property-for-sale/advanced-search/results/p{page}?sp=pid%3d1%2c5%2c6%2c9%2c7%2c8%2c2%2c3%2c14%26so%3dNewest': 'industrial property',
}
################################################################################


# ________________________ Get URL Total ________________________
def get_url_total(url: str) -> int:
    retry     = 0
    web_total = None

    while retry < RETRY_LIMIT:
        try:
            res       = session.get(url, timeout=15)
            soup      = BeautifulSoup(res.content, 'html.parser')
            tot_div   = soup.find('div', class_='panel-body').text
            total_str = re.sub(r"[\s\xa0]", "", tot_div.split('of ')[1])
            web_total = int(total_str)
            break
        except Exception as e:
            retry += 1
            print(f"[Total Listings Error]: {e} | Attempt ({retry})")
            time.sleep(2)

    return web_total
# ________________________ END ________________________


# ________________________ Get & Upload Web Total ________________________
def get_web_total():
    retry     = 0
    web_total = None

    while retry < RETRY_LIMIT:
        try:
            res       = session.get('{base_url}/for-sale/advanced-search/results?sp=pid%3d1%2c5%2c6%2c9%2c7%2c8%2c2%2c3%2c14%26so%3dNewest')
            soup      = BeautifulSoup(res.content, 'html.parser')
            tot_div   = soup.find('div', class_='panel-body').text
            total_str = re.sub(r"[\s\xa0]", "", tot_div.split('of ')[1])
            web_total = int(total_str)
            break
        except Exception as e:
            retry += 1
            print(f"[Total Listings Error]: {e} | Attempt ({retry})")
            time.sleep(2)

    if web_total is None:
        print(f"Failed to get total listings after {retry} attempts.")
        return

    timestamp = datetime.now().strftime('%Y-%m-%d')
    filename  = f"Prop24total_{timestamp}.csv"
    rows      = [{"total_listings": web_total, "Time_stamp": timestamp}]

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["total_listings", "Time_stamp"])
        writer.writeheader()
        writer.writerows(rows)

    blob_client = BlobClient.from_connection_string(con_str, 'webtotals', filename)
    with open(filename, "rb") as f:
        blob_client.upload_blob(f, overwrite=True)

    print(f"Total Listings ({web_total}) uploaded to Blob.")
# ________________________ END ________________________


# ________________________ Extractor ________________________
def extract_page(soup: BeautifulSoup, property_type: str, page_num: int) -> list[dict]:
    page_data = []

    # ── Identify fake ad class names ──────────────────────────────────────
    fake_class_names = []
    for style_tag in soup.find_all('style', type='text/css'):
        for line in style_tag.text.strip().splitlines():
            line = line.strip()
            if line.startswith('.'):
                fake_class_names.append(line.split('{')[0].strip()[1:])
                break

    for tile in soup.find_all('div', class_='p24_tileContainer'):

        # ── Skip fake/sponsored tiles ──────────────────────────────────────
        if any(cls in tile.get('class', []) for cls in fake_class_names):
            a_tag = tile.find('a', href=True)
            if a_tag:
                tile_urls_with_fake_class.append("{base_url}" + a_tag['href'])
            continue

        try:
            listing_number = tile.get('data-listing-number')
            estate_agency_span = tile.find('span', class_='p24_branding')
            estate_agency      = estate_agency_span.get('title', '').replace('Estate Agency profile for ', '').strip() if estate_agency_span else None

            agent_name_span = tile.find('span', class_='p24_brandingAgent')
            agent_name      = agent_name_span.find('span', class_='p24_brandingAgentName').get_text(strip=True) if agent_name_span else None

            a_tag          = tile.find('a', href=True)
            url            = a_tag['href'] if a_tag else None
            price          = tile.find(class_='p24_price').get_text(strip=True) if tile.find(class_='p24_price') else None

            title = (
                tile.find(class_='p24_description').get_text(strip=True)
                if tile.find(class_='p24_description')
                else tile.find('span', class_='p24_title').get_text(strip=True)
                if tile.find('span', class_='p24_title')
                else None
            )

            location = tile.find('span', class_='p24_location').get_text(strip=True) if tile.find('span', class_='p24_location') else None
            address  = tile.find('span', class_='p24_address').get_text(strip=True)  if tile.find('span', class_='p24_address')  else None

            bedrooms = tile.find('span', class_='p24_featureDetails', title='Bedrooms').find('span').get_text(strip=True) \
                if tile.find('span', class_='p24_featureDetails', title='Bedrooms') else None
            bathrooms = tile.find('span', class_='p24_featureDetails', title='Bathrooms').find('span').get_text(strip=True) \
                if tile.find('span', class_='p24_featureDetails', title='Bathrooms') else None
            parking_spaces = tile.find('span', class_='p24_featureDetails', title='Parking Spaces').find('span').get_text(strip=True) \
                if tile.find('span', class_='p24_featureDetails', title='Parking Spaces') else None

            erf_size = tile.find('span', class_='p24_size', title='Erf Size').find('span').get_text(strip=True) \
                if tile.find('span', class_='p24_size', title='Erf Size') else None
            if erf_size is None:
                img_el = tile.find(class_='p24_sizeIcon')
                if img_el:
                    sib      = img_el.find_next_sibling('span')
                    erf_size = sib.text.strip() if sib else None

            floor_size = tile.find('span', class_='p24_size', title='Floor Size').find('span').get_text(strip=True) \
                if tile.find('span', class_='p24_size', title='Floor Size') else None

            page_data.append({
                'listing_number': listing_number,
                'title':          title,
                'property_type':  property_type,
                'price':          price,
                'estate_agency':  estate_agency,
                'agent_name':     agent_name,
                'location':       location,
                'address':        address,
                'bedrooms':       bedrooms,
                'bathrooms':      bathrooms,
                'parking_spaces': parking_spaces,
                'erf_size':       erf_size,
                'floor_size':     floor_size,
                'url':            url,
                'Timestamp':      datetime.now().strftime('%Y-%m-%d'),
            })

        except Exception as e:
            print(f"[Tile Error] page {page_num}: {e}")

    return page_data
# ________________________ END ________________________


# ________________________ Worker ________________________
def worker():
    while True:
        try:
            url_template, page_num, property_type, total_pages = page_queue.get(timeout=5)
        except Empty:
            break

        retry     = 0
        extracted = False

        while not extracted and retry < RETRY_LIMIT:
            try:
                url       = url_template.format(page=page_num)
                res       = session.get(url, timeout=20)
                soup      = BeautifulSoup(res.content, 'html.parser')
                page_data = extract_page(soup, property_type, page_num)

                if len(page_data) < 20 and page_num < total_pages:
                    raise ValueError(f"Only {len(page_data)} listings on page {page_num}, expected ~20")

                with results_lock:
                    data_list.extend(page_data)

                print(f"[{property_type}] Page {page_num}/{total_pages} → {len(page_data)} listings")
                extracted = True

            except Exception as e:
                retry += 1
                print(f"[Worker Error] [{property_type}] page {page_num} attempt {retry}: {e}")
                time.sleep(5 * retry)

        if not extracted:
            failed_pages.append((url_template, page_num, property_type))
            print(f"[Skipped] [{property_type}] page {page_num} after {RETRY_LIMIT} retries.")

        page_queue.task_done()
# ________________________ END ________________________


# ________________________ Upload Merged CSV to Blob ________________________
def upload_merged_to_blob():
    if not data_list:
        print("No data to upload.")
        return

    timestamp  = datetime.now().strftime('%Y-%m-%d')
    filename   = f"prop24_outside_data{timestamp}.csv.gz"
    # fieldnames = list(data_list[0].keys())

    with gzip.open(filename, "wt", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=['title', 'property_type', 'listing_number', 'price', 'estate_agency', 'agent_name',
                                         'location', 'address', 'bedrooms', 'bathrooms', 'parking_spaces',
                                    'erf_size', 'floor_size', 'url','Timestamp'], extrasaction="ignore", restval="")
        writer.writeheader()
        writer.writerows(data_list)

    blob_client = BlobClient.from_connection_string(con_str, 'prop24data', filename)
    with open(filename, "rb") as f:
        blob_client.upload_blob(f, overwrite=True)

    print(f"Uploaded {len(data_list)} records → {filename}")
# ________________________ END ________________________


# ________________________ MAIN ________________________
if __name__ == "__main__":
    start_time = time.time()

    get_web_total()

    # ── Populate queue ────────────────────────────────────────────────────
    for url_template, property_type in url_to_property_type.items():
        total       = get_url_total(url_template.format(page=1))
        total_pages = math.ceil(total / 20)
        print(f"{property_type}: {total} listings → {total_pages} pages")

        for pg in range(1, total_pages + 1):
            page_queue.put((url_template, pg, property_type, total_pages))

    print(f"\nTotal pages queued: {page_queue.qsize()} across {NUM_THREADS} threads.\n")

    # ── Launch threads ────────────────────────────────────────────────────
    threads = []
    for _ in range(NUM_THREADS):
        t = threading.Thread(target=worker, daemon=True)
        t.start()
        threads.append(t)

    page_queue.join()

    upload_merged_to_blob()

    elapsed = (time.time() - start_time) / 60
    print(f"\nDone. {len(data_list)} records extracted in {elapsed:.2f} mins.")
    print(f"Failed pages : {len(failed_pages)}")
    print(f"Fake ad URLs : {len(tile_urls_with_fake_class)}")
# ________________________ END ________________________
