import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
import os
from urllib.parse import urljoin
import uuid
import json

# ----------------------------
# CONFIGURATION
# ----------------------------
BASE = "https://www.rogerebert.com"
START = BASE + "/reviews"
AJAX_URL = START
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}

# ----------------------------
# 1. AJAX  (Seeds)
# ----------------------------
def get_ajax_review_links(paged=1):
    """Get the urls from the site"""
    payload = {
        "action": "facetwp_refresh",
        "data": {
            "facets": {
                "search_reviews": "",
                "rating_filter": [], "genre": [], 
                "exclude_non_rated": []
            },
            "template": "wp",
            "paged": paged
        }
    }

    try:
        r = requests.post(AJAX_URL, headers=HEADERS, json=payload, timeout=10)
        if r.status_code != 200: return []
        
        data = r.json()
        soup = BeautifulSoup(data.get("template", ""), "html.parser")
        links = []
        
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "/reviews/" in href:
                full_url = urljoin(BASE, href)
                if "rogerebert.com" in full_url:
                    links.append(full_url)
                
        return list(set(links))
    except Exception as e:
        print(f"⚠️ Error AJAX: {e}")
        return []

# ----------------------------
# 2. MAIN FUNCTION (Stock of seeds)
# ----------------------------
def launch_scraping_roger_ebert(limit=100):
    print(f"--- 🕵️‍♂️ Launching Roger Ebert Scraping : {limit} articles ---")

    to_scrape = [] 
    visited_urls = set()
    results = []
    
    current_seed_page = 1 

    while len(results) < limit:
        
        # --- QUEUE LOGIC ---
        # If the queue is empty, we take new seeds
        if not to_scrape:
            print(f"\n--- 🚜 Empty queue. Collecting new seeds (Page {current_seed_page})... ---")
            new_seeds = get_ajax_review_links(paged=current_seed_page)
            
            # We only add new ones
            count_added = 0
            for link in new_seeds:
                if link not in visited_urls:
                    # We add it
                    to_scrape.append((link, 0, "ROOT (AJAX)"))
                    count_added += 1
            
            print(f"   -> {count_added} new seeds added.")
            current_seed_page += 1
            
            # Security if nothing left on the site or Error
            if not to_scrape:
                print("❌ No more seeds available on the website.")
                break
            
            time.sleep(1) 

        # --- FIFO ---
        # We take the next url to scrape
        url, depth, parent_url = to_scrape.pop(0) 
        
        if url in visited_urls: continue
        visited_urls.add(url)
        
        # Display
        clean_url = url.split('/')[-1][:25] + "..."
        if depth == 0:
            print(f"[{len(results)+1}/{limit}] (P0) 🌱 Seed : {clean_url}")
        else:
            parent_clean = parent_url.split('/')[-1][:15] + "..."
            print(f"[{len(results)+1}/{limit}] (P{depth}) 🔗 Link : {parent_clean} -> {clean_url}")

        try:
            r = requests.get(url, headers=HEADERS, timeout=10)
            if r.status_code != 200: continue
            
            soup = BeautifulSoup(r.text, "html.parser")
            
            # --- INITIALIZATION ---
            row = {
                'review_id': str(uuid.uuid4()),
                'site_name': 'Roger Ebert',
                'parent_review_url': parent_url, # <--- New colum to the check the parent review
                'article_url': url,
                'film_title': None,
                'review_score': None,
                'film_year': None,
                'film_director': None,
                'film_main_actors': None,
                'film_genre': None,
                'film_duration': None,
                'article_text_full': None,
                'word_count': 0,
                'article_author': None,
                'article_date': None,
                'cited_works_list': [] 
            }

            # --- EXTRACTION ---

            # 1. TITLE
            h1 = soup.find("h1", class_="page-title")
            if h1: row['film_title'] = h1.get_text(strip=True)

            # 2. SCORE
            star_box = soup.find("div", class_="star-box")
            if star_box:
                star_img = star_box.find("img", class_=lambda x: x and "filled" in x and "star" in x)
                if star_img:
                    classes = star_img.get("class", [])
                    for cls in classes:
                        if cls.startswith("star") and cls[4:].isdigit():
                            row['review_score'] = int(cls[4:]) / 10
                            break

            # 3. METADATA
            meta_div = soup.find("div", class_="text-meta-grey")
            if meta_div:
                meta_text = meta_div.get_text(" ", strip=True)
                year_match = re.search(r'\b(19|20)\d{2}\b', meta_text)
                if year_match: row['film_year'] = year_match.group(0)
                dur_match = re.search(r'(\d+)\s*minutes', meta_text)
                if dur_match: row['film_duration'] = dur_match.group(1)

            # 4. CAST & DIRECTOR
            for h4 in soup.find_all("h4", class_="font-heading-serif"):
                section_title = h4.get_text(strip=True).lower()
                ul_list = h4.find_next("ul") 
                if not ul_list: continue
                if "cast" in section_title:
                    actors = [a.get_text(strip=True) for a in ul_list.find_all("a")]
                    row['film_main_actors'] = ", ".join(actors[:5])
                elif "director" in section_title:
                    directors = [a.get_text(strip=True) for a in ul_list.find_all("a")]
                    row['film_director'] = ", ".join(directors)

            # 5. GENRE 
            # We look for <a> 
            genre_tags = soup.find_all("a", class_="border-primary-gold")
            
            # We only collect the genre, nothing else
            genres_list = [tag.get_text(strip=True) for tag in genre_tags]
            
            # We check to have zero duplicates 
            if genres_list:
                row['film_genre'] = ", ".join(sorted(list(set(genres_list))))
            else:
                row['film_genre'] = "N/A"


            # 6. AUTHOR
            author_tag = soup.find("a", href=lambda x: x and "/contributors/" in x)
            if author_tag: row['article_author'] = author_tag.get_text(strip=True)

            # 7. DATE (JSON-LD, Metadata)
            json_ld_scripts = soup.find_all('script', type='application/ld+json')
            for script in json_ld_scripts:
                try:
                    data = json.loads(script.get_text())
                    if "@graph" in data:
                        for item in data["@graph"]:
                            if item.get("@type") == "Review" and "datePublished" in item:
                                row['article_date'] = item["datePublished"].split("T")[0]
                                break
                    elif "datePublished" in data:
                        row['article_date'] = data["datePublished"].split("T")[0]
                    if row['article_date']: break
                except: continue
            
            if not row['article_date']:
                date_div = soup.find("div", class_="font-heading-sans text-meta-grey")
                if date_div and "minutes" not in date_div.get_text():
                    row['article_date'] = date_div.get_text(strip=True)

            # 8. TEXT & LINK ANALYSIS (BFS CORE)
            content_div = soup.find("div", class_="entry-content")
            if content_div:
                for bad_div in content_div.select("div[id^='roger-'], div[id*='Video']"):
                    bad_div.decompose()

                paragraphs = content_div.find_all("p")
                text_content = " ".join([p.get_text(strip=True) for p in paragraphs])
                row['article_text_full'] = text_content
                row['word_count'] = len(text_content.split())

                # --- BFS LOGIC ---
                if depth < 2:
                    links_found = []
                    for a in content_div.find_all("a", href=True):
                        href = a['href']
                        if "/reviews/" in href and "rogerebert.com" in urljoin(BASE, href):
                            full_link = urljoin(BASE, href)
                            cited_title = a.get_text(strip=True)
                            
                            if cited_title and len(cited_title) < 50:
                                links_found.append(cited_title)
                            
                            if full_link not in visited_urls and full_link != url:
                                # url becomes the parent of full_link 
                                to_scrape.append((full_link, depth + 1, url))
                    
                    row['cited_works_list'] = ", ".join(links_found)

            if row['film_title']:
                results.append(row)
            
            time.sleep(0.5)

        except Exception as e:
            print(f"❌ Erreur sur {url}: {e}")
            continue

    if results:
        df = pd.DataFrame(results)
        # We display the parent column first for better understanding
        cols = ['parent_review_url', 'article_url', 'film_title'] + [c for c in df.columns if c not in ['parent_review_url', 'article_url', 'film_title']]
        df = df[cols]
        
        print(f"\n✅ Finish ! {len(df)} reviews collected.")
        os.makedirs("data/raw", exist_ok=True)
        df.to_csv("data/raw/roger_ebert_debug.csv", index=False)
        df.to_excel("data/raw/roger_ebert_debug.xlsx", index = False)
        df.to_json("data/raw/roger_ebert_debug.json", index = False)
        return df
    else:
        return pd.DataFrame()

if __name__ == "__main__":
    launch_scraping_roger_ebert(limit=900)