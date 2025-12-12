import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import uuid
import re

INDEX_URL = "https://lwlies.com/reviews"
BASE_URL = "https://lwlies.com"
OUTPUT_FILE_CSV = 'data/raw/Arthur.csv'
OUTPUT_FILE_XLSX = 'data/raw/Arthur.xlsx'
URL_BLACKLIST = ["https://lwlies.com/reviews/p2"]

def extract_text_by_label(soup, label_text):
    """We are looking for a paragraph with a label. We return the text cleaned."""
    p_tag = soup.find(lambda tag: tag.name == 'p' and label_text in tag.get_text())
    if p_tag:
        # If there are some labels, we join them
        links = p_tag.find_all('a')
        if links:
            return ", ".join([l.get_text(strip=True) for l in links])
        # Else, we return the cleaned text, without the label
        return p_tag.get_text(strip=True).replace(label_text, "").strip()
    return None

def extract_score(soup):
    """We want to extract the average of the 3 ratings given."""
    # 1. Via the figures
    score_spans = soup.find_all(attrs={re.compile(r"^data-flatplan-review-score"): True})
    scores = [int(t.get_text(strip=True)) for t in score_spans if t.get_text(strip=True).isdigit()]
    
    if not scores:
        # 2. Via the black bubbles (on the website)
        rating_divs = soup.find_all('div', class_=lambda x: x and 'w-14 h-14 rounded-full bg-black' in x)
        for div in rating_divs:
            span = div.find('span')
            if span and span.get_text(strip=True).isdigit():
                scores.append(int(span.get_text(strip=True)))
    
    if scores:
        avg = sum(scores) / 3 if len(scores) == 3 else sum(scores) / len(scores)
        return round(avg, 2)
    return None

def extract_article_title(soup):
    h1 = soup.find('h1')
    if not h1: return None
    article_title = h1.get_text(strip=True)
    return article_title

def extract_title(soup):
    """Extraction of the title (spaces and dashes excluded)."""
    h1 = soup.find('h1')
    if not h1: return None
    full_title = h1.get_text(strip=True)
    
    # Case 1 : We have a dash in the title 
    match_dash = re.search(r'review\s*[–\u2013-]', full_title, flags=re.IGNORECASE)
    if match_dash: return full_title[:match_dash.start()].strip()
    
    # Case 2 : We have a space in the title
    match_space = re.search(r'(\s+)review\b', full_title, flags=re.IGNORECASE)
    if match_space: return full_title[:match_space.start()].strip()
    
    return full_title


# --- MAIN FUNCTIONS ---

def get_review_links(start_url, target_count):
    """Navigation through the pages, collecting the review links."""
    print(f"-> Recherche de {target_count} liens de critiques...")    
    links = set()
    current_url = start_url
    page_count = 1
    
    while current_url and len(links) < target_count:
        print(f"   Scraping Page {page_count}: {current_url}")
        try:
            response = requests.get(current_url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')

            # 1. Links collection
            count_before = len(links)
            for link in soup.find_all('a', href=True):
                href = link['href']
                # Filter : We exclude pagination
                if '/reviews/' in href and href.strip('/') != 'reviews' and not re.search(r'\/p\d+\/?$', href):
                    full_url = href if href.startswith('http') else BASE_URL + href
                    if full_url not in URL_BLACKLIST:
                        links.add(full_url)
            
            print(f"      -> {len(links) - count_before} new links found.")

            # 2. "Next Button, specific to this website"
            next_btn = soup.find('a', attrs={'data-nova-track-data-label': 'pagination_next'})
            if next_btn and 'href' in next_btn.attrs:
                current_url = next_btn['href'] if next_btn['href'].startswith('http') else BASE_URL + next_btn['href']
                time.sleep(0.1)
                page_count += 1
            else:
                print("      -> No more 'Next Button', that's it!")
                break
                
        except Exception as e:
            print(f"❌ Erreur Index: {e}")
            break
            
    return list(links)

def scrape_review_page(url):
    """Extraction of the data from a review page."""
    print(f"   Scraping Review: {url}")
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200: return None
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # --- DATA INITIALIZATION ---
        data = {
            'review_id': str(uuid.uuid4()),
            'site_name': 'Little White Lies',
            'article_title': None,
            'article_url': url,
            'article_author': None,
            'article_date': None,
            'film_title': None,
            'film_year': None,
            'film_director': None,
            'film_main_actors': None,
            'film_genre': None,      # Empty for this site
            'film_duration': None,
            'article_text_full': None,
            'main_image_url': None,
            'image_alt': None,
            'word_count': None,
            'review_score': None,
            'links_to_other_reviews': None, # Empty for this site (maybe Trailer Link)
            #'cited_works_list': [] #Not in our plan but could be very important for the next parts
        }

        # --- FILLING ---

        # Title and Author
        data['article_title']= extract_article_title(soup)
        data['film_title'] = extract_title(soup)
        author_tag = soup.find('a', href=lambda x: x and '/contributor/' in x, class_=lambda x: x and 'font-bold' in x)
        data['article_author'] = author_tag.get_text(strip=True) if author_tag else 'N/A'
        
        # Date of publishing
        date_span = soup.find('span', class_=lambda x: x and 'uppercase' in x and 'not-italic' in x)
        p_parent = date_span.find_parent('p') if date_span else None
        data['article_date'] = date_span.get_text(strip=True) if p_parent and 'Published' in p_parent.get_text() else 'N/A'

        # Metadata
        data['film_director'] = extract_text_by_label(soup, 'Directed by')
        data['film_main_actors'] = extract_text_by_label(soup, 'Starring')
        
        # Duration (Regex)
        raw_runtime = extract_text_by_label(soup, 'Runtime')
        data['film_duration'] = re.search(r'\d+', raw_runtime).group(0) if raw_runtime and re.search(r'\d+', raw_runtime) else None
        
        # Year of release
        raw_release = extract_text_by_label(soup, 'Released')
        data['film_year'] = re.search(r'\d{4}', raw_release).group(0) if raw_release and re.search(r'\d{4}', raw_release) else None

        # Rating
        data['review_score'] = extract_score(soup)

        # Image
        img = soup.find('img', class_="w-full h-auto relative")
        if img:
            data['main_image_url'] = img.get('src')
            data['image_alt'] = img.get('alt')

        # Body & Word Count
        content_div = soup.find('div', class_=lambda x: x and ('text-prose' in x or 'column' in x))
        if content_div:
            # Cleaning (ads, etc)
            for junk in content_div.find_all('div', class_=lambda x: x and ('bg-[var' in x or 'ad' in x)):
                junk.decompose()
            
            # Clean text extraction
            paragraphs = content_div.find_all('p')
            clean_text = " ".join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])
            
            if len(clean_text) > 50:
                data['article_text_full'] = clean_text
                data['word_count'] = len(clean_text.split())
            else:
                data['article_text_full'] = None
                data['word_count'] = 0
            
            # Link Analysis (quoted films)
            cited = [tag.get_text(strip=True) for tag in content_div.find_all('i') if tag.get_text(strip=True) != data['film_title']]
            data['cited_works_list'] = ", ".join(set(cited))
        else:
            data['article_text_full'] = None
            data['word_count'] = 0

        return data

    except Exception as e:
        print(f"⚠️ Erreur Scraping {url}: {e}")
        return None

# --- LAUNCHING FUNCTION ---
def launch_scraping_arthur(limit = 300):
    print(f"--- Lancement du scraping Arthur (Little White Lies) pour {limit} critiques ---")
    links = get_review_links(INDEX_URL, target_count=limit)
    
    # Number of links needed
    links_to_scrape = links[:limit] 
    
    print(f"\n Treating {len(links_to_scrape)} reviews...")
    results = []
    
    for i, link in enumerate(links_to_scrape):
        print(f"[{i+1}/{len(links_to_scrape)}]", end=" ")
        review = scrape_review_page(link)
        
        # On sauvegarde seulement si on a réussi à avoir un titre
        if review and review['film_title']:
            results.append(review)
        elif review:
            print(f" (⚠️ Missing title: {link})")
        time.sleep(0.1)

    if results:
        df = pd.DataFrame(results)
        df.to_csv(OUTPUT_FILE_CSV, index=False)
        df.to_excel(OUTPUT_FILE_XLSX, index=False)
        print(f"\n✅ Finished ! {len(df)} reviews backed up in {OUTPUT_FILE_CSV} and in {OUTPUT_FILE_XLSX}")

        return df #To avoid any problems with the global function
    else:
        print("\n❌ No data has been collected.")
        return pd.DataFrame()

# --- RUNNING  ---
if __name__ == "__main__":
    launch_scraping_arthur(limit=300)