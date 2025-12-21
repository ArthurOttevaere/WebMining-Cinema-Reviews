import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from datetime import datetime
import re
import os
from urllib.parse import urljoin

# ----------------------------
# CONFIGURATION
# ----------------------------
BASE = "https://www.rogerebert.com"
START = BASE + "/reviews"
AJAX_URL = START
OUTPUT_FILE_CSV = "data/raw/roger_ebert.csv"
OUTPUT_FILE_XLSX = "data/raw/roger_ebert.xlsx"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/123.0 Safari/537.36",
    "Content-Type": "application/json",
    "Accept": "application/json"
}

COLONNES_CIBLE = [
    "site_name", "article_title", "article_url", "article_author", "article_date",
    "article_tags", "film_title", "film_year", "film_director", "film_main_actors",
    "film_genre", "film_duration", "article_text_full",
    "article_blockquotes", "article_subtitles", "main_image_url", "all_image_urls",
    "image_alt", "word_count", "image_count", "review_score", "links_to_other_reviews"
]

# ----------------------------
# FONCTION AJAX
# ----------------------------
def get_ajax_review_links(paged=2):
    payload = {
        "action": "facetwp_refresh",
        "data": {
            "facets": {
                "search_reviews": "",
                "order_by": [], "filter_by_reviewer": [], "year_filter": [],
                "rating_filter": ["-1.0","40"], "genre": [], "great_movies": [],
                "exclude_non_rated": [], "loader": []
            },
            "frozen_facets": {},
            "http_params": {"get": [], "uri": "reviews", "url_vars": []},
            "template": "wp",
            "extras": {"sort": "default"},
            "soft_refresh": 1, "is_bfcache": 1, "first_load": 0,
            "paged": paged
        }
    }

    try:
        r = requests.post(AJAX_URL, headers=HEADERS, json=payload, timeout=15)
        r.raise_for_status()
        data = r.json()
        html_content = data.get("template", "")
        if not html_content.strip():
            return [], False
        soup = BeautifulSoup(html_content, "html.parser")
        links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.startswith("/reviews/"):
                full_url = urljoin(BASE, href)
            elif "rogerebert.com/reviews/" in href:
                full_url = href
            else:
                continue
            if full_url not in links:
                links.append(full_url)
        load_more = bool(soup.find("button", class_="facetwp-load-more"))
        return links, load_more
    except Exception as e:
        print(f"Erreur AJAX: {e}")
        return [], False

# ----------------------------
# FONCTION PRINCIPALE BFS
# ----------------------------
def launch_scrap_roger_ebert(max_articles=60, max_depth=2):
    print(f"--- Lancement BFS Roger Ebert pour {max_articles} articles, profondeur max {max_depth} ---")

    to_scrape = [(START, 0)]  # liste de tuples (URL, profondeur)
    visited_urls = set()
    donnees_finales = []

    # ----------------------------
    # Récupération des URLs AJAX
    # ----------------------------
    paged = 2
    while len(to_scrape) < max_articles:
        new_links, load_more = get_ajax_review_links(paged=paged)
        for link in new_links:
            to_scrape.append((link, 0))  # profondeur 0 pour AJAX
        if not load_more or not new_links:
            break
        paged += 1
        time.sleep(0.5)

    print(f"Total URLs à scraper après AJAX : {len(to_scrape)}")

    # ----------------------------
    # BFS avec contrôle de profondeur
    # ----------------------------
    while to_scrape and len(donnees_finales) < max_articles:
        url, depth = to_scrape.pop(0)
        if url in visited_urls:
            continue
        visited_urls.add(url)
        print(f"Scraping ({len(donnees_finales)+1}/{max_articles}) profondeur {depth}: {url}")

        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
        except Exception as e:
            print(f"Erreur scraping URL: {e}")
            continue

        # ----------------------------
        # Extraction simplifiée
        # ----------------------------
        site_name = "Roger Ebert"
        h1 = soup.find("h1")
        film_title = h1.get_text(strip=True) if h1 else ""
        article_title = film_title
        author_tag = soup.find("a", href=lambda val: val and "/contributors/" in val)
        article_author = author_tag.get_text(strip=True) if author_tag else ""

        article_date = ""
        time_tag = soup.find("time")
        if time_tag and time_tag.has_attr("datetime"):
            article_date = time_tag["datetime"].split("T")[0]

        # Texte complet
        body = soup.find("div", class_="entry-content") or soup.find("article")
        paragraphs = [p.get_text(strip=True) for p in body.find_all("p")] if body else []
        article_text_full = " ".join(paragraphs)

        # Liens internes (seulement si profondeur < max_depth)
        links_internal = []
        if body and depth < max_depth:
            for a in body.find_all("a", href=True, attrs={"data-type": "review"}):
                href = a["href"]
                if href.startswith("/reviews/"):
                    href = urljoin(BASE, href)
                if href not in visited_urls:
                    to_scrape.append((href, depth + 1))
                links_internal.append(href)

        row = {
            "site_name": site_name,
            "article_title": article_title,
            "article_url": url,
            "article_author": article_author,
            "article_date": article_date,
            "article_text_full": article_text_full,
            "links_to_other_reviews": ", ".join(links_internal)
        }

        donnees_finales.append(row)
        time.sleep(0.1)

    # ----------------------------
    # Sauvegarde CSV
    # ----------------------------
    if donnees_finales:
        df = pd.DataFrame(donnees_finales)
        os.makedirs(os.path.dirname(OUTPUT_FILE_CSV), exist_ok=True)
        df.to_csv(OUTPUT_FILE_CSV, index=False, encoding="utf-8")
        print(f"\n✅ Terminé ! {len(df)} reviews sauvegardées dans {OUTPUT_FILE_CSV}")
    else:
        print("\n❌ Aucune donnée collectée.")

# ----------------------------
# EXECUTION
# ----------------------------
if __name__ == "__main__":
    launch_scrap_roger_ebert(max_articles=20, max_depth=2)
