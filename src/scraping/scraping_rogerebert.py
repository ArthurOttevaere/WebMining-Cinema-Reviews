import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from datetime import datetime
import re
import csv

# ----------------------------
# CONFIGURATION
# ----------------------------
BASE = "https://www.rogerebert.com"
START = BASE + "/reviews"

MAX_REVIEWS = 15
MAX_DEPTH = 2

OUTPUT_CSV = "scrap_reviews.csv"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/123.0 Safari/537.36"
}

# ✅ TES COLONNES + out_degree + depth
COLONNES_CIBLE = [
    "site_name", "article_title", "article_url", "article_author", "article_date",
    "article_tags", "film_title", "film_year", "film_director", "film_main_actors",
    "film_genre", "film_duration", "film_synopsis", "article_text_full",
    "article_blockquotes", "article_subtitles", "main_image_url", "all_image_urls",
    "image_alt", "word_count", "image_count", "review_score",
    "links_to_other_reviews", "out_degree", "depth"
]

# ----------------------------
# HTTP
# ----------------------------
def get_soup(url):
    for _ in range(3):
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            r.raise_for_status()
            return BeautifulSoup(r.text, "html.parser")
        except Exception:
            time.sleep(2)
    return None

# ----------------------------
# URLS FALLBACK
# ----------------------------
def get_urls(n=50):
    urls = []
    page = 1
    while len(urls) < n:
        list_url = START if page == 1 else f"{START}/page/{page}"
        soup = get_soup(list_url)
        if not soup:
            break
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "/reviews/" in href and "/page/" not in href:
                full = href if href.startswith("http") else BASE + href
                if full not in urls:
                    urls.append(full)
            if len(urls) >= n:
                break
        page += 1
        time.sleep(1)
    return urls

# ----------------------------
# SCRAPE REVIEW
# ----------------------------
def scrape(url, depth):
    soup = get_soup(url)
    if not soup:
        return None, set()

    site_name = "Roger Ebert"

    # --- TITRE ---
    h1 = soup.find("h1")
    film_title = h1.get_text(strip=True) if h1 else ""
    article_title = film_title

    # --- AUTEUR ---
    author_tag = soup.find("a", href=lambda x: x and "/contributors/" in x)
    article_author = author_tag.get_text(strip=True) if author_tag else ""

    # ----------------------------
    # DATE (LOGIQUE VALIDÉE)
    # ----------------------------
    article_date = ""
    time_tag = soup.find("time")
    if time_tag and time_tag.has_attr("datetime"):
        article_date = time_tag["datetime"].split("T")[0]
    else:
        date_div = soup.find("div", class_="font-heading-sans text-meta-grey")
        if date_div:
            raw_date = date_div.get_text(strip=True)
            if "ago" in raw_date.lower():
                article_date = datetime.today().strftime("%Y-%m-%d")
            else:
                try:
                    dt = datetime.strptime(raw_date, "%B %d, %Y")
                    article_date = dt.strftime("%Y-%m-%d")
                except ValueError:
                    article_date = raw_date

    # ----------------------------
    # TEXTE
    # ----------------------------
    body = soup.find("div", class_="entry-content") or soup.find("article")
    paragraphs = [p.get_text(strip=True) for p in body.find_all("p")] if body else []
    article_text_full = " ".join(paragraphs)

    # ----------------------------
    # MÉTADONNÉES FILM (TA LOGIQUE)
    # ----------------------------
    header_section = soup.find("section", class_="review-header") or soup.find("div", class_="container")
    header_text = header_section.get_text(" ", strip=True) if header_section else soup.get_text(" ", strip=True)

    year_match = re.search(r'\b(19|20)\d{2}\b', header_text)
    film_year = year_match.group(0) if year_match else ""

    duration_match = re.search(r'(\d+)\s*minutes', header_text, re.I)
    if duration_match:
        film_duration = f"{duration_match.group(1)} minutes"
    else:
        duration_match_h = re.search(r'(\d+)h\s*(\d+)m', header_text, re.I)
        film_duration = duration_match_h.group(0) if duration_match_h else ""

    genre_tag = soup.select_one("a.uppercase")
    film_genre = genre_tag.get_text(strip=True) if genre_tag else ""

    # --- DIRECTOR / ACTORS ---
    def get_list_items(header_text):
        header = soup.find(lambda tag: tag.name == "h4" and header_text in tag.get_text())
        if header:
            sibling = header.find_next(["ul", "p"])
            if sibling and sibling.name == "ul":
                return [li.get_text(strip=True) for li in sibling.find_all("li")]
            elif sibling:
                return [sibling.get_text(strip=True)]
        return []

    film_director = ", ".join(get_list_items("Director"))

    actors = []
    cast_header = soup.find(lambda tag: tag.name == "h4" and "Cast" in tag.get_text())
    if cast_header and cast_header.find_next("ul"):
        actors = [
            li.find("a").get_text(strip=True)
            for li in cast_header.find_next("ul").find_all("li")
            if li.find("a")
        ]
    film_main_actors = ", ".join(actors)

    # ----------------------------
    # SCORE (TA LOGIQUE)
    # ----------------------------
    review_score = ""

    # Cas 1 : étoiles pleines (images)
    score_imgs = soup.find_all("img", src=lambda x: x and "star-full" in x)
    if score_imgs:
        review_score = str(len(score_imgs))
    else:
        # Cas 2 : score via classes CSS (ex: star40 = 4.0)
        score_tag = soup.find("img", class_=lambda x: x and "star" in x)
        if score_tag:
            for c in score_tag.get("class", []):
                if c.startswith("star"):
                    try:
                        review_score = str(int(c.replace("star", "")) / 10)
                    except ValueError:
                        pass

    # ----------------------------
    # CITATIONS BODY
    # ----------------------------
    cited_reviews = set()
    if body and depth < MAX_DEPTH:
        for a in body.find_all("a", href=True):
            if a.get("data-type") == "review":
                href = a["href"]
                full = href if href.startswith("http") else BASE + href
                if full != url:
                    cited_reviews.add(full)

    # ----------------------------
    # RETOUR
    # ----------------------------
    data = {
        "site_name": site_name,
        "article_title": article_title,
        "article_url": url,
        "article_author": article_author,
        "article_date": article_date,
        "article_tags": "",
        "film_title": film_title,
        "film_year": film_year,
        "film_director": film_director,
        "film_main_actors": film_main_actors,
        "film_genre": film_genre,
        "film_duration": film_duration,
        "film_synopsis": "",
        "article_text_full": article_text_full,
        "article_blockquotes": "",
        "article_subtitles": "",
        "main_image_url": "",
        "all_image_urls": "",
        "image_alt": "",
        "word_count": len(article_text_full.split()),
        "image_count": 0,
        "review_score": review_score,
        "links_to_other_reviews": ", ".join(sorted(cited_reviews)),
        "out_degree": len(cited_reviews),
        "depth": depth
    }

    return data, cited_reviews

# ----------------------------
# MAIN — BFS AVEC DEPTH
# ----------------------------
if __name__ == "__main__":

    fallback_urls = get_urls(50)
    visited = set()
    nouvelles_donnees = []

    to_scrape = [(fallback_urls.pop(0), 0)]

    while to_scrape and len(nouvelles_donnees) < MAX_REVIEWS:
        url, depth = to_scrape.pop(0)
        if url in visited:
            continue

        print(f"\nSCRAPING {len(nouvelles_donnees)+1}/{MAX_REVIEWS} | depth={depth}")
        res, cited = scrape(url, depth)
        visited.add(url)

        if not res:
            continue

        if depth < MAX_DEPTH:
            for c in cited:
                if c not in visited:
                    to_scrape.append((c, depth + 1))

        if not to_scrape and fallback_urls:
            to_scrape.append((fallback_urls.pop(0), 0))

        nouvelles_donnees.append(res)
        time.sleep(1)

    df = pd.DataFrame(nouvelles_donnees, columns=COLONNES_CIBLE)

    df.to_csv(
        OUTPUT_CSV,
        index=False,
        sep=";",
        encoding="utf-8-sig",
        quoting=csv.QUOTE_ALL
    )

    print(f"\n✅ TERMINÉ — {len(df)} reviews sauvegardées")
