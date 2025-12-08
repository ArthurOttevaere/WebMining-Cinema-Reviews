import requests
from bs4 import BeautifulSoup
import csv
from urllib.parse import urljoin
from datetime import datetime

# Configuration for SlashFilm scraping
slashfilm = {
    "name": "SlashFilm",
    "base_url": "https://www.slashfilm.com/category/reviews/",
    "ajax_url": "https://www.slashfilm.com/category/reviews/?ajax=1&action=more-stories&offset={}",
    "article_selector": "h3 a",
    "title_selector": "h1.title-gallery",
    "content_selector": "div#content.article",
    "date_selector": ".byline-timestamp time",
    "rating_selector": "div#content.article p strong",
    "author_selector": "a.byline-author",
    "image_selector": "img.gallery-image"
}

all_rows = []
article_links = set()

# Step 1: Collect article links using AJAX infinite scroll
# Each batch seems to load ~27 articles, so we increment offset by 27
for offset in range(0, 500, 27):  # adjust upper limit depending on how many articles you want
    response = requests.get(slashfilm["ajax_url"].format(offset))
    if response.status_code != 200:
        break

    soup = BeautifulSoup(response.text, "html.parser")
    articles = soup.select(slashfilm["article_selector"])
    for a in articles:
        href = a.get("href")
        if href:
            article_links.add(urljoin(slashfilm["base_url"], href))

print("Nombre de liens collectés :", len(article_links))  # ✅ Vérification

# Step 2: Scrape each article in detail
for url in list(article_links)[:600]:  # limit to 600 articles
    art_resp = requests.get(url)
    art_soup = BeautifulSoup(art_resp.text, "html.parser")

    # Main title
    title_elem = art_soup.select_one(slashfilm["title_selector"])
    title = title_elem.get_text(strip=True) if title_elem else ""

    # Film title (before "Review")
    film_title = title.split("Review")[0].strip() if "Review" in title else ""

    # Date (parse ISO 8601 and keep DD-MM-YYYY)
    date = ""
    time_elem = art_soup.select_one(slashfilm["date_selector"])
    if time_elem:
        raw_date = time_elem.get("datetime", "").strip()
        # On garde juste la partie avant le "T"
        date = raw_date.split("T")[0] if raw_date else ""

    # Author
    author_elem = art_soup.select_one(slashfilm["author_selector"])
    author = author_elem.get_text(strip=True) if author_elem else ""

    # Main text (subtitles + paragraphs)
    content_block = art_soup.select_one(slashfilm["content_selector"])
    text = " ".join(
        elem.get_text(" ", strip=True)
        for elem in content_block.find_all(["h2", "p"])
    ) if content_block else ""

    # Review score (/Film Rating)
    rating_elem = art_soup.find("strong", string=lambda s: s and "/Film Rating:" in s)
    rating = rating_elem.get_text(strip=True).replace("/Film Rating:", "").strip() if rating_elem else ""
       

    # Film info (last <em> tag)
    em_tags = art_soup.find_all("em")
    film_info = em_tags[-1].get_text(" ", strip=True) if em_tags else ""

    # Images
    all_images = art_soup.find_all("img", class_="gallery-image")
    image_url = all_images[0].get("src") if all_images else ""
    

    # Word count
    word_count = len((title + " " + text).split())

    # External links (one per line)
    external_links_str = "\n".join(
        link["href"] for link in content_block.find_all("a", href=True)
        if link["href"].startswith("http")
    ) if content_block else ""

    # Add row
    # Prépare la ligne
    row = [
        slashfilm["name"], title, film_title, url, date, author, rating,
        text, film_info, word_count, image_url, external_links_str
    ]

    # Vérifie que les champs essentiels sont remplis
    required_fields = [title, film_title, date, author, rating, text, film_info, image_url]
    if all(required_fields):
        all_rows.append(row)
        print("✅ Scraping:", url, "→", title)
    else:
        print("❌ Review ignorée (incomplète):", url)

print("Nombre de lignes ajoutées :", len(all_rows))  # ✅ Vérification


# Step 3: Save results to CSV
with open("slashfilm_critiques.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "site_name", "article_title", "film_title", "article_url",
        "article_date", "article_author", "review_score",
        "article_full_text", "film_info", "word_count",
        "main_image_url","external_links"
    ])
    writer.writerows(all_rows)

print("SlashFilm scraping completed with infinite scroll simulation!")

