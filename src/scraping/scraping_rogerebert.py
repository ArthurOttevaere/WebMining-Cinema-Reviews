import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from datetime import datetime
import re
import os

# --- CONFIGURATION ---
BASE = "https://www.rogerebert.com"
START = BASE + "/reviews"
# Nom du fichier qui sera créé (à modifier si besoin)
NOM_FICHIER = "Scrap_Amine.xlsx" 

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/123.0 Safari/537.36"
}

# Liste des colonnes
COLONNES_CIBLE = [
    "site_name", "article_title", "article_url", "article_author", "article_date", 
    "article_tags", "film_title", "film_year", "film_director", "film_main_actors", 
    "film_genre", "film_duration", "article_text_full", 
    "article_blockquotes", "article_subtitles", "main_image_url", "all_image_urls", 
    "image_alt", "word_count", "image_count", "review_score", "links_to_other_reviews"
]

def launch_scrap_amine(x):
    """
    Scrape 'x' articles et crée un nouveau fichier Excel avec les résultats.
    Toute la logique est contenue dans cette unique fonction.
    """
    print(f"--- Lancement du script pour {x} articles ---")
    print(f"Attention : Le fichier '{NOM_FICHIER}' sera créé ou écrasé s'il existe.")

    # ---------------------------------------------------------
    # ETAPE 1 : RÉCUPÉRATION DES URLS
    # ---------------------------------------------------------
    print("\nRécupération des URLs...")
    soup_index = None
    
    # Tentative de connexion (3 essais)
    for attempt in range(3):
        try:
            r = requests.get(START, headers=HEADERS, timeout=15)
            r.raise_for_status()
            soup_index = BeautifulSoup(r.text, "html.parser")
            break
        except Exception:
            time.sleep(2)
    
    if not soup_index:
        print("Erreur : Impossible de charger la page index.")
        return

    urls = []
    # On cherche les liens
    for a in soup_index.find_all("a", href=True):
        href = a["href"]
        if href.startswith("/reviews/"):
            full = BASE + href
        elif "rogerebert.com/reviews/" in href:
            full = href
        else:
            continue
        
        if full not in urls:
            urls.append(full)
        
        # On s'arrête dès qu'on a le nombre demandé 'x'
        if len(urls) >= x:
            break
            
    print(f"{len(urls)} URLs trouvées à scraper.")

    # ---------------------------------------------------------
    # ETAPE 2 : BOUCLE DE SCRAPING
    # ---------------------------------------------------------
    donnees_finales = []
    
    for i, url in enumerate(urls):
        print(f"Scraping ({i+1}/{len(urls)}): {url}")
        
        # --- Connexion à l'article ---
        soup = None
        for attempt in range(3):
            try:
                r = requests.get(url, headers=HEADERS, timeout=15)
                r.raise_for_status()
                soup = BeautifulSoup(r.text, "html.parser")
                break
            except Exception:
                time.sleep(2)
        
        if not soup:
            continue 

        # --- Extraction ---
        
        # Info de base
        site_name = "Roger Ebert"
        h1 = soup.find("h1")
        film_title = h1.get_text(strip=True) if h1 else ""
        article_title = film_title 

        author_tag = soup.find("a", href=lambda val: val and "/contributors/" in val)
        article_author = author_tag.get_text(strip=True) if author_tag else ""

        # Date
        article_date = ""
        time_tag = soup.find("time")
        if time_tag and time_tag.has_attr("datetime"):
            article_date = time_tag["datetime"].split("T")[0]
        else:
            date_tag = soup.find("div", attrs={"class": "font-heading-sans text-meta-grey"})
            if date_tag:
                raw_date = date_tag.get_text(strip=True)
                if any(kw in raw_date.lower() for kw in ["ago", "less than"]):
                    article_date = datetime.today().strftime("%Y-%m-%d")
                else:
                    try:
                        dt_obj = datetime.strptime(raw_date, "%B %d, %Y")
                        article_date = dt_obj.strftime("%Y-%m-%d")
                    except ValueError:
                        article_date = raw_date

        # Métadonnées
        header_section = soup.find("section", class_="review-header") or soup.find("div", class_="container")
        header_text = header_section.get_text(" ", strip=True) if header_section else soup.get_text(" ", strip=True)

        year_match = re.search(r'\b(19|20)\d{2}\b', header_text)
        film_year = year_match.group(0) if year_match else ""

        duration_match = re.search(r'(\d+)\s*minutes', header_text, re.IGNORECASE)
        if duration_match:
            film_duration = f"{duration_match.group(1)} minutes"
        else:
            duration_match_h = re.search(r'(\d+)h\s*(\d+)m', header_text, re.IGNORECASE)
            film_duration = duration_match_h.group(0) if duration_match_h else ""

        genre_tag = soup.select_one("a.uppercase")
        film_genre = genre_tag.get_text(strip=True) if genre_tag else ""

        # Listes
        directors = []
        dir_header = soup.find(lambda tag: tag.name == "h4" and "Director" in tag.get_text())
        if dir_header:
            sibling = dir_header.find_next(["ul", "p"])
            if sibling.name == "ul":
                directors = [li.get_text(strip=True) for li in sibling.find_all("li")]
            else:
                directors = [sibling.get_text(strip=True)]
        
        actors = []
        cast_header = soup.find(lambda tag: tag.name == "h4" and "Cast" in tag.get_text())
        if cast_header and cast_header.find_next("ul"):
            actors = [li.find("a").get_text(strip=True) for li in cast_header.find_next("ul").find_all("li") if li.find("a")]

        tags = []
        tag_section = soup.find("section", class_="tags")
        if tag_section:
            tags = [a.get_text(strip=True) for a in tag_section.find_all("a")]
        
        # Contenu
        body = soup.find("div", class_="entry-content") or soup.find("article")
        paragraphs = [p.get_text(strip=True) for p in body.find_all("p")] if body else []
        article_text_full = " ".join(paragraphs)
        
        links_internal = []
        if body:
            for a in body.find_all("a", href=True):
                if "facebook" not in a["href"] and "twitter" not in a["href"]:
                    links_internal.append(a["href"])

        score_imgs = soup.find_all("img", src=lambda val: val and "star-full" in val)
        if score_imgs:
            review_score = str(len(score_imgs))
        else:
            score_tag = soup.find("img", class_="filled")
            review_score = ""
            if score_tag:
                 for c in score_tag.get("class", []):
                    if c.startswith("star"):
                        review_score = str(int(c.replace("star", "")) / 10)

        # Extras
        imgs = soup.find_all("img")
        all_image_urls = [img.get("src") for img in imgs if img.get("src")]
        image_alt = [img.get("alt") for img in imgs if img.get("alt")]
        blockquotes = [b.get_text(strip=True) for b in soup.find_all("blockquote")]
        subtitles = [h2.get_text(strip=True) for h2 in soup.find_all("h2")]

        # Création de la ligne (SANS SYNOPSIS)
        row = {
            "site_name": site_name,
            "article_title": article_title,
            "article_url": url,
            "article_author": article_author,
            "article_date": article_date,
            "article_tags": ", ".join(tags),
            "film_title": film_title,
            "film_year": film_year,
            "film_director": ", ".join(directors),
            "film_main_actors": ", ".join(actors),
            "film_genre": film_genre,
            "film_duration": film_duration,
            # "film_synopsis" a été supprimé ici
            "article_text_full": article_text_full,
            "article_blockquotes": " | ".join(blockquotes),
            "article_subtitles": " | ".join(subtitles),
            "main_image_url": all_image_urls[0] if all_image_urls else "",
            "all_image_urls": ", ".join(all_image_urls),
            "image_alt": ", ".join(image_alt),
            "word_count": len(article_text_full.split()),
            "image_count": len(all_image_urls),
            "review_score": review_score,
            "links_to_other_reviews": ", ".join(links_internal)
        }
        
        donnees_finales.append(row)
        time.sleep(1)

    # ---------------------------------------------------------
    # ETAPE 3 : CRÉATION DU FICHIER (Ecrasement)
    # ---------------------------------------------------------
    if donnees_finales:
        df = pd.DataFrame(donnees_finales)
        
        # On s'assure d'avoir toutes les colonnes
        for col in COLONNES_CIBLE:
            if col not in df.columns:
                df[col] = ""
        
        # On remet dans l'ordre
        df = df[COLONNES_CIBLE]
        
        try:
            # index=False pour ne pas avoir la colonne 0,1,2...
            df.to_excel(NOM_FICHIER, index=False)
            print(f"\nSuccès ! Fichier '{NOM_FICHIER}' créé avec {len(donnees_finales)} articles.")
        except PermissionError:
            print(f"ERREUR : Impossible d'écrire. Fermez le fichier '{NOM_FICHIER}' !")
    else:
        print("\nAucune donnée récupérée.")

# Appel du script
launch_scrap_amine(5)