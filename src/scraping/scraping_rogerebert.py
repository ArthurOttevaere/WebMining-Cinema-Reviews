import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from datetime import datetime
import re
import os # Nécessaire pour vérifier si le fichier existe

# --- CONFIGURATION ---
BASE = "https://www.rogerebert.com"
START = BASE + "/reviews"
# Le nom du fichier est défini ici
NOM_FICHIER_EXISTANT = "Scrap.xlsx" 

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/123.0 Safari/537.36"
}

# Liste exacte des colonnes demandées
# J'ai remis film_synopsis pour garder la structure, mais on le laissera vide.
COLONNES_CIBLE = [
    "site_name", "article_title", "article_url", "article_author", "article_date", 
    "article_tags", "film_title", "film_year", "film_director", "film_main_actors", 
    "film_genre", "film_duration", "film_synopsis", "article_text_full", 
    "article_blockquotes", "article_subtitles", "main_image_url", "all_image_urls", 
    "image_alt", "word_count", "image_count", "review_score", "links_to_other_reviews"
]

def get_soup(url):
    """Gère la requête HTTP et renvoie l'objet BeautifulSoup"""
    for attempt in range(3):
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            r.raise_for_status()
            return BeautifulSoup(r.text, "html.parser")
        except Exception:
            time.sleep(2)
    return None

def get_urls(n=10):
    """Récupère les URLs en gérant correctement le format /page/X"""
    urls = []
    page_num = 1
    
    while len(urls) < n:
        if page_num == 1:
            current_list_url = START
        else:
            current_list_url = f"{START}/page/{page_num}"
            
        print(f"Chargement de la liste : page {page_num} ({current_list_url})...")
        
        soup = get_soup(current_list_url)
        if not soup:
            print("Impossible de charger la page ou fin des pages.")
            break

        count_before = len(urls)
        
        for a in soup.find_all("a", href=True):
            href = a["href"]
            full = ""
            
            # Filtrage des liens de critiques
            if href.startswith("/reviews/"):
                full = BASE + href
            elif "rogerebert.com/reviews/" in href:
                full = href
            else:
                continue
            
            if "/page/" in full:
                continue

            if full not in urls:
                urls.append(full)
            
            if len(urls) >= n:
                break
        
        if len(urls) == count_before:
            print(f"Aucune nouvelle URL sur la page {page_num}. Arrêt.")
            break
            
        page_num += 1
        time.sleep(1)

    return urls

def scrape(url):
    """Extrait les données d'une page de critique"""
    soup = get_soup(url)
    if not soup:
        return {}

    # --- 1. INFO DE BASE ---
    site_name = "Roger Ebert"
    h1 = soup.find("h1")
    film_title = h1.get_text(strip=True) if h1 else ""
    article_title = film_title 

    author_tag = soup.find("a", href=lambda x: x and "/contributors/" in x)
    article_author = author_tag.get_text(strip=True) if author_tag else ""

    # --- 2. DATE (Logique unifiée) ---
    article_date = ""
    time_tag = soup.find("time")
    if time_tag and time_tag.has_attr("datetime"):
        article_date = time_tag["datetime"].split("T")[0]
    else:
        date_tag = soup.find("div", attrs={"class": "font-heading-sans text-meta-grey"})
        if date_tag:
            raw_date = date_tag.get_text(strip=True)
            if any(x in raw_date.lower() for x in ["ago", "less than"]):
                article_date = datetime.today().strftime("%Y-%m-%d")
            else:
                try:
                    dt_obj = datetime.strptime(raw_date, "%B %d, %Y")
                    article_date = dt_obj.strftime("%Y-%m-%d")
                except ValueError:
                    article_date = raw_date

    # --- 3. METADONNÉES FILM (Regex & Selectors) ---
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

    # --- 4. LISTES (Réalisateurs, Acteurs, Tags) ---
    def get_list_items(header_text):
        header = soup.find(lambda tag: tag.name == "h4" and header_text in tag.get_text())
        if header:
            sibling = header.find_next(["ul", "p"])
            if sibling.name == "ul":
                return [li.get_text(strip=True) for li in sibling.find_all("li")]
            return [sibling.get_text(strip=True)]
        return []

    directors = get_list_items("Director")
    
    actors = []
    cast_header = soup.find(lambda tag: tag.name == "h4" and "Cast" in tag.get_text())
    if cast_header and cast_header.find_next("ul"):
        actors = [li.find("a").get_text(strip=True) for li in cast_header.find_next("ul").find_all("li") if li.find("a")]

    tags = []
    tag_section = soup.find("section", class_="tags")
    if tag_section:
        tags = [a.get_text(strip=True) for a in tag_section.find_all("a")]
    
    # --- 5. CONTENU & SCORE ---
    body = soup.find("div", class_="entry-content") or soup.find("article")
    paragraphs = [p.get_text(strip=True) for p in body.find_all("p")] if body else []
    article_text_full = " ".join(paragraphs)
    
    links_internal = []
    if body:
        for a in body.find_all("a", href=True):
            if "facebook" not in a["href"] and "twitter" not in a["href"]:
                links_internal.append(a["href"])

    score_imgs = soup.find_all("img", src=lambda x: x and "star-full" in x)
    if score_imgs:
        review_score = str(len(score_imgs))
    else:
        score_tag = soup.find("img", class_="filled")
        review_score = ""
        if score_tag:
             for c in score_tag.get("class", []):
                if c.startswith("star"):
                    review_score = str(int(c.replace("star", "")) / 10)

    # --- 6. EXTRAS ---
    imgs = soup.find_all("img")
    all_image_urls = [img.get("src") for img in imgs if img.get("src")]
    image_alt = [img.get("alt") for img in imgs if img.get("alt")]
    blockquotes = [b.get_text(strip=True) for b in soup.find_all("blockquote")]
    subtitles = [h2.get_text(strip=True) for h2 in soup.find_all("h2")]

    # --- RETOUR DICTIONNAIRE ---
    return {
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
        "film_synopsis": "", # On force explicitement une chaine vide
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

if __name__ == "__main__":
    # 1. Charger le fichier existant
    if os.path.exists(NOM_FICHIER_EXISTANT):
        print(f"Chargement du fichier existant : {NOM_FICHIER_EXISTANT}")
        try:
            df_existant = pd.read_excel(NOM_FICHIER_EXISTANT)
            
            # Nettoyage : on supprime les lignes complètement vides
            original_len = len(df_existant)
            df_existant.dropna(how='all', inplace=True)
                        
        except Exception as e:
            print(f"Erreur lors de la lecture du fichier : {e}")
            df_existant = pd.DataFrame(columns=COLONNES_CIBLE)
    else:
        print(f"Le fichier {NOM_FICHIER_EXISTANT} n'existe pas, il sera créé.")
        df_existant = pd.DataFrame(columns=COLONNES_CIBLE)

    # 2. Scraper les nouvelles données
    print("\nRécupération des URLs...")
    urls = get_urls(100) 
    print(f"{len(urls)} URLs trouvées à scraper.")

    nouvelles_donnees = []
    
    # Création liste URLs existantes pour éviter doublons
    urls_existantes = []
    if 'article_url' in df_existant.columns:
        urls_existantes = df_existant['article_url'].astype(str).tolist()

    for i, url in enumerate(urls):
        if url in urls_existantes:
            print(f"Article déjà présent, on passe : {url}")
            continue

        print(f"Scraping ({i+1}/{len(urls)}): {url}")
        res = scrape(url)
        if res:
            nouvelles_donnees.append(res)
        time.sleep(1)

    # 3. Fusionner et Sauvegarder
    if nouvelles_donnees:
        df_nouveau = pd.DataFrame(nouvelles_donnees)
        
        # Remplissage colonnes manquantes
        for col in COLONNES_CIBLE:
            if col not in df_nouveau.columns:
                df_nouveau[col] = ""
        
        df_nouveau = df_nouveau[COLONNES_CIBLE]
        df_final = pd.concat([df_existant, df_nouveau], ignore_index=True)
    else:
        # Même si aucune nouvelle donnée, on sauvegarde pour appliquer le nettoyage du synopsis
        df_final = df_existant
        
    try:
        df_final.to_excel(NOM_FICHIER_EXISTANT, index=False)
        print(f"Succès ! {len(nouvelles_donnees)} nouvelles lignes ajoutées.")
        print(f"Total lignes dans {NOM_FICHIER_EXISTANT} : {len(df_final)}")
        print("Le fichier a été mis à jour.")
    except PermissionError:
        print(f"ERREUR : Veuillez fermer le fichier {NOM_FICHIER_EXISTANT} avant de lancer le script !")