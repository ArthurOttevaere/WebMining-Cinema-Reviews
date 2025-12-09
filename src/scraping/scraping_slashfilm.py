import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from urllib.parse import urljoin

def launch_scraping_lenny(limite_liens=10):
    """
    Scraping of  SlashFilm 
    """
    
    # --- Configuration of the output files ---
    OUTPUT_FILE_CSV = "data/raw/Lenny.csv"
    OUTPUT_FILE_XLSX = "data/raw/Lenny.xlsx"
    
    # Configuration for Slashfilm scraping
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

    # --- STEP 1 : Collect the links (articles) using AJAX ---
    print(f"--- Launching the scraping of Slashfilm ---")
    print("Collecting links...")
    
    article_links = set()
    
    # We make a loop to collect enough links 
    # We stop after having enough links (even a bit more)
    max_offset = 500 if limite_liens < 500 else limite_liens + 100
    
    for offset in range(0, max_offset, 27):
        try:
            response = requests.get(slashfilm["ajax_url"].format(offset))
            if response.status_code != 200:
                break

            soup = BeautifulSoup(response.text, "html.parser")
            articles = soup.select(slashfilm["article_selector"])
            
            if not articles: # If there is no more articles, we stop
                break
                
            for a in articles:
                href = a.get("href")
                if href:
                    article_links.add(urljoin(slashfilm["base_url"], href))
            
            # We stop if we have enough links
            if len(article_links) >= limite_liens:
                break
                
        except Exception as e:
            print(f"Error while collecting the links: {e}")
            break

    # --- STEP 2 : Scraping of article's details ---
    # We transforme the set in a list
    links_to_scrape = list(article_links)[:limite_liens]
    
    print(f"Number of links collected : {len(article_links)}")
    print(f"\nTreating {len(links_to_scrape)} reviews...")
    
    results = []

    for i, url in enumerate(links_to_scrape):
        # Progression display [1/10]
        print(f"[{i+1}/{len(links_to_scrape)}]", end=" ")
        
        try:
            art_resp = requests.get(url)
            if art_resp.status_code != 200:
                print(f" (❌ Error HTTP: {art_resp.status_code})")
                continue

            art_soup = BeautifulSoup(art_resp.text, "html.parser")

            # --- Scraping the details ---
            
            # Main title
            title_elem = art_soup.select_one(slashfilm["title_selector"])
            title = title_elem.get_text(strip=True) if title_elem else ""

            # Film title (before "Review")
            film_title = title.split("Review")[0].strip() if "Review" in title else ""

            # Date
            date = ""
            time_elem = art_soup.select_one(slashfilm["date_selector"])
            if time_elem:
                raw_date = time_elem.get("datetime", "").strip()
                date = raw_date.split("T")[0] if raw_date else ""

            # Author
            author_elem = art_soup.select_one(slashfilm["author_selector"])
            author = author_elem.get_text(strip=True) if author_elem else ""

            # Main text
            content_block = art_soup.select_one(slashfilm["content_selector"])
            text = " ".join(
                elem.get_text(" ", strip=True)
                for elem in content_block.find_all(["h2", "p"])
            ) if content_block else ""

            # Rating
            rating_elem = art_soup.find("strong", string=lambda s: s and "/Film Rating:" in s)
            rating = rating_elem.get_text(strip=True).replace("/Film Rating:", "").strip() if rating_elem else ""

            # Film info
            em_tags = art_soup.find_all("em")
            film_info = em_tags[-1].get_text(" ", strip=True) if em_tags else ""

            # Images
            all_images = art_soup.find_all("img", class_="gallery-image")
            image_url = all_images[0].get("src") if all_images else ""

            # Word count
            word_count = len((title + " " + text).split())

            # External links
            external_links_str = "\n".join(
                link["href"] for link in content_block.find_all("a", href=True)
                if link["href"].startswith("http")
            ) if content_block else ""

            # --- End of the details scraping ---

            # We create a dictionnary
            review = {
                "site_name": slashfilm["name"],
                "article_title": title,
                "article_url": url,
                "article_author": author,
                "article_date": date,
                "film_title": film_title,
                "film_year": "",
                "film_director": "",
                "film_main_actors": "",
                "film_genre": "",
                "film_duration": "",
                "article_text_full": text,
                "article_blockquotes": "",
                "main_image_url": image_url,
                "image_caption": "",
                "word_count": word_count,
                "review_score": rating,
                "links_to_other_reviews": external_links_str,
                #"film_info": film_info,
                
            }

            # We check for the required fields
            required_values = [title, film_title, date, author, rating, text, film_info, image_url]
            
            if all(required_values):
                results.append(review)
                # We display a part of the title
                safe_title = (title[:30] + '..') if len(title) > 30 else title
                print(f"✅ {safe_title}")
            else:
                print(f" (⚠️ Incomplete/Ignored Review)")

        except Exception as e:
            print(f" (❌ Code Error: {e})")

        # Time sleep for the server
        time.sleep(0.1)

    # --- STEP 3 : Saving with pandas (CSV and Excel) ---
    if results:
        df = pd.DataFrame(results)
        
        import os
        os.makedirs(os.path.dirname(OUTPUT_FILE_CSV), exist_ok=True)

        df.to_csv(OUTPUT_FILE_CSV, index=False, encoding="utf-8")
        
        # Excel save
        try:
            df.to_excel(OUTPUT_FILE_XLSX, index=False)
            print(f"\n✅ Finished ! {len(df)} reviews backed up in {OUTPUT_FILE_CSV} and {OUTPUT_FILE_XLSX}")
        except Exception as e:
            print(f"\n✅ Finished CSV! (Excel error: {e}) Saved in {OUTPUT_FILE_CSV}")
    else:
        print("\n❌ No data has been collected.")

# --- RUNNING ---
if __name__ == "__main__":
    launch_scraping_lenny(limite_liens=30)