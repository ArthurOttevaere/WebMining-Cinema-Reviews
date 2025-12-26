"""
Launching script for the project of Web Mining
This script alloww the launching of the diffrent steps of the project (scraping, text mining and link analysis).
It works by calling all of the required functions.
"""

# --- PARAMETERS ---
RUN_SCRAPER = False
LIMIT_FILMS = 900


def main():
    print("🚀 Processing the launching...")

    if RUN_SCRAPER:
        print('🌐 STEP 1 : Scraping new reviews...')
        # We call the scraper 
        #from src.scraping.bfs_arthur.py import launch_scraping_roger_ebert() 
        #df = launch_scraping_roger_ebert(limit=LIMIT_FILMS)
    else:
        print("📂 STEP 1 : Loading the original dataset (used for the analysis if the project)...")
        import pandas as pd
        try:
            # We load the original csv file (900 reviews) 
            df = pd.read_csv("data/processed/reviews_final_900.csv")
        except FileNotFoundError:
            print("❌ Error : Original file can not be found.")
            return
        
    # --- STEP 2 : Text Mining ---
    print("🧠 STEP 2 : Text Mining...")
    # Call the text mining function 

    # --- STEP 3 : Link Analysis ---
    print("📊 STEP 3 : Processing the metrics...")
    # Call the link analysis function 

    print("✅ Finished.")


if __name__ == "__main__":
    main()