"""
PIPELINE - PROJECT ROGER EBERT (Cinema Reviews) - Web Mining
"""

import pandas as pd
import os

# Functions importation
from src.scraping.scraping_bfs import launch_scraping_roger_ebert
from src.text_mining.text_mining_visuals import run_text_mining
from src.link_analysis.graph_builder import main as build_graph
from src.link_analysis.link_analysis import main as run_link_analysis

# --- FORBIDDEN TO CHANGE ANYTHING IN CONFIGURATION ---
RUN_SCRAPER = False      # IMPORTANT : False allows to have the same results as report, using the data scraped at the beginning
                        # IMPORTANT bis : Selecting True, will change all the results because it scrapes new reviews.
SHOW_PLOTS = True       # True to see all the graphs
LIMIT_SCRAPING = 900
DATA_PATH = "data/processed/reviews_final_900.csv"


def main():
    print("═" * 50)
    print("🎬 PROCESSING THE LAUNCH...")    
    print("═" * 50)

    if RUN_SCRAPER:
        print('🌐 STEP 1 : Scraping new reviews...')
        # We call the scraper 
        df_raw = launch_scraping_roger_ebert(limit=LIMIT_SCRAPING)
    else:
        print(f"\n📂 STEP 1 : Loading the original dataset ({DATA_PATH})...")
        if os.path.exists(DATA_PATH):
            df_raw = pd.read_csv(DATA_PATH)
        else:
            print(f"❌ Error : File {DATA_PATH} not found !")
            return
        
    # --- STEP 2 : TEXT MINING & CLUSTERING ---
    print("\n🧠 STEP 2 : Text mining analysis...")
    df_processed = run_text_mining(df_raw, show_plots=SHOW_PLOTS)
    
    # We save the enriched results
    os.makedirs("data/processed", exist_ok=True)
    df_processed.to_csv("data/processed/reviews_with_clusters.csv", index=False)
    print("✅ Improved data saved.")


    # --- STEP 3.1 : BUILDING THE GRAPH ---
    print("\n🕸️  STEP 3.1 : Building the network (Nodes & Edges)...")
    try:
        build_graph() # Appelle la fonction main() de graph_builder
        print("✅ Graph structure built (nodes_tfidf.csv & edges_tfidf.csv created).")
    except Exception as e:
        print(f"❌ Error during Graph Building: {e}")
        return

    # --- STEP 3.2 : Link Analysis ---
    print("📊 STEP 3 : Processing the metrics...")
    # Call the link analysis function 
    try:
        run_link_analysis()
        print("✅ Link Analysis completed successfully.")
    except Exception as e:
        print(f"❌ Error during Link Analysis: {e}")
        return

    print("\n" + "═" * 50)
    print("🏆 SUCCESSFULLY FINISHED")
    print("═" * 50)

if __name__ == "__main__":
    main()