import pandas as pd
import os
import time

# --- IMPORT ---
from scraping_lwlies import launch_scraping_arthur

# Configuration
OUTPUT_RAW_PATH = 'data/raw/corpus_global_raw.csv'

def main():
    print("🎬 DÉMARRAGE DE LA COLLECTE GLOBALE 🎬")
    
    # List for the dfs
    dfs_to_merge = []

    # --- 1. COLLECTion : LITTLE WHITE LIES (Arthur) ---
    try:
        print("\n⚡ [1/3] Lancement du scraping Arthur...")
        df_arthur = launch_scraping_arthur(limit=10) 
        
        if not df_arthur.empty:
            df_arthur['source_site'] = 'Little White Lies'
            dfs_to_merge.append(df_arthur)
            
    except Exception as e:
        print(f"❌ Erreur critique Arthur : {e}")

    # --- 2. COLLECTION : AMINE ---
   

    # --- 3. COLLECTION : LENNY ---


    # --- 4. MERGE AND SAVE ---
    print("\n------------------------------------------------")
    
    if dfs_to_merge:
        print(f"🔄 Fusion de {len(dfs_to_merge)} sources...")
        
        # Colomns must be identical
        final_df = pd.concat(dfs_to_merge, ignore_index=True)
        
        # We remove the duplicates
        initial_len = len(final_df)
        final_df.drop_duplicates(subset=['article_url'], inplace=True)
        dedup_len = len(final_df)
        
        if initial_len != dedup_len:
            print(f"🧹 {initial_len - dedup_len} doublons supprimés.")

        # Save
        final_df.to_csv(OUTPUT_RAW_PATH, index=False)
        final_df.to_excel(OUTPUT_RAW_PATH.replace('.csv', '.xlsx'), index=False)
        
        print(f"\n✅ TERMINÉ ! Le corpus global est prêt.")
        print(f"📊 Total critiques : {len(final_df)}")
        print(f"📁 Fichier : {OUTPUT_RAW_PATH}")
        
        # Repartition of the scraping
        print("\nRépartition par source :")
        print(final_df['source_site'].value_counts())
        
    else:
        print("❌ Aucune donnée récupérée. Vérifiez les scrapers individuels.")

if __name__ == "__main__":
    main()