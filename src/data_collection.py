import pandas as pd
import os
import time

# --- IMPORTS DES SCRAPERS ---
# Assurez-vous que les fichiers de vos camarades sont bien dans le dossier src/
# et qu'ils ont bien une fonction principale qui retourne un DataFrame
from scraping_lwlies import launch_scraping_arthur
# from scraping_amine import lancer_scraping_amine  # Décommenter quand prêt
# from scraping_lenny import lancer_scraping_lenny  # Décommenter quand prêt

# Configuration
OUTPUT_RAW_PATH = 'data/raw/corpus_global_raw.csv'

def main():
    print("🎬 DÉMARRAGE DE LA COLLECTE GLOBALE 🎬")
    
    # Liste pour stocker les DataFrames
    dfs_to_merge = []

    # --- 1. COLLECTE : LITTLE WHITE LIES (Arthur) ---
    try:
        print("\n⚡ [1/3] Lancement du scraping Arthur...")
        # On peut limiter pour le test, ou mettre 300 pour la prod
        df_arthur = launch_scraping_arthur(limit=10) 
        
        if not df_arthur.empty:
            # IMPORTANT : On marque l'origine des données avant la fusion
            df_arthur['source_site'] = 'Little White Lies'
            dfs_to_merge.append(df_arthur)
            
    except Exception as e:
        print(f"❌ Erreur critique Arthur : {e}")

    # --- 2. COLLECTE : SITE AMINE ---
    # try:
    #     print("\n⚡ [2/3] Lancement du scraping Amine...")
    #     df_amine = lancer_scraping_amine(limit=300)
    #     if not df_amine.empty:
    #         df_amine['source_site'] = 'Nom Site Amine' # À adapter
    #         dfs_to_merge.append(df_amine)
    # except Exception as e:
    #     print(f"❌ Erreur critique Amine : {e}")

    # --- 3. COLLECTE : SITE LENNY ---
    # (Même logique...)

    # --- 4. FUSION ET SAUVEGARDE ---
    print("\n------------------------------------------------")
    
    if dfs_to_merge:
        print(f"🔄 Fusion de {len(dfs_to_merge)} sources...")
        
        # C'est ici que la magie opère grâce à vos colonnes identiques
        final_df = pd.concat(dfs_to_merge, ignore_index=True)
        
        # Petit nettoyage de sécurité (doublons exacts)
        initial_len = len(final_df)
        final_df.drop_duplicates(subset=['article_url'], inplace=True)
        dedup_len = len(final_df)
        
        if initial_len != dedup_len:
            print(f"🧹 {initial_len - dedup_len} doublons supprimés.")

        # Sauvegarde
        final_df.to_csv(OUTPUT_RAW_PATH, index=False)
        final_df.to_excel(OUTPUT_RAW_PATH.replace('.csv', '.xlsx'), index=False)
        
        print(f"\n✅ TERMINÉ ! Le corpus global est prêt.")
        print(f"📊 Total critiques : {len(final_df)}")
        print(f"📁 Fichier : {OUTPUT_RAW_PATH}")
        
        # Aperçu de la répartition
        print("\nRépartition par source :")
        print(final_df['source_site'].value_counts())
        
    else:
        print("❌ Aucune donnée récupérée. Vérifiez les scrapers individuels.")

if __name__ == "__main__":
    main()