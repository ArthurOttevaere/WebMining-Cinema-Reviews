import pandas as pd
from sklearn.decomposition import TruncatedSVD
import os

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
dossier = r"C:\Users\33778\Desktop\WM (Amine)"
fichier_entree = os.path.join(dossier, "matrice_vectorielle_tf-idf.csv")
fichier_sortie = os.path.join(dossier, "matrice_lsa_reduite.csv")

# ==============================================================================
# 2. CHARGEMENT
# ==============================================================================
print(f"Chargement de la matrice TF-IDF : {fichier_entree}")

# index_col=0 est crucial pour garder les titres des films (ID) en index
try:
    df = pd.read_csv(fichier_entree, sep=';', encoding='utf-8-sig', index_col=0)
except:
    print("⚠️ Echec lecture UTF-8, tentative CP1252...")
    df = pd.read_csv(fichier_entree, sep=';', encoding='cp1252', index_col=0)

print(f"Dimensions originales : {df.shape}")
# Doit être (899, 16383) ou similaire

# ==============================================================================
# 3. RÉDUCTION (LSA / SVD)
# ==============================================================================
n_composantes = 50 # On garde 50 concepts
print(f"Compression en cours vers {n_composantes} dimensions...")

svd = TruncatedSVD(n_components=n_composantes, random_state=42)
matrice_reduite = svd.fit_transform(df)

# Information sur la qualité
variance = svd.explained_variance_ratio_.sum()
print(f"✅ Variance expliquée (Information conservée) : {variance:.2%}")

# ==============================================================================
# 4. RECONSTRUCTION DU DATAFRAME PROPRE
# ==============================================================================
# On remet ça dans un format lisible avec les titres de films
colonnes = [f"Concept_{i}" for i in range(n_composantes)]
df_lsa = pd.DataFrame(matrice_reduite, index=df.index, columns=colonnes)

print(f"Dimensions finales : {df_lsa.shape}")
# Doit être (899, 50)

# ==============================================================================
# 5. SAUVEGARDE
# ==============================================================================
print(f"Sauvegarde dans : {fichier_sortie}")
df_lsa.to_csv(fichier_sortie, sep=';', encoding='utf-8-sig')

print("✅ Terminé ! Tu peux maintenant utiliser ce fichier pour le Clustering et la Similarité.")