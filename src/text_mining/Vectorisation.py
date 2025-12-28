import pandas as pd
from collections import Counter
import ast
import os

# --- CONFIGURATION ---
#dossier = r"C:\Users\33778\Desktop\WM (Amine)"
#chemin_entree = os.path.join(dossier, "resultat_avec_tokens.csv")
#chemin_sortie = os.path.join(dossier, "matrice_vectorielle.csv")

# We go up to the file root 
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

chemin_entree = os.path.join(BASE_DIR, "data", "raw", "results_with_tokens.csv")
chemin_sortie = os.path.join(BASE_DIR, "data", "raw", "vector_matrix.csv")

print(f"📂 Lecture de : {chemin_entree}")
print(f"💾 Sauvegarde prévue dans : {chemin_sortie}")

print(f"Lecture du fichier : {chemin_entree}")

# 1. CHARGEMENT SIMPLE
print(f"Lecture du fichier : {chemin_entree}")
df = pd.read_csv(chemin_entree, sep=';', encoding='utf-8-sig')

# 2. CONVERSION DES LISTES
print("Conversion du texte en listes...")
df['tokens'] = df['tokens'].apply(ast.literal_eval)


# IDENTIFIANTS SIMPLIFIÉS

print("Création des identifiants (Titre + Année)...")

# On force l'année en texte et le titre en texte

df['unique_id'] = df['film_title'].astype(str) + " (" + df['film_year'].astype(str) + ")"

# Création du dictionnaire
documents = dict(zip(df['unique_id'], df['tokens']))
print(f"--> {len(documents)} documents prêts.")

#Vectorisation


# ÉTAPE 1 : VOCABULAIRE
print("Étape 1 : Vocabulaire...")
vocabulary = set()
for liste_mots in documents.values():
    vocabulary.update(liste_mots)
print(f"--> {len(vocabulary)} mots uniques.")

# ÉTAPE 2 : FRÉQUENCES
print("Étape 2 : Comptage...")
term_frequencies = {}

for doc, tokens in documents.items():
    term_frequencies[doc] = Counter(tokens)

# ÉTAPE 3 : MATRICE (TF)
print("Étape 3 : Matrice...")
td_matrix_dict = {}

for term in vocabulary:
    column_values = []
    for doc in documents:
        column_values.append(term_frequencies[doc].get(term, 0))
    td_matrix_dict[term] = column_values

td_matrix = pd.DataFrame(
    td_matrix_dict,
    index=documents.keys()
).fillna(0)

# ÉTAPE 4 : FILTRE MOTS RARES (Min DF >= 2)
print("Étape 4 : Filtre mots rares...")
compteur = (td_matrix > 0).sum(axis=0)
filtered_td_matrix = td_matrix.loc[:, compteur >= 2]

# ÉTAPE 5 : FILTRE MOTS TROP FRÉQUENTS (Max DF < 50%)
print("Étape 5 : Filtre mots fréquents...")
seuil_max = len(documents) * 0.5
compteur = (filtered_td_matrix > 0).sum(axis=0)
filtered_td_matrix = filtered_td_matrix.loc[:, compteur < seuil_max]

# SAUVEGARDE
print(f"Sauvegarde : {chemin_sortie}")
filtered_td_matrix.to_csv(chemin_sortie, sep=';', encoding='utf-8-sig')
print("✅ Terminé !")