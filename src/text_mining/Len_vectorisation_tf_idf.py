import pandas as pd
import numpy as np
from collections import Counter
import ast
import os

# ==============================================================================
# CONFIGURATION
# ==============================================================================
dossier = r"C:\Users\33778\Desktop\WM (Amine)"
chemin_entree = os.path.join(dossier, "resultat_tokens_lemmatized.csv")
chemin_sortie_tfidf = os.path.join(dossier, "matrice_vectorielle_tf-idf.csv")

# ==============================================================================
# 1. CHARGEMENT ROBUSTE
# ==============================================================================
print(f"Lecture du fichier : {chemin_entree}")
if not os.path.exists(chemin_entree):
    print("❌ ERREUR : Fichier introuvable.")
    exit()

try:
    df = pd.read_csv(
        chemin_entree,
        sep=';',
        encoding='utf-8-sig',
        engine='python',
        on_bad_lines='warn'
    )
except:
    print("⚠️ Lecture UTF-8 échouée, tentative Windows-1252...")
    df = pd.read_csv(
        chemin_entree,
        sep=';',
        encoding='cp1252',
        engine='python',
        on_bad_lines='warn'
    )

# ==============================================================================
# 2. CONVERSION TEXTE → LISTE
# ==============================================================================
print("Conversion des tokens...")
df['tokens'] = df['tokens'].fillna("[]").apply(ast.literal_eval)

# ==============================================================================
# 3. CRÉATION DES IDENTIFIANTS UNIQUES
# ==============================================================================
print("Création des identifiants uniques...")

df['film_year'] = df['film_year'].astype(str)
df['film_title'] = df['film_title'].astype(str)

def creer_id(row):
    titre = row['film_title'].strip()
    annee = row['film_year'].replace('.0', '').strip()
    if annee and annee != '0' and annee.lower() != 'nan':
        return f"{titre} ({annee})"
    else:
        return f"{titre}_{row.name}"

df['unique_id'] = df.apply(creer_id, axis=1)

documents = dict(zip(df['unique_id'], df['tokens']))
print(f"--> {len(documents)} documents uniques retenus.")

# ==============================================================================
# 4. CONSTRUCTION DU VOCABULAIRE
# ==============================================================================
print("Construction du vocabulaire...")
vocabulary = set()
for tokens in documents.values():
    vocabulary.update(tokens)

print(f"--> {len(vocabulary)} mots uniques.")

# ==============================================================================
# 5. MATRICE TERM-DOCUMENT
# ==============================================================================
print("Construction de la matrice brute...")
term_frequencies = {
    doc: Counter(tokens)
    for doc, tokens in documents.items()
}

td_matrix = pd.DataFrame(
    {
        term: [term_frequencies[doc].get(term, 0) for doc in documents]
        for term in vocabulary
    },
    index=documents.keys()
).fillna(0)

# ==============================================================================
# 6. FILTRAGE (AVANT TF-IDF)
# ==============================================================================
print("Filtrage des mots rares et trop fréquents...")

# Min DF >= 2
df_counts = (td_matrix > 0).sum(axis=0)
td_matrix = td_matrix.loc[:, df_counts >= 2]

# Max DF < 50 %
seuil_max = len(documents) * 0.5
df_counts = (td_matrix > 0).sum(axis=0)
td_matrix = td_matrix.loc[:, df_counts < seuil_max]

print(f"--> Dimension après filtrage : {td_matrix.shape}")

# ==============================================================================
# 7. CALCUL TF-IDF (ROBUSTE)
# ==============================================================================
print("Calcul du TF-IDF...")

# TF
row_sums = td_matrix.sum(axis=1)
row_sums = row_sums.replace(0, 1)   # sécurité division par zéro
tf = td_matrix.div(row_sums, axis=0)

# IDF (version stable standard NLP)
N = td_matrix.shape[0]
df_count = (td_matrix > 0).sum(axis=0)
idf = np.log((N + 1) / (df_count + 1)) + 1

# TF-IDF final
tf_idf = tf.mul(idf, axis=1).fillna(0)

# ==============================================================================
# 8. SAUVEGARDE
# ==============================================================================
print(f"Sauvegarde : {chemin_sortie_tfidf}")
tf_idf.to_csv(chemin_sortie_tfidf, sep=';', encoding='utf-8-sig')

print("✅ Terminé — TF-IDF propre, stable et prêt pour le clustering.")

# ==============================================================================
# 9. STATISTIQUES CLAIRES : TOKENS vs DIMENSIONS
# ==============================================================================

# Nombre total de tokens (après tokenisation + fusion)
nb_tokens_total = df["tokens"].apply(len).sum()

# Nombre moyen de tokens par document
nb_tokens_moyen = df["tokens"].apply(len).mean()

# Taille du vocabulaire (avant filtrage)
taille_vocabulaire = len(vocabulary)

# Dimensions finales TF-IDF (après filtrage)
nb_documents, nb_dimensions = tf_idf.shape

print("\n--- RÉCAPITULATIF ---")
print(f"Nombre de documents          : {nb_documents}")
print(f"Nombre total de tokens       : {nb_tokens_total}")
print(f"Tokens moyens / document     : {nb_tokens_moyen:.2f}")
print(f"Taille du vocabulaire brut   : {taille_vocabulaire}")
print(f"Dimensions TF-IDF finales    : {nb_dimensions}")
