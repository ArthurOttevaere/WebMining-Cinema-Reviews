import pandas as pd
import nltk
from nltk.corpus import stopwords
import os
import re

# --- 1. CONFIGURATION ---
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

stop_words = list(set(stopwords.words('english'))) + ["'s"]
stem = nltk.stem.SnowballStemmer("english")

# --- 2. FONCTION DE NETTOYAGE ---
def extract_tokens(text):
    if not isinstance(text, str):
        return []

    # Tout en minuscules
    text = text.lower()

    # On garde tout SAUF (^) les mots et espaces
    text = re.sub(r'[^\w\s]', '', text)

    # Découpage
    mots_bruts = nltk.word_tokenize(text)

    # Filtrage
    mots_propres = []
    for mot in mots_bruts:
        if mot not in stop_words:
            # E. Racinisation
            racine = stem.stem(mot)
            if racine.strip() != '':
                mots_propres.append(racine)

    return mots_propres

# CHARGEMENT
#dossier = r"C:\Users\33778\Desktop\WM (Amine)"
#fichier_entree = os.path.join(dossier, "roger_ebert_debug.csv")
#fichier_sortie = os.path.join(dossier, "resultat_avec_tokens.csv")

# We go up to the file root 
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

fichier_entree = os.path.join(BASE_DIR, "data", "processed", "reviews_final_900.csv")
fichier_sortie = os.path.join(BASE_DIR, "data", "raw", "results_with_tokens.csv")

print(f"📂 Lecture de : {fichier_entree}")
print(f"💾 Sauvegarde prévue dans : {fichier_sortie}")

print(f"Lecture du fichier : {fichier_entree}")

#df = pd.read_csv(fichier_entree, sep=';', encoding='utf-8-sig', engine='python')
df = pd.read_csv(fichier_entree, sep=',', encoding='utf-8-sig', engine='python', on_bad_lines='warn')

# --- 4. CORRECTION FORMAT ANNÉE (NOUVEAU !) ---
if 'film_year' in df.columns:
    print("Correction du format de l'année (suppression du .0)...")
    # On remplit les vides par 0 et on force le type "Entier" (int)
    df['film_year'] = pd.to_numeric(df['film_year'], errors='coerce').fillna(0).astype(int)

# --- 5. TOKENISATION ET SAUVEGARDE ---
# --- 5. TOKENISATION DIRECTE ---
print("Tokenisation en cours sur la colonne 'article_text_full'...")

# On attaque directement la colonne sans vérifier si elle existe
df['tokens'] = df['article_text_full'].apply(extract_tokens)

print(f"Sauvegarde dans : {fichier_sortie}")
df.to_csv(fichier_sortie, index=False, sep=';', encoding='utf-8-sig')
print("✅ Terminé !")