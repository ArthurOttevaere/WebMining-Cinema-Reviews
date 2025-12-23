import pandas as pd
import os
import re
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
dossier = r"C:\Users\33778\Desktop\WM (Amine)"
fichier_entree = os.path.join(dossier, "roger_ebert_debug.csv")
fichier_sortie = os.path.join(dossier, "resultat_tokens_lemmatized.csv")

# ==============================================================================
# 2. NLTK SETUP
# ==============================================================================
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# ==============================================================================
# 3. OUTILS WORDNET
# ==============================================================================
def get_wordnet_pos(tag):
    if tag.startswith("J"):
        return wordnet.ADJ
    if tag.startswith("V"):
        return wordnet.VERB
    if tag.startswith("N"):
        return wordnet.NOUN
    if tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN

def canonical_synonym(word):
    synsets = wordnet.synsets(word)
    if not synsets:
        return word
    return synsets[0].lemmas()[0].name().lower()


# 4. TOKENISATION + LEMMATISATION + FUSION

def preprocess_and_tokenize(text):
    if not isinstance(text, str):
        return []

    # nettoyage
    text = re.sub(r"[^A-Za-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)

    result = []
    for word, tag in tagged:
        w = word.lower()

        if w in stop_words or len(w) <= 2:
            continue
        # suppression des noms propres
        if tag in ("NNP", "NNPS"):
            continue
        # lemmatisation
        lemma = lemmatizer.lemmatize(w, get_wordnet_pos(tag))

        # fusion synonymes
        canon = canonical_synonym(lemma)

        if canon not in stop_words and len(canon) > 2:
            result.append(canon)

    return result


# 5. CHARGEMENT CSV

print("Chargement du fichier...")
df = pd.read_csv(
    fichier_entree,
    sep=";",
    encoding="utf-8-sig",
    engine="python"
)

print(f"Dimensions initiales : {df.shape}")


# 6. APPLICATION TOKENISATION

print("Tokenisation + lemmatisation + fusion WordNet en cours...")
df["tokens"] = df["article_text_full"].apply(preprocess_and_tokenize)


# 7. SAUVEGARDE

print(f"Sauvegarde dans : {fichier_sortie}")
df.to_csv(fichier_sortie, sep=";", encoding="utf-8-sig", index=False)

print("✅ Terminé — tokens propres, fusionnés et interprétables.")


