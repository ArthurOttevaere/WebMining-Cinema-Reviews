# =============================================================================
# PIPELINE NLP COMPLET
# Tokenisation → TF-IDF MANUEL → LSA → Similarité → K-Means → Visualisations
# =============================================================================

import pandas as pd
import numpy as np
import os
import re
import nltk
import matplotlib.pyplot as plt

from collections import Counter
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag

from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.preprocessing import Normalizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# =============================================================================
# 0. CONFIGURATION
# =============================================================================
print("\n===== INITIALISATION =====")

dossier = r"C:\Users\33778\Desktop\WM (Amine)"
fichier_csv = os.path.join(dossier, "roger_ebert_debug.csv")

# =============================================================================
# 1. SETUP NLTK
# =============================================================================
print("\n[1/9] Setup NLTK")

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# =============================================================================
# 2. FONCTIONS NLP (SUPPRESSION DES NOMS PROPRES)
# =============================================================================
def get_wordnet_pos(tag):
    if tag.startswith("J"): return wordnet.ADJ
    if tag.startswith("V"): return wordnet.VERB
    if tag.startswith("N"): return wordnet.NOUN
    if tag.startswith("R"): return wordnet.ADV
    return wordnet.NOUN


def tokenize_text(text):
    """Tokenisation légère, sans regex agressive, sans NNP"""
    if not isinstance(text, str):
        return []

    text = re.sub(r"\s+", " ", text).strip()
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)

    clean_tokens = []
    for token, tag in tagged:
        t = token.lower()

        if (
            t in stop_words or
            len(t) <= 2 or
            not t.isalpha() or
            tag in ("NNP", "NNPS")
        ):
            continue

        lemma = lemmatizer.lemmatize(t, get_wordnet_pos(tag))
        clean_tokens.append(lemma)

    return clean_tokens

# =============================================================================
# 3. CHARGEMENT DES DONNÉES
# =============================================================================
print("\n[2/9] Chargement du CSV")

df = pd.read_csv(fichier_csv, sep=';', encoding='utf-8-sig', engine='python')

df["film_year"] = pd.to_numeric(df["film_year"], errors="coerce").fillna(0).astype(int)

df["unique_id"] = df.apply(
    lambda r: f"{r['film_title']} ({r['film_year']})"
    if r["film_year"] > 0 else f"{r['film_title']}_{r.name}",
    axis=1
)

# =============================================================================
# 4. TOKENISATION
# =============================================================================
print("\n[3/9] Tokenisation")

df["tokens"] = df["article_text_full"].apply(tokenize_text)
documents = dict(zip(df["unique_id"], df["tokens"]))

print(f"Documents uniques : {len(documents)}")

# =============================================================================
# 5. TF-IDF MANUEL (STRICT)
# =============================================================================
print("\n[4/9] Construction TF-IDF MANUEL")

# Vocabulaire
vocabulary = set()
for tokens in documents.values():
    vocabulary.update(tokens)

# Matrice terme-document brute
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

# ---------------- FILTRAGE DF ----------------
df_counts = (td_matrix > 0).sum(axis=0)
td_matrix = td_matrix.loc[:, df_counts >= 2]

seuil_max = len(documents) * 0.5
df_counts = (td_matrix > 0).sum(axis=0)
td_matrix = td_matrix.loc[:, df_counts < seuil_max]

print(f"Dimensions après filtrage : {td_matrix.shape}")

# ---------------- TF ----------------
row_sums = td_matrix.sum(axis=1).replace(0, 1)
tf = td_matrix.div(row_sums, axis=0)

# ---------------- IDF ----------------
N = td_matrix.shape[0]
df_count = (td_matrix > 0).sum(axis=0)
idf = np.log((N + 1) / (df_count + 1)) + 1

# ---------------- TF-IDF FINAL ----------------
tfidf = tf.mul(idf, axis=1).fillna(0)

# =============================================================================
# 6. LSA + NORMALISATION
# =============================================================================
print("\n[5/9] LSA")

svd = TruncatedSVD(n_components=200, random_state=42)
X_lsa = svd.fit_transform(tfidf)

print(f"Variance expliquée : {svd.explained_variance_ratio_.sum():.2%}")

X = Normalizer().fit_transform(X_lsa)

# =============================================================================
# 7. MATRICE DE SIMILARITÉ (AVEC SCORES)
# =============================================================================
print("\n[6/9] Similarité cosinus")

similarity = cosine_similarity(X)

nb_films = 20
subset = similarity[:nb_films, :nb_films]
labels = tfidf.index[:nb_films]

plt.figure(figsize=(10, 8))
plt.imshow(subset, cmap="viridis")
plt.colorbar(label="Similarité cosinus")
plt.title("Matrice de similarité (20 premiers films)")

plt.xticks(range(nb_films), labels, rotation=90, fontsize=8)
plt.yticks(range(nb_films), labels, fontsize=8)

for i in range(nb_films):
    for j in range(nb_films):
        val = subset[i, j]
        plt.text(j, i, f"{val:.2f}", ha="center", va="center",
                 color="white" if val < 0.5 else "black", fontsize=7)

plt.tight_layout()
plt.show()

# =============================================================================
# 8. K-MEANS
# =============================================================================
print("\n[7/9] K-Means (choix automatique de K)")

scores = []
for k in range(2, 15):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_k = km.fit_predict(X)
    score = silhouette_score(X, labels_k, metric="cosine")
    scores.append((k, score))
    print(f"K={k} → Silhouette={score:.4f}")

K_opt, best_score = max(scores, key=lambda x: x[1])
print(f"\nK optimal retenu : {K_opt} (silhouette={best_score:.4f})")

kmeans = KMeans(n_clusters=K_opt, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X)

# =============================================================================
# 9. VISUALISATION CLUSTERING
# =============================================================================
print("\n[8/9] Visualisation clustering")

coords = PCA(n_components=2).fit_transform(X)

plt.figure(figsize=(10, 7))
plt.scatter(coords[:, 0], coords[:, 1], c=clusters, cmap="tab10", alpha=0.7)
plt.title(f"Clustering des films (K={K_opt})")
plt.grid(alpha=0.3)
plt.show()

# =============================================================================
# 10. MOTS DOMINANTS
# =============================================================================
print("\n[9/9] Mots dominants par cluster")

centres_tfidf = svd.inverse_transform(kmeans.cluster_centers_)
mots = tfidf.columns

for i in range(K_opt):
    idx = centres_tfidf[i].argsort()[-10:][::-1]
    print(f"Cluster {i} ({(clusters==i).sum()} films) :",
          ", ".join(mots[j] for j in idx))

print("\n✅ PIPELINE FINAL TERMINÉ")
