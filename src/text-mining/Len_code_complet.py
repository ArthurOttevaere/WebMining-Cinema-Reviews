# ==============================================================================
# PIPELINE NLP COMPLET — SORTIES TERMINAL (AUCUN FICHIER)
# ==============================================================================

import os, re, nltk
import numpy as np
import pandas as pd

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

import matplotlib.pyplot as plt

# ==============================================================================
# 1. CHARGEMENT
# ==============================================================================
print("\n================= ÉTAPE 1 : CHARGEMENT =================")

dossier = r"C:\Users\33778\Desktop\WM (Amine)"
fichier_source = os.path.join(dossier, "roger_ebert_debug.csv")

df = pd.read_csv(fichier_source, sep=";", encoding="utf-8-sig", engine="python")
print(f"Documents chargés : {df.shape[0]}")
print(f"Colonnes : {list(df.columns)}")

# ==============================================================================
# 2. TOKENISATION
# ==============================================================================
print("\n================= ÉTAPE 2 : TOKENISATION =================")

for pkg in ["punkt", "stopwords", "wordnet", "omw-1.4", "averaged_perceptron_tagger"]:
    nltk.download(pkg, quiet=True)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def get_pos(tag):
    return wordnet.ADJ if tag.startswith("J") else \
           wordnet.VERB if tag.startswith("V") else \
           wordnet.NOUN if tag.startswith("N") else \
           wordnet.ADV if tag.startswith("R") else wordnet.NOUN

def canonical(word):
    syn = wordnet.synsets(word)
    return syn[0].lemmas()[0].name().lower() if syn else word

def tokenize(text):
    if not isinstance(text, str):
        return []
    text = re.sub(r"[^A-Za-z0-9\s]", " ", text)
    tokens = pos_tag(word_tokenize(text))
    out = []
    for w, t in tokens:
        w = w.lower()
        if w in stop_words or len(w) <= 2 or t in ("NNP","NNPS"):
            continue
        lemma = lemmatizer.lemmatize(w, get_pos(t))
        canon = canonical(lemma)
        if canon not in stop_words and len(canon) > 2:
            out.append(canon)
    return out

df["tokens"] = df["article_text_full"].apply(tokenize)

nb_tokens_total = df["tokens"].apply(len).sum()
nb_tokens_moy = df["tokens"].apply(len).mean()

print(f"Tokens totaux      : {nb_tokens_total}")
print(f"Tokens / document  : {nb_tokens_moy:.2f}")

# ==============================================================================
# 3. TF-IDF
# ==============================================================================
print("\n================= ÉTAPE 3 : TF-IDF =================")

df["unique_id"] = df.apply(
    lambda r: f"{r['film_title']} ({str(r['film_year']).replace('.0','')})",
    axis=1
)

docs = dict(zip(df["unique_id"], df["tokens"]))
vocab = set(t for toks in docs.values() for t in toks)

print(f"Taille vocabulaire brut : {len(vocab)}")

tf = {d: Counter(toks) for d, toks in docs.items()}

td = pd.DataFrame(
    {w: [tf[d].get(w,0) for d in docs] for w in vocab},
    index=docs.keys()
)

dfc = (td > 0).sum(axis=0)
td = td.loc[:, (dfc >= 2) & (dfc < 0.5 * len(docs))]

print(f"Dimensions après filtrage : {td.shape}")

tf_norm = td.div(td.sum(axis=1).replace(0,1), axis=0)
idf = np.log((len(td)+1)/((td>0).sum(axis=0)+1)) + 1
tfidf = tf_norm.mul(idf, axis=1)

print(f"Matrice TF-IDF finale : {tfidf.shape}")

# ==============================================================================
# 4. LSA (SVD)
# ==============================================================================
print("\n================= ÉTAPE 4 : LSA / SVD =================")

svd = TruncatedSVD(n_components=50, random_state=42)
X_lsa = svd.fit_transform(tfidf)

print(f"Dimensions LSA : {X_lsa.shape}")
print(f"Variance expliquée : {svd.explained_variance_ratio_.sum():.2%}")

# ==============================================================================
# 5. SIMILARITÉ COSINUS
# ==============================================================================
print("\n================= ÉTAPE 5 : SIMILARITÉ =================")

similarite = cosine_similarity(X_lsa)
print(f"Matrice similarité : {similarite.shape}")

# Test jumeaux
premier = list(docs.keys())[0]
sim_scores = pd.Series(similarite[0], index=docs.keys()).sort_values(ascending=False)

print(f"\nFilms les plus proches de : {premier}")
print(sim_scores.head(5))

# ==============================================================================
# 6. CLUSTERING
# ==============================================================================
print("\n================= ÉTAPE 6 : CLUSTERING =================")

X = Normalizer().fit_transform(X_lsa)

scores = []
for k in range(4,6):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    sil = silhouette_score(X, labels, metric="cosine")
    scores.append((k, km.inertia_, sil))
    print(f"K={k} | Inertie={km.inertia_:.2f} | Silhouette={sil:.4f}")

K_opt = max(scores, key=lambda x: x[2])[0]
print(f"\nK optimal retenu : {K_opt}")

kmeans = KMeans(n_clusters=K_opt, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X)

# ==============================================================================
# 7. MOTS DOMINANTS PAR CLUSTER
# ==============================================================================
print("\n================= MOTS DOMINANTS =================")

centres_lsa = kmeans.cluster_centers_
centres_tfidf = svd.inverse_transform(centres_lsa)
mots = tfidf.columns

for i in range(K_opt):
    top_idx = centres_tfidf[i].argsort()[-10:][::-1]
    top_mots = [mots[j] for j in top_idx]
    taille = (clusters == i).sum()
    print(f"Cluster {i} ({taille} films) : {', '.join(top_mots)}")

# ==============================================================================
# 8. VISUALISATION PCA
# ==============================================================================
coords = PCA(n_components=2).fit_transform(X)

plt.figure(figsize=(10,7))
plt.scatter(coords[:,0], coords[:,1], c=clusters, cmap="tab10", alpha=0.7)
plt.title(f"Clustering LSA (K={K_opt})")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.grid(alpha=0.3)
plt.show()

print("\n✅ PIPELINE TERMINÉ — SORTIES IDENTIQUES, AUCUN FICHIER CRÉÉ.")
