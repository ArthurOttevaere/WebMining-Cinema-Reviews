import pandas as pd
import re
import nltk
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import ngrams, pos_tag

from itertools import combinations
from collections import Counter

from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from wordcloud import WordCloud

# -------------------------------------------------------------------
# 1) Load CSV
# -------------------------------------------------------------------
df = pd.read_csv("roger_ebert_debug.csv")
df["article_text_full"] = df["article_text_full"].astype(str)
total_docs = len(df)

# -------------------------------------------------------------------
# 2) NLTK setup
# -------------------------------------------------------------------
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# -------------------------------------------------------------------
# 3) Preprocessing + tokenization + lemmatization
# -------------------------------------------------------------------
def preprocess_and_tokenize(text):
    text = re.sub(r"[^A-Za-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = word_tokenize(text)
    tagged_tokens = pos_tag(tokens)

    processed_tokens = []
    for token, tag in tagged_tokens:
        token = lemmatizer.lemmatize(token.lower())

        if (
            token in stop_words or
            len(token) <= 2 or
            re.search(r"\d", token) or
            tag in ("NNP", "NNPS")
        ):
            continue

        processed_tokens.append(token)

    return processed_tokens

df["tokens"] = df["article_text_full"].apply(preprocess_and_tokenize)

# -------------------------------------------------------------------
# 4) Automatic filtering (document frequency)
# -------------------------------------------------------------------
min_doc_freq = 10
max_doc_frac = 0.5

doc_freq = Counter()
for tokens in df["tokens"]:
    for t in set(tokens):
        doc_freq[t] += 1

filtered_tokens = {
    w for w, f in doc_freq.items()
    if f >= min_doc_freq and f / total_docs <= max_doc_frac
}

df["tokens"] = df["tokens"].apply(
    lambda toks: [t for t in toks if t in filtered_tokens]
)

df["clean_text"] = df["tokens"].apply(lambda toks: " ".join(toks))

print(f"Number of documents: {total_docs}")
print(f"Total tokens after filtering: {sum(len(t) for t in df['tokens'])}")

# -------------------------------------------------------------------
# 5) TF-IDF vectorization
# -------------------------------------------------------------------

tfidf_vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=10,
    max_df=0.7
)

X_tfidf = tfidf_vectorizer.fit_transform(df["clean_text"])
print(f"TF-IDF dimensions (raw): {X_tfidf.shape}")

# -------------------------------------------------------------------
# 6) Dimensionality reduction (SVD) + scaling
# -------------------------------------------------------------------
svd = TruncatedSVD(n_components=150, random_state=42)
X_tfidf_svd = svd.fit_transform(X_tfidf)

scaler = StandardScaler()
X_tfidf_final = scaler.fit_transform(X_tfidf_svd)

print(f"TF-IDF dimensions after SVD: {X_tfidf_final.shape}")

# -------------------------------------------------------------------
# 7) Doc2Vec vectorization
# -------------------------------------------------------------------
tagged_docs = [
    TaggedDocument(words=toks, tags=[str(i)])
    for i, toks in enumerate(df["tokens"])
]

d2v_model = Doc2Vec(
    dm=0,                # PV-DBOW
    vector_size=200,
    window=8,
    min_count=5,
    epochs=60,
    workers=4,
    seed=42
)

d2v_model.build_vocab(tagged_docs)
d2v_model.train(
    tagged_docs,
    total_examples=d2v_model.corpus_count,
    epochs=d2v_model.epochs
)

X_d2v = np.array([d2v_model.dv[str(i)] for i in range(total_docs)])
X_d2v = StandardScaler().fit_transform(X_d2v)

print(f"Doc2Vec dimensions: {X_d2v.shape}")

# -------------------------------------------------------------------
# 8) Silhouette-based choice of K
# -------------------------------------------------------------------
def best_k_silhouette(X, k_min=2, k_max=12):
    scores = {}
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = km.fit_predict(X)
        score = silhouette_score(X, labels)
        scores[k] = score
    return scores

tfidf_scores = best_k_silhouette(X_tfidf_final)
d2v_scores = best_k_silhouette(X_d2v)

best_k_tfidf = max(tfidf_scores, key=tfidf_scores.get)
best_k_d2v = max(d2v_scores, key=d2v_scores.get)

print(f"\nBest K TF-IDF: {best_k_tfidf} | silhouette = {tfidf_scores[best_k_tfidf]:.4f}")
print(f"Best K Doc2Vec: {best_k_d2v} | silhouette = {d2v_scores[best_k_d2v]:.4f}")

# -------------------------------------------------------------------
# 9) Final clustering
# -------------------------------------------------------------------
kmeans_tfidf = KMeans(n_clusters=best_k_tfidf, random_state=42, n_init=20)
labels_tfidf = kmeans_tfidf.fit_predict(X_tfidf_final)

kmeans_d2v = KMeans(n_clusters=best_k_d2v, random_state=42, n_init=20)
labels_d2v = kmeans_d2v.fit_predict(X_d2v)

df["cluster_tfidf"] = labels_tfidf
df["cluster_d2v"] = labels_d2v

# -------------------------------------------------------------------
# 10) FINAL SILHOUETTE SCORES
# -------------------------------------------------------------------
sil_tfidf_final = silhouette_score(X_tfidf_final, labels_tfidf)
sil_d2v_final = silhouette_score(X_d2v, labels_d2v)

print("\n===== FINAL SILHOUETTE SCORES =====")
print(f"TF-IDF Silhouette score : {sil_tfidf_final:.4f}")
print(f"Doc2Vec Silhouette score: {sil_d2v_final:.4f}")

# -------------------------------------------------------------------
# 11) PCA visualization
# -------------------------------------------------------------------
pca = PCA(n_components=2, random_state=42)

X_tfidf_2d = pca.fit_transform(X_tfidf_final)
X_d2v_2d = pca.fit_transform(X_d2v)

plt.figure(figsize=(8,6))
sns.scatterplot(x=X_tfidf_2d[:,0], y=X_tfidf_2d[:,1],
                hue=labels_tfidf, palette="tab10")
plt.title("TF-IDF + SVD Clustering (PCA)")
plt.show()

plt.figure(figsize=(8,6))
sns.scatterplot(x=X_d2v_2d[:,0], y=X_d2v_2d[:,1],
                hue=labels_d2v, palette="tab10")
plt.title("Doc2Vec Clustering (PCA)")
plt.show()

# -------------------------------------------------------------------
# 12) Descriptive analysis (word frequency + word cloud)
# -------------------------------------------------------------------
token_counter = Counter([t for sublist in df["tokens"] for t in sublist])

top20 = token_counter.most_common(20)
words, freqs = zip(*top20)

plt.figure(figsize=(12,6))
sns.barplot(x=list(freqs), y=list(words))
plt.title("Top 20 Most Frequent Words")
plt.tight_layout()
plt.show()

wc = WordCloud(width=800, height=400, background_color="white") \
    .generate_from_frequencies(token_counter)

plt.figure(figsize=(12,6))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud")
plt.show()

# -------------------------------------------------------------------
# 13) N-grams analysis (bigrams and trigrams)
# -------------------------------------------------------------------
all_tokens_flat = [t for sublist in df["tokens"] for t in sublist]
bigram_counter = Counter(ngrams(all_tokens_flat, 2))
trigram_counter = Counter(ngrams(all_tokens_flat, 3))

top10_bigrams = bigram_counter.most_common(10)
bigram_labels = [" ".join(bg) for bg, _ in top10_bigrams]
bigram_values = [freq for _, freq in top10_bigrams]

plt.figure(figsize=(10,6))
sns.barplot(x=bigram_values, y=bigram_labels, dodge=False)
plt.title("Top 10 Bigrams")
plt.xlabel("Frequency")
plt.ylabel("Bigram")
plt.show()

top10_trigrams = trigram_counter.most_common(10)
trigram_labels = [" ".join(tg) for tg, _ in top10_trigrams]
trigram_values = [freq for _, freq in top10_trigrams]

plt.figure(figsize=(10,6))
sns.barplot(x=trigram_values, y=trigram_labels, dodge=False)
plt.title("Top 10 Trigrams")
plt.xlabel("Frequency")
plt.ylabel("Trigram")
plt.show()

# -------------------------------------------------------------------
# 14) Co-occurrence matrix analysis
# -------------------------------------------------------------------
top_n_words = [w for w, _ in token_counter.most_common(30)]
cooc_matrix = pd.DataFrame(0, index=top_n_words, columns=top_n_words)

for tokens in df["tokens"]:
    tokens_set = set(tokens)
    for w1, w2 in combinations(top_n_words, 2):
        if w1 in tokens_set and w2 in tokens_set:
            cooc_matrix.loc[w1, w2] += 1
            cooc_matrix.loc[w2, w1] += 1

plt.figure(figsize=(12,10))
sns.heatmap(cooc_matrix, cmap="YlGnBu", annot=True, fmt="d")
plt.xticks(rotation=45, ha='right', fontstyle='italic')
plt.yticks(rotation=0, va='center', fontstyle='italic')
plt.title("Co-occurrence Matrix of Top 30 Words")
plt.tight_layout()
plt.show()

# -------------------------------------------------------------------
# 15) Descriptive analysis by genre
# -------------------------------------------------------------------
if "film_genre" in df.columns:
    print("Analyzing by genre...")
    for genre, subdf in df.groupby("film_genre"):
        words = [w for lst in subdf["tokens"] for w in lst]
        counter = Counter(words).most_common(20)

        top_df = pd.DataFrame(counter, columns=["word", "freq"])
        plt.figure(figsize=(10,6))
        sns.barplot(data=top_df, x="freq", y="word", palette="viridis")
        plt.title(f"Top 20 Most Frequent Words — Genre: {genre}")
        plt.xlabel("Frequency")
        plt.ylabel("Word")
        plt.tight_layout()
        plt.show()
else:
    print("⚠ No 'film_genre' field detected in dataset. Skipping genre analysis.")
