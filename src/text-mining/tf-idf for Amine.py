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
min_doc_freq = 2
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
    min_df=5,
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
# 7) Silhouette-based choice of K
# -------------------------------------------------------------------
def best_k_silhouette(X, k_min=2, k_max=12):
    scores = {}
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = km.fit_predict(X)
        scores[k] = silhouette_score(X, labels)
    return scores

tfidf_scores = best_k_silhouette(X_tfidf_final)
best_k = max(tfidf_scores, key=tfidf_scores.get)

print(f"\nBest K TF-IDF: {best_k} | silhouette = {tfidf_scores[best_k]:.4f}")

# -------------------------------------------------------------------
# 8) Final clustering
# -------------------------------------------------------------------
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=20)
df["cluster"] = kmeans.fit_predict(X_tfidf_final)

sil_final = silhouette_score(X_tfidf_final, df["cluster"])
print(f"\nFinal silhouette score: {sil_final:.4f}")

# -------------------------------------------------------------------
# 9) PCA visualization
# -------------------------------------------------------------------
pca = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(X_tfidf_final)

plt.figure(figsize=(8,6))
sns.scatterplot(x=X_2d[:,0], y=X_2d[:,1], hue=df["cluster"], palette="tab10")
plt.title("TF-IDF + SVD Clustering (PCA)")
plt.tight_layout()
plt.show()

# -------------------------------------------------------------------
# 10) Descriptive analysis (word frequency + word cloud)
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
# 11) Distribution of movie scores
# -------------------------------------------------------------------
if "review_score" in df.columns:
    score_counts = df["review_score"].value_counts().sort_index()
    percentages = score_counts / len(df) * 100

    x_pos = np.arange(len(score_counts)) * 1.2

    plt.figure(figsize=(12,6))
    bars = plt.bar(x_pos, percentages, width=1.0, edgecolor="black")

    for bar, pct in zip(bars, percentages):
        plt.text(bar.get_x() + bar.get_width()/2, pct + 0.5,
                 f"{pct:.1f}%", ha="center", va="bottom")

    plt.xticks(x_pos, score_counts.index)
    plt.xlabel("Review score")
    plt.ylabel("Percentage of reviews (%)")
    plt.title("Distribution of movie review scores")
    plt.tight_layout()
    plt.show()

# -------------------------------------------------------------------
# Boxplot: Review score distribution by genre
# -------------------------------------------------------------------
if "film_genre" in df.columns and "review_score" in df.columns:

    plt.figure(figsize=(14,6))
    sns.boxplot(
        data=df,
        x="film_genre",
        y="review_score",
        showfliers=True
    )

    plt.title("Distribution of Review Scores by Film Genre")
    plt.xlabel("Film genre")
    plt.ylabel("Review score")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

else:
    print("⚠ 'film_genre' or 'review_score' not found. Skipping genre boxplot.")


# -------------------------------------------------------------------
# 12) N-grams analysis
# -------------------------------------------------------------------
all_tokens = [t for sublist in df["tokens"] for t in sublist]

for n in [2, 3]:
    counter = Counter(ngrams(all_tokens, n)).most_common(10)
    labels = [" ".join(ng) for ng, _ in counter]
    values = [freq for _, freq in counter]

    plt.figure(figsize=(10,6))
    sns.barplot(x=values, y=labels)
    plt.title(f"Top 10 {n}-grams")
    plt.tight_layout()
    plt.show()

# -------------------------------------------------------------------
# 13) Co-occurrence matrix
# -------------------------------------------------------------------
top_words = [w for w, _ in token_counter.most_common(30)]
cooc = pd.DataFrame(0, index=top_words, columns=top_words)

for tokens in df["tokens"]:
    tokens = set(tokens)
    for w1, w2 in combinations(top_words, 2):
        if w1 in tokens and w2 in tokens:
            cooc.loc[w1, w2] += 1
            cooc.loc[w2, w1] += 1

plt.figure(figsize=(12,10))
sns.heatmap(cooc, cmap="YlGnBu", annot=True, fmt="d")
plt.title("Co-occurrence Matrix of Top 30 Words")
plt.tight_layout()
plt.show()

# -------------------------------------------------------------------
# 14) Analysis by genre
# -------------------------------------------------------------------
if "film_genre" in df.columns:
    for genre, subdf in df.groupby("film_genre"):
        words = [w for lst in subdf["tokens"] for w in lst]
        counter = Counter(words).most_common(20)

        top_df = pd.DataFrame(counter, columns=["word", "freq"])
        plt.figure(figsize=(10,6))
        sns.barplot(data=top_df, x="freq", y="word")
        plt.title(f"Top 20 Words — Genre: {genre}")
        plt.tight_layout()
        plt.show()
