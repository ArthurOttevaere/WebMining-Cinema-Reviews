import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from itertools import combinations
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk import ngrams
import seaborn as sns

# -------------------------------------------------------------------
# 1) Load CSV
# -------------------------------------------------------------------
df = pd.read_csv("roger_ebert_debug.csv")
texts = df["article_text_full"].astype(str)
total_docs = len(df)

# -------------------------------------------------------------------
# 2) NLTK setup
# -------------------------------------------------------------------
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# -------------------------------------------------------------------
# 3) Preprocessing + tokenization + lemmatization
# -------------------------------------------------------------------
def preprocess_and_tokenize(text):
    text = re.sub(r"[^A-Za-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = word_tokenize(text)
    processed_tokens = []
    for token in tokens:
        token_lower = lemmatizer.lemmatize(token.lower())
        if token_lower in stop_words or len(token_lower) <= 2:
            continue
        processed_tokens.append(token_lower)
    return processed_tokens

df["tokens"] = texts.apply(preprocess_and_tokenize)

# -------------------------------------------------------------------
# 4) Automatic filtering of rare or too frequent words
# -------------------------------------------------------------------
min_doc_freq = 2       # appear in at least 2 documents
max_doc_frac = 0.7     # appear in at most 70% of documents

# Count document frequency
doc_freq = Counter()
for tokens in df["tokens"]:
    for t in set(tokens):  # set to count only once per document
        doc_freq[t] += 1

# Filter tokens
filtered_tokens = {w for w, dfreq in doc_freq.items()
                   if dfreq >= min_doc_freq and dfreq / total_docs <= max_doc_frac}

df["tokens"] = df["tokens"].apply(lambda toks: [t for t in toks if t in filtered_tokens])

# Final token counter after filtering
token_counter = Counter([t for sublist in df["tokens"] for t in sublist])

# -------------------------------------------------------------------
# 5) Descriptive analysis: word frequency + word cloud
# -------------------------------------------------------------------
# Top 20 most frequent words
top20 = token_counter.most_common(20)
words, freqs = zip(*top20)

# Bar chart
plt.figure(figsize=(12,6))
sns.barplot(x=list(freqs), y=list(words), palette="viridis")
plt.title("Top 20 Most Frequent Words")
plt.xlabel("Frequency")
plt.ylabel("Word")
plt.tight_layout()
plt.show()

# Word cloud
wc = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(token_counter)
plt.figure(figsize=(12,6))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud of Most Frequent Words")
plt.show()

# -------------------------------------------------------------------
# 6) N-grams analysis (bigrams and trigrams)
# -------------------------------------------------------------------
all_tokens_flat = [t for sublist in df["tokens"] for t in sublist]
bigram_counter = Counter(ngrams(all_tokens_flat, 2))
trigram_counter = Counter(ngrams(all_tokens_flat, 3))

# Top 10 bigrams
top10_bigrams = bigram_counter.most_common(10)
bigram_labels = [" ".join(bg) for bg, _ in top10_bigrams]
bigram_values = [freq for _, freq in top10_bigrams]

plt.figure(figsize=(10,6))
sns.barplot(x=bigram_values, y=bigram_labels, dodge=False)
plt.title("Top 10 Bigrams")
plt.xlabel("Frequency")
plt.ylabel("Bigram")
plt.show()

# Top 10 trigrams
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
# 7) Co-occurrence matrix analysis
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
# 8) Descriptive analysis by genre
# -------------------------------------------------------------------
if "film_genre" in df.columns:
    print("Analyzing by genre...")

    for genre, subdf in df.groupby("film_genre"):
        # Get all tokens for this genre
        words = [w for lst in subdf["tokens"] for w in lst]
        counter = Counter(words).most_common(20)  # top 20 words

        top_df = pd.DataFrame(counter, columns=["word", "freq"])

        # Bar chart
        plt.figure(figsize=(10,6))
        sns.barplot(data=top_df, x="freq", y="word", palette="viridis")
        plt.title(f"Top 20 Most Frequent Words — Genre: {genre}")
        plt.xlabel("Frequency")
        plt.ylabel("Word")
        plt.tight_layout()
        plt.show()

else:
    print("⚠ No 'film_genre' field detected in dataset. Skipping genre analysis.")
