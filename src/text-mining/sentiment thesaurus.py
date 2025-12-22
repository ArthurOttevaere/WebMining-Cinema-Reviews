import pandas as pd 
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns

# -------------------------------------------------------------------
# 1) Load CSV
# -------------------------------------------------------------------
df = pd.read_csv("roger_ebert_debug.csv")
texts = df["article_text_full"].astype(str)
scores = df["review_score"].astype(float)

# -------------------------------------------------------------------
# 2) NLTK setup
# -------------------------------------------------------------------
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("averaged_perceptron_tagger")
nltk.download("vader_lexicon")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))
negations = {"not","no","never","hardly","none"}

# -------------------------------------------------------------------
# 3) Preprocessing + tokenization + POS tagging + lemmatization
# -------------------------------------------------------------------
def preprocess_and_tokenize(text):
    text = re.sub(r"[^A-Za-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    processed_tokens = []
    for token, tag in tagged:
        if token.lower() in stop_words or len(token) <= 2:
            continue
        token_proc = token if tag in ("NNP", "NNPS") else lemmatizer.lemmatize(token.lower())
        processed_tokens.append((token_proc, tag))
    return processed_tokens

df["tokens_tagged"] = texts.apply(preprocess_and_tokenize)

# -------------------------------------------------------------------
# 4) Extract evaluative words
# -------------------------------------------------------------------
sia = SentimentIntensityAnalyzer()
lexicon = sia.lexicon
eval_tags = {"JJ","JJR","JJS","RB","RBR","RBS","VB","VBD","VBG","VBN","VBP","VBZ","NN","NNS"}

def extract_eval_words(tokens_tagged):
    return [t for t, tag in tokens_tagged if t.lower() in lexicon and tag in eval_tags]

df["eval_words"] = df["tokens_tagged"].apply(extract_eval_words)

# -------------------------------------------------------------------
# 4b) Automatic filtering of too rare or too frequent terms
# -------------------------------------------------------------------
all_eval_words = [w.lower() for sublist in df["eval_words"] for w in sublist]
eval_counter = Counter(all_eval_words)
n_docs = len(df)

# Thresholds
min_doc_freq = 2       # must appear in at least 2 documents
max_doc_frac = 0.8     # must appear in at most 80% of documents

# Compute document frequency
doc_freq = Counter()
for words in df["eval_words"]:
    for w in set([t.lower() for t in words]):
        doc_freq[w] += 1

# Apply filtering
filtered_words = {w for w, dfreq in doc_freq.items() if dfreq >= min_doc_freq and dfreq/n_docs <= max_doc_frac}
print(f"Number of evaluative words after filtering: {len(filtered_words)} (out of {len(eval_counter)})")

def filter_eval_words(words):
    return [w for w in words if w.lower() in filtered_words]

df["eval_words_filtered"] = df["eval_words"].apply(filter_eval_words)

# -------------------------------------------------------------------
# 5) Positive / negative words + negation handling
# -------------------------------------------------------------------
def filter_sentiment_words(tokens):
    pos_words, neg_words = [], []
    for i, token in enumerate(tokens):
        prev_neg = i>0 and tokens[i-1].lower() in negations
        score = lexicon.get(token.lower(), 0)
        if score != 0:
            if prev_neg:
                score = -score
            if score > 0:
                pos_words.append(token)
            elif score < 0:
                neg_words.append(token)
    return pos_words, neg_words

df["pos_words"], df["neg_words"] = zip(*df["eval_words_filtered"].apply(filter_sentiment_words))

# -------------------------------------------------------------------
# 6) Essential calculations: density & pos/neg ratio
# -------------------------------------------------------------------
df["review_word_count"] = texts.apply(lambda x: len(x.split()))
df["eval_count"] = df["eval_words_filtered"].apply(len)

df["eval_density"] = df["eval_count"] / df["review_word_count"]
df["eval_density"] = df["eval_density"].fillna(0)

df["pos_count"] = df["pos_words"].apply(len)
df["neg_count"] = df["neg_words"].apply(len)

df["pos_neg_ratio"] = df["pos_count"] / (df["neg_count"] + 1)
df["pos_neg_ratio"] = df["pos_neg_ratio"].fillna(0)

# -------------------------------------------------------------------
# 7) Word clouds
# -------------------------------------------------------------------
pos_counter = Counter([w for sublist in df["pos_words"] for w in sublist])
neg_counter = Counter([w for sublist in df["neg_words"] for w in sublist])

wc_pos = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(pos_counter)
wc_neg = WordCloud(width=800, height=400, background_color="black").generate_from_frequencies(neg_counter)

plt.figure(figsize=(10,5))
plt.imshow(wc_pos, interpolation="bilinear")
plt.axis("off")
plt.title("Positive Words Word Cloud")
plt.show()

plt.figure(figsize=(10,5))
plt.imshow(wc_neg, interpolation="bilinear")
plt.axis("off")
plt.title("Negative Words Word Cloud")
plt.show()

# -------------------------------------------------------------------
# 8) Compound sentiment score
# -------------------------------------------------------------------
df["compound"] = df["eval_words_filtered"].apply(lambda toks: sia.polarity_scores(" ".join(toks))["compound"])

plt.figure(figsize=(8,6))
sns.scatterplot(x=df["review_score"], y=df["compound"])
plt.title("Review Score vs Compound Sentiment")
plt.xlabel("Review Score")
plt.ylabel("Compound Score")
plt.show()

# -------------------------------------------------------------------
# 9) Top 20 reliable evaluative words
# -------------------------------------------------------------------
all_eval_words_filtered = [t for sublist in df["eval_words_filtered"] for t in sublist]
freq_counter_filtered = Counter(all_eval_words_filtered)
print("Top 20 filtered evaluative words:", freq_counter_filtered.most_common(20))

# -------------------------------------------------------------------
# 10) Analysis: Vocabulary vs Score
# -------------------------------------------------------------------
plt.figure(figsize=(8,6))
sns.regplot(x=df["review_score"], y=df["eval_density"])
plt.title("Evaluative Word Density vs Review Score")
plt.xlabel("Review Score")
plt.ylabel("Evaluative Density")
plt.show()

plt.figure(figsize=(8,6))
sns.regplot(x=df["review_score"], y=df["pos_neg_ratio"])
plt.title("Positive/Negative Word Ratio vs Review Score")
plt.xlabel("Review Score")
plt.ylabel("Pos/Neg Ratio")
plt.show()

# -------------------------------------------------------------------
# 11) Analysis by genres
# -------------------------------------------------------------------
if "film_genre" in df.columns:
    print("Analyzing by genre...")

    for genre, subdf in df.groupby("film_genre"):
        words = [w for lst in subdf["eval_words_filtered"] for w in lst]
        counter = Counter(words).most_common(20)
        top_df = pd.DataFrame(counter, columns=["word", "freq"])

        plt.figure(figsize=(10, 6))
        sns.barplot(data=top_df, x="freq", y="word", palette="viridis")
        plt.title(f"Top 20 Evaluative Words — Genre: {genre}")
        plt.xlabel("Frequency")
        plt.ylabel("Evaluative Word")
        plt.tight_layout()
        plt.show()
else:
    print("⚠ No 'film_genre' field detected in dataset. Skipping genre analysis.")
