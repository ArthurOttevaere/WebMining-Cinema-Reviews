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
# 1) Chargement du CSV
# -------------------------------------------------------------------
df = pd.read_csv("roger_ebert_debug.csv")
texts = df["article_text_full"].astype(str)
scores = df["review_score"].astype(float)

# -------------------------------------------------------------------
# 2) NLTK
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
# 3) Prétraitement + tokenisation + POS + lemmatisation
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
# 4) Extraire mots évaluatifs
# -------------------------------------------------------------------
sia = SentimentIntensityAnalyzer()
lexicon = sia.lexicon

eval_tags = {"JJ","JJR","JJS","RB","RBR","RBS","VB","VBD","VBG","VBN","VBP","VBZ","NN","NNS"}

def extract_eval_words(tokens_tagged):
    return [t for t, tag in tokens_tagged if t.lower() in lexicon and tag in eval_tags]

df["eval_words"] = df["tokens_tagged"].apply(extract_eval_words)

# -------------------------------------------------------------------
# 5) Positifs / négatifs + correction des négations
# -------------------------------------------------------------------
def filter_sentiment_words(tokens):
    pos_words, neg_words = [], []
    for i, token in enumerate(tokens):
        prev_neg = i>0 and tokens[i-1].lower() in negations
        score = lexicon.get(token.lower(), 0)
        if score != 0:
            if prev_neg:
                score = -score
            if score>0:
                pos_words.append(token)
            elif score<0:
                neg_words.append(token)
    return pos_words, neg_words

df["pos_words"], df["neg_words"] = zip(*df["eval_words"].apply(filter_sentiment_words))

# -------------------------------------------------------------------
# 6) Calculs essentiels : densité & ratio pos/neg
# -------------------------------------------------------------------
df["review_word_count"] = texts.apply(lambda x: len(x.split()))
df["eval_count"] = df["eval_words"].apply(len)

df["eval_density"] = df["eval_count"] / df["review_word_count"]
df["eval_density"] = df["eval_density"].fillna(0)

df["pos_count"] = df["pos_words"].apply(len)
df["neg_count"] = df["neg_words"].apply(len)

df["pos_neg_ratio"] = df["pos_count"] / (df["neg_count"] + 1)
df["pos_neg_ratio"] = df["pos_neg_ratio"].fillna(0)

# -------------------------------------------------------------------
# 7) Nuages de mots
# -------------------------------------------------------------------
pos_counter = Counter([w for sublist in df["pos_words"] for w in sublist])
neg_counter = Counter([w for sublist in df["neg_words"] for w in sublist])

wc_pos = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(pos_counter)
wc_neg = WordCloud(width=800, height=400, background_color="black").generate_from_frequencies(neg_counter)

plt.figure(figsize=(10,5))
plt.imshow(wc_pos, interpolation="bilinear")
plt.axis("off")
plt.title("Nuage de mots positifs")
plt.show()

plt.figure(figsize=(10,5))
plt.imshow(wc_neg, interpolation="bilinear")
plt.axis("off")
plt.title("Nuage de mots négatifs")
plt.show()

# -------------------------------------------------------------------
# 8) Compound
# -------------------------------------------------------------------
df["compound"] = df["eval_words"].apply(lambda toks: sia.polarity_scores(" ".join(toks))["compound"])

plt.figure(figsize=(8,6))
sns.scatterplot(x=df["review_score"], y=df["compound"])
plt.title("Review Score vs Compound")
plt.xlabel("Review Score")
plt.ylabel("Compound Score")
plt.show()

# -------------------------------------------------------------------
# 9) Top 20 mots évaluatifs fiables
# -------------------------------------------------------------------
all_eval_words_strict = [t for sublist in df["eval_words"] for t in sublist]
freq_counter_strict = Counter(all_eval_words_strict)
print("20 mots évaluatifs stricts les plus fréquents :", freq_counter_strict.most_common(20))

# -------------------------------------------------------------------
# 10) Analyse (3) : Relation vocabulaire ↔ note
# -------------------------------------------------------------------
plt.figure(figsize=(8,6))
sns.regplot(x=df["review_score"], y=df["eval_density"])
plt.title("Densité évaluative vs Note")
plt.xlabel("Review Score")
plt.ylabel("Densité évaluative")
plt.show()

plt.figure(figsize=(8,6))
sns.regplot(x=df["review_score"], y=df["pos_neg_ratio"])
plt.title("Ratio mots positifs / négatifs vs Note")
plt.xlabel("Review Score")
plt.ylabel("Ratio pos/neg")
plt.show()

# -------------------------------------------------------------------
# 11) Analyse par genres
# -------------------------------------------------------------------
if "film_genre" in df.columns:
    print("Analyse par genre en cours...")

    for genre, subdf in df.groupby("film_genre"):
        words = [w for lst in subdf["eval_words"] for w in lst]
        counter = Counter(words).most_common(20)
        top_df = pd.DataFrame(counter, columns=["word", "freq"])

        plt.figure(figsize=(10, 6))
        sns.barplot(data=top_df, x="freq", y="word", palette="viridis")
        plt.title(f"Top 20 mots évaluatifs — {genre}")
        plt.xlabel("Fréquence")
        plt.ylabel("Mot évaluatif")
        plt.tight_layout()
        plt.show()

else:
    print("⚠ Aucun champ 'film_genre' détecté dans votre dataset. Analyse ignorée.")
