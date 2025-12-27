import pandas as pd
import re
import nltk
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import ngrams, pos_tag
from itertools import combinations
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

def run_text_mining(df, show_plots=False):

    # -------------------------------------------------------------------
    # Load CSV
    # -------------------------------------------------------------------

    #df = pd.read_csv("roger_ebert_debug.csv")  # read CSV file into dataframe
    df["article_text_full"] = df["article_text_full"].astype(str)  # ensure text is string
    total_docs = len(df)  # number of documents

    # -------------------------------------------------------------------
    # NLTK setup
    # -------------------------------------------------------------------

    nltk.download("punkt", quiet=True)  # tokenizer models
    nltk.download("stopwords", quiet=True)  # stopwords
    nltk.download("wordnet", quiet=True)  # lemmatizer dictionary
    nltk.download("averaged_perceptron_tagger", quiet=True)  # POS tagger
    nltk.download("vader_lexicon", quiet=True)  # sentiment lexicon

    lemmatizer = WordNetLemmatizer()  # initialize lemmatizer
    stop_words = set(stopwords.words("english"))  # load English stopwords
    negations = {"not","no","never","hardly","none"}  # negation words

    # -------------------------------------------------------------------
    # Preprocessing + tokenization + lemmatization
    # -------------------------------------------------------------------

    def preprocess_and_tokenize(text):
        text = re.sub(r"[^A-Za-z\s]", " ", text)  # remove non-letter characters
        text = re.sub(r"\s+", " ", text).strip()  # remove extra spaces

        tokens = word_tokenize(text)  # split text into tokens
        tagged_tokens = pos_tag(tokens)  # part-of-speech tagging

        processed_tokens = []
        for token, tag in tagged_tokens:
            token = lemmatizer.lemmatize(token.lower())  # lemmatize & lowercase

            if (
                token in stop_words or  # remove stopwords
                len(token) <= 2 or  # remove short words
                re.search(r"\d", token) or  # remove numbers
                tag in ("NNP", "NNPS")  # remove proper nouns
            ):
                continue

            processed_tokens.append(token)  # keep token

        return processed_tokens

    df["tokens"] = df["article_text_full"].apply(preprocess_and_tokenize)  # apply preprocessing

    # -------------------------------------------------------------------
    # Automatic filtering (document frequency)
    # -------------------------------------------------------------------

    min_doc_freq = 2  # minimum document frequency
    max_doc_frac = 0.5  # maximum fraction of documents a token can appear in

    doc_freq = Counter()
    for tokens in df["tokens"]:
        for t in set(tokens):
            doc_freq[t] += 1  # count unique tokens per document

    filtered_tokens = {
        w for w, f in doc_freq.items()
        if f >= min_doc_freq and f / total_docs <= max_doc_frac  # apply DF thresholds
    }

    df["tokens"] = df["tokens"].apply(
        lambda toks: [t for t in toks if t in filtered_tokens]  # filter tokens
    )

    df["clean_text"] = df["tokens"].apply(lambda toks: " ".join(toks))  # join tokens back to text

    print(f"Number of documents: {total_docs}")  # print number of docs
    print(f"Total tokens after filtering: {sum(len(t) for t in df['tokens'])}")  # print remaining tokens

    # -------------------------------------------------------------------
    # TF-IDF vectorization
    # -------------------------------------------------------------------

    tfidf_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),  # use unigrams + bigrams
        min_df=2,  # min document frequency
        max_df=0.5  # max document fraction
    )

    X_tfidf = tfidf_vectorizer.fit_transform(df["clean_text"])  # compute TF-IDF
    print(f"TF-IDF dimensions (raw): {X_tfidf.shape}")  # print TF-IDF shape

    # -------------------------------------------------------------------
    # Dimensionality reduction (SVD) + scaling
    # -------------------------------------------------------------------

    svd = TruncatedSVD(n_components=150, random_state=42)  # reduce dimensions
    X_tfidf_svd = svd.fit_transform(X_tfidf)  # apply SVD

    X_tfidf_final = Normalizer(norm="l2").fit_transform(X_tfidf_svd)  # normalize vectors

    S = cosine_similarity(X_tfidf_final)  # compute cosine similarity matrix

    # ------------------------------------------------------------------- 
    # Cosine similarity — TABLE (All pairs with similarity > 0.7)
    # -------------------------------------------------------------------

    # 1) Remove self-similarity
    S_no_diag = S.copy()
    np.fill_diagonal(S_no_diag, 0)  # set diagonal to 0

    # 2) Extract all unique review pairs (i < j) ensuring different titles
    pairs = []
    titles = df["film_title"].values

    for i in range(S_no_diag.shape[0]):
        for j in range(i + 1, S_no_diag.shape[1]):
            if titles[i] != titles[j] and S_no_diag[i, j] > 0.7:  # keep only high similarity pairs
                pairs.append((i, j, S_no_diag[i, j]))

    # 3) Sort pairs by similarity (descending)
    pairs = sorted(pairs, key=lambda x: x[2], reverse=True)  # sort pairs

    # 4) Build DataFrame (5 columns)
    similarity_table = pd.DataFrame({
        "Review 1": [titles[i] for i, j, _ in pairs],  # first review
        "Score 1":  [df["review_score"].iloc[i] for i, j, _ in pairs],  # first score
        "Review 2": [titles[j] for i, j, _ in pairs],  # second review
        "Score 2":  [df["review_score"].iloc[j] for i, j, _ in pairs],  # second score
        "Cosine similarity": [round(sim, 3) for _, _, sim in pairs]  # similarity
    })

    # -------------------------------------------------------------------
    # Render table as an image
    # -------------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(18, 0.6 * len(similarity_table)))  # figure size
    ax.axis("off")  # hide axes

    # Column widths (relative)
    col_widths = [0.34, 0.08, 0.34, 0.08, 0.16]

    table = ax.table(
        cellText=similarity_table.values,  # data
        colLabels=similarity_table.columns,  # column names
        loc="center",  # center table
        cellLoc="center",  # center text
        colWidths=col_widths  # column widths
    )

    # Styling
    table.auto_set_font_size(False)
    table.set_fontsize(9)  # set font size
    table.scale(1, 2.0)  # scale table

    # Header styling
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")  # bold header
            cell.set_facecolor("#EAEAEA")  # grey header background

    plt.title(
        "Film Reviews with Cosine Similarity > 0.7",
        fontsize=14,
        pad=5  
    )

    plt.tight_layout()
    if show_plots:
        plt.show()
    else:
        plt.close()

    # -------------------------------------------------------------------
    # Silhouette-based choice of K
    # -------------------------------------------------------------------

    def best_k_silhouette(X, k_min=2, k_max=12):
        scores = {}
        for k in range(k_min, k_max + 1):
            km = KMeans(n_clusters=k, random_state=42, n_init=20)  # k-means clustering
            labels = km.fit_predict(X)  # assign clusters
            scores[k] = silhouette_score(X, labels, metric="cosine")  # compute silhouette
        return scores

    tfidf_scores = best_k_silhouette(X_tfidf_final)  # evaluate best K
    best_k = max(tfidf_scores, key=tfidf_scores.get)  # select best K
    print(f"\nBest K TF-IDF: {best_k} | silhouette = {tfidf_scores[best_k]:.4f}")  # print best K

    # -------------------------------------------------------------------
    # Final clustering
    # -------------------------------------------------------------------

    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=20)  # final clustering
    df["cluster"] = kmeans.fit_predict(X_tfidf_final)  # assign cluster labels

    # -------------------------------------------------------------------
    # Dominant words per cluster
    # -------------------------------------------------------------------

    centroids_tfidf = svd.inverse_transform(kmeans.cluster_centers_)  # map centroids to original TF-IDF
    terms = tfidf_vectorizer.get_feature_names_out()  # get feature names
    top_n = 10  # top terms per cluster

    for i in range(best_k):
        top_idx = centroids_tfidf[i].argsort()[-top_n:][::-1]  # indices of top terms
        top_terms = [terms[j] for j in top_idx]  # get term names
        print(f"\nCluster {i} ({(df['cluster']==i).sum()} documents)")  # print cluster size
        print("Dominant words:", ", ".join(top_terms))  # print dominant words

    sil_final = silhouette_score(X_tfidf_final, df["cluster"])  # final silhouette
    print(f"\nFinal silhouette score: {sil_final:.4f}")  # print final silhouette

    # -------------------------------------------------------------------
    # Thesaurus analysis based on filtered tokens
    # -------------------------------------------------------------------

    sia = SentimentIntensityAnalyzer()  # initialize VADER sentiment analyzer
    lexicon = sia.lexicon  # get VADER lexicon

    def extract_eval_words(tokens):
        return [t for t in tokens if t in lexicon]  # keep only tokens present in lexicon

    df["eval_words"] = df["tokens"].apply(extract_eval_words)  # apply extraction to all reviews

    def filter_sentiment_words(tokens, window=3):
        
        pos_words, neg_words = [], []  # lists to store positive and negative words
        for i, token in enumerate(tokens):
            score = lexicon.get(token, 0)  # get VADER sentiment score
            if score != 0:
                # Check for negation within 'window' preceding words
                prev_neg = any(t in negations for t in tokens[max(i - window, 0):i])
                if prev_neg:
                    score = -score  # invert score if negation detected
                if score > 0:
                    pos_words.append(token)  # add to positive words
                elif score < 0:
                    neg_words.append(token)  # add to negative words
        return pos_words, neg_words

    df["pos_words"], df["neg_words"] = zip(*df["eval_words"].apply(filter_sentiment_words))  # apply sentiment filtering

    # -------------------------------------------------------------------
    # Essential calculations: density & pos/neg ratio
    # -------------------------------------------------------------------

    df["review_word_count"] = df["tokens"].apply(len)  # total words per review
    df["eval_count"] = df["eval_words"].apply(len)  # total evaluative words
    df["eval_density"] = df["eval_count"] / df["review_word_count"]  # ratio of evaluative words
    df["pos_count"] = df["pos_words"].apply(len)  # count of positive words
    df["neg_count"] = df["neg_words"].apply(len)  # count of negative words
    df["pos_neg_ratio"] = df["pos_count"] / (df["neg_count"] + 1)  # pos/neg ratio (avoid division by zero)
    df["pos_neg_ratio"] = df["pos_neg_ratio"].fillna(0)  # fill missing values

    # -------------------------------------------------------------------
    # Compound sentiment score with trend line
    # -------------------------------------------------------------------

    df["compound"] = df["eval_words"].apply(lambda toks: sia.polarity_scores(" ".join(toks))["compound"])  # compute compound score

    plt.figure(figsize=(8,6))
    sns.regplot(
        x=df["review_score"],  # review scores on x-axis
        y=df["compound"]  # compound sentiment on y-axis
    )
    plt.title("Review Score vs Compound Sentiment")
    plt.xlabel("Review Score")
    plt.ylabel("Compound Score")
    plt.tight_layout()
    if show_plots:
        plt.show()
    else:
        plt.close()

    # -------------------------------------------------------------------
    # Top 20 reliable evaluative words — Barplot
    # -------------------------------------------------------------------

    # Count all evaluative words
    all_eval_words_filtered = [t for sublist in df["eval_words"] for t in sublist]  # flatten list of evaluative words
    freq_counter_filtered = Counter(all_eval_words_filtered)  # count word frequencies

    # Top 20 words
    top_n = 20
    top_words = freq_counter_filtered.most_common(top_n)  # top 20 evaluative words
    words, freqs = zip(*top_words)

    # Plot
    plt.figure(figsize=(10,6))
    sns.barplot(x=list(freqs), y=list(words), palette="viridis")  # horizontal barplot
    plt.title(f"Top {top_n} Filtered Evaluative Words")
    plt.xlabel("Frequency")
    plt.ylabel("Word")
    plt.tight_layout()
    if show_plots:
        plt.show()
    else:
        plt.close()

    # -------------------------------------------------------------------
    # Analysis: Vocabulary vs Score
    # -------------------------------------------------------------------

    plt.figure(figsize=(8,6))
    sns.regplot(x=df["review_score"], y=df["pos_neg_ratio"])  # plot pos/neg ratio vs review score
    plt.title("Positive/Negative Word Ratio vs Review Score")
    plt.xlabel("Review Score")
    plt.ylabel("Pos/Neg Ratio")
    if show_plots:
        plt.show()
    else:
        plt.close()

    # -------------------------------------------------------------------
    # Descriptive analysis (word frequency + word cloud)
    # -------------------------------------------------------------------

    token_counter = Counter([t for sublist in df["tokens"] for t in sublist])  # count all tokens
    top20 = token_counter.most_common(20)  # top 20 most frequent tokens
    words, freqs = zip(*top20)

    plt.figure(figsize=(12,6))
    sns.barplot(x=list(freqs), y=list(words))  # barplot for top 20 words
    plt.title("Top 20 Most Frequent Words")
    plt.tight_layout()
    if show_plots:
        plt.show()
    else:
        plt.close()

    wc = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(token_counter)  # generate word cloud
    plt.figure(figsize=(12,6))
    plt.imshow(wc, interpolation="bilinear")  # display word cloud
    plt.axis("off")
    plt.title("Word Cloud")
    plt.tight_layout()
    if show_plots:
        plt.show()
    else:
        plt.close()

    # -------------------------------------------------------------------
    # Distribution of movie scores
    # -------------------------------------------------------------------

    if "review_score" in df.columns:
        score_counts = df["review_score"].value_counts().sort_index()  # count scores
        percentages = score_counts / len(df) * 100  # convert to percentage
        x_pos = np.arange(len(score_counts)) * 1.2
        plt.figure(figsize=(12,6))
        bars = plt.bar(x_pos, percentages, width=1.0, edgecolor="black")  # plot histogram
        for bar, pct in zip(bars, percentages):
            plt.text(bar.get_x() + bar.get_width()/2, pct + 0.5, f"{pct:.1f}%", ha="center", va="bottom")  # annotate bars
        plt.xticks(x_pos, score_counts.index)
        plt.xlabel("Review score")
        plt.ylabel("Percentage of reviews (%)")
        plt.title("Distribution of movie review scores")
        plt.tight_layout()
        if show_plots:
            plt.show()
        else:
            plt.close()

    # -------------------------------------------------------------------
    # Boxplot: Review length (word_count) by review score
    # -------------------------------------------------------------------

    if "review_score" in df.columns:
        df["word_count"] = df["tokens"].apply(len)  # count words per review
        plt.figure(figsize=(12,6))
        sns.boxplot(
            data=df,
            x="review_score",
            y="word_count",
            showfliers=True  # show outliers
        )
        plt.title("Review Length Distribution by Review Score")
        plt.xlabel("Review score")
        plt.ylabel("Number of words in review")
        plt.tight_layout()
        if show_plots:
            plt.show()
        else:
            plt.close()
    else:
        print("⚠ 'review_score' or 'word_count' not found. Skipping boxplot.")  # skip if missing

    # -------------------------------------------------------------------
    # N-grams analysis 
    # -------------------------------------------------------------------

    all_tokens = [t for sublist in df["tokens"] for t in sublist]  # flatten token list
    for n in [2,3]:
        counter = Counter(ngrams(all_tokens, n)).most_common(10)  # top n-grams
        labels = [" ".join(ng) for ng, _ in counter]  # convert tuple to string
        values = [freq for _, freq in counter]
        plt.figure(figsize=(10,6))
        sns.barplot(x=values, y=labels)
        plt.title(f"Top 10 {n}-grams")
        plt.tight_layout()
        if show_plots:
            plt.show()
        else:
            plt.close()

    # -------------------------------------------------------------------
    # Co-occurrence matrix
    # -------------------------------------------------------------------

    top_words = [w for w, _ in token_counter.most_common(30)]  # select top 30 words
    cooc = pd.DataFrame(0, index=top_words, columns=top_words)  # initialize co-occurrence matrix
    for tokens in df["tokens"]:
        tokens = set(tokens)  # unique tokens
        for w1, w2 in combinations(top_words, 2):
            if w1 in tokens and w2 in tokens:
                cooc.loc[w1, w2] += 1  # increment co-occurrence
                cooc.loc[w2, w1] += 1  # symmetric entry

    plt.figure(figsize=(12,10))
    sns.heatmap(cooc, cmap="YlGnBu", annot=True, fmt="d")  # heatmap with counts
    plt.title("Co-occurrence Matrix of Top 30 Words")
    plt.tight_layout()
    if show_plots:
        plt.show()
    else:
        plt.close()

    # -------------------------------------------------------------------
    # Average score & sentiment per cluster — Table
    # -------------------------------------------------------------------

    if "review_score" in df.columns and "compound" in df.columns:
        cluster_stats = df.groupby("cluster").agg(
            avg_score=("review_score", "mean"),  # average score
            avg_sentiment=("compound", "mean"),  # average sentiment
            n_docs=("review_score", "count")  # number of documents
        ).reset_index()

        # Convert cluster column to string to avoid .0 in table
        cluster_stats["cluster"] = cluster_stats["cluster"].astype(int).astype(str)  # convert cluster to string
        
        # Format numeric columns
        cluster_stats["avg_score"] = cluster_stats["avg_score"].round(2)  # round scores
        cluster_stats["avg_sentiment"] = cluster_stats["avg_sentiment"].round(3)  # round sentiment

        # Render as image
        fig, ax = plt.subplots(figsize=(8, 0.6 * len(cluster_stats)))
        ax.axis("off")

        table = ax.table(
            cellText=cluster_stats.values,
            colLabels=cluster_stats.columns,
            loc="center",
            cellLoc="center"
        )

        # Styling
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)  # increase row height

        # Header styling
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_text_props(weight="bold")  # bold header
                cell.set_facecolor("#EAEAEA")  # header color

        plt.title("Average Score & Compound Sentiment per Cluster", fontsize=14, pad=20)
        plt.tight_layout()
        if show_plots:
            plt.show()
        else:
            plt.close()

    # -------------------------------------------------------------------
    # Type-Token Ratio (TTR) per review
    # -------------------------------------------------------------------

    df["ttr"] = df["tokens"].apply(lambda x: len(set(x)) / len(x) if len(x) > 0 else 0)  # compute TTR

    # Histogram of TTR
    plt.figure(figsize=(8,6))
    sns.histplot(df["ttr"], bins=20, kde=True)  # histogram of TTR
    plt.title("Distribution of Type-Token Ratio (TTR)")
    plt.xlabel("TTR")
    plt.ylabel("Number of Reviews")
    plt.tight_layout()
    if show_plots:
        plt.show()
    else:
        plt.close()

    # Correlation of TTR with review score and sentiment
    if "review_score" in df.columns:
        plt.figure(figsize=(8,6))
        sns.regplot(x=df["review_score"], y=df["ttr"], scatter_kws={"alpha":0.6})  # scatter plot
        plt.title("TTR vs Review Score")
        plt.xlabel("Review Score")
        plt.ylabel("Type-Token Ratio")
        plt.tight_layout()
        if show_plots:
            plt.show()
        else:
            plt.close()

    if "compound" in df.columns:
        plt.figure(figsize=(8,6))
        sns.regplot(x=df["compound"], y=df["ttr"], scatter_kws={"alpha":0.6})
        plt.title("TTR vs Compound Sentiment")
        plt.xlabel("Compound Sentiment")
        plt.ylabel("Type-Token Ratio")
        plt.tight_layout()
        if show_plots:
            plt.show()
        else:
            plt.close()

    # -------------------------------------------------------------------
    # Sentiment trajectory analysis (narrative progression)
    # -------------------------------------------------------------------

    # We reuse the existing VADER sentiment analyzer (sia)
    # and the already cleaned full review text stored in the DataFrame.

    def sentiment_trajectory(text, n_parts=5):
        """
        Splits a review into n_parts equal segments and computes
        the VADER compound sentiment score for each segment.
        """
        words = text.split()
        
        # If the review is empty, return neutral sentiment values
        if len(words) == 0:
            return [0] * n_parts

        # Split the review into n_parts consecutive chunks
        chunks = np.array_split(words, n_parts)
        scores = []

        # Compute sentiment score for each chunk
        for chunk in chunks:
            chunk_text = " ".join(chunk)
            sentiment = sia.polarity_scores(chunk_text)["compound"]
            scores.append(sentiment)

        return scores


    # Thresholds used to distinguish review groups
    LOW_SCORE = 2.0     # low-rated (negative) reviews
    HIGH_SCORE = 3.0    # high-rated (positive) reviews

    records = []

    # Iterate through all reviews in the DataFrame
    for _, row in df.iterrows():
        score = row["review_score"]

        # Assign reviews to groups based on their rating
        if score <= LOW_SCORE:
            group = "Disappointing films (0–2)"
        elif score >= HIGH_SCORE:
            group = "Outstanding films (3–4)"
        else:
            continue  # neutral reviews are excluded from this analysis

        # Compute sentiment trajectory for the review
        traj = sentiment_trajectory(row["article_text_full"])

        # Store sentiment values for each narrative stage
        for step, value in enumerate(traj):
            records.append([group, step, value])


    # Create a DataFrame suitable for visualization
    df_plot = pd.DataFrame(
        records,
        columns=["Group", "Narrative stage", "Sentiment"]
    )

    # -------------------------------------------------------------------
    # Visualization of sentiment trajectories
    # -------------------------------------------------------------------

    plt.figure(figsize=(12, 7))

    # Line plot showing average sentiment evolution for each group
    sns.lineplot(
        data=df_plot,
        x="Narrative stage",
        y="Sentiment",
        hue="Group",
        style="Group",
        markers=True,
        dashes=False,
        linewidth=3
    )

    # Label narrative stages explicitly
    plt.xticks(
        [0, 1, 2, 3, 4],
        ["Introduction", "Early development", "Middle", "Climax", "Conclusion"]
    )

    # Reference line for neutral sentiment
    plt.axhline(0, color="black", linestyle="--", alpha=0.5)

    plt.title("Sentiment trajectory: outstanding vs disappointing films")
    plt.xlabel("Narrative progression")
    plt.ylabel("Average VADER sentiment (compound)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if show_plots:
        plt.show()
    else:
        plt.close()

    return df 

 # --- LAUNCHING ---
if __name__ == "__main__":
    # Allows to test the file we want
    df_test = pd.read_csv("data/processed/reviews_final_900.csv")
    run_text_mining(df_test, show_plots=True)

