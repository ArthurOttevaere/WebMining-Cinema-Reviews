import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# --------------------------------------------------
# 1) FILE LOADING
# --------------------------------------------------
file_path = r"C:\Users\33778\Desktop\WM (Amine)\roger_ebert_debug.csv"

try:
    df = pd.read_csv(
        file_path,
        sep=';',
        encoding='utf-8-sig',
        on_bad_lines='skip',
        engine='python'
    )
except UnicodeDecodeError:
    df = pd.read_csv(
        file_path,
        sep=';',
        encoding='latin1',
        on_bad_lines='skip',
        engine='python'
    )

# --------------------------------------------------
# 2) DATA CLEANING
# --------------------------------------------------
df["article_text_full"] = df["article_text_full"].astype(str)

df = df.dropna(subset=["review_score"])

print("Valid rows:", len(df))

# --------------------------------------------------
# 3) SENTIMENT ANALYSIS (VADER)
# --------------------------------------------------
nltk.download("vader_lexicon", quiet=True)
sia = SentimentIntensityAnalyzer()

def sentiment_trajectory(text, n_parts=5):
    words = text.split()
    scores = []

    chunks = np.array_split(words, n_parts)

    for chunk in chunks:
        chunk_text = " ".join(chunk)
        sentiment = sia.polarity_scores(chunk_text)["compound"]
        scores.append(sentiment)

    return scores

# --------------------------------------------------
# 4) DATA STRUCTURE FOR PLOTTING
# --------------------------------------------------
LOW_SCORE = 2.0
HIGH_SCORE = 3

records = []

for index, row in df.iterrows():
    score = row["review_score"]

    if score <= LOW_SCORE:
        group = "Disappointing films (0–2)"
    elif score >= HIGH_SCORE:
        group = "Outstanding films (3–4)"
    else:
        continue

    traj = sentiment_trajectory(row["article_text_full"])

    step = 0
    for value in traj:
        records.append([group, step, value])
        step += 1

df_plot = pd.DataFrame(records, columns=["Group", "Narrative stage", "Sentiment"])

# --------------------------------------------------
# 5) VISUALIZATION
# --------------------------------------------------
plt.figure(figsize=(12, 7))

sns.lineplot(
    data=df_plot,
    x="Narrative stage",
    y="Sentiment",
    hue="Group",
    style="Group",
    markers=True,
    dashes=False,
    linewidth=3,
    palette={
        "Disappointing films (0–2)": "#D62728",
        "Outstanding films (3–4)": "#2CA02C"
    }
)

plt.xticks(
    [0, 1, 2, 3, 4],
    ["Introduction", "Early development", "Middle", "Climax", "Conclusion"]
)

plt.axhline(0, color="black", linestyle="--", alpha=0.5)
plt.title("Sentiment trajectory: outstanding vs disappointing films")
plt.xlabel("Narrative progression")
plt.ylabel("Average VADER sentiment (compound)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
