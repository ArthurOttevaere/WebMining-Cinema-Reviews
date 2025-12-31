import pandas as pd
import numpy as np
import os
import warnings
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from collections import Counter

# --- IMPORTING SKLEARN ---
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ==========================================
# 1. CONFIGURATION
# ==========================================

NUM_THEMES = 12     # Based on the results of the text-mining code      
INTRA_LINKS = 4          
INTER_LINKS = 1          

# Lowered tresholds because with TF-IDF/SVD it is often lower than with embeddings 
MIN_SIM_INTRA = 0.30     
MIN_SIM_INTER = 0.50     

INPUT_FILE = "data/processed/reviews_final_900.csv"
NODES_OUTPUT = "data/processed/nodes_tfidf.csv"
EDGES_OUTPUT = "data/processed/edges_tfidf.csv"

# ==========================================
# 2. NLP PREPROCESSING
# ==========================================

def setup_nltk():
    resources = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']
    for res in resources:
        try:
            nltk.data.find(f'tokenizers/{res}')
        except LookupError:
            nltk.download(res, quiet=True)

setup_nltk()

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

my_cinema_stops = {
    'film', 'movie', 'cinema', 'story', 'director', 'plot', 'scene', 
    'character', 'actor', 'role', 'time', 'life', 'people', 'world', 
    'way', 'thing', 'lot', 'new', 'big', 'little', 'good', 'bad', 
    'great', 'best', 'real', 'really', 'just', 'make', 'watch', 'seen', 
    'look', 'know', 'think', 'feel', 'want', 'human', 'shot', 'want', 'moment'
    'play', 'series'
}
stop_words.update(my_cinema_stops) 

# --- CLEANING FUNCTION ---
def preprocess_and_tokenize(text):
    # 1. Regex clean
    text = re.sub(r"[^A-Za-z\s]", " ", str(text))
    text = re.sub(r"\s+", " ", text).strip()

    tokens = word_tokenize(text)
    tagged_tokens = pos_tag(tokens)

    processed_tokens = []
    for token, tag in tagged_tokens:
        token = lemmatizer.lemmatize(token.lower())

        if (token in stop_words or 
            len(token) <= 2 or 
            re.search(r"\d", token) or 
            tag in ("NNP", "NNPS")):
            continue

        processed_tokens.append(token)

    return processed_tokens

# --- NAMING FUNCTION ---
def get_smart_label(texts, n_top=3):
    try:
        # 1. We create a blacklist ONLY for the display. 
        # It helps the reader to better understand the themes.
        DISPLAY_STOPS = [
            'point', 'effect', 'woman', 'end', 'scene', 'plot',
            'man', 'good', 'bad', 'story', 'film', 'movie', 'something',
            'another', 'well', 'set', 'sequence', 'show', 'play', 'still',
            'seems', 'made', 'white', 'find', 'often', 'tell', 'whose'
        ]
        
        # We combine our blacklist with the stopwords. 
        from sklearn.feature_extraction.text import TfidfVectorizer
        from nltk.corpus import stopwords
        all_stops = list(stopwords.words('english')) + DISPLAY_STOPS

        # 2. Launching TF-IDF
        tfidf = TfidfVectorizer(stop_words=all_stops, max_features=500)
        tfidf_matrix = tfidf.fit_transform(texts)
        
        # 3. Extracting the most important words remaining
        mean_scores = np.array(tfidf_matrix.mean(axis=0)).flatten()
        top_indices = mean_scores.argsort()[-n_top:][::-1]
        feature_names = np.array(tfidf.get_feature_names_out())
        
        return ", ".join(feature_names[top_indices])
    except ValueError:
        return "diverse"

# ==========================================
# 3. MAIN PIPELINE
# ==========================================

def main(num_themes=NUM_THEMES, input_path=INPUT_FILE):
    print(f"📂 Reading : {input_path}")
    if not os.path.exists(input_path):
        print("❌ Error : CSV file not found.")
        return
    
    df = pd.read_csv(input_path, sep=',', encoding='utf-8')

    # --- REMOVING DUPLICATES ---
    df = df.drop_duplicates(subset=['article_text_full'])
    df = df.drop_duplicates(subset=['film_title'])
    df = df.reset_index(drop=True)
    total_docs = len(df)
    print(f"🧹 Data loaded. {total_docs} films.")

    # --- STEP A : TOKENIZATION ---
    print("⚙️  Preprocessing & Tokenization (This takes time)...")
    df["tokens"] = df["article_text_full"].apply(preprocess_and_tokenize)

    # --- STEP B : FILTERING by FREQUENCY ---
    print("📉 Filtering tokens by frequency...")
    min_doc_freq = 2
    max_doc_frac = 0.5

    doc_freq = Counter()
    for tokens in df["tokens"]:
        for t in set(tokens):
            doc_freq[t] += 1

    filtered_tokens_set = {
        w for w, f in doc_freq.items()
        if f >= min_doc_freq and f / total_docs <= max_doc_frac
    }

    # We only keep filtered words and we create a srtring for TF-IDF. 
    df["clean_text"] = df["tokens"].apply(
        lambda toks: " ".join([t for t in toks if t in filtered_tokens_set])
    )

    # --- STEP C : VECTORIZATION (TF-IDF + SVD) ---
    print("🧮 Vectorization (TF-IDF -> SVD -> Normalizer)...")
    
    # 1. TF-IDF
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.5)
    X_tfidf = tfidf_vectorizer.fit_transform(df["clean_text"])
    
    # 2. SVD (Reducting dimension)
    svd = TruncatedSVD(n_components=150, random_state=42) # Comme eux
    X_svd = svd.fit_transform(X_tfidf)
    
    # 3. Normalization (Important for consine similarity)
    normalizer = Normalizer(norm="l2")
    X_final = normalizer.fit_transform(X_svd)
        
    # --- CLUSTERING ---
    print("🎨 Clustering (K-Means on SVD matrix)...")
    kmeans = KMeans(n_clusters=num_themes, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_final)
    df['cluster_id'] = clusters
    
    # --- NAMING ---
    print("🏷️  Naming themes...")
    cluster_names = {}
    for cid in range(num_themes):
        # We use cleaned text to find our keywords
        texts = df[df['cluster_id'] == cid]['clean_text']
        keywords = get_smart_label(texts)
        cluster_names[cid] = keywords
        print(f"   - Theme {cid}: {keywords}")
    
    df['Theme_Label'] = df['cluster_id'].map(cluster_names)
    
    # --- NODES EXPORT ---
    nodes = df[['review_id', 'film_title', 'Theme_Label', 'review_score']].copy()
    nodes.columns = ['Id', 'Label', 'Theme', 'Score']
    nodes.to_csv(NODES_OUTPUT, index=False)
    
    # --- EDGES EXPORT ---
    print("🔗 Computing Cosine Similarity & Edges...")
    
    # We process similarity on SVD matrix. 
    sim_matrix = cosine_similarity(X_final) 
    edges_list = []
    
    for i in range(len(df)):
        my_cluster = df.iloc[i]['cluster_id']
        sorted_indices = np.argsort(sim_matrix[i])[::-1]
        
        intra_count = 0
        inter_count = 0
        
        for neighbor_idx in sorted_indices[1:]: # Skip self (index 0)
            if intra_count >= INTRA_LINKS and inter_count >= INTER_LINKS:
                break
                
            score = sim_matrix[i][neighbor_idx]
            neighbor_cluster = df.iloc[neighbor_idx]['cluster_id']
            
            # Links INTRA (Same cluster)
            if neighbor_cluster == my_cluster:
                if intra_count < INTRA_LINKS and score > MIN_SIM_INTRA:
                    edges_list.append({
                        'Source': df.iloc[i]['review_id'], 
                        'Target': df.iloc[neighbor_idx]['review_id'],
                        'Weight': round(score, 4), 
                        'Type': 'Undirected'
                    })
                    intra_count += 1
            
            # Links INTER (Bridges between clusters)
            else:
                if inter_count < INTER_LINKS and score > MIN_SIM_INTER:
                    edges_list.append({
                        'Source': df.iloc[i]['review_id'], 
                        'Target': df.iloc[neighbor_idx]['review_id'],
                        'Weight': round(score, 4), 
                        'Type': 'Undirected'
                    })
                    inter_count += 1

    pd.DataFrame(edges_list).to_csv(EDGES_OUTPUT, index=False)
    print(f"✅ FINISHED ! Nodes: {NODES_OUTPUT} | Edges: {EDGES_OUTPUT}")

if __name__ == "__main__":
    main()