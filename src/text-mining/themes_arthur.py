import pandas as pd
import numpy as np
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

# ==========================================
# 1. CONFIGURATION
# ==========================================

NUM_THEMES = 8          
INTRA_LINKS = 4          
INTER_LINKS = 1          
MIN_SIM_INTRA = 0.40     
MIN_SIM_INTER = 0.60     

# Will be useful for implementation in GitHub
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_FILE = os.path.join(BASE_DIR, "data", "processed", "reviews_final_900.csv")
NODES_OUTPUT = os.path.join(BASE_DIR, "data", "raw", "gephi_nodes_clustered.csv")
EDGES_OUTPUT = os.path.join(BASE_DIR, "data", "raw", "gephi_edges_clustered.csv")

# ==========================================
# 2. NLP Cleaning 
# ==========================================

# We download the required packages of nltk
def download_nltk():
    resources = ['punkt', 'stopwords', 'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng']
    for res in resources:
        try:
            nltk.data.find(f'tokenizers/{res}')
        except LookupError:
            nltk.download(res, quiet=True)

download_nltk()

# --- UPDATE OF BLACK LIST ---
CUSTOM_STOPS = [
    'film', 'films', 'movie', 'movies', 'cinema', 'story', 'director', 
    'character', 'characters', 'scene', 'scenes', 'plot', 'screen', 
    'cast', 'actor', 'actress', 'role', 'performance', 'review',
    'time', 'life', 'people', 'world', 'year', 'years', 'way', 'thing', 'things',
    'lot', 'new', 'big', 'little', 'good', 'bad', 'great', 'best', 
    'real', 'really', 'just', 'make', 'made', 'watch', 'seen', 'look',
    'know', 'think', 'feel', 'say', 'come', 'go', 'going', 'end', 'start',
    'bit', 'kind', 'quite', 'actually', 'audience', 'hollywood', 'feature',
    'work', 'long', 'short', 'old', 'young', 'man', 'woman', 'guy', 'girl',
    'does', 'did', 'original', 'action', 'star', 'screenplay', 'something',
    'animation', 'many', 'day', 'night', 'american', 'family', 'point',
    'first', 'much', 'men', 'women', 
    'park', 'house', 'home', 'street', 'city', 'town', 
    'black', 'white', 'red', 'blue', 'green', 
    'john', 'david', 'lee', 'michael', 'james', 
    'mr', 'mrs', 'dr', 'st', 
    'series', 'documentary', 'sense' 
]
FINAL_STOP_WORDS = list(stopwords.words('english')) + CUSTOM_STOPS

# We filter the nouns, adjectives
def filter_nouns_adjectives(text):
    if not isinstance(text, str): return ""
    try:
        tokens = word_tokenize(text.lower())
        tagged = pos_tag(tokens)
        relevant_words = [word for word, tag in tagged 
                          if tag in ['NN', 'NNS', 'JJ'] 
                          and word not in FINAL_STOP_WORDS
                          and len(word) > 2] 
        return " ".join(relevant_words)
    except:
        return ""

# --- NAMING (conditions) ---
def get_smart_keywords(texts, n_top=3):
    try:
        cleaned_texts = [filter_nouns_adjectives(t) for t in texts]
        
        # We can adjust the parameters wether we want to be stricter or not 
        tfidf = TfidfVectorizer(
            max_features=500,
            max_df=0.5,       
            min_df=6          
        )
        
        tfidf_matrix = tfidf.fit_transform(cleaned_texts)
        mean_scores = np.array(tfidf_matrix.mean(axis=0)).flatten()
        top_indices = mean_scores.argsort()[-n_top:][::-1]
        feature_names = np.array(tfidf.get_feature_names_out())
        
        return ", ".join(feature_names[top_indices])
    except ValueError:
        return "diverse"

# ==========================================
# 3. PIPELINE
# ==========================================

def main(num_themes = NUM_THEMES, input_path = INPUT_FILE):
    print(f"📂 Reading : {input_path}")
    if not os.path.exists(input_path):
        # Fallback au cas où le chemin est différent
        print("⚠️ Unlocated file, trying local path... ")
        INPUT_FILE_LOCAL = "reviews_final_900.csv" 
        if os.path.exists(INPUT_FILE_LOCAL):
             df = pd.read_csv(INPUT_FILE_LOCAL)
        else:
            print("❌ Error : No CSV file.")
            return
    else:
        df = pd.read_csv(input_path)
    
    # --- REMOVING DUPLICATES ---
    initial_len = len(df)
    df = df.drop_duplicates(subset=['article_text_full'])
    df = df.drop_duplicates(subset=['film_title'])
    
    df = df.reset_index(drop=True) 
    
    print(f"🧹 Duplicates removed. {len(df)} films remaining.")
    
    # --- EMBEDDINGS ---
    print("🧠 Porcessing Embeddings...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    sentences = df['article_text_full'].fillna("").tolist()
    embeddings = model.encode(sentences, show_progress_bar=True)
    
    # --- CLUSTERING ---
    print("🎨 Clustering...")
    kmeans = KMeans(n_clusters=num_themes, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(embeddings)
    df['cluster_id'] = clusters
    
    # --- NAMING ---
    print("🏷️  Naming...")
    cluster_names = {}
    for cid in range(num_themes):
        texts = df[df['cluster_id'] == cid]['article_text_full'].fillna("")
        keywords = get_smart_keywords(texts)
        cluster_names[cid] = keywords
        print(f"   - Theme {cid}: {keywords}")
    
    df['Theme_Label'] = df['cluster_id'].map(cluster_names)
    
    # --- NODES EXPORT ---
    nodes = df[['review_id', 'film_title', 'Theme_Label', 'review_score']].copy()
    nodes.columns = ['Id', 'Label', 'Theme', 'Score']
    nodes.to_csv(NODES_OUTPUT, index=False)
    
    # --- EDGES EXPORT ---
    print("🔗 Crafting the links...")
    sim_matrix = cosine_similarity(embeddings)
    edges_list = []
    
    for i in range(len(df)):
        my_cluster = df.iloc[i]['cluster_id']
        sorted_indices = np.argsort(sim_matrix[i])[::-1]
        
        intra_count = 0
        inter_count = 0
        
        for neighbor_idx in sorted_indices[1:]:
            if intra_count >= INTRA_LINKS and inter_count >= INTER_LINKS:
                break
                
            score = sim_matrix[i][neighbor_idx]
            
            neighbor_cluster = df.iloc[neighbor_idx]['cluster_id']
            
            if neighbor_cluster == my_cluster:
                if intra_count < INTRA_LINKS and score > MIN_SIM_INTRA:
                    edges_list.append({
                        'Source': df.iloc[i]['review_id'], 'Target': df.iloc[neighbor_idx]['review_id'],
                        'Weight': round(score, 4), 'Type': 'Undirected'
                    })
                    intra_count += 1
            else:
                if inter_count < INTER_LINKS and score > MIN_SIM_INTER:
                    edges_list.append({
                        'Source': df.iloc[i]['review_id'], 'Target': df.iloc[neighbor_idx]['review_id'],
                        'Weight': round(score, 4), 'Type': 'Undirected'
                    })
                    inter_count += 1

    pd.DataFrame(edges_list).to_csv(EDGES_OUTPUT, index=False)
    print(f"✅ TERMINÉ ! Nodes: {NODES_OUTPUT} | Edges: {EDGES_OUTPUT}")

# RUNNING LOCALLY
if __name__ == "__main__":
    main()