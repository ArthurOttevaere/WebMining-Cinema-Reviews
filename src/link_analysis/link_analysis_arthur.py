import pandas as pd
import numpy as np
import os
from tabulate import tabulate 

# ==========================================
# 1. NUMPY FUNCTIONS (MATRICIAL)
# ==========================================

def degree_matrix(A: np.ndarray, direction: str = "out") -> np.ndarray:
    if direction == "out": 
        deg = A.sum(axis=1)
    elif direction == "in": 
        deg = A.sum(axis=0)
    return np.diag(deg)

def transition_matrix(A: np.ndarray, direction: str = "out") -> np.ndarray:
    D = degree_matrix(A, direction=direction)
    deg = np.diag(D)
    P = np.zeros_like(A, dtype=float)
    
    for i, d in enumerate(deg):
        if d > 0:
            P[i, :] = A[i, :] / d
    return P

def shortest_path_matrix(A: np.ndarray) -> np.ndarray:
    """Implentation of Floyd-Warshall"""
    n = A.shape[0]
    dist = np.full((n, n), 100000.0) 
    rows, cols = np.where(A > 0) 
    dist[rows, cols] = 1.0
    np.fill_diagonal(dist, 0)
    
    print("   ⏳ Processing Floyd-Warshall (can take 1 to 2 min)...")
    for k in range(n):
        dist = np.minimum(dist, dist[:, k][:, None] + dist[k, :])
    return dist

def closeness_centrality(SP: np.ndarray) -> np.ndarray:
    n = SP.shape[0]
    CC = np.zeros(n, dtype=float)
    for i in range(n):
        dists = SP[i, :]
        valid_dists = dists[dists < 90000]
        if len(valid_dists) > 1:
            dist_sum = valid_dists.sum()
            CC[i] = (len(valid_dists) - 1) / dist_sum
    return CC  

def eccentricity_centrality(SP: np.ndarray) -> np.ndarray:
    n = SP.shape[0]
    ecc_cent = np.zeros(n, dtype=float)
    for i in range(n):
        dists = SP[i, :]
        valid_dists = dists[dists < 90000] # We ignore the "infinite"
        if len(valid_dists) > 0:
            max_dist = valid_dists.max() # Distance to the farest node. 
            if max_dist > 0:
                ecc_cent[i] = 1.0 / max_dist
    return ecc_cent  
    
def laplacian_matrix(A: np.ndarray) -> np.ndarray:
    D = degree_matrix(A, "out")
    L = D - A
    return L

def laplacian_pseudoinverse(A: np.ndarray) -> np.ndarray:
    print("   ⏳ Processing L+ (Pseudo-Inverse)...")
    L = laplacian_matrix(A)
    n = A.shape[0]
    e = np.ones((n, 1))
    E = (e @ e.T)/n
    M = L - E
    M_inv = np.linalg.inv(M)
    L_plus = M_inv + E
    return L_plus

def information_centrality(L_plus: np.ndarray) -> np.ndarray:
    diag_L_plus = np.diag(L_plus)
    info_cent = 1.0 / (diag_L_plus + 1e-10) 
    info_cent = (info_cent - info_cent.min()) / (info_cent.max() - info_cent.min())
    return info_cent

def get_spectral_partition(L: np.ndarray) -> np.ndarray:
    """
    Fiedler Vector' to cut the graph into two communities
    """
    print("   ⏳ Processing the eigen values...")
    # eigh is optimized for symetric matrix such as L
    eigenvals, eigenvecs = np.linalg.eigh(L)
    
    # Fiedler vector is the seconde column (first one is for the values)
    fiedler_vector = eigenvecs[:, 1]
    
    # We split (if sign is "+" or "-").
    partition = np.where(fiedler_vector > 0, 1, 0)
    return partition

def pagerank_power_iteration(A: np.ndarray, alpha: float = 0.85, max_iter: int = 100) -> np.ndarray:
    n = A.shape[0]
    P = transition_matrix(A)
    row_sums = P.sum(axis=1)
    P[row_sums == 0] = np.ones(n) / n
    E = np.ones((n, n)) / n
    pr = np.ones((n,1))/n
    G = alpha * P + (1 - alpha) * E

    for _ in range(max_iter):
        prev_pr = pr.copy()
        pr = G.T @ pr
        pr = pr/pr.sum()
        if np.linalg.norm(pr - prev_pr) < 1e-6:
            break
    return pr.flatten()

# ==========================================
# 2. MAIN PIPELINE
# ==========================================

INPUT_EDGES = "data/processed/edges_tfidf.csv"
INPUT_NODES = "data/processed/nodes_tfidf.csv"
OUTPUT_FILE = "data/processed/numpy_metrics_results.csv"

def main():
    print("📂 Loading data...")
    if not os.path.exists(INPUT_EDGES):
        print(f"❌ No files found : {INPUT_EDGES}")
        return

    df_edges = pd.read_csv(INPUT_EDGES)
    df_nodes = pd.read_csv(INPUT_NODES)
    
    # Building the A matrix. 
    node_ids = df_nodes['Id'].unique()
    n = len(node_ids)
    id_map = {id_: i for i, id_ in enumerate(node_ids)}
    
    A = np.zeros((n, n), dtype=float)
    
    print(f"🏗️  Building the A matrix ({n}x{n})...")
    for _, row in df_edges.iterrows():
        try:
            u, v = id_map[row['Source']], id_map[row['Target']]
            A[u, v] = row['Weight']
            A[v, u] = row['Weight']
        except KeyError:
            pass

    # --- ANALYSIS ---
    print("1️⃣ Degree Centrality...")
    A_binary = (A > 0).astype(int)
    degrees = np.diag(degree_matrix(A_binary, "out"))

    print("2️⃣ Shortest Path & Closeness...")
    SP = shortest_path_matrix(A_binary) 
    closeness = closeness_centrality(SP)
    eccentricity = eccentricity_centrality(SP)
    
    real_dists = SP[SP < 90000]
    avg_path = real_dists.mean() if len(real_dists) > 0 else 0
    print(f"   👉 Average distance (jumps) : {avg_path:.2f}")

    print("3️⃣ PageRank (Power Iteration)...")
    pagerank = pagerank_power_iteration(A, alpha=0.85)

    print("4️⃣ Information Centrality (via Laplacian L+)...")
    L_plus = laplacian_pseudoinverse(A)
    info_centrality = information_centrality(L_plus)

    # --- NEW FEATURE 1: SPECTRAL CLUSTERING ---
    print("5️⃣ Spectral Clustering (Fiedler Partition)...")
    L = laplacian_matrix(A) # On recalcule L ou on le récupère
    communities = get_spectral_partition(L)
    
    # --- EXPORT & DISPLAY ---
    
    print("💾 Preparing the results...")
    results = pd.DataFrame({
        'Id': node_ids,
        'Degree': degrees,
        'Closeness': closeness,
        'Eccentricity': eccentricity,
        'PageRank': pagerank,
        'InfoCent': info_centrality,
        'SpectralGroup': communities
    })
    
    final_df = pd.merge(df_nodes, results, on='Id')
    
    # We round up for a better readability
    cols_float = ['Closeness', 'Eccentricity', 'PageRank', 'InfoCent']
    final_df[cols_float] = final_df[cols_float].round(4)
    
    # CSV Export
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ CSV exported : {OUTPUT_FILE}")

    # --- NEW FEATURE 2: CORRELATION MATRIX ---
    print("\n" + "═"*80)
    print("      🧩 META-ANALYSIS : CORRELATION BETWEEN METRICS")
    print("      (Does popularity mean influence ?)")
    print("═"*80)
    
    # Spearman correlation on the ranks, not the values.
    corr_matrix = final_df[['Degree', 'Closeness', 'PageRank', 'InfoCent']].corr(method='spearman')
    print(tabulate(corr_matrix, headers='keys', tablefmt='fancy_grid', floatfmt=".2f"))
    
    # --- TABULATE DISPLAY (TOP N PER METRIC) ---
    
    # We choose the top "n" to display.
    TOP_N = 10
    
    # Metrics to analyze.
    metrics_to_show = [
        ('Degree', 'Local influence (Number of direct connections)'),
        ('Closeness', 'Closeness (Ability to reach quickly the network.)'),
        ('Eccentricity', 'Graphic centrality (Distance to the farest point)'),
        ('InfoCent', 'Strategic points (Bridges and information traffic)'),
        ('PageRank', 'Global Influence (Importance of neighbours)')

    ]

    print("\n" + "═"*80)
    print("      📊 CENTRALITY ANALYSIS : TOP {} BY MEASURE".format(TOP_N))
    print("═"*80)

    for col, description in metrics_to_show:
        print(f"\n🔥 {description} [{col}]")
        
        # Sorting by the displayed metric
        top_df = final_df.sort_values(by=col, ascending=False).head(TOP_N)
        
        # We select the columns to display
        display_cols = ['Label', 'Theme', col]
        
        print(tabulate(top_df[display_cols], headers='keys', tablefmt='fancy_grid', showindex=False))

    # --- NEW FEATURE 3: GLOBAL GRAPH STATS ---
    print("\n" + "═"*80)
    print("      🌍 NETWORK'S GLOBAL HEALTH")
    print("═"*80)
    
    # 1. We filter to avoid "0" values
    valid_scores = eccentricity[eccentricity > 0]
    
    if len(valid_scores) > 0:
        # Radius 
        min_eccentricity_value = 1.0 / valid_scores.max()
        
        # Diameter
        max_eccentricity_value = 1.0 / valid_scores.min()
        
        print(f"📏 Graph diameter : {int(max_eccentricity_value)} jumps (Max eccentricity)")
        print(f"🎯 Graph radius   : {int(min_eccentricity_value)} jumps (Min eccentricity / Center)")
    else:
        print("⚠️ Graph appears to be fully disconnected.")

    # We display the average distance
    if 'avg_path' in locals():
        print(f"🏃 Average distance : {avg_path:.2f} jumps")
        
    # Quick analysis of the spectral groups
    group_0_size = final_df[final_df['SpectralGroup'] == 0].shape[0]
    group_1_size = final_df[final_df['SpectralGroup'] == 1].shape[0]
    print(f"🌗 Spectral partition : Group A ({group_0_size} films) vs Group B ({group_1_size} films)")
    print("═"*80)

    # --- NAMING THE SPECTRAL GROUPS ---
    print("\n🕵️‍♂️ IDENTITY OF SPECTRAL GROUPS (Who are they?)")
    print("─"*60)
    
    for group_id in [0, 1]:
        # We take all the films in this group.
        subset = final_df[final_df['SpectralGroup'] == group_id]
        
        # We count the themes appearing the most in this group.
        top_themes = subset['Theme'].value_counts().head(3)
        
        print(f"\n🔵 GROUP {group_id} ({len(subset)} films) is dominated by:")
        for theme, count in top_themes.items():
            print(f"   - {theme} ({count} films)")

if __name__ == "__main__":
    main()