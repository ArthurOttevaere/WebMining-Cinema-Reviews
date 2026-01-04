import pandas as pd
import numpy as np
import os
from tabulate import tabulate 
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx


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

def betweenness_centrality(A: np.ndarray) -> np.ndarray:
    print("   ⏳ Processing Betweenness Centrality (NetworkX)...")
    # 1. Matrix Conversion -> Graph NetworkX
    G = nx.from_numpy_array(A)
    
    # 2. Computing
    bc_dict = nx.betweenness_centrality(G, weight=None, normalized=True)
    
    # 3. Converting into a numpy array (sorted) 
    n = A.shape[0]
    bc_values = np.zeros(n)
    for i in range(n):
        bc_values[i] = bc_dict.get(i, 0.0)
        
    return bc_values


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
    print("   ⏳ Processing L+ (Pseudo-Inverse via Moore-Penrose)...")
    L = laplacian_matrix(A)
    L_plus = np.linalg.pinv(L)
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

def plot_cluster_distance_heatmap(df_nodes, SP, id_map):
    """
    Display the average number of jumps needed between themes.
    """
    print("\n   🎨 Generating Cluster Distance Heatmap...")
    
    # 1. Extarct the themes
    themes = df_nodes['Theme'].unique()
    themes = sorted([t for t in themes if pd.notna(t)])
    
    n_themes = len(themes)
    dist_matrix = np.zeros((n_themes, n_themes))
    
    # 2. Fill in the matrix
    for i, theme_a in enumerate(themes):
        # Ids of the fils from the cluster "A".
        ids_a = df_nodes[df_nodes['Theme'] == theme_a]['Id']
        indices_a = [id_map[x] for x in ids_a if x in id_map]
        
        for j, theme_b in enumerate(themes):
            if i == j:
                dist_matrix[i, j] = 0 # Distance from itself
                continue
                
            # Ids of the fils from the cluster "B".
            ids_b = df_nodes[df_nodes['Theme'] == theme_b]['Id']
            indices_b = [id_map[x] for x in ids_b if x in id_map]
            
            # We take the distances between those 2 groups.
            # We use np.ix_ to extract the sub matrix. 
            sub_sp = SP[np.ix_(indices_a, indices_b)]
            
            # We only keep valid ways (< 90000).
            valid_paths = sub_sp[sub_sp < 90000]
            
            if len(valid_paths) > 0:
                dist_matrix[i, j] = valid_paths.mean()
            else:
                dist_matrix[i, j] = np.nan # No connection 
    
    # 3. Display
    plt.figure(figsize=(12, 10))
    # We shorten the names, only for display reasons 
    short_labels = [f"{t.split(',')[0]}..." for t in themes]
    
    sns.heatmap(
        dist_matrix, 
        annot=True, 
        fmt=".1f", 
        cmap="viridis_r", # Short distance = good -> inverted
        xticklabels=short_labels, 
        yticklabels=short_labels,
        cbar_kws={'label': 'Average Jumps (Shortest Path)'}
    )
    plt.title("Average Distance Between the Clusters")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

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

    print("4️⃣-bis Betweenness Centrality (Freeman)...")
    betweenness = betweenness_centrality(A)

    print("4️⃣ Information Centrality (via Laplacian L+)...")
    L_plus = laplacian_pseudoinverse(A)
    info_centrality = information_centrality(L_plus)

    # --- SPECTRAL CLUSTERING ---
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
        'Betweenness': betweenness,
        'InfoCent': info_centrality,
        'SpectralGroup': communities
    })
    
    final_df = pd.merge(df_nodes, results, on='Id')
    
    # We round up for a better readability
    cols_float = ['Closeness', 'Eccentricity', 'PageRank', 'Betweenness', 'InfoCent']
    final_df[cols_float] = final_df[cols_float].round(4)
    
    # CSV Export
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ CSV exported : {OUTPUT_FILE}")

    # --- CORRELATION MATRIX ---
    print("\n" + "═"*80)
    print("      🧩 META-ANALYSIS : CORRELATION BETWEEN METRICS")
    print("      (Does popularity mean influence ?)")
    print("═"*80)
    
    # Spearman correlation on the ranks, not the values.
    corr_matrix = final_df[['Degree', 'Closeness', 'PageRank', 'Betweenness', 'InfoCent']].corr(method='spearman')
    print(tabulate(corr_matrix, headers='keys', tablefmt='fancy_grid', floatfmt=".2f"))
    
    # --- TABULATE DISPLAY (TOP N PER METRIC) ---
    
    # We choose the top "n" to display.
    TOP_N = 10
    
    # Metrics to analyze.
    metrics_to_show = [
        ('Degree', 'Local influence (Number of direct connections)'),
        ('Closeness', 'Closeness (Ability to reach quickly the network.)'),
        ('Eccentricity', 'Graphic centrality (Distance to the farest point)'),
        ('Betweenness', 'Bridges (Shortest path bottlenecks)'),
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

    # --- CLUSTER HEATMAP ---
    plot_cluster_distance_heatmap(final_df, SP, id_map)
    
    print("\n" + "═"*80)

if __name__ == "__main__":
    main()