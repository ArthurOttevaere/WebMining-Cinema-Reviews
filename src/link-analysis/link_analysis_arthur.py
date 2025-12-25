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
        valid_dists = dists[dists < 90000] # On ignore les infinis
        if len(valid_dists) > 0:
            max_dist = valid_dists.max() # Distance vers le noeud le plus loin
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

INPUT_EDGES = "data/raw/edges_tfidf.csv"
INPUT_NODES = "data/raw/nodes_tfidf.csv"
OUTPUT_FILE = "data/raw/numpy_metrics_results.csv"

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
    
    # --- EXPORT & DISPLAY ---
    
    print("💾 Preparing the results...")
    results = pd.DataFrame({
        'Id': node_ids,
        'Degree': degrees,
        'Closeness': closeness,
        'Eccentricity': eccentricity,
        'PageRank': pagerank,
        'InfoCent': info_centrality 
    })
    
    final_df = pd.merge(df_nodes, results, on='Id')
    
    # We round up for a better readability
    cols_float = ['Closeness', 'Eccentricity', 'PageRank', 'InfoCent']
    final_df[cols_float] = final_df[cols_float].round(4)
    
    # CSV Export
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ CSV exported : {OUTPUT_FILE}")
    
    # --- TABULATE DISPlAY ---
    print("\n" + "="*60)
    print("🏆 TOP 20 FILMS (Ranked by PageRank)")
    print("="*60)
    
    # We select only the interesting columns for the display. 
    cols_to_show = ['Label', 'Theme', 'Degree', 'Closeness', 'Eccentricity', 'PageRank', 'InfoCent']
    
    # We extract top 20
    top_20 = final_df.sort_values(by='PageRank', ascending=False).head(20)[cols_to_show]
    
    # We display with tabulate
    print(tabulate(top_20, headers='keys', tablefmt='fancy_grid', showindex=False))
    print("="*60)

    # Adding the average path of the network
    if 'closeness' in locals():
        print(f"INFO RÉSEAU : Un film est en moyenne à {avg_path:.2f} sauts de n'importe quel autre.")

if __name__ == "__main__":
    main()