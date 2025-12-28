import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.metrics import silhouette_score
import umap.umap_ as umap
import os

# ==============================================================================
# 1. CHARGEMENT DE LA MATRICE TF-IDF
# ==============================================================================
dossier = r"C:\Users\33778\Desktop\WM (Amine)"
chemin_tfidf = os.path.join(dossier, "matrice_vectorielle_tf-idf.csv")
chemin_sortie = os.path.join(dossier, "resultats_clustering.csv")

print(f"Lecture TF-IDF : {chemin_tfidf}")
tfidf = pd.read_csv(chemin_tfidf, sep=';', index_col=0, encoding='utf-8-sig')
print(f"Matrice TF-IDF : {tfidf.shape}")  # (899, ~14000)

# ==============================================================================
# 2. LSA (SVD)
# ==============================================================================
n_components = 50
print(f"LSA en cours ({n_components} dimensions)...")

svd = TruncatedSVD(n_components=n_components, random_state=42)
X_lsa = svd.fit_transform(tfidf)

print(f"Variance expliquée : {svd.explained_variance_ratio_.sum():.2%}")

# ==============================================================================
# 3. NORMALISATION L2
# ==============================================================================
scaler = Normalizer()
X = scaler.fit_transform(X_lsa)

# ==============================================================================
# 4. CHOIX DU NOMBRE DE CLUSTERS
# ==============================================================================
K_range = range(4, 6)
inerties = []
silhouettes = []

print("\nCalcul Elbow & Silhouette...")
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)

    inerties.append(km.inertia_)
    silhouettes.append(silhouette_score(X, labels, metric="cosine"))

evaluation = pd.DataFrame({
    "K": list(K_range),
    "Inertie": inerties,
    "Silhouette": silhouettes
})
print(evaluation)

# ==============================================================================
# 5. K-MEANS FINAL
# ==============================================================================
K_OPTIMAL = evaluation.loc[evaluation["Silhouette"].idxmax(), "K"]
print(f"\nK optimal retenu : {K_OPTIMAL}")

kmeans = KMeans(n_clusters=int(K_OPTIMAL), random_state=42, n_init=10)
clusters = kmeans.fit_predict(X)

df_resultats = pd.DataFrame(index=tfidf.index)
df_resultats["Groupe"] = clusters
df_resultats.to_csv(chemin_sortie, sep=';', encoding='utf-8-sig')

print(f"Résultats sauvegardés : {chemin_sortie}")

# ==============================================================================
# 6. MOTS DOMINANTS PAR CLUSTER (IMPORTANT)
# ==============================================================================
print("\n--- Mots dominants par cluster ---")

centres_lsa = kmeans.cluster_centers_
centres_tfidf = svd.inverse_transform(centres_lsa)
mots = tfidf.columns

for i in range(int(K_OPTIMAL)):
    top_idx = centres_tfidf[i].argsort()[-10:][::-1]
    top_mots = [mots[j] for j in top_idx]
    taille = (clusters == i).sum()
    print(f"Cluster {i} ({taille} films) : {', '.join(top_mots)}")

# ==============================================================================
# 7. VISUALISATION PCA 2D
# ==============================================================================
pca = PCA(n_components=2)
coords_pca = pca.fit_transform(X)

plt.figure(figsize=(12, 8))
plt.scatter(coords_pca[:, 0], coords_pca[:, 1], c=clusters, cmap="tab10", alpha=0.7)
plt.title(f"PCA – Clustering LSA (K={int(K_OPTIMAL)})")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.grid(alpha=0.3)
plt.show()
