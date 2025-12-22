import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import os


# CONFIGURATION
#dossier = r"C:\Users\33778\Desktop\WM (Amine)"
#chemin_entree = os.path.join(dossier, "matrice_vectorielle_tf-idf.csv")  # Ta matrice TF-IDF
#chemin_sortie = os.path.join(dossier, "matrice_similarite.csv")

# We go up to the file root 
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

chemin_entree = os.path.join(BASE_DIR, "data", "raw", "vector_matrix_tf-idf.csv")
chemin_sortie = os.path.join(BASE_DIR, "data", "raw", "similarity_matrix.csv")

print(f"📂 Lecture de : {chemin_entree}")
print(f"💾 Sauvegarde prévue dans : {chemin_sortie}")

print(f"Lecture du fichier : {chemin_entree}")

# 1. CHARGEMENT DE LA MATRICE TF-IDF
print(f"Lecture de la matrice : {chemin_entree}")
if not os.path.exists(chemin_entree):
    print("❌ ERREUR : Fichier introuvable. Lance Vectorisation.py d'abord.")
    exit()

# index_col=0 est TRES important : cela dit que la première colonne (Noms des films)
# est l'étiquette des lignes, pas une donnée mathématique.
try:
    tf_idf = pd.read_csv(chemin_entree, sep=';', index_col=0, encoding='utf-8-sig')
except:
    tf_idf = pd.read_csv(chemin_entree, sep=';', index_col=0, encoding='cp1252')

print(f"--> Matrice chargée : {tf_idf.shape}")


# CALCUL DE LA SIMILARITÉ (COSINUS)
print("Calcul de la similarité Cosinus (Cela compare chaque film avec tous les autres)...")

# Le calcul magique de sklearn
similarity_matrix = cosine_similarity(tf_idf)

# On remet ça dans un joli tableau avec les titres en lignes et en colonnes
similarity_df = pd.DataFrame(similarity_matrix, index=tf_idf.index, columns=tf_idf.index)

# Sauvegarde du résultat complet
print(f"Sauvegarde de la matrice complète dans : {chemin_sortie}")
similarity_df.to_csv(chemin_sortie, sep=';', encoding='utf-8-sig')


# VISUALISATION

print("\nGénération du graphique (Heatmap)...")


def plot_similarity_matrix(similarity_df, nb_films=20):
    #On ne prend que les 'nb_films' premiers pour que le graphique soit lisible
    
    subset_df = similarity_df.iloc[:nb_films, :nb_films]

    plt.figure(figsize=(10, 8))  # Taille de l'image
    plt.imshow(subset_df, interpolation='nearest', cmap='viridis')  # 'viridis' ou 'hot' ou 'Blues'
    plt.colorbar(label='Similarité Cosinus (0=Différent, 1=Identique)')
    plt.title(f'Matrice de Similarité (Zoom sur les {nb_films} premiers films)')

    # Gestion des étiquettes (Titres des films)
    plt.xticks(ticks=range(len(subset_df.columns)), labels=subset_df.columns, rotation=90, fontsize=8)
    plt.yticks(ticks=range(len(subset_df.index)), labels=subset_df.index, fontsize=8)

    plt.xlabel('Films')
    plt.ylabel('Films')

    # Affichage des valeurs dans les cases
    # On le fait uniquement parce qu'on a réduit à 20 films.
    for i in range(len(subset_df)):
        for j in range(len(subset_df)):
            valeur = subset_df.iloc[i, j]
            # On change la couleur du texte selon que la case est foncée ou claire
            couleur_texte = 'white' if valeur < 0.7 else 'black'
            plt.text(j, i, f"{valeur:.2f}", ha='center', va='center', color=couleur_texte, fontsize=7)

    plt.tight_layout()
    plt.show()


# Lancer l'affichage sur un échantillon de 20 films
plot_similarity_matrix(similarity_df, nb_films=20)


# PETIT TEST : TROUVER LES JUMEAUX

premier_film = similarity_df.index[0]
print(f"\n--- Test : Qui ressemble le plus à '{premier_film}' ? ---")

# On trie les scores du plus grand au plus petit
similaires = similarity_df[premier_film].sort_values(ascending=False)

# On affiche les 5 premiers
print(similaires.head(5))