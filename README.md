# 🎬 Projet Web Mining : Analyse des Critiques de Cinéma

**Cours :** MLSMM2153 - Web Mining (2025-2026)  
**Professeurs :** Corentin Vande Kerckhove & Sylvain Courtain  
**Sujet 4 :** Analyse des critiques culturelles sur des blogs (Cinéma)

---

## 👥 L'Équipe

* **Arthur Ottevaere**
* **Mohamed Amine El Mohicine**
* **Lenny Andry**

---

## 📖 Contexte et Objectifs

Ce projet a pour but d'analyser les critiques cinématographiques se trouvant sur des blogs en ligne. Dans ce projet, nous collectons et analysons de nombreuses critiques provenant de trois blogs cinématographiques anglophones distincts afin d'identifier des tendances sémantiques et structurelles.

Le projet suit le même cheminement que le cours de Web Mining, à savoir :

1. **Collecte de données (Scraping) :** Récupération automatique de corpus massifs (textes, notes, métadonnées, casting).
2. **Text Mining :** A compléter quand nous arriverons à cette étape.
3. **Link Analysis :** A compléter quand nous arriverons à cette étape.

---

## 📂 Structure du Projet

L'architecture respecte la séparation entre code source, données brutes et résultats. Dans le but de faciliter la réplication des analyses.

```text
.
├── src/                    # Code source Python
│   ├── scraping            # Scripts de collecte des données (RogerEbert)
│   ├── text_mining         # Scripts de transformation et d'analyse du contenu textuel des critiques
│   ├── link_analysis       # Scripts de construction du graph et d'analyses des liens
│
├── data/
│   ├── raw/                # Données brutes issues du scraping, text-mining et link-analysis (.csv/.xlsx)
│   │                       # Note : Ces fichiers ne sont pas versionnés sur GitHub (via .gitignore)
│   └── processed/          # Données nettoyées prêtes pour l'analyse
│
├── results/                # Graphiques, visualisations et rapports
├── .gitignore              # Configuration des fichiers exclus (env, données lourdes)
├── requirements.txt        # Liste des dépendances Python nécessaires
└── README.md               # Documentation du projet
```

---

## 🚀 Guide d'Utilisation (Pipeline)

### 1. Installation

Assurez-vous d'avoir Python 3.9+ installé. Clonez le repo et installez les dépendances :

```Bash
git clone https://github.com/ArthurOttevaere/WebMining-Cinema-Reviews.git
cd WebMining-Cinema-Reviews
pip install -r requirements.txt
```

### 2. Exécution des analyses

L'ensemble du pipeline (Scraping, Text mining et Link analysis) est orchestrée par un scrpit unique afin d'assurer une meilleure réplicabilité. Alors, pour lancer l'analyse complète, il suffit d'entrer la commande suivante dans votre terminal :

```Bash
python main.py
```

Ce script exécute, en arrière plan, les étapes suivantes :

* **Chargement des données :** Par défaut, le script charge le dataset fourni `data/processed/reviews_final_900.csv` pour éviter une nouvelle collecte longue des données. Cela permet également d'obtenir les mêmes résultats que ceux illustrés dans le rapport et dans l'ensemble de l'analyse.

* **Text mining :** Nettoyage, vectorisation TF-IDF et clustering des critiques cinématographiques. Des visuels relatifs à l'analyse sémantique apparaitront au lancement du code.

* **Construction du graphe :** Génère des noeuds et des arrêtes sur base de la similarité cosinus. Ces "Nodes" et "Edges" sont directement calculées via le corpus de données scrapé (`data/processed/reviews_final_900.csv)`.

* **Link analysis :** Calcul des métriques avancées (Centralité, PareRank, etc.).

### **⚠️ Note importante concernant le Scraping (`RUN_SCRAPER = False`)**

Par défaut, la collecte de nouvelles données est désactivée pour garantir la **stricte réplicabilité des résultats** présentés dans notre rapport.

Bien que le module de scraping soit complet et fonctionnel (importé via `src.scraping`), nous vous recommandons vivement de **ne pas passer cette variable à `True`**, car :

1. **Cohérence :** Le site *RogerEbert.com* étant dynamique, une nouvelle collecte modifierait le corpus. Les clusters et métriques de graphe divergeraient alors de ceux analysés dans le PDF rendu.

2. **Performance :** L'analyse s'exécute ici instantanément sur le jeu de données figé (`reviews_final_900.csv`), alors qu'un nouveau scraping prendrait un temps plus conséquent.

Le code de scraping est inclus dans le projet à des fins de démonstration méthodologique et de vérification technique uniquement.

---

## 🧠 Méthodologie et Concepts Clés

### Text Mining

La phase de text mining repose sur un pipeline complet de traitement linguistique et de modélisation vectorielle appliqué aux critiques collectées. Après un nettoyage systématique du texte, les critiques ont été tokenisées, lemmatisées et filtrées à l’aide de critères linguistiques et statistiques (stopwords, noms propres, fréquence documentaire). Le corpus ainsi normalisé a été représenté sous forme de vecteurs TF-IDF intégrant unigrams et bigrams, puis soumis à une réduction dimensionnelle par SVD et à une normalisation L2. Cette représentation permet de mesurer efficacement la similarité sémantique entre critiques via la similarité cosinus.

### Link Analysis (Approche Matricielle)

Contrairement aux approches classiques utilisant des librairies haut niveau, nous avons implémenté les mesures de centralité via les concepts d'algèbre linéaire et de calcul matriciel, tout deux abordés lors des cours théoriques :

* **Centralité de Degré :** Calculée via la matrice d'adjacence.

* **PageRank :** Implémenté par la méthode des puissances (Power Iteration).

* **Information Centrality :** Calculée à partir de la Pseudo-Inverse du Laplacien (L +) pour identifier les nœuds ponts.

* **Closeness, Eccentricity & Shortest Path :** Basés sur l'algorithme de Floyd-Warshall.

* **Diamètre et rayon du graphe :** Calculés sur base de ..., ils nous renseigne sur la santé globale du graphe.

* **Partitionnement spectral :** Grâce à une coupe du graphe en deux, il renseigne sur la cohésion interne des groupes, relativement à leur dissociation les uns des autres.

---

## 📊 Aperçu des Résultats

### Visualisation Gephi

![/Users/arthurottevaere/Downloads/605446336_1517578152628690_6745418421955372632_n.png]
*Légende* : Les couleurs représentent les thèmes (Clusters) identifiés par TF-IDF.

### Top Films (Link Analysis)

Voici un extrait des films les plus influents identifiés par nos algorithmes : Mettre ici une capture d'écran ou un petit tableau du rendu Tabulate.
