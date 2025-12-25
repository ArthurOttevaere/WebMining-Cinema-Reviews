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
1.**Collecte de données (Scraping) :** Récupération automatique de corpus massifs (textes, notes, métadonnées, casting).
2.**Text Mining :** A compléter quand nous arriverons à cette étape.
3.**Link Analysis :** A compléter quand nous arriverons à cette étape.

---

## 📂 Structure du Projet

L'architecture respecte la séparation entre code source, données brutes et résultats. Dans le but de faciliter la réplication des analyses.

-data/ : Contient les datasets (bruts et traités)
-src/ :
    - scraping/ : Scripts de collecte de données.
    - text-mining/ : Scripts de prétraitement, TF-IDF et clustering.
    - link-analysis/ : Construction du graphe, implémentation matricielle et analyse des liens.
-results/ : Résultats exportés (CSV, screenshots, etc.).
-requirements.txt : Listes des dépendances Python nécessaires.

```text
.
├── src/                    # Code source Python
│   ├── scraping-lwlies.py  # Script de collecte pour Little White Lies
│   ├── scraping_amine.py   # Script de collecte pour [Site 2]
│   ├── scraping_lenny.py   # Script de collecte pour [Site 3]
│   └── 2_data_prep.py      # (À venir) Script de fusion et nettoyage
├── data/
│   ├── raw/                # Données brutes issues du scraping (.csv/.xlsx)
│   │                       # Note : Ces fichiers ne sont pas versionnés sur GitHub
│   └── processed/          # Données nettoyées prêtes pour l'analyse
├── results/                # Graphiques, visualisations et rapports
├── .gitignore              # Configuration des fichiers exclus (env, données lourdes)
├── requirements.txt        # Liste des dépendances Python
└── README.md               # Documentation du projet
