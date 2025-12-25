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

```text
.
├── src/                    # Code source Python
│   ├── scraping            # Scripts de collecte des données (RogerEbert)
│   ├── text-mining         # Scripts de transformation et d'analyse du contenu textuel des critiques
│   ├── link-analysis       # Scripts de construction du graph et d'analyses des liens
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
git clone https://github.com/votre-compte/votre-repo.git
cd votre-repo
pip install -r requirements.txt
```

### 2. Exécution des analyses

Pour répliquer l'analyse complète, exécutez les scripts dans l'ordre suivant :

*Collecte : python src/scraping/scraper.py (Génère le fichier brut).

*Traitement & Graphe : python src/text_mining/generate_gephi_linked.py (Génère les nœuds et les arêtes).

*Analyse des métriques : python src/link_analysis/link_analysis_numpy.py (Calcule les centralités matricielles).
