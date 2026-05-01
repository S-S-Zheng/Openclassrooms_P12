# Projet 12: Concevez un système de recommandations pour une agriculture optimisée par les données

> Ce projet vise à restituer un maximum des compétences techniques acquises au cours de la formation.

<!-- Balise d'en-tête -->
<a id="readme-top"></a>

<!-- PROJET -->
<br />
<div align="center">
  <a href="https://github.com/S-S-Zheng/Openclassrooms_P12.git">
    <!-- <img src="images/logo.png" alt="Logo" width="80" height="80"> -->
  </a>

<h3 align="center">Projet 12: Concevez un système de recommandations pour une agriculture optimisée par les données</h3>

  <p align="center">
    Projet 12 de la formation d'OpenClassrooms: Data scientist Machine Learning (projet débuté le 22/04/2026)
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Sommaire</summary>
  <ol>
    <li><a href="#Overview">Overview</a></li>
    <li><a href="#Partie-1">Partie 1</a></li>
      <ul>
        <li><a href="#Contexte-et-enjeux">Contexte et enjeux</a></li>
        <li><a href="#Plus-value-IA">Plus-value IA</a></li>
        <li><a href="#Approche-technique-et-données-du-PoC">Approche technique et données du PoC</a></li>
        <li><a href="#Critères-de-validation-du-PoC">Critères de validation du PoC</a></li>
        <li><a href="#Timeline-et-resources">Timeline et resources</a></li>
      </ul>
    <li><a href="#Partie-2">Partie 2</a></li>
      <ul>
        <li><a href="#System-Design">System Design</a></li>
        <li><a href="#Timeline-de-livraison-de-loutil-de-recommandation">Timeline de livraison de l'outil de recommandation</a></li>
        <li><a href="#Dimensionnement-coût-projet">Dimensionnement coût projet</a></li>
      </ul>
    <li><a href="#Partie-3">Partie 3</a></li>
      <ul>
        <li><a href="#Anticiper-autour-du-traitement-des-données-personnelles">Anticiper autour du traitement des données personnelles</a></li>
      </ul>
  </ol>
</details>

## Overview

Vous êtes **Data Scientist Machine Learning junior** dans l’entreprise **Agritech Answers**, spécialisée dans l’**optimisation agricole et l’innovation agrotechnologique**.

**L'objectif est de développer une application web simple et intuitive pour aider nos clients agriculteurs à prendre de meilleures décisions**.

## Fonctionnalités

- **Fonction de prédiction** : Permettre à un utilisateur de sélectionner une culture spécifique, de renseigner les conditions de sa parcelle (température, usage de pesticides, etc.) et d'obtenir une estimation chiffrée du rendement attendu.
- **Fonction de recommandation** : L'utilisateur renseigne uniquement les conditions de sa parcelle, et l'application lui recommande la culture la plus rentable en simulant le rendement pour toutes les cultures possibles et en affichant un classement.

## Prérequis

- Python 3.12+
- Clef API Mistral (obtenue sur [console.mistral.ai](https://console.mistral.ai/))
- Créer un compte logfire (eventuellement clef API) : [logfire](https://logfire-eu.pydantic.dev/)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Installation

1. **Cloner le dépôt**

    ```bash
    git clone https://github.com/S-S-Zheng/Openclassrooms_P12.git
    cd Openclassrooms_P12.git
    ```

2. **Créer un environnement virtuel**

    ```bash
    # Création de l'environnement virtuel
    python -m venv venv

    # Activation de l'environnement virtuel
    # Sur Windows
    venv\Scripts\activate
    # Sur macOS/Linux
    source venv/bin/activate
    ```

3. **Installer les dépendances (Poetry)**

    ```bash
    poetry init
    poetry install
    poetry shell

    # Configurer les variables d'environnement
    cp .env.example .env
    # Éditer .env
    # Variables (filtres...) ET MISTRAL_API_KEY
    ```

4. **Configurer la clé API**

    Toujours dans le `.env`:

    ```bash
    MISTRAL_API_KEY=votre_clé_api_mistral
    ```

5. **Logfire quickstart**

    ```bash
    # Install SDK
    poetry add logfire

    # En environnement de dev:
    # Authentification de l'environnement local
    poetry run logfire auth
    # Set up du dossier logfire
    poetry run logfire projects use nom-dossier-logfire

    # En environnement de prod:
    export LOGFIRE_TOKEN='__YOUR_LOGFIRE_WRITE_TOKEN__'
    ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Structure du projet

```text
.
├── notebooks                  # Travaux d'exploration et brouillons
├── datas                      # Données d'entrée et de sortie
├── src/
│   └── livrable_p12/          # Package principal
│       ├── backend
│       │   ├── main.py            # Point d'entrée FastAPI
│       │   ├── core/              # Logique métier (Domaine)
│       │   │   ├── entities/      # Modèles de données purs
│       │   │   └── ports/         # Interfaces (Abstractions)
│       │   ├── adapters/          # Implémentations techniques
│       │   │   ├── api/           # Routes et dépendances FastAPI
│       │   │   ├── repository/    # Client Supabase
│       │   │   ├── inference_models/    # Modèles d'inférence (ONNX Runtime)
│       │   │   └── llm_client/    # Client Mistral (LangChain)
│       │   └── infrastructure/    # Configuration Logfire & Environnement
│       └── frontend
│           ├── main.py                # Configuration de la page et navigation (tabs)
│           ├── modules/
│           │   ├── ui_prediction.py   # Logique d'affichage de l'onglet Prédiction
│           │   ├── ui_recommandation.py # Logique d'affichage de l'onglet Recommandation
│           │   └── components.py      # Composants réutilisables (headers, footers, tooltips)
│           └── utils/
│               ├── api_client.py      # Fonctions requests FastAPI (gestion erreur 404/500)
│               └── data_processors.py # Formatage des données API pour Plotly
├── tests/                     # Suite de tests automatisée (Pytest)
├── coverage.ini               # Configuration du rapport de couverture
├── Dockerfile                 # Instructions de conteneurisation
├── docker-compose.yml         # Constructeur du docker
├── .gitignore                 # Elements à ignorer pour le dépôt GitHub
├── .dockerignore              # Elements à ignorer pour le/les conteneurs
├── .env.example               # Exemple mini du .env
├── LICENSE                    # Licence MIT du projet
├── pytest.ini                 # Configuration globale de l'env de test
├── README.md                  # Documentation principale du projet
└── pyproject.toml             # Dépendances complètes
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- 
## Utilisation

### Ajouter des documents

Placez vos documents dans le dossier `datas/raw/` à la racine.

### Indexer les documents

Exécutez le script ``ìndexer.py`` pour traiter les documents et créer l'index FAISS :

```bash
# En vous plaçant à la racine du projet
python CLI/indexer.py
```

Le workflow du script est le suivant :

1. *Initialisation et Récupération des Arguments*
    Le script démarre en récupérant les paramètres d'entrée via argparse :
    - Le répertoire source des documents (--input-dir).
    - Une éventuelle URL de téléchargement (--data-url).
    - Configuration du logging pour suivre l'avancement en temps réel.

2. *Extraction de la Donnée (Extraction)*
    Le script gère l'arrivée des fichiers :
    - Si une URL est fournie, le fichier est téléchargé, dézippé et stocké dans le répertoire d'entrée.
    - Si aucune URL n'est fournie : Le script bascule sur l'utilisation des fichiers locaux déjà présents dans le dossier.

3. *Parsing et Gestion du Cache (Transformation)*
    Vérification du cache (évite de refaire l'OCR). Si le cache est vide, lit et parse les données non-structurées, les transforme en "documents" bruts et une copie est sauvegardée au cas où.

4. *Nettoyage Sémantique (Processing)*
    Une fois le texte extrait, il est nettoyé pour améliorer la qualité différents patterns détéctés et ``blacklist.txt`` pour supprimer le bruit (mentions inutiles, headers répétitifs, mots spécifiques). Les documents sont "titrés" ou restructurés pour que l'index contienne une donnée sémantiquement riche.

5. *Construction de l'Index Vectoriel (Loading)*
    Le script appel ensuite ``VectorStoreManager``, pour vectoriser, créer l'index et persister sur le disque.

### Remplir la base de données

Exécutez le script ``load_excel_to_db.py`` pour remplir la base de données.

```bash
# En vous plaçant à la racine du projet
python CLI/load_excel_to_db.py
```

Workflow du script :

1. *Initialisation et Reset de la Base*
    Le script appelle ``init_db(reset_tables=True)`` pour supprimer les anciennes tables et recréer un schéma SQL vierge, garantissant l'idempotence du processus.

2. *Extraction et Nettoyage (Pandas)*
    Lecture des feuilles Excel. Le script nettoie les en-têtes, supprime les colonnes "fantômes" (Unnamed) et applique un strip() sur les données textuelles.

3. *Validation et Normalisation (Pydantic)*
    Chaque ligne est validée par NBAInputSchema ou TeamInputSchema. Cette étape assure l'intégrité des types et gère la conversion des valeurs (ex: NaN vers None).

4. *Ingestion et agregation des tables*
    Remplis les tables ``Player``, ``Stat``, ``Team`` et réalise certaine agrégation.

5. *Validation de la Transaction*
    Le script effectue un ``commit()`` final pour persister les données. En cas d'erreur, un ``rollback()`` est déclenché pour prévenir toute corruption de la base.

### Lancer l'évaluation RAGAS

Exécutez le script ``evaluate_ragas.py`` pour lancer une évaluation RAGAS du RAG.

```bash
# En vous plaçant à la racine du projet
python CLI/evaluate_ragas.py
```

L'évaluateur va regarder les métriques suivantes:

- *Faithfulness (Fidélité)*
    Mesure si la réponse de l'IA est factuellement soutenue par les documents extraits (détection d'hallucinations).

- *Answer Relevancy (Pertinence)*
    Évalue si la réponse reste pertinente par rapport à la question posée.

- *Context Recall (Rappel du Contexte)*
    Vérifie si toutes les informations nécessaires pour répondre (définies dans la "Ground Truth") sont bien présentes dans les documents extraits de la base vectorielle.

Workflow du script :

1. *Initialisation de l'Agent*
    Le script encapsule l'Agent NBA dans un ``RAGPrototypeWrapper`` pour isoler les composants de recherche (contexte) et de génération (réponse).

2. *Inférence sur Dataset de Test*
    Charge un fichier JSON de Q\&A ``qa_pairs.json``. L'agent traite chaque question pour générer une réponse réelle et extraire les contextes associés.

3. *Configuration du Juge LLM*
    Utilisation de Mistral-Small comme "Juge" pour attribuer des scores.

4. *Calcul des Scores*
    Le script itère sur le dataset, calcule les métriques et gère les pauses (async sleep) pour respecter les limites de l'API (Rate Limiting).

5. *Rapport et Sauvegarde*
    Génère un scoring en std et une sauvegarde des résultats dans ``ragas.json`` pour analyse ultérieure.

### Lancer l'application

```bash
# A la racine du projet
streamlit run src/livrable_p10/app/main.py
```

L'application sera accessible à l'adresse <http://localhost:8501> dans votre navigateur.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Modules principaux

### Orchestration

``src/livrable_p10/app/agents/nba_agent.py``

Cerveau de l'application basé sur Pydantic AI :

1. *Aiguillage des outils*
    Détermine s'il doit utiliser l'outil SQL ``ask_database`` ou l'outil Sémantique ``ask_index``

2. *Mémoire court terme*
    Gère l'historique de la conversation pour le contexte.

### Outil SQL

``src/livrable_p10/app/tools/sql/sql_tool.py & sql_pipeline.py``

Transforme le langage naturel en requêtes complexes :

1. *NLP -> SQL*
    Traduit les questions en requêtes SQLite via Mistral.

2. *Sécurité \& Nettoyage*
    Mode de lecture seule, validation pydantic, limite les instructions.

3. *Monitoring*
    Remplit la table ``Report`` pour suivre les requêtes.

4. *Exécution*
    Récupère les statistiques brutes (points, rebonds, victoires) en base.

### Outil sémantique

``src/livrable_p10/app/tools/semantic/vector_store.py``

Gère toutes les fonctionnalités liées à l'index :

1. *Charge et chunk*
    A l'instanciation de ``VectoreStoreManager``, va tenter de charger et chunker les documents brutes.

2. *Embedding et persistence*
    Utilise un modèle HuggingFace local pour transformer le texte en vecteurs. ``build_index()`` va transformer en objet document puis splitter, emmbeder, et sauvegarder les vecteurs générés par les documents.

3. *Recherche*
    Effectue une recherche de similarité pour extraire les passages les plus pertinents via sa méthode ``search()``.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Personnalisation

Vous pouvez personnaliser l'application en modifiant les paramètres dans `config.py` :

- Modèles Mistral utilisés
- Des hyperparamètres de FAISS et du LLM comme la température, le chunking, le nombre de contexte à fournir au LLM
- Les chemins de lecture et de persistence

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## License

Distributed under the project_license. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p> -->
