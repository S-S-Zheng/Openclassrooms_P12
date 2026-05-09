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
    <li><a href="#overview">Overview</a></li>
    <li><a href="#fonctionnalités">Fonctionnalités</a></li>
    <li><a href="#prérequis">Prérequis</a></li>
    <li>
      <a href="#installation">Installation</a>
      <ul>
        <li><a href="#localement">Configuration Locale</a></li>
        <li><a href="#distance">Configuration Distante</a></li>
      </ul>
    </li>
    <li><a href="#structure-du-projet">Structure du Projet</a></li>
    <li>
      <a href="#lancer-les-services">Lancer les services</a>
      <ul>
        <li><a href="#backend-fastapi-swagger">Backend FastAPI</a></li>
        <li><a href="#frontend-streamlit">Frontend Streamlit</a></li>
        <li><a href="#frontend-and-backend">Lancement Docker</a></li>
      </ul>
    </li>
    <li><a href="#utilisations">Exemples d'utilisation (API)</a></li>
    <li><a href="#deploiement">Déploiement</a></li>
    <li><a href="#license">License</a></li>
  </ol>
</details>

## Overview

**L'objectif est de développer une application web simple et intuitive pour aider nos clients agriculteurs à prendre de meilleures décisions**.

- **Eco-conception et Performance** : L'application utilise le format ONNX, permettant des prédictions ultra-rapides et une consommation de ressources minimale, idéale pour un hébergement green et gratuit.

- **Transparence de l'IA (XAI)** : L'outil ne se contente pas de prédire ; il explique ses décisions en affichant l'importance des variables (via SHAP), permettant à l'agriculteur de comprendre l'influence de la température ou des pesticides sur son rendement.

- **Architecture Robuste**: Conçu selon les principes de l'Architecture Hexagonale, le système est modulaire : on peut changer de base de données ou de modèle d'IA sans réécrire l'application.

## Fonctionnalités

- **Fonction de prédiction** : Permettre à un utilisateur de sélectionner une culture spécifique, de renseigner les conditions de sa parcelle (température, usage de pesticides, etc.) et d'obtenir une estimation chiffrée du rendement attendu.
- **Fonction de recommandation** : L'utilisateur renseigne uniquement les conditions de sa parcelle, et l'application lui recommande la culture la plus rentable en simulant le rendement pour toutes les cultures possibles et en affichant un classement avec une analyse sémantique faite par un LLM.

## Prérequis

- Python 3.12+
- Clef API Mistral (obtenue sur [console.mistral.ai](https://console.mistral.ai/))
- Se créer un espace sur HuggingFace (obtenue sur [huggingface](https://huggingface.co/))
- Se créer une base Supabase (obtenue sur [supabase](https://supabase.com/))
- Se créer une base Postgres locale

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
    # Variables (filtres...)
    ```

4. **Configurer la clé API**

    Toujours dans le `.env`:

    ```bash
    MISTRAL_API_KEY=votre_clé_api_mistral
    HUGGINGFACE_TOKEN=votre_clé_api_huggingface
    POSTGRES_PASSWORD=votre_clé_api_postgres
    SB_PASSWORD=votre_clé_api_supabase
    ```

### Localement

1. **Création d'une base PostgreSQL**

Pour exécuter les tests d'intégration, vous devez disposer d'une instance PostgreSQL locale.

  1. *Installation* :

      ```bash
      sudo apt install postgresql postgresql-contrib
      ```

  2. *Accès au terminal psql* :

      ```bash
      sudo -u postgres psql
      ```

  3. *Initialisation de la DB* :

      ```sql
      CREATE DATABASE ml_test_db;
      CREATE USER test_user WITH PASSWORD 'votre_mot_de_passe';
      GRANT ALL PRIVILEGES ON DATABASE ml_test_db TO test_user;
      ```

### Distance

- **Création d'un dépôt distant GitHub** :

    Créez un compte sur [GitHub](https://github.com/), créez un nouveau dépôt vide et connectez votre projet local :

    ```bash
    git remote add origin https://github.com/votre-user/votre-projet.git
    git push -u origin main
    ```

    Pour les secrets, allez dans Settings > Secrets and variables > Actions pour ajouter vos secrets (HUGGINGFACE_TOKEN, SB_HOST, etc.).

- **Création d'un espace sur Hugging Face**:

    Créez un compte sur Hugging Face, cliquez sur "New Space", choisissez le SDK Docker et un nom pour votre projet.Une fois l'espace créé, dans vos paramètres de profil, créez un "Write Token" pour permettre à GitHub de pousser le code.

    Concernant les secrets, allez dans les paramètres de votre Space, ajoutez les variables d'environnement de votre base de données Supabase pour que l'API puisse s'y connecter au runtime.

- **Création d'une base PostgreSQL Supabase**:

    Créez un compte et un projet sur Supabase puis cliquez sur le bouton Connect sur la barre de tâche supérieure à côté du nom de la base pour récupérez les informations de connexion. A noté que vous pourrez reset votre mot de passe de la base si celui-ci ne vous convient plus dans Project Settings > Database.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Structure du projet

Le projet suit une architecture découplée pour garantir la scalabilité et la maintenabilité :

**Backend (FastAPI)** : API de service exposant les prédictions du modèle XGBoost converti en format ONNX (interopérabilité et performance).

**Frontend (Streamlit)** : Interface utilisateur interactive permettant de simuler des scénarios climatiques et de visualiser l'importance des caractéristiques.

**Core (Architecture Hexagonale)** : Logique métier isolée des frameworks techniques.

```text
.
├── configs
│   └── prompts.yaml                            # Prompts pour le LLM
├── coverage.ini                                # Sources à exclure des tests de couverture
├── datas                                       # Données d'entrée et de sortie
│   ├── raw
│   └── results
│       ├── best_model
│       │   └── best_model.onnx                 # Modele ML
│       └── shap
│           └── shap_metadata.json              # métadonnées issue de SHAP
├── deploy_hf.sh                                # Shell pour déploiement sur HuggingFace
├── docker_start.sh                             # Shell pour init back et front (FastAPI/Streamlit)
├── coverage.ini                                # Configuration du rapport de couverture
├── Dockerfile                                  # Instructions de conteneurisation
├── docker-compose.yml                          # Constructeur du docker
├── .gitignore                                  # Elements à ignorer pour le dépôt GitHub
├── .dockerignore                               # Elements à ignorer pour le/les conteneurs
├── .env.example                                # Exemple mini du .env
├── LICENSE                                     # Licence MIT du projet
├── pytest.ini                                  # Configuration globale de l'env de test
├── README.md                                   # Documentation principale du projet
├── mlflow.db                                   # DB MLFlow
├── mlruns                                      # Résultats MLFlow
├── notebooks                                   # Travaux d'exploration et brouillons
├── poetry.lock                 
├── poetry.toml
├── pyproject.toml                              # Dépendances complètes
├── smoke_test.py                               # Smoke Test
├── src                                         # Src Layout
│   └── livrable_p12                            # Package principal
│       ├── backend                             # --------- Backend ---------
│       │   ├── adapters                        # Implémentations techniques
│       │   │   ├── api                         # Routes et dépendances FastAPI
│       │   │   │   ├── prediction.py           # Logique de prédiction simple
│       │   │   │   └── recommendation.py       # Logique de recommandation
│       │   │   ├── database                    # Client Supabase
│       │   │   │   ├── base.py                 # Base
│       │   │   │   ├── connection.py           # Moteur DB
│       │   │   │   ├── create_db.py            # Instancie/Clean les tables
│       │   │   │   ├── orm.py                  # Schemas des tables
│       │   │   │   └── supabase_repository.py  # Adaptateur Supabase
│       │   │   ├── inference_models            # Modèles d'inférence (ONNX Runtime)
│       │   │   │   └── onnx_predictor.py       # Adaptateur ONNX
│       │   │   └── llm_client                  # Client Mistral (LangChain)
│       │   │       └── mistral_response.py     # Adaptateur Mistral
│       │   ├── core                            # Logique métier (Domaine)
│       │   │   ├── entities                    # Modèles de données purs
│       │   │   │   └── models.py               # Données pures métier
│       │   │   ├── ports                       # Interfaces (Abstractions)
│       │   │   │   ├── llm_response.py         # Port LLM
│       │   │   │   └── yield_predictor.py      # Port ML
│       │   │   └── services
│       │   │       └── advisor.py              # Service de connexion ML-LLM
│       │   ├── main.py                         # Point d'entrée FastAPI
│       │   ├── settings.py                     # Settings du backend
│       │   └── utils
│       └── frontend                            # --------- Frontend ---------
│           ├── adapters
│           │   └── api
│           │       └── requests_api.py         # Fonctions requests FastAPI
│           ├── core
│           │   ├── entities
│           │   │   └── models.py               # Entités
│           │   └── services
│           │       └── crops_list.py           # Service de dynamisation des cultures
│           ├── main.py                         # Configuration de la page et navigation
│           ├── settings.py                     # Settings pour frontend
│           ├── ui                              # UI/UX
│           │   ├── pages
│           │   │   ├── prediction_page.py      # Logique d'affichage de l'onglet Prédiction
│           │   │   └── recommendation_page.py  # Logique d'affichage de l'onglet Recommandation
│           │   └── visuals
│           │       └── visualization.py        # Logique des graphiques
│           └── utils
└── tests                                       # Suite de tests automatisée (Pytest)
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Lancer les services

Par défaut et dans le CI/CD, on a programmé de sorte que le système soit conteneurisé sur HuggingFace cependant, on peut tout à fait les utilisé localement.

### Backend FastAPI Swagger

L'API est motorisée par FastAPI et Uvicorn. Pour démarrer le service :

```bash
# Depuis la racine du projet
uvicorn livrable_p12.backend.main:app --reload
```

Le serveur sera accessible sur `http://localhost:8000`

### Frontend Streamlit

L'interface API Utilisateur se fera sur Streamlit. Pour démarrer le service :

```bash
# Depuis la racine du projet
streamlit run livrable_p12/frontend/main.py
```

Le serveur sera accessible sur `http://localhost:8501`

### Frontend et backend

- **Prérequis**: Docker & Docker Compose, un fichier .env (voir .env.example)
- Pour démarrer l'ensemble de l'écosystème (API + Interface) :

```bash
docker compose up --build
```

[Interface Web](http://localhost:8501)
[Documentation API](http://localhost:8000/docs)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Utilisations

### Prédiction simple

Estime le rendement pour une culture spécifique selon les conditions climatiques.

```bash
curl -X POST "http://localhost:8000/predict?crop=Maize" \
     -H "Content-Type: application/json" \
     -d '{"country": "France", "year": 2024, "rainfall_mm": 1000, "temperature_celcius": 18, "pesticides_tons": 500, "temp_anomaly": 0.5}'
```

### Recommandation

Identifie les k meilleures cultures en terme de rendement à partir de conditions pedoclimatique données et fournit une explication agronomique via LLM.

```bash
curl -X POST "http://localhost:8000/recommend?top_k=3" \
     -H "Content-Type: application/json" \
     -d '{"country": "France", "year": 2024, "rainfall_mm": 1000, "temperature_celcius": 18, "pesticides_tons": 500, "temp_anomaly": 0.5}'
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Deploiement

Le déploiement est entièrement automatisé via une architecture MLOps :

- **Hook pré-commit (filtre local)**:
  Le hook de pré-commit s'exécute automatiquement sur ta machine à chaque tentative de git commit. Son rôle est de s'assurer qu'aucun code "sale" ou mal formaté ne quitte ton poste de travail.

- **CI (GitHub Actions)** :
  Chaque modification push/pull request sur la branche main déclenche automatiquement le pipeline.
  Il est constitué de deux jobs:
  
  *lint* : Vérifie la conformité du code en passant par Ruff.

  *tests* : Ne se lance que si le linting est validé, ce job a pour fonction de tester le comportement unitaire et fonctionnel du code. Il lance un conteneur éphemère PostgreSQL 15 de test et execute pytest en suivant les directives du pytest.ini. Il fera en premier lieu un `Smoke test` pour vérifier que le modèle se charge correctement et est utilisable.

- **CD (GitHub Actions)** :
  Le déploiement ne se lance que si le pipeline de CI a réussi (workflow_run success). Il est restreint à la branche main pour garantir que seul le code de production est déployé.

  Au lieu de pousser tout le dépôt (ce qui serait lourd et risqué), le script sélectionne les fichiers afin de garantir une conteneurisation optimisée, il force aussi le push au sein de HF en créant une branche orpheline à cause de soucis avec les fichiers `LFS`. La construction de HF oblige aussi à mettre front et back ensemble.

  Le script génère aussi dynamiquement un README.md avec un bloc YAML (frontmatter). C'est ce fichier qui configure Hugging Face (SDK Docker, port 7860, version Python, licence).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## License

Distributed under the project_license. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>
