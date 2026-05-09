#!/bin/bash

# Initialisation de la base de données (Création des tables si absentes)
echo "Vérification et création du schéma de base de données..."
# On utilise python directement pour lancer le module de création
python -m livrable_p12.db.create_db

# Démarrer le Backend en arrière-plan (Port 8000)
echo "Démarrage du Backend..."
uvicorn livrable_p12.backend.main:app --host 0.0.0.0 --port 8000 &

# Attendre que le Backend soit prêt
sleep 5

# Démarrer le Frontend (Port 7860 pour Hugging Face)
echo "Démarrage du Frontend..."
streamlit run src/livrable_p12/frontend/main.py --server.port 7860 --server.address 0.0.0.0