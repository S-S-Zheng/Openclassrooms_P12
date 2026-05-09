#!/bin/bash

# Démarrer le Backend en arrière-plan (Port 8000)
uvicorn livrable_p12.backend.main:app --host 0.0.0.0 --port 8000 &

# Attendre que le Backend soit prêt
sleep 5

# Démarrer le Frontend (Port 7860 pour Hugging Face)
streamlit run src/livrable_p12/frontend/main.py --server.port 7860 --server.address 0.0.0.0