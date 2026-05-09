# --- Étape 1 : Build (Installation des dépendances) ---
FROM python:3.12-slim AS builder

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=true\
    POETRY_VIRTUALENVS_CREATE=true

# Installation des dépendances système nécessaires à la compilation (gcc, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential && rm -rf /var/lib/apt/lists/*

# Installation de Poetry 2.0
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# Copie des fichiers de config uniquement (pour le cache Docker)
COPY pyproject.toml poetry.lock ./ 
# Installation dépendances
RUN poetry install --only main --no-root

# --- Étape 2 : Runtime (Image finale légère) ---
FROM python:3.12-slim AS runtime

# On définit le dossier de travail à la racine de l'application
WORKDIR /app
# Installation des dépendances système pour l'exécution (copie du builder)
COPY --from=builder /app/.venv /app/.venv

# On propage le PYTHONPATH pour que livrable_p12 soit reconnu
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="app/src" \
    PYTHONUNBUFFERED=1

# Copie des élements
COPY src/ ./src/
COPY datas/results/best_model/best_model.onnx ./datas/results/best_model/
COPY datas/results/shap/shap_metadata.json ./datas/results/shap/
COPY configs/prompts.yaml ./configs/
COPY docker_start.sh ./

# Droits pour Hugging Face (User 1000)
RUN useradd -m -u 1000 agriuser && chown -R agriuser:agriuser /app
RUN chmod +x docker_start.sh
# utilisateur non-root pour la sécurité
USER agriuser

# Port par défaut pour HF
EXPOSE 7860

# ---------------- EN LOCAL ----------------
# # Commande de lancement
# CMD python -m livrable_p12.db.create_db && \
#     python -m livrable_p12.db.import_dataset_to_db && \
#     uvicorn livrable_p12.main:app --host 0.0.0.0 --port 7860
# On ne garde que le lancement de l'API.
# ou
# CMD ["uvicorn", "livrable_p12.main:app", "--host", "0.0.0.0", "--port", "8000"]
# ---------------- AVEC SUPABASE ---------------------
# L'initialisation de la DB se fait une seule fois manuellement ou via une migration.
# CMD ["uvicorn", "livrable_p12.main:app", "--host", "0.0.0.0", "--port", "7860"]
# Script de lancement
CMD ["./docker_start.sh"]