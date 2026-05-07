"""
Point d'entrée principal de l'application FastAPI.

Ce module assemble les différents composants de l'architecture :
1. Orchestre le cycle de vie de l'application (Lifespan) pour le chargement du modèle.
2. Centralise l'inclusion des routeurs.
3. Définit les endpoints de base comme la vérification de l'état (Healthcheck).

Notes:
-------
L'utilisation de ``@asynccontextmanager`` + ``lifespan=`` (FastAPI ≥ 0.93) permettent:
* De charger entièrement les ressources avant que le serveur ne se mettent a accepter des requetes.
* Les ressources occupées par le service sont correctement déchargées à l'extinction du serveur.
"""

# Imports
import logging

# import logging.config
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import APIRouter, FastAPI
from fastapi.responses import RedirectResponse

from livrable_p12.backend.settings import get_settings

# from app.api.routes.ask import router as ask_router
# from app.api.routes.rebuild import router as rebuild_router
# from app.rag.rag_pipeline import EventRAGPipeline

# ======================= Logging configuration =======================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ======================= Lifespan ===============================================
# assynccontextmanager est un décorateur qui permet de définir une fonction
# capable de gérer une phase avant de démarrage et une après d'arrêt.
# Ici, tout ce qui est écrit avant yield s'éxé UNE SEULE FOIS au lancement
# du serveur ce qui permet de maintenit l'état tant que le serveur est en ON
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Gère le cycle de vie de l'application (Démarrage et Arrêt).

    Tout le code situé avant l'instruction 'yield' est exécuté une seule fois
    lors du lancement du serveur. Cela permet d'initialiser les ressources lourdes
    et de les maintenir en mémoire RAM tout au long de la session.

    Actions au démarrage :
        - Cache les settings
        - Instancie les ressources lourdes (ML et données)
        - Injection de ``settings`` et ``predictor`` dans ``app.state`` pour un accès global
            via les requêtes.

    Args:
        app (FastAPI): L'instance de l'application.
    """

    # ============== Phase de démarrage ================
    logger.info("Démarrage de l'API AgriTech - Chargement des ressources...")

    # Cache singleton
    settings = get_settings()
    # Instanciation du modele ML
    # Instanciation du client LLM

    # Stockage settings et ML et LLM
    app.state.settings = settings

    yield  # le serveur accepte les requêtes à partir d'ici

    # ================== Phase d'arrêt =================
    logger.info("Arrêt du serveur, nettoyage...")


# ================= Montage des Routers ==================================
# Instancie un router spécific pour les routes par défaut
generic_router = APIRouter()


# /health
# Test auto CI/CD, debug rapide
# FONDAMENTAL + NE DOIT JAMAIS DEPENDRE DE QUOIQUE CE SOIT
@generic_router.get("/health", tags=["Health"])
async def healthcheck():
    """
    Vérifie la disponibilité opérationnelle du service.\n
    Ce endpoint est crucial pour les outils de monitoring.
    Il doit rester indépendant des ressources externes pour
    isoler les pannes réseau/modèle de la panne serveur.

    Returns:
        dict: Un dictionnaire indiquant le statut opérationnel.
    """
    return {"status": "ok"}


# / (root)
# Feedback immédiat, debug, UX minimale
@generic_router.get("/", tags=["Root"], include_in_schema=False)
async def root():
    """
    Point d'entrée racine.\n
    Redirige automatiquement l'utilisateur vers la documentation Swagger
    interactive (/docs) pour faciliter l'exploration de l'API.

    Returns:
        RedirectResponse: Redirection vers l'interface utilisateur Swagger.
    """
    return RedirectResponse(url="/docs")


# ======================= API factory ==============================================
# factory `create_app()` plutôt que `app = FastAPI()` directement car les tests appellent
# `create_app()` pour créer une instance nouvelle de l'application pour chaque test.
# Sans factory, tous les tests partageraient la même instance.
def create_app() -> FastAPI:
    """
    Créée et configure l'app FastAPI.

    Returns
    -------
    FastAPI
    """
    settings = get_settings()  # noqa: F841

    app = FastAPI(
        title="AgriTech Yield Prediction API",
        description=("API de prédiction et de recommandation agricole"),
        version="1.0.0",
        # Si on avait un front-end d'actif:
        # docs_url="/docs" if settings.app_env != "production" else None,
        # redoc_url="/redoc" if settings.app_env != "production" else None,
        swagger_ui_parameters={
            "persistAuthorization": False
        },  # Ne garde pas clé /rebuild en mémoire
        lifespan=lifespan,
    )

    # Applique la configuration des routes
    app.include_router(generic_router)
    # app.include_router(ask_router)
    # app.include_router(rebuild_router)

    return app


# ===========================================================================


# Module-level app instance used by uvicorn / gunicorn
app = create_app()
