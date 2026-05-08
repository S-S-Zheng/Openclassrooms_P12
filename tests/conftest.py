"""
Script pour initier mocks et fixtures pytest pour l'ensemble de la suite de tests.

Toutes les entrées/sorties externes (API Mistral, E/S)
sont remplacées par des bouchons (stubs) légers afin que les tests s'exécutent
hors-ligne, rapidement et de manière déterministe.
"""

# imports
# Permet d'écrire des indices de type (ex: list[Document]) même si la classe n'est pas encore
# totalement définie, améliorant la compatibilité.
from __future__ import annotations

from unittest.mock import MagicMock

# patch: Remplace temporairement une classe ou une fonction par un mock.
# MagicMock: Un objet "caméléon" qui accepte n'importe quel appel de méthode.
# AsyncMock: La version asynchrone pour simuler, par exemple, une base de données ou un client HTTP.
import pytest

# import pytest_asyncio # permet d'écrire des tests qui peuvent "attendre" (await) des réponses
# Client spécialisé qui simule des requêtes HTTP sur FastAPI sans devoir lancer un vrai serveur Web.
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker

from livrable_p12.backend.adapters.database.base import Base
from livrable_p12.backend.adapters.database.connection import get_db
from livrable_p12.backend.core.entities.models import (
    CropYieldContext,
    FeatureImportance,
    PredictionResult,
    YieldResponse,
)
from livrable_p12.backend.main import app, create_app

# --------------------
from livrable_p12.frontend.adapters.api.requests_api import RequestAgriAPIAdapter
from livrable_p12.frontend.core.entities.models import AgriResult
from livrable_p12.frontend.core.entities.models import FeatureImportance as FeatureImportanceFront
from livrable_p12.frontend.core.entities.models import PredictionResult as PredictionResultFront


# ---------------------------- DATAS ----------------------------
@pytest.fixture
def predict_context():
    """Fournit un contexte de données valide pour la recommandation"""
    return CropYieldContext(
        country="France",
        crop="Wheat",
        year=2026,
        rainfall_mm=500.0,
        temperature_celcius=15.0,
        pesticides_tons=10.0,
        temp_anomaly=1.5,
    )


@pytest.fixture
def recommendation_context():
    """Fournit un contexte de données valide pour la recommandation"""
    return CropYieldContext(
        country="France",
        crop=None,
        year=2026,
        rainfall_mm=500.0,
        temperature_celcius=15.0,
        pesticides_tons=10.0,
        temp_anomaly=1.5,
    )


@pytest.fixture
def mock_yield_response():
    """Simule la sortie du modèle ML."""
    return YieldResponse(
        primary_prediction=PredictionResult(crop="Potatoes", yield_val=15.5, unit="tons/ha"),
        recommendations=[
            PredictionResult(crop="Potatoes", yield_val=17.4),
            PredictionResult(crop="Wheat", yield_val=6.2),
        ],
        top_features=[
            FeatureImportance(feature="crop", impact=5.0),
            FeatureImportance(feature="temperature_celsius", impact=1.5),
        ],
        llm_analysis="Analyse des résultats ML par le LLM test.",
    )


# ---------------------------- API ----------------------------
@pytest.fixture
def client(db_session_for_tests):
    """
    Fournit un TestClient FastAPI configuré avec une session de base de données de test.
    Gère le cycle de vie (lifespan) de l'application pour chaque test.
    """
    """
    Fournit un client HTTP configuré pour tester les endpoints de l'API.

    Cette fixture utilise le 'TestClient' de FastAPI au sein d'un gestionnaire
    de contexte afin de déclencher les événements 'lifespan' (startup/shutdown).
    Cela permet notamment de charger en mémoire ce qui est nécéssaire avant l'exécution
    des tests.

    Yields:
        TestClient: Une instance de client capable d'effectuer des requêtes (GET, POST, etc.).
    """
    app = create_app()
    # override DIRECTEMENT sur cette instance
    app.dependency_overrides[get_db] = lambda: db_session_for_tests

    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


# ---------------------------- DB ----------------------------
DATABASE_URL_TEST = "postgresql+psycopg2://postgres:12345@localhost:5432/test_db"


@pytest.fixture(scope="session")
def test_engine():
    """
    Fixture de session : Crée les tables au démarrage et les supprime à la fin de la session.
    Utilisation automatique de la fixture.
    """
    engine = create_engine(DATABASE_URL_TEST)
    try:
        # On vérifie si on peut se connecter avant de tenter le create_all
        with engine.connect():
            pass
        Base.metadata.create_all(bind=engine)
        yield engine  # On renvoi l'objet engine pour les autres fixtures
        # Vide la base db à la fin de la session != rollback()
        Base.metadata.drop_all(bind=engine)
    except OperationalError:
        # pytest.skip("Base de données de test non disponible")
        # On ne fait rien : les tests d'intégration DB échoueront d'eux-mêmes
        # mais les tests unitaires et de routes mockées passeront !
        yield


@pytest.fixture
def db_session_for_tests(test_engine):
    """
    Fournit une session de base de données isolée pour chaque test.
    Utilise une transaction SQL pour effectuer un rollback systématique à la fin
    du test, garantissant une base propre pour le test suivant.
    """
    connection = test_engine.connect()
    transaction = connection.begin()
    Session = sessionmaker(autocommit=False, autoflush=False, bind=connection)
    test_session = Session()

    yield test_session

    test_session.close()
    # Annule l'insertion pour le test suivant plus efficace de delete
    if transaction.is_active:
        transaction.rollback()
    connection.close()


# Evite aux routes d'utiliser la session normale plutot que le test à cause de Depend(get_db)
# ==> les client.post, get etc seront automatiquement envoyées vers le test.
@pytest.fixture(autouse=True)
def override_db(db_session_for_tests):
    """
    Assure l'isolation systématique de la base de données pour l'application
    ==> Remplace la dépendance get_db par la session de test

    En utilisant 'autouse=True', cette fixture garantit que n'importe quelle route
    faisant appel à la dépendance 'get_db' recevra la session de test en cours,
    évitant ainsi toute écriture accidentelle dans la base de production ou de dev.

    Args:
        db_session_for_tests: La session SQLAlchemy transactionnelle définie plus haut.
    """
    app.dependency_overrides[get_db] = lambda: db_session_for_tests
    yield
    app.dependency_overrides.clear()


# Session db qui saute pendant une transaction
@pytest.fixture
def db_session_broken_for_tests(db_session_for_tests):
    """
    Simule une panne de base de données (OperationalError).
    Utilisé pour tester la robustesse des rollbacks et la gestion des erreurs API.
    """
    # Le flush() entrainera un crash
    db_session_for_tests.flush = MagicMock(
        side_effect=OperationalError("Unexpected Crash", params=None, orig=None)  # type:ignore
    )
    # On "espionne" le rollback
    db_session_for_tests.rollback = MagicMock(wraps=db_session_for_tests.rollback)
    return db_session_for_tests


# ================================================================================
# ================================ FRONT END ================================================
# ================================================================================

# ---------------------------- DATAS ----------------------------

# ---------------------------- API ----------------------------
BASE_URL_TEST = "http://api_test:8000"


@pytest.fixture
def request_adapter():
    return RequestAgriAPIAdapter(base_url=BASE_URL_TEST)


@pytest.fixture
def sample_agri_result():
    return AgriResult(
        primary_prediction=PredictionResultFront(crop="Wheat", yield_val=25.0, unit="tons/ha"),
        recommendations=[
            PredictionResultFront(crop="Potatoes", yield_val=30.0, unit="tons/ha"),
            PredictionResultFront(crop="Wheat", yield_val=20.0, unit="tons/ha"),
            PredictionResultFront(crop="Maize", yield_val=10.0, unit="tons/ha"),
        ],
        top_features=[
            FeatureImportanceFront(feature="crop", impact=10.0),
            FeatureImportanceFront(feature="country", impact=6.0),
            FeatureImportanceFront(feature="pesticides_tons", impact=5.8),
            FeatureImportanceFront(feature="rainfall_mm", impact=2.8),
            FeatureImportanceFront(feature="temperature_celsius", impact=0.8),
        ],
        llm_analysis="bla" * 10,
    )
