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

# patch: Remplace temporairement une classe ou une fonction par un mock.
# MagicMock: Un objet "caméléon" qui accepte n'importe quel appel de méthode.
# AsyncMock: La version asynchrone pour simuler, par exemple, une base de données ou un client HTTP.
import pytest

# import pytest_asyncio # permet d'écrire des tests qui peuvent "attendre" (await) des réponses
# Client spécialisé qui simule des requêtes HTTP sur FastAPI sans devoir lancer un vrai serveur Web.
from fastapi.testclient import TestClient

from livrable_p12.backend.core.entities.models import (
    CropYieldContext,
    FeatureImportance,
    PredictionResult,
    YieldResponse,
)
from livrable_p12.backend.main import create_app


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
        recommendations=[
            PredictionResult(crop="Potatoes", yield_val=17.4),
            PredictionResult(crop="Wheat", yield_val=6.2),
        ],
        top_features=[
            FeatureImportance(feature="crop", impact=5.0),
            FeatureImportance(feature="temperature_celsius", impact=1.5),
        ],
    )


@pytest.fixture
def client():
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

    with TestClient(app) as test_client:
        yield test_client
