"""
Suite de tests pour le point d'entrée principal (main.py) et le cycle de vie de l'application.

Ce module valide les fonctionnalités de base de l'infrastructure FastAPI :
1. La disponibilité opérationnelle via le endpoint de santé (Healthcheck).
2. La redirection de l'URL racine vers l'interface de documentation Swagger.
3. Le bon fonctionnement du 'lifespan', garantissant que le RAG et les paramètres sont
    correctement chargés en mémoire au démarrage du serveur.
"""

# Imports

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from livrable_p12.backend.main import create_app

# =================== Health =======================


@pytest.mark.unit
# On s'assure que /health est fonctionnelle: code 200
# On s'assure que la réponse est bien status:ok
def test_healthcheck(client):
    """
    Vérifie que le point d'entrée de santé est opérationnel.

    Indispensable pour les sondes de disponibilité (Liveness/Readiness probes)
    dans les environnements de déploiement type Docker ou Kubernetes.
    """
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# =================== Root =======================


@pytest.mark.unit
def test_root_redirects_to_docs(client):
    """
    Vérifie la redirection automatique de la racine.

    S'assure que tout utilisateur accédant à l'URL de base est immédiatement
    orienté vers la documentation interactive de l'API (Swagger UI).
    """
    # follow_redirects=False permet de vérifier le code 307 de redirection
    response = client.get("/", follow_redirects=False)

    assert response.status_code == 307
    assert response.headers["location"] == "/docs"


# =================== Lifespan =======================


# ---------------- Test du Lifespan réussi
@pytest.mark.integration
def test_lifespan_startup_success():
    """
    Valide l'initialisation du contexte de l'application.

    Vérifie que le mécanisme 'lifespan' a correctemt injecté ce qu'il faut dans les ``app.state``
    """
    # On mock les adaptateurs pour éviter de charger de vrais fichiers/réseau
    with (
        patch(
            "livrable_p12.backend.adapters.inference_models.onnx_predictor.ONNXYieldPredictorAdapter"
        ) as MockPredictor,
        patch(
            "livrable_p12.backend.adapters.llm_client.mistral_response.MistralResponseAdapter"
        ) as MockLLM,
    ):
        # Configuration des mocks
        MockPredictor.return_value = MagicMock()
        MockLLM.return_value = MagicMock()

        app = create_app()

        # TestClient déclenche le lifespan à l'entrée du bloc 'with'
        with TestClient(app):
            # Vérification que l'état de l'application contient notre service
            assert hasattr(app.state, "advisor_service")
            assert app.state.advisor_service.predictor is not None
            assert app.state.advisor_service.llm_engine is not None


# ---------------- Test que le shutdown s'execute bien
@pytest.mark.integration
def test_lifespan_shutdown(caplog):
    """Vérifie que le log de nettoyage est présent à la fermeture."""
    app = create_app()

    with TestClient(app):
        pass  # Startup... puis Shutdown automatique à la sortie du bloc

    # Assertions
    assert "Arrêt du serveur, nettoyage..." in caplog.text
