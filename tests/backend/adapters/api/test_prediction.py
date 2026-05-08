# Imports
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from livrable_p12.backend.adapters.database.orm import Prediction


@pytest.mark.integration
def test_predict_response(client: TestClient, predict_context):
    """
    Vérifie que l'endpoint /predict renvoie un format de donnée valide.
    """
    # Préparation du payload (exclusion du crop car passé en query param)
    payload = predict_context.model_dump()
    payload.pop("crop", None)

    # Route
    # Note: On passe 'crop' en query param comme défini dans la route
    response = client.post("/predict?crop=Maize", json=payload)

    # Assertions
    assert response.status_code == 200
    data = response.json()
    assert "primary_prediction" in data
    assert data["primary_prediction"]["crop"] == "Maize"
    assert isinstance(data["primary_prediction"]["yield_val"], float)


@pytest.mark.integration
def test_predict_persistence(client: TestClient, predict_context, db_session_for_tests):
    """
    Vérifie qu'une prédiction réussie est bien enregistrée en base de données.
    """
    payload = predict_context.model_dump()
    payload.pop("crop", None)

    # Exécution de l'appel
    client.post("/predict?crop=Maize", json=payload)

    # Persistence
    # On vérifie l'ajout dans la table Prédiction
    feature_saved = select(Prediction).where(Prediction.country == "France")
    db_record = db_session_for_tests.execute(feature_saved).scalar_one_or_none()

    # Assertions
    assert db_record is not None
    assert db_record.year == 2026
    assert db_record.crop == "Maize"
    assert db_record.yield_val is not None
