# Imports
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from livrable_p12.backend.adapters.database.orm import Recommendation


@pytest.mark.integration
def test_recommendation_response(client: TestClient, recommendation_context):
    payload = recommendation_context.model_dump()
    payload.pop("crop", None)

    # Appel de la route
    response = client.post("/recommend?top_k=5", json=payload)

    # Assertions
    assert response.status_code == 200
    data = response.json()
    assert "recommendations" in data
    assert len(data["recommendations"]) > 0


@pytest.mark.integration
def test_recommendation_persistence(
    client: TestClient, recommendation_context, db_session_for_tests
):
    payload = recommendation_context.model_dump()
    payload.pop("crop", None)

    # Appel de la route
    client.post("/recommend?top_k=5", json=payload)

    feature_saved = select(Recommendation).where(
        Recommendation.input_context["country"].astext == "France"
    )
    db_record = db_session_for_tests.execute(feature_saved).scalars().first()
    crops_presents = [res["crop"] for res in db_record.all_results]

    # Assertions
    assert db_record is not None
    assert "Potatoes" in crops_presents
    assert db_record.llm_analysis is not None
    assert db_record.top_features is not None
