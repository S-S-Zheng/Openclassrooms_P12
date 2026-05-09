# Imports
from unittest.mock import MagicMock, patch

import pytest
from requests.exceptions import HTTPError

from livrable_p12.frontend.core.entities.models import AgriResult

BASE_URL_TEST = "http://api_test:8000"


@pytest.mark.unit
def test_map_to_entity_success(request_adapter, mock_yield_response):
    """Vérifie que le mapping transforme correctement le dictionnaire en objets typés."""
    # conversion en dict
    data_dict = mock_yield_response.model_dump()
    result = request_adapter._map_to_entity(data_dict)

    # Assertions
    assert isinstance(result, AgriResult)
    assert result.primary_prediction is not None
    assert result.primary_prediction.crop == "Potatoes"
    assert len(result.recommendations) == 2
    assert result.recommendations[1].crop == "Wheat"
    assert result.top_features[0].feature == "crop"
    assert result.llm_analysis == "Analyse des résultats ML par le LLM test."


@pytest.mark.unit
def test_map_to_entity_empty_recommendations(request_adapter, mock_yield_response):
    """Vérifie la robustesse si les listes sont vides."""
    data_dict = mock_yield_response.model_dump()
    data_dict.pop("recommendations")
    data_dict.pop("top_features")
    data_dict.pop("llm_analysis")
    result = request_adapter._map_to_entity(data_dict)

    # Assertions
    assert result.recommendations == []
    assert result.top_features == []
    assert result.llm_analysis is None


# @patch("requests.post") => la bibliothèque unittest.mock intercepte l'appel et crée
# automatiquement un objet MagicMock qu'elle injecte en premier argument dans la fonction de test.
@pytest.mark.integration
@patch("requests.post")  # Rattaché à mock_post donc
def test_get_prediction_success(mock_post, request_adapter, mock_yield_response, predict_context):
    """Vérifie le flux complet d'une requête de prédiction réussie."""
    payload = predict_context.model_dump()
    crop = payload.get("crop", "Maize")
    # Configuration du mock de 'requests'
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_yield_response.model_dump()
    mock_post.return_value = mock_response

    result = request_adapter.get_prediction(payload)

    # Assertions
    # Vérification de l'appel réseau
    mock_post.assert_called_once_with(
        f"{BASE_URL_TEST}/predict?crop={crop}", json=payload, timeout=15
    )
    assert result.primary_prediction.crop == "Potatoes"


@pytest.mark.integration
@patch("requests.post")
def test_get_recommendation_network_error(mock_post, request_adapter):
    """Vérifie que l'adapter propage les erreurs HTTP (404, 500, etc.)."""
    # On simule une erreur 500 du serveur
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.raise_for_status.side_effect = HTTPError("Internal Server Error")
    mock_post.return_value = mock_response

    with pytest.raises(HTTPError):
        request_adapter.get_recommendation({"fake": "data"})
