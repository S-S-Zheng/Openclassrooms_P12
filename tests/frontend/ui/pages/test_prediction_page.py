# Imports
from unittest.mock import MagicMock, patch

import pytest

from livrable_p12.frontend.adapters.api.requests_api import RequestAgriAPIAdapter
from livrable_p12.frontend.ui.pages.prediction_page import render_prediction_page


@pytest.mark.unit
def test_prediction_page_calls_adapter(predict_context, sample_agri_result):
    """Vérifie que le clic sur le bouton déclenche l'appel API avec le bon payload."""
    # On simule le clic sur le bouton Streamlit
    with patch("streamlit.button", return_value=True), patch("streamlit.spinner"):
        mock_api = MagicMock(spec=RequestAgriAPIAdapter)
        mock_api.get_prediction.return_value = sample_agri_result
        context = predict_context.model_dump()

        # On exécute la fonction de rendu
        render_prediction_page(mock_api, context, ["Wheat", "Maize", "Potatoes", "Soybeans"])

        # Assertions
        mock_api.get_prediction.assert_called_once()
        args, _ = mock_api.get_prediction.call_args
        assert args[0]["crop"] in ["Wheat", "Maize", "Potatoes", "Soybeans"]


@pytest.mark.unit
def test_prediction_page_error_handling():
    """Vérifie que la page affiche une erreur si l'API crash."""
    with patch("streamlit.button", return_value=True), patch("streamlit.error") as mock_st_error:
        mock_api = MagicMock(spec=RequestAgriAPIAdapter)
        mock_api.get_prediction.side_effect = Exception("API Down")
        render_prediction_page(mock_api, {"test": "data"}, ["Wheat", "Maize", "Potatoes"])

        # Assertions
        mock_st_error.assert_called_once()
        assert "Une erreur est survenue" in mock_st_error.call_args[0][0]
