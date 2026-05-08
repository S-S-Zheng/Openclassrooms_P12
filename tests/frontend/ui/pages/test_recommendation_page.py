# Imports
from unittest.mock import MagicMock, patch

import pytest

from livrable_p12.frontend.adapters.api.requests_api import RequestAgriAPIAdapter
from livrable_p12.frontend.ui.pages.recommendation_page import render_recommandation_page


@pytest.mark.integration
def test_recommendation_page_renders_graphs(sample_agri_result):
    """Vérifie que les composants graphiques sont sollicités quand des données arrivent."""
    with (
        patch("streamlit.button", return_value=True),
        patch("streamlit.plotly_chart") as mock_chart,
    ):
        mock_api = MagicMock(spec=RequestAgriAPIAdapter)
        mock_api.get_recommendation.return_value = sample_agri_result
        render_recommandation_page(mock_api, {"country": "France"})

        # On attend au moins 2 graphiques (reco + features)
        assert mock_chart.call_count >= 2
