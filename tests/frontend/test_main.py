# Imports
from unittest.mock import MagicMock, patch

import pytest

from livrable_p12.frontend.main import get_api_adapter, main

BASE_URL_TEST = "http://localhost:8000"


@pytest.mark.unit
def test_get_api_adapter_caching():
    """Vérifie que l'adapter est bien instancié avec la bonne URL."""
    with patch("streamlit.cache_resource"):
        adapter = get_api_adapter()
        assert adapter._base_url == BASE_URL_TEST


@pytest.mark.unit
@patch("streamlit.tabs")
@patch("streamlit.sidebar")
def test_main_layout_initialization(mock_sidebar, mock_tabs):
    """Vérifie que le layout (onglets, sidebar) est initialisé au lancement."""
    mock_tabs.return_value = [MagicMock(), MagicMock()]

    with patch("streamlit.title") as mock_title:
        main()
        mock_title.assert_called_once_with("AgritechAnswersAssistant - Aide à la décision agricole")
        mock_tabs.assert_called_once()
        # On vérifie que la sidebar a bien été sollicitée pour créer les inputs
        # Note : Streamlit appelle les méthodes sur 'st', mais le patch sidebar
        # permet de s'assurer qu'on est bien dans le bloc de menu.
        assert mock_sidebar.__enter__.called
        assert mock_sidebar.__exit__.called
