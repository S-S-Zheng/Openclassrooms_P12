# Imports
import os

import pytest

from livrable_p12.backend.adapters.llm_client.mistral_response import MistralResponseAdapter

# from livrable_p12.backend.core.entities.models import (
#     YieldResponse, PredictionResult, FeatureImportance
# )


@pytest.mark.functional
def test_mistral_adapter_response(mock_yield_response):
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        pytest.skip("Besoin de la vraie clé pour ce test")

    adapter = MistralResponseAdapter()

    # On teste la seule méthode qui compte : la génération finale
    analysis = adapter.generate_analysis(mock_yield_response)

    assert isinstance(analysis, str)
    assert len(analysis) > 20
