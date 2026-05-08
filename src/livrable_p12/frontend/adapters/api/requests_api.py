# Imports
import requests

from livrable_p12.frontend.core.entities.models import (
    AgriResult,
    FeatureImportance,
    PredictionResult,
)


class RequestAgriAPIAdapter:
    def __init__(self, base_url: str):
        self._base_url = base_url

    def _map_to_entity(self, data: dict) -> AgriResult:
        """Transforme le dictionnaire JSON de l'API en objet métier AgriResult."""
        return AgriResult(
            primary_prediction=(
                PredictionResult(**data["primary_prediction"])
                if data.get("primary_prediction")
                else None
            ),
            recommendations=[PredictionResult(**reco) for reco in data.get("recommendations", [])],
            top_features=[FeatureImportance(**feat) for feat in data.get("top_features", [])],
            llm_analysis=data.get("llm_analysis"),
        )

    def get_prediction(self, payload: dict, crop: str) -> AgriResult:
        """Route dédiée à la prédiction simple."""
        response = requests.post(f"{self._base_url}/predict?crop={crop}", json=payload, timeout=15)
        response.raise_for_status()

        return self._map_to_entity(response.json())

    def get_recommendation(self, payload: dict) -> AgriResult:
        """Route dédiée à la recommandation (plusieurs cultures)."""
        response = requests.post(f"{self._base_url}/recommend", json=payload, timeout=30)
        response.raise_for_status()

        return self._map_to_entity(response.json())
