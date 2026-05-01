# Imports
from abc import ABC, abstractmethod

from livrable_p12.backend.core.entities.models import YieldResponse


class LLMResponsePort(ABC):
    """Port pour l'interprétation sémantique des résultats par le LLM."""

    @abstractmethod
    def generate_analysis(self, results: YieldResponse) -> str:
        """Génère une explication textuelle en français basée sur les résultats du modèle."""
        pass
