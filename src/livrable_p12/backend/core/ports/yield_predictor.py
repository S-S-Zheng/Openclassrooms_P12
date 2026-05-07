from abc import ABC, abstractmethod

from livrable_p12.backend.core.entities.models import CropYieldContext, YieldResponse


class YieldPredictorPort(ABC):
    """
    Interface définissant les capacités du moteur d'IA.
    """

    @abstractmethod
    def predict_yield(self, context: CropYieldContext, crop: str) -> YieldResponse:
        """Réalise une prédiction ponctuelle pour une culture donnée."""
        pass

    @abstractmethod
    def get_recommendations(self, context: CropYieldContext, top_k: int = 3) -> YieldResponse:
        """
        Réalise une inférence de masse sur le contexte pedoclimatique et renvoi les meilleures
        cultures possibles suivant le rendement par ordre décroissant.
        """
        pass
