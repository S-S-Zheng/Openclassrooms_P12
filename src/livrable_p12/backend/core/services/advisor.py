"""
Service de coordination entre les Adapters ML et LLM via l'utilisation des Ports pour créer
l'abstraction
--------

- `Injection de dépendances` : Le __init__ accepte les Ports.
Cela signifie que dans mes tests unitaires,
je peux passer des "Mocks" à la place de Mistral pour ne pas payer de jetons API à chaque test.

- `Séparation des responsabilités` : Si on décide demain de ne plus utiliser Mistral
mais un modèle local, on change l'objet passé au service, mais le service lui-même ne change pas.

- `Clarté` : Le service retourne un tuple (données_brutes, texte_vulgarisé).
C'est l'API qui décidera comment formater ça pour le Front.
"""

# Imports
import logging

from livrable_p12.backend.core.entities.models import CropYieldContext, YieldResponse
from livrable_p12.backend.core.ports.llm_response import LLMResponsePort
from livrable_p12.backend.core.ports.yield_predictor import YieldPredictorPort

logger = logging.getLogger(__name__)


class AgriAdvisorService:
    """
    Orchestre la liaison entre le prédicteur ML et le vulgarisateur LLM.
    """

    def __init__(self, predictor: YieldPredictorPort, llm_engine: LLMResponsePort):
        """
        Initialise le service avec les adaptateurs nécessaires.

        Args:
            predictor (YieldPredictorPort): Adaptateur pour le moteur de prédiction ML.
            llm_engine (LLMResponsePort): Adaptateur pour le moteur de synthèse textuelle.
        """
        self.predictor = predictor
        self.llm_engine = llm_engine

    async def simple_prediction(self, context: CropYieldContext, crop: str) -> YieldResponse:
        """Réalise une prédiction de rendement à partir de la culture et du contexte par le ML"""
        logger.info(f"Prédiction de rendement pour la culture: {crop}...")
        return self.predictor.predict_yield(context, crop)

    async def llm_recommendation(
        self, context: CropYieldContext, top_k: int = 3
    ) -> tuple[YieldResponse, str]:
        """
        Réalise un prédiction pour toutes les cultures à partir du contexte, renvoi ensuite
        les meilleurs choix suivant leur rendement. L'inférence est ensuite annalyser par le LLM
        qui vulgarise la réponse pour l'utilisateur.

        Args:
            context (CropYieldContext): Le contexte pédoclimatique de la requête.
            top_k (int, optional): Nombre de cultures à recommander. Defaults to 3.

        Returns:
            tuple[YieldResponse, str]: Un tuple contenant les données structurées
                et l'analyse du LLM.
        """
        logger.info(f"Recommandation complète (top_k={top_k})...")
        # Inférence ML (Bloquant mais rapide via ONNX)
        results = self.predictor.get_recommendations(context, top_k=top_k)

        # Analyse LLM (On délègue l'exécution pour ne pas bloquer l'Event Loop)
        # Bien que l'adapter soit bloquant en interne, nous l'appelons ici
        analysis = self.llm_engine.generate_analysis(results)

        return results, analysis
