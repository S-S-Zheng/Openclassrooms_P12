"""
entities
-----
Classes (généralement via Pydantic ou des dataclasses) qui représentent objets métier. Exemple :
    Une entité YieldPrediction contiendra juste yield_val, unit et confidence.
    Elle ne contient aucune logique de base de données.
    C'est le "langage commun" entre le Data Scientist et le Backend.

On a choisit ici de partir sur pydantic pour 3 raisons:
    - Validation automatique : dataclass ne vérifie pas au runtime si rainfall_mm
        est bien un nombre. Pydantic le fait et rejette la requête si elle est mal formée.
    - Sérialisation JSON : FastAPI utilise Pydantic pour générer automatiquement
        la documentation (Swagger/OpenAPI).
    - Coût technique : Utiliser des dataclasses forcerait le développeur Backend
        à ré-écrire des validateurs manuels. On garde Pydantic pour assurer le "Contrat"
        entre le modèle ML et l'API.
"""

# Imports
from datetime import datetime, timezone
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class CropYieldContext(BaseModel):
    """
    Entité de base partagée pour la prédiction et la recommandation.
    Utilise les variables communes du dataset historique.
    """

    country: str = Field(..., examples=["france", "albania"])
    year: int = Field(..., ge=1900, le=2100)
    crop: str = Field(..., examples=["wheat", "rice", "maize"])
    rainfall_mm: float = Field(..., ge=0.0, le=10000.0, alias="rainfall")
    temperature_celcius: float = Field(..., ge=-50.0, le=50.0, alias="temp")
    pesticide_tons: float = Field(..., ge=0, alias="pest")
    temp_anomaly: float = Field(
        ..., ge=0.0, le=2.0, description="Ratio temperature du pays / moyenne du pays"
    )


class PredictionResult(BaseModel):
    """Résultat pour une culture spécifique."""

    crop: str
    yield_val: float
    unit: str = "tons/ha"


class YieldResponse(BaseModel):
    """Objet final renvoyé au Backend."""

    primary_prediction: Optional[PredictionResult] = None
    recommendations: List[PredictionResult] = []

    # Explicabilité et Monitoring
    top_features: List[dict] = []
    drift_detected: bool = False

    # LLM Summary (Mistral)
    analysis_fr: Optional[str] = None

    # Métadonnées
    datetime_utc: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    model_version: str = "agritech_answers_V1"

    model_config = ConfigDict(from_attributes=True)
