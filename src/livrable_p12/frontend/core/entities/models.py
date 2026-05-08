# Imports
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class PredictionResult:
    """Résultat pour une culture spécifique."""

    crop: str
    yield_val: float
    unit: str = "tons/ha"


@dataclass(frozen=True)
class FeatureImportance:
    feature: str
    impact: float


@dataclass(frozen=True)  # frozen rend l'objet immuable != sans car dataclasse par défaut mutable
class AgriResult:
    """Modèle unique pour Prediction et Recommandation."""

    primary_prediction: Optional[PredictionResult] = None
    recommendations: List[PredictionResult] = field(default_factory=list)
    llm_analysis: Optional[str] = None

    # Explicabilité et Monitoring
    top_features: List[FeatureImportance] = field(default_factory=list)
