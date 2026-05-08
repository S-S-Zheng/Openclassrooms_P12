# Imports
from dataclasses import dataclass
from typing import List, Optional, Union


@dataclass
class FeatureImportance:
    feature: str
    impact: float


@dataclass
class AgriResult:
    """Modèle unique pour Prediction et Recommandation."""

    yield_value: Union[float, List[float]]
    crop: Union[str, List[str]]
    top_features: List[FeatureImportance]
    llm_analysis: Optional[str] = None
