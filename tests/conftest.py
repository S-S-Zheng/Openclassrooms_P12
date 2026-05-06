# imports

import pytest

from livrable_p12.backend.core.entities.models import (
    CropYieldContext,
    FeatureImportance,
    PredictionResult,
    YieldResponse,
)


@pytest.fixture
def predict_context():
    """Fournit un contexte de données valide pour la recommandation"""
    return CropYieldContext(
        country="France",
        crop="Wheat",
        year=2026,
        rainfall_mm=500.0,
        temperature_celcius=15.0,
        pesticides_tons=10.0,
        temp_anomaly=1.5,
    )


@pytest.fixture
def recommendation_context():
    """Fournit un contexte de données valide pour la recommandation"""
    return CropYieldContext(
        country="France",
        crop=None,
        year=2026,
        rainfall_mm=500.0,
        temperature_celcius=15.0,
        pesticides_tons=10.0,
        temp_anomaly=1.5,
    )


@pytest.fixture
def mock_yield_response():
    """Simule la sortie du modèle ML."""
    return YieldResponse(
        recommendations=[
            PredictionResult(crop="Potatoes", yield_val=17.4),
            PredictionResult(crop="Wheat", yield_val=6.2),
        ],
        top_features=[
            FeatureImportance(feature="crop", impact=5.0),
            FeatureImportance(feature="temperature_celsius", impact=1.5),
        ],
    )
