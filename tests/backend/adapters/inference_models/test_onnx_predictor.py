# Imports
import os

import pytest

from livrable_p12.backend.adapters.inference_models.onnx_predictor import ONNXYieldPredictor


@pytest.mark.unit
def test_onnx_predictor_initialization():
    """
    Vérifie que l'adapter charge correctement le modèle et les métadonnées JSON.
    """
    # Chemins relatifs depuis la racine du projet
    model_path = "datas/results/best_model/best_model.onnx"
    metadata_path = "datas/results/shap/shap_metadata.json"

    # On vérifie que les fichiers existent avant de tester
    if not os.path.exists(model_path) or not os.path.exists(metadata_path):
        pytest.skip("Fichier manquant pour le test")

    predictor = ONNXYieldPredictor(model_path=model_path, metadata_path=metadata_path)

    # Assertions
    assert predictor.all_crops is not None
    assert len(predictor.global_importance) > 0


@pytest.mark.unit
def test_predict_yield_logic(predict_context):
    """
    Vérifie la logique de prédiction.
    """
    # Chemins relatifs depuis la racine du projet
    model_path = "datas/results/best_model/best_model.onnx"
    metadata_path = "datas/results/shap/shap_metadata.json"

    # On vérifie que les fichiers existent avant de tester
    if not os.path.exists(model_path) or not os.path.exists(metadata_path):
        pytest.skip("Fichier manquant pour le test")

    predictor = ONNXYieldPredictor(model_path=model_path, metadata_path=metadata_path)

    # On force la culture pour ce test
    response = predictor.predict_yield(predict_context, crop="Wheat")

    # Assertions
    assert response.primary_prediction is not None
    assert response.primary_prediction.crop == "Wheat"
    assert isinstance(response.primary_prediction.yield_val, float)
    assert response.primary_prediction.yield_val > 0


@pytest.mark.integration
def test_get_recommendations_logic(recommendation_context):
    """
    Vérifie la logique d'inférence de masse et le tri des résultats.
    """
    # Chemins relatifs depuis la racine du projet
    model_path = "datas/results/best_model/best_model.onnx"
    metadata_path = "datas/results/shap/shap_metadata.json"

    # On vérifie que les fichiers existent avant de tester
    if not os.path.exists(model_path) or not os.path.exists(metadata_path):
        pytest.skip("Fichier manquant pour le test")

    predictor = ONNXYieldPredictor(model_path=model_path, metadata_path=metadata_path)

    # Test du batch inference (get_recommendations)
    top_k = 3
    response = predictor.get_recommendations(recommendation_context, top_k=top_k)

    # Assertions
    assert len(response.recommendations) == top_k
    assert response.recommendations[0].yield_val >= response.recommendations[1].yield_val
    assert len(response.top_features) > 0
