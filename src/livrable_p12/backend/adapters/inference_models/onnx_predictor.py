import json
from typing import List

import numpy as np
import onnxruntime as ort

from livrable_p12.backend.core.entities.models import (
    CropYieldContext,
    FeatureImportance,
    PredictionResult,
    YieldResponse,
)
from livrable_p12.backend.core.ports.yield_predictor import YieldPredictorPort


class ONNXYieldPredictor(YieldPredictorPort):
    def __init__(self, model_path: str, metadata_path: str):
        # Chargement du runtime ONNX
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        # Liste exhaustive des cultures du dataset
        self.all_crops = [
            "Potatoes",
            "Maize",
            "Wheat",
            "Rice, paddy",
            "Sorghum",
            "Soybeans",
            "Sweet potatoes",
            "Cassava",
            "Plantains and others",
            "Yams",
        ]
        # On injecte ici les résultats SHAP globale faites en amont
        self.metadata_path = metadata_path
        self.global_importance = self._load_metadatas()

    def _load_metadatas(self):
        try:
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            return [FeatureImportance(**item) for item in metadata]
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Erreur lors du chargement de la métadonnée: {e}")
            return []

    def _prepare_input(self, context: CropYieldContext, crops: List[str]) -> dict:
        """
        Transforme le contexte et la culture en matrice numpy pour ONNX.
        'crops' est une liste, même si elle ne contient qu'un seul élément.
        Note: Doit refléter EXACTEMENT l'ordre des colonnes X_train.
        """
        # ONNX avec pipeline attend un dictionnaire de [1, 1] pour chaque feature
        n_rows = len(crops)

        return {
            "country": np.array([[context.country]] * n_rows, dtype=object),
            "crop": np.array([[crop] for crop in crops], dtype=object),
            "year": np.array([[context.year]] * n_rows, dtype=np.int64),
            "rainfall_mm": np.full((n_rows, 1), context.rainfall_mm, dtype=np.float32),
            "pesticide_tons": np.full((n_rows, 1), context.pesticide_tons, dtype=np.float32),
            "temperature_celcius": np.full(
                (n_rows, 1), context.temperature_celcius, dtype=np.float32
            ),
            "temp_anomaly": np.full((n_rows, 1), context.temp_anomaly, dtype=np.float32),
        }

    def predict_yield(self, context: CropYieldContext, crop: str) -> YieldResponse:
        """Implémentation pour une seule culture."""
        inputs = self._prepare_input(context, [crop])
        # onnxruntime renvoie une liste de outputs. Le 1er est la prédiction.
        prediction = self.session.run(None, inputs)[0]

        return YieldResponse(
            primary_prediction=PredictionResult(crop=crop, yield_val=float(prediction[0])),
            top_features=self.global_importance,
        )

    def get_recommendations(self, context: CropYieldContext, top_k: int = 3) -> YieldResponse:
        """Implémentation pour la recommandation."""
        # On passe toute la liste des cultures
        batch_inputs = self._prepare_input(context, self.all_crops)
        # UNE SEULE EXÉCUTION pour toutes les cultures (Gain de performance massif)
        all_preds = self.session.run(None, batch_inputs)[0]  # Shape (n_crops, 1)

        recos = [
            PredictionResult(crop=self.all_crops[i], yield_val=float(all_preds[i][0]))
            for i in range(len(self.all_crops))
        ]

        # Tri et extraction du Top K
        sorted_results = sorted(recos, key=lambda x: x.yield_val, reverse=True)

        return YieldResponse(
            recommendations=sorted_results[:top_k], top_features=self.global_importance
        )
