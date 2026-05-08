"""_summary_

Returns:
    _type_: _description_
"""

# Imports
import logging
import os

import psutil
from sqlalchemy.orm import Session

from livrable_p12.backend.adapters.database.orm import Monitoring, Prediction, Recommendation
from livrable_p12.backend.core.entities.models import (
    CropYieldContext,
    YieldResponse,
)

logger = logging.getLogger(__name__)


class SupabaseRepositoryAdapter:
    @staticmethod
    def _get_system_metrics() -> dict:
        """
        Capture les métriques de performance système (CPU/RAM).
        """
        # RAM utilisée par le worker API (en Mo)
        process = psutil.Process(os.getpid())
        # pas d'intervalle pour ne pas bloquer l'API (non-blocking)
        return {
            "cpu_usage": float(psutil.cpu_percent(interval=None)),
            "memory_usage": float(process.memory_info().rss / (1024 * 1024)),
        }

    @staticmethod
    def _create_monitoring_log(
        request_hash: str,
        endpoint: str,
        duration: float,
        metrics: dict,
        version: str,
        model_type: str,
        status_code: int = 200,
    ) -> Monitoring:
        """
        Instancie l'objet ORM Monitoring.
        """
        return Monitoring(
            request_hash=request_hash,
            endpoint_type=endpoint,
            status_code=status_code,
            response_time_ms=duration,
            cpu_usage=metrics["cpu_usage"],
            memory_usage=metrics["memory_usage"],
            model_type=model_type,
            version=version,
        )

    @classmethod
    def save_prediction_trade(
        cls,
        db: Session,
        request_hash: str,
        context: CropYieldContext,
        result: YieldResponse,
        duration: float,
        version: str,
        model_type: str,
        status_code: int = 200,
    ):
        """
        Orchestre la sauvegarde d'une prédiction de rendement simple.

        Args:
            db (Session): Session SQL active.
            request_hash (str): Hash unique de la requête.
            context (CropYieldContext): Entité métier d'entrée.
            result (YieldResponse): Résultat du modèle.
            duration (float): Temps de réponse en ms.
        """
        metrics = cls._get_system_metrics()

        try:
            # Création des objets ORM
            monitoring_log = cls._create_monitoring_log(
                request_hash, "predict", duration, metrics, version, model_type, status_code
            )

            data = context.model_dump()
            prediction_entry = Prediction(
                request_hash=request_hash,
                # **context.model_dump(),
                country=data.get("country"),
                year=data.get("year"),
                crop=result.primary_prediction.crop,  # type:ignore
                rainfall_mm=data.get("rainfall_mm"),
                pesticides_tons=data.get("pesticides_tons"),
                temperature_celcius=data.get("temperature_celcius"),
                temp_anomaly=data.get("temp_anomaly"),
                yield_val=result.primary_prediction.yield_val,  # type:ignore
                unit=result.primary_prediction.unit,  # type:ignore
                top_features=[feature.model_dump() for feature in result.top_features],
            )
            # Persistance atomique
            db.add_all([monitoring_log, prediction_entry])
            db.commit()
            logger.info(f"Prédiction sauvegardée avec succès (Hash: {request_hash[:8]}...)")

        except Exception as e:
            db.rollback()
            logger.error(f"Erreur lors de la sauvegarde de la prédiction : {e}")
            raise RuntimeError(f"Erreur Database : {e}") from e

    @classmethod
    def save_recommendation_trade(
        cls,
        db: Session,
        request_hash: str,
        context: CropYieldContext,
        results: YieldResponse,
        analysis: str,
        duration: float,
        version,
        model_type,
        status_code,
    ):
        """
        Orchestre la sauvegarde d'une recommandation complexe avec analyse LLM.

        Args:
            db (Session): Session SQL active.
            request_hash (str): Hash unique de la requête.
            context (CropYieldContext): Entité métier d'entrée.
            result (YieldResponse): Résultat du batch.
            analysis (str): Réponse du LLM
            duration (float): Temps de réponse en ms.
        """
        metrics = cls._get_system_metrics()

        try:
            monitoring_log = cls._create_monitoring_log(
                request_hash, "recommend", duration, metrics, version, model_type, status_code
            )

            recommendation_entry = Recommendation(
                request_hash=request_hash,
                input_context=context.model_dump(),
                all_results=[result.model_dump() for result in results.recommendations],
                top_features=[feature.model_dump() for feature in results.top_features],
                llm_analysis=analysis,
            )

            db.add_all([monitoring_log, recommendation_entry])
            db.commit()
            logger.info(f"Recommandation sauvegardée avec succès (Hash: {request_hash[:8]}...)")

        except Exception as e:
            db.rollback()
            logger.error(f"Erreur lors de la sauvegarde de la recommandation : {e}")
            raise RuntimeError(f"Erreur Database : {e}") from e
