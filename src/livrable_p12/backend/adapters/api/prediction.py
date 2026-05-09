# Imports
import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from livrable_p12.backend.adapters.database.connection import get_db
from livrable_p12.backend.adapters.database.supabase_repository import SupabaseRepositoryAdapter
from livrable_p12.backend.core.entities.models import CropYieldContext, YieldResponse
from livrable_p12.backend.utils.hash_id import generate_feature_hash
from livrable_p12.backend.utils.system_metrics import get_duration

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Predictions"])


@router.post("/predict", response_model=YieldResponse, status_code=status.HTTP_200_OK)
async def predict_yield(
    payload: CropYieldContext,
    db: Annotated[Session, Depends(get_db)],
    request: Request,  # Pour accéder à app.state
    crop: str,  # La culture spécifique demandée
) -> YieldResponse:
    """
    Simple prédiction du rendement agricole
    """
    # On récupère le service pré-instancié dans le lifespan
    advisor = request.app.state.advisor_service
    settings = request.app.state.settings
    request_hash = generate_feature_hash(payload.model_dump())

    # On utilise run_in_threadpool pour ne pas bloquer l'event loop
    monitor_entry = await run_in_threadpool(
        SupabaseRepositoryAdapter.cache_hit_or_miss, db, request_hash
    )
    if monitor_entry and monitor_entry.predicts:
        logger.info(f"CACHE HIT trouvée pour {request_hash[:8]}")
        pred = monitor_entry.predicts
        return YieldResponse(
            primary_prediction=[
                {"crop": pred.crop, "yield_val": pred.yield_val, "unit": pred.unit}
            ],  # type:ignore
            top_features=pred.top_features,
            version=settings.version,
            model_type=advisor.predictor.model_type,
        )

    try:
        # ML
        inference_result, duration = await run_in_threadpool(
            get_duration(advisor.predictor.predict_yield), payload, crop
        )
        #  On complète la YieldResponse inference_result
        inference_result.version = settings.version
        inference_result.model_type = advisor.predictor.model_type
        # Hashing d'identification
        request_hash = generate_feature_hash(payload.model_dump())
        # Persistence
        await run_in_threadpool(
            SupabaseRepositoryAdapter.save_prediction_trade,
            db=db,
            request_hash=request_hash,
            context=payload,
            result=inference_result,
            duration=duration,
            version=settings.version,
            model_type=inference_result.model_type,
        )
        return inference_result

    except Exception as e:
        logger.error(f"Erreur Route Predict: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erreur lors de la prédiction.",
        ) from e
