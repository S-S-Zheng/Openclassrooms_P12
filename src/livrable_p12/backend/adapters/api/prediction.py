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

    try:
        # ML
        result, duration = await run_in_threadpool(
            get_duration(advisor.predictor.predict_yield), payload, crop
        )
        # Hashing d'identification
        request_hash = generate_feature_hash(payload.model_dump())
        # Persistence
        await run_in_threadpool(
            SupabaseRepositoryAdapter.save_prediction_trade,
            db=db,
            request_hash=request_hash,
            context=payload,
            result=result,
            duration=duration,
            version=settings.version,
            model_type=advisor.predictor.model_type,
        )
        return result

    except Exception as e:
        logger.error(f"Erreur Route Predict: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erreur lors de la prédiction.",
        ) from e
