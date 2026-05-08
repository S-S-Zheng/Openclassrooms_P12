# Imports
import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from livrable_p12.backend.adapters.database.connection import get_db
from livrable_p12.backend.adapters.database.supabase_repository import SupabaseRepositoryAdapter
from livrable_p12.backend.core.entities.models import CropYieldContext, YieldResponse
from livrable_p12.backend.utils.hash_id import generate_feature_hash
from livrable_p12.backend.utils.system_metrics import get_duration_async

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Recommendations"])


@router.post("/recommend", response_model=YieldResponse)
async def get_recommendation(
    payload: CropYieldContext,
    db: Annotated[Session, Depends(get_db)],
    request: Request,
    top_k: int = Query(default=3, ge=1, le=10),
) -> YieldResponse:
    """
    Recommandation en calculant les rendements pour TOUTES les cultures puis on utilise
    le LLM pour générer des conseils agronomiques.
    """
    advisor = request.app.state.advisor_service
    settings = request.app.state.settings

    try:
        # ML + LLM
        (results, analysis), duration = await get_duration_async(advisor.llm_recommendation)(
            payload, top_k=top_k
        )
        # On prépare l'objet de réponse final
        response_data = YieldResponse(
            recommendations=results.recommendations
            if hasattr(results, "recommendations")
            else results,
            llm_analysis=analysis,
            model_type=advisor.predictor.model_type,
            version=settings.version,
        )
        # Persistence
        await run_in_threadpool(
            SupabaseRepositoryAdapter.save_recommendation_trade,
            db=db,
            request_hash=generate_feature_hash(payload.model_dump()),
            context=payload,
            results=response_data,
            analysis=analysis,
            duration=duration,
            version=settings.version,
            model_type=advisor.predictor.model_type,
            status_code=200,
        )
        return response_data

    except Exception as e:
        logger.error(f"Erreur Route Recommend: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erreur lors de la recommandation.",
        ) from e
