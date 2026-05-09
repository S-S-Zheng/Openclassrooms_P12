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
    request_hash = generate_feature_hash(payload.model_dump())

    # On utilise run_in_threadpool pour ne pas bloquer l'event loop
    monitor_entry = await run_in_threadpool(
        SupabaseRepositoryAdapter.cache_hit_or_miss, db, request_hash
    )
    # Si le hash existe ET qu'il y a une recommandation associée
    if monitor_entry and monitor_entry.recos:
        logger.info(f"CACHE HIT pour {request_hash[:8]}")
        reco = monitor_entry.recos  # C'est l'objet Recommendation lié
        return YieldResponse(
            recommendations=reco.all_results,
            top_features=reco.top_features,
            llm_analysis=reco.llm_analysis,
            version=settings.version,
            model_type=advisor.predictor.model_type,
        )

    try:
        # ML + LLM
        (inference_result, analysis), duration = await get_duration_async(
            advisor.llm_recommendation
        )(payload, top_k=top_k)
        # on complète inference_result
        inference_result.llm_analysis = analysis
        inference_result.version = settings.version
        inference_result.model_type = advisor.predictor.model_type
        # Persistence
        await run_in_threadpool(
            SupabaseRepositoryAdapter.save_recommendation_trade,
            db=db,
            request_hash=request_hash,
            context=payload,
            results=inference_result,
            analysis=analysis,
            duration=duration,
            version=settings.version,
            model_type=inference_result.model_type,
            status_code=200,
        )
        return inference_result

    except Exception as e:
        logger.error(f"Erreur Route Recommend: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erreur lors de la recommandation.",
        ) from e
