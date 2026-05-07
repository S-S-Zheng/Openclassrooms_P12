# Imports


from livrable_p12.backend.adapters.database.orm import Monitoring, Prediction, Recommendation
from livrable_p12.backend.adapters.database.supabase_repository import SupabaseRepositoryAdapter


def test_save_prediction_creates_monitoring(
    db_session_for_tests, predict_context, mock_yield_response
):
    """Vérifie uniquement la création de la ligne de monitoring."""
    hash_id = "test_hash_123"
    duration = 2.3
    version = "v1.0.0"
    model_type = "XGBoost"

    SupabaseRepositoryAdapter.save_prediction_trade(
        db=db_session_for_tests,
        request_hash=hash_id,
        context=predict_context,
        result=mock_yield_response,
        duration=duration,
        version=version,
        model_type=model_type,
    )
    log = db_session_for_tests.query(Monitoring).filter_by(request_hash=hash_id).first()
    assert log is not None
    assert log.endpoint_type == "predict"


def test_save_prediction_stores_correct_values(
    db_session_for_tests, predict_context, mock_yield_response
):
    """Vérifie que les données métier (Yield) sont correctes en base."""
    hash_id = "test_hash_123"
    duration = 2.3
    version = "v1.0.0"
    model_type = "XGBoost"

    SupabaseRepositoryAdapter.save_prediction_trade(
        db=db_session_for_tests,
        request_hash=hash_id,
        context=predict_context,
        result=mock_yield_response,
        duration=duration,
        version=version,
        model_type=model_type,
    )
    pred = db_session_for_tests.query(Prediction).filter_by(request_hash=hash_id).first()
    assert pred.crop == "Potatoes"
    assert pred.yield_val == 15.5


def test_monitoring_cascade_delete(db_session_for_tests, predict_context, mock_yield_response):
    """Vérifie que la suppression du Monitoring entraîne celle de la Prediction."""
    hash_id = "test_hash_123"
    duration = 2.3
    version = "v1.0.0"
    model_type = "XGBoost"

    SupabaseRepositoryAdapter.save_prediction_trade(
        db=db_session_for_tests,
        request_hash=hash_id,
        context=predict_context,
        result=mock_yield_response,
        duration=duration,
        version=version,
        model_type=model_type,
    )
    log = db_session_for_tests.query(Monitoring).filter_by(request_hash=hash_id).first()
    db_session_for_tests.delete(log)
    db_session_for_tests.commit()

    assert db_session_for_tests.query(Prediction).filter_by(request_hash=hash_id).first() is None


def test_save_recommendation_stores_json_data(
    db_session_for_tests, recommendation_context, mock_yield_response
):
    """Vérifie que le stockage JSON fonctionne (compatibilité SQLite/Postgres)."""
    hash_id = "test_hash_123"
    duration = 2.3
    version = "v1.0.0"
    model_type = "XGBoost"
    status_code = 200
    analysis = "Texte semantique du LLM à partir des résultats ML"

    SupabaseRepositoryAdapter.save_recommendation_trade(
        db=db_session_for_tests,
        request_hash=hash_id,
        context=recommendation_context,
        results=mock_yield_response,
        analysis=analysis,
        duration=duration,
        version=version,
        model_type=model_type,
        status_code=status_code,
    )

    reco = db_session_for_tests.query(Recommendation).filter_by(request_hash=hash_id).first()
    assert reco.llm_analysis == "Texte semantique du LLM à partir des résultats ML"
    assert isinstance(reco.all_results, list)  # Vérifie que le JSON est bien parsé
