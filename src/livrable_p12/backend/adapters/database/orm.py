"""
Module de définition des modèles de données ORM (Object-Relational Mapping).

Ce module contient les schémas SQL pour la base via SQLAlchemy. Il définit
l'organisation des données stockées, incluant les enregistrements de prédictions
détaillés et le système de journalisation (logging) pour la traçabilité des requêtes.
"""

# Pydantic définit la forme des données qui entrent/sortent,
# SQLAlchemy définit la forme des données qui dorment en base.

# imports
from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB  # Pour Supabase plus tard si besoin
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from livrable_p12.backend.adapters.database.base import Base


# ================= DONNÉES MÉTIER (RAG / SQL Tool) =================
class Prediction(Base):
    """
    Table pour la prédiction simple.\n
    """

    __tablename__ = "predictions"

    # Identifications
    id = Column(Integer, primary_key=True)
    request_hash = Column(String(64), ForeignKey("monitoring.request_hash"))

    # Inputs
    country = Column(String(50))
    crop = Column(String(30))
    year = Column(Integer)
    rainfall_mm = Column(Float)
    pesticides_tons = Column(Float)
    temperature_celcius = Column(Float)
    temp_anomaly = Column(Float)

    # Outputs
    yield_val = Column(Float)
    unit = Column(String(15))
    top_features = Column(JSONB)  # feature importance globale

    # Relations
    # Crée une dépendance des ID avec la table monitoring via request_hash (permet la jointure)
    # Créée une relation bidirectionnelle entre monitoring et predictions
    monitoring = relationship("Monitoring", back_populates="predicts")


class Recommendation(Base):
    """
    Table pour la recommandation.\n
    """

    __tablename__ = "recommendations"

    # Identifications
    id = Column(Integer, primary_key=True)
    request_hash = Column(String(64), ForeignKey("monitoring.request_hash"))

    # Inputs
    input_context = Column(JSONB)  # Le contexte

    # Outputs
    all_results = Column(JSONB)  # Top K des cultures
    top_features = Column(JSONB)  # feature importance globale
    llm_analysis = Column(Text)  # La vulgarisation Mistral

    # Crée une dépendance des ID avec la table monitoring via request_hash (permet la jointure)
    # Créée une relation bidirectionnelle entre monitoring et recommendations
    monitoring = relationship("Monitoring", back_populates="recos")


# ================= MONITORING =================


class Monitoring(Base):
    """
    Table de monitoring des requêtes et cache\n
    """

    __tablename__ = "monitoring"

    # Identifications
    # ID hashé SHA-256 de la requête pour le cache
    request_hash = Column(String(64), primary_key=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    endpoint_type = Column(String(30))
    status_code = Column(Integer)
    response_time_ms = Column(Float)
    cpu_usage = Column(Float)  # % CPU global
    memory_usage = Column(Float)  # RAM utilisée par le process en Mo
    model_type = Column(String(30))
    version = Column(String(30))

    # Relations
    predicts = relationship(
        "Prediction", back_populates="monitoring", uselist=False, cascade="all, delete-orphan"
    )
    recos = relationship(
        "Recommendation", back_populates="monitoring", uselist=False, cascade="all, delete-orphan"
    )
