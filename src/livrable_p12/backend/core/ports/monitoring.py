# Imports
from abc import ABC, abstractmethod
from typing import Any, Protocol


# Approche structurelle
class DataFrameLike(Protocol):
    def __getitem__(self, key: Any) -> Any: ...
    def head(self, n: int = 5) -> Any: ...


class MonitoringPort(ABC):
    """Interface pour le logging et le monitoring de dérive (Drift)."""

    @abstractmethod
    def log_request_metadata(
        self,
        status_code: int,
        endpoint: str,
        inference_time_ms: float,
        response_time_ms: float,
        cpu_usage: float,
        memory_usage: float,
        n_features: list,
        model_type: str,
        version: str,
    ):
        """Enregistre les métadonnées de la requête dans la table Report."""
        pass

    @abstractmethod
    def run_drift_analysis(self, hist_data: DataFrameLike, curr_data: DataFrameLike) -> dict:
        """Génère un rapport d'évolution entre données historiques et réelles."""
        pass
