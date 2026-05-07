# Imports
from abc import ABC, abstractmethod


class MonitoringPort(ABC):
    """Interface pour le logging"""

    @abstractmethod
    def log_request_metadata(
        self,
        status_code: int,
        endpoint: str,
        response_time_ms: float,
        cpu_usage: float,
        memory_usage: float,
        model_type: str,
        version: str,
    ):
        """Enregistre les métadonnées de la requête dans la table Report."""
        pass
