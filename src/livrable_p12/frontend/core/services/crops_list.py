# Imports
import json
from pathlib import Path
from typing import List


class CropListService:
    def __init__(self, data_path: str):
        # On définit le chemin vers le dossier datas à la racine du projet
        self.data_path = Path(__file__).resolve().parents[5] / data_path

    def _get_fallback_crops(self) -> List[str]:
        """Liste de secours en cas d'absence de fichier."""
        return [
            "Potatoes",
            "Maize",
            "Wheat",
            "Rice",
            "paddy",
            "Sorghum",
            "Soybeans",
            "Sweet potatoes",
            "Cassava",
            "Yams",
        ]

    def get_crops(self) -> List[str]:
        """
        Récupère dynamiquement la liste des cultures supportées par le modèle.
        """
        try:
            if not self.data_path.exists():
                return self._get_fallback_crops()

            with open(self.data_path, "r", encoding="utf-8") as f:
                data = json.load(f)

                crops = data.get("all_crops")
                if crops:
                    return sorted([str(crop) for crop in crops])
                return self._get_fallback_crops()

        except (json.JSONDecodeError, IOError):
            return self._get_fallback_crops()
