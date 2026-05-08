"""

Centralise les variables d'environnement. Au lieu que chaque fichier appel ses propres variables
d'environnement, on utilise ce fichier qui va lire les .env et fallback si ceux-ci n'ont pas la
la variable ou est mal entré avec une valeur par défaut.\n
On se protège avec Pydantic.

Tous les modules importent get_settings() au lieu de faire os.environ.

Le fichier .env n'est qu'un fichier texte brut.\n
Utiliser une classe Settings (via pydantic-settings) apporte trois avantages majeurs
en Data Science et Production :

1. Validation de Type
    Si openagenda_max_events=BEAUCOUP dans .env, Pydantic lèvera
    une erreur immédiatement au démarrage car il attend un int.

2. Auto-complétion (IDE)
    Dans le code, quand on écrit settings.,
    l'éditeur va proposer mistral_api_key != avec un simple os.getenv, on navigues à
    l'aveugle donc risque d'erreur.

3. Valeurs par défaut
    On peut mettre des valeurs fallback (ex: app_port: int = 8000).
    Si la variable est absente du .env, l'app ne plante pas.

4. Conversion automatique
    Il transforme la chaîne "99.99" du .env en un véritable float
    utilisable pour tes calculs géographiques.
"""

# imports
from functools import lru_cache
from urllib.parse import quote_plus  # Import indispensable pour les caractères spéciaux

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# =====================================================================


class Settings(BaseSettings):
    """
    Classe paramétrique qui centralise les variables d'environnement. Lit automatiquement le .env
    dans le dossier courant si l'argument env_file est déclaré dans model_config via
    Pydantic-settings ce qui permet en plus un contrôle.
    """

    # ===================== Paths ======================================
    model_path: str = "datas/results/best_model/best_model.onnx"
    metadata_path: str = "datas/results/shap/shap_metadata.json"
    prompt_config_path: str = "configs/prompts.yaml"
    # ======================== LLM =====================================
    mistral_base_url: str = "https://api.mistral.ai/v1"
    """Via pydantic-settings, Pydantic va automatiquement chercher une variable
    d'environnement nommée MISTRAL_API_KEY"""
    mistral_api_key: str
    """'mistral-small-latest', 'mistral-medium-latest' ou 'mistral-latest'"""
    llm_model: str = "mistral-small-latest"
    """Contrôle le côté factuel (faible valeur, conseillé 0.2) ou imaginaire du modèle """
    llm_temperature: float = 0.2
    """Nb max de tokens à générer """
    llm_max_tokens: int = 512
    """Nucleus sampling". Le modèle ne choisit ses mots que parmi les n% les plus probables."""
    llm_top_p: float = 0.9
    """Si l'API Mistral est surchargée (erreur 503),
    LangChain réessaie automatiquement n fois avant d'échouer. """
    llm_max_retries: int = 2
    """Si Mistral ne répond pas après n sec,
    coupe la connexion pour ne pas bloquer l'utilisateur."""
    llm_timeout: int = 30
    # ================ Securité /rebuild =======================================
    # rebuild_api_key: str
    # """OBLIGATOIRE: CLEF API POUR POUVOIR INDEXER"""
    # ================ Base de données =======================================
    db_user: str = "postgres.iemmbmmrjvdsrtfjwhwc"
    db_password: str = Field(validation_alias="sb_password")
    db_host: str = "aws-1-eu-west-1.pooler.supabase.com"
    db_port: str = "6543"
    db_name: str = "postgres"

    @property
    def database_url(self) -> str:
        """Génère dynamiquement l'URL de connexion sécurisée."""
        encoded_pass = quote_plus(self.db_password)
        options = "?sslmode=require" if "supabase.co" in self.db_host else ""
        return (
            f"postgresql+psycopg2://{self.db_user}:{encoded_pass}@{self.db_host}:"
            f"{self.db_port}/{self.db_name}{options}"
        )

    # ================ Serveur ===============================================
    # app_host: str = "0.0.0.0"
    # app_port: int = 8000

    # ================ Métadonnées ===============================================
    version: str = "agritech_V1"
    # ================= CONFIG ===============================================
    model_config = SettingsConfigDict(
        env_file=(".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # Ignore les variables inconnues
        coerce_numbers_to_str=True,  # Permet la conversion des int en str auto
    )


# ====================================================================
# C'est un cache qui garde en mémoire le résultat du premier appel.
# La première fois qu'on appelle 'get_settings()', Python lit le fichier '.env' et
# crée l'objet 'Settings'. Toutes les fois suivantes, il retourne **exactement le même objet**
# sans relire le fichier. C'est le pattern **Singleton** — un seul objet de configuration pour
# toute l'application.
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Retourne le singleton 'Settings' caché.\n
    L'utilisation de lru_cache garanti que l'.env n'est lu qu'une fois et que tous les modules
    recoivent exactement le même objet.
    """
    return Settings()  # type:ignore # pragma: no cover
