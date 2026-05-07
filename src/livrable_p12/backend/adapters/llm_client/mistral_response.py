# Imports
import logging

import yaml
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI

from livrable_p12.backend.core.entities.models import YieldResponse
from livrable_p12.backend.core.ports.llm_response import LLMResponsePort
from livrable_p12.backend.settings import get_settings

logger = logging.getLogger(__name__)


class MistralResponseAdapter(LLMResponsePort):
    def __init__(self):
        # le singleton de config
        self.settings = get_settings()
        # Chargement des prompts du LLM
        self.prompts = self._load_yaml(self.settings.prompt_config_path)
        # Les prompts
        self.system_prompt = self.prompts["system_prompt"]
        self.user_template = self.prompts["user_template"]
        # le client
        self.llm = ChatMistralAI(
            base_url=self.settings.mistral_base_url,
            model_name=self.settings.llm_model,
            api_key=self.settings.mistral_api_key,  #  type:ignore
            max_tokens=self.settings.llm_max_tokens,
            temperature=self.settings.llm_temperature,
            top_p=self.settings.llm_top_p,
            max_retries=self.settings.llm_max_retries,
            timeout=self.settings.llm_timeout,
        )

    def _load_yaml(self, path: str) -> dict:
        """Chargement sécurisée en yaml pour la config et les prompts."""
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _format_context(self, results: YieldResponse) -> dict:
        """
        Transforme les résultats ML en texte pour le prompt.
        Découple la structure des résultas de la structure du prompt.
        """
        reco_text = "\n".join(
            [
                f"- {result.crop}: {result.yield_val:.2f} {result.unit}"
                for result in results.recommendations
            ]
        )
        feature_text = ", ".join([f.feature for f in results.top_features])

        return {"recos": reco_text, "features": feature_text}

    def generate_analysis(self, results: YieldResponse) -> str:
        """Génère la réponse NLP à partir des résultats du ML"""
        formatted_data = self._format_context(results)

        prompt_template = ChatPromptTemplate.from_messages(
            [("system", self.system_prompt), ("user", self.user_template)]
        )

        chain = prompt_template | self.llm

        try:
            response = chain.invoke(formatted_data)
            return str(response.content)
        except Exception as e:
            return f"L'analyse n'a pas pu être générée : {str(e)}"
