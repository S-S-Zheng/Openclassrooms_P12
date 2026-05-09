# Imports
import logging

import streamlit as st

from livrable_p12.frontend.adapters.api.requests_api import RequestAgriAPIAdapter
from livrable_p12.frontend.core.services.crops_list import CropListService
from livrable_p12.frontend.settings import get_settings
from livrable_p12.frontend.ui.pages.prediction_page import render_prediction_page
from livrable_p12.frontend.ui.pages.recommendation_page import render_recommandation_page

settings = get_settings()
# =============================================================================
# CONFIGURATION DE LA PAGE (Doit être la première commande Streamlit)
# ATTENTION OBLIGATOIREMENT JUSTE APRÈS LES IMPORTS!
st.set_page_config(page_title=settings.ui_title, layout="wide", initial_sidebar_state="auto")

# =========================================================
# LOGS ET RESSOURCES
# =================== Configuration logs ===================
# Logging & Monitoring (inspiré de ton code)
logging.basicConfig(level=logging.INFO)


# =================== Initialisation des ressources ===================
@st.cache_resource
def get_api_adapter():
    base_url = f"http://{settings.app_host}:{settings.app_port}"
    return RequestAgriAPIAdapter(base_url=base_url)


adapter = get_api_adapter()


@st.cache_data
def load_crops_metadata() -> list:
    service = CropListService(data_path=settings.crop_path)
    return service.get_crops()


supported_crops = load_crops_metadata()


# =====================================================
# INTERFACE UTILISATEUR (UI)
def main():
    st.title("AgritechAnswersAssistant - Aide à la décision agricole")
    # Utilisation du session_state pour persister les résultats
    if "prediction_cache" not in st.session_state:
        st.session_state.prediction_cache = None
    if "reco_cache" not in st.session_state:
        st.session_state.reco_cache = None

    # Sidebar pour le contexte (commune aux deux modes)
    with st.sidebar:
        st.header("Paramètres Climatiques")
        context = {
            "country": st.text_input("Pays", "France", max_chars=50),
            "year": st.slider("Année", min_value=1980, max_value=2050, value=2026, step=1),
            "rainfall_mm": st.slider(
                "Précipitations", min_value=0, max_value=5000, value=1000, step=50
            ),
            "temperature_celcius": st.slider(
                "Température", min_value=-10, max_value=50, value=15, step=1
            ),
            "pesticides_tons": st.slider(
                "Pesticides", min_value=0, max_value=200000, value=15000, step=1000
            ),
            "temp_anomaly": st.number_input(
                "Anomalie Temp.", min_value=0.0, max_value=2.0, value=1.0, step=0.1
            ),
        }

    tab1, tab2 = st.tabs(["Prédiction", "Recommandation"])

    with tab1:
        render_prediction_page(adapter, context, supported_crops)
    with tab2:
        render_recommandation_page(adapter, context)


if __name__ == "__main__":  # pragma: no cover
    main()
