# Imports
import streamlit as st

from livrable_p12.frontend.adapters.api.requests_api import RequestAgriAPIAdapter
from livrable_p12.frontend.ui.visuals.visualization import plot_feature_importance


def render_prediction_page(
    request_adapter: RequestAgriAPIAdapter, context: dict, available_crops: list
):
    """Affiche l'interface de prédiction simple."""
    st.header("Prédiction du rendement d'une culture suivant les conditions pedoclimatiques.")

    # Champ spécifique à la prédiction (la culture)
    crop_choice = st.selectbox(
        "Sélectionnez la culture à analyser",
        # ["Wheat", "Maize", "Potatoes", "Soybeans"],
        options=available_crops,
        index=0,
        help="Choisissez la culture dont la prédiction de rendement vous interesserait.",
    )

    if st.button("Prédire", type="primary"):
        # On fusionne le contexte de la sidebar avec le choix de la culture
        payload = {**context, "crop": crop_choice}

        with st.spinner(f"Calcul du rendement pour : {crop_choice}..."):
            try:
                result = request_adapter.get_prediction(payload)
                if result.primary_prediction and result.top_features:
                    # Métrique principale
                    st.success("Selon le modèle, voici les résultats:")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(
                            label=f"Rendement estimé ({result.primary_prediction.crop})",
                            value=(
                                f"{result.primary_prediction.yield_val:.2f} "
                                f"{result.primary_prediction.unit}"
                            ),
                        )
                    # # Visualisation de l'importance des caractéristiques
                    # if result.top_features:
                    with col2:
                        plot_feature_importance(result.top_features)
                else:
                    st.warning("Aucune donnée de prédiction renvoyée par le modèle.")
            except Exception as e:
                st.error(f"Une erreur est survenue lors de l'appel API : {e}")
