# Imports
import streamlit as st

from livrable_p12.frontend.adapters.api.requests_api import RequestAgriAPIAdapter
from livrable_p12.frontend.ui.visuals.visualization import (
    plot_feature_importance,
    plot_yield_comparison,
)


def render_recommandation_page(request_adapter: RequestAgriAPIAdapter, context: dict):
    """Affiche l'interface de recommandation (classement des cultures)."""
    st.header("Recommandation de culture suivant les conditions pedoclimatiques")

    if st.button("Recommandation", type="primary"):
        with st.spinner("Analyse des cultures offrant le meilleur rendement..."):
            try:
                result = request_adapter.get_recommendation(context)

                if result.recommendations and result.top_features:
                    col1, col2 = st.columns(2)

                    with col1:
                        plot_yield_comparison(result.recommendations)

                    with col2:
                        plot_feature_importance(result.top_features)
                # Analyse sémantique LLM
                if result.llm_analysis:
                    st.subheader("Analyse LLM")
                    st.success(result.llm_analysis)

            except Exception as e:
                st.error(f"Erreur technique : {e}")
