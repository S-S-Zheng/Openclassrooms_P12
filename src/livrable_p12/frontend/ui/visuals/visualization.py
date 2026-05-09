# livrable_p12/frontend/ui/visualizations.py

# Imports
import pandas as pd
import plotly.express as px
import streamlit as st


def plot_yield_comparison(recommendations):
    """Génère le graphique en barres pour comparer les rendements."""
    if not recommendations:
        return st.warning("Pas de recommandations à afficher.")

    df = pd.DataFrame(
        [{"Culture": res.crop, "Rendement": res.yield_val} for res in recommendations]
    )
    fig = px.bar(
        df,
        x="Culture",
        y="Rendement",
        color="Rendement",
        title="Classement des cultures par rendement estimé",
        labels={"Rendement": "tonnes/ha"},
        template="plotly_white",
    )
    return st.plotly_chart(fig, width="stretch")


def plot_feature_importance(features, title="Facteurs d'influence"):
    """Génère le graphique horizontal de l'importance des caractéristiques."""
    if not features:
        return st.info("Données d'importance non disponibles.")

    df = pd.DataFrame([{"Facteur": f.feature, "Impact": f.impact} for f in features]).sort_values(
        by="Impact", ascending=True
    )

    fig = px.bar(
        df,
        x="Impact",
        y="Facteur",
        orientation="h",
        title=title,
        color="Impact",
        color_continuous_scale="RdYlGn",  # Rouge à Vert pour l'impact
    )
    return st.plotly_chart(fig, width="stretch")
