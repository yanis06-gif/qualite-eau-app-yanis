
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import io
from PIL import Image
from datetime import datetime
import matplotlib.pyplot as plt
import altair as alt

# Configuration de la page
st.set_page_config(page_title="Qualité de l'eau potable", page_icon="💧", layout="wide")

# Normes algériennes pour certains paramètres
normes = {
    'pH': {"min": 6.5, "max": 8.5, "conseil": "Ajuster le pH avec des agents acidifiants ou basifiants."},
    'Turbidity': {"max": 5, "conseil": "Filtrer l'eau pour réduire la turbidité."},
    'Free Chlorine': {"min": 0.2, "max": 0.5, "conseil": "Réguler le dosage du chlore."},
    'Nitrate': {"max": 50, "conseil": "Réduire les apports agricoles et industriels."},
    'Temperature': {"max": 30, "conseil": "Conserver l’eau à l’abri de la chaleur."}
}

# Paramètres standards
parametres = [
    'Total Coliform', 'Escherichia Coli', 'Faecal Streptococci', 'Turbidity', 'pH', 'Temperature',
    'Free Chlorine', 'Chlorates', 'Sulfate', 'Magnesium', 'Calcium', 'Conductivity', 'Dry Residue',
    'Complete Alkaline Title', 'Nitrite', 'Ammonium', 'Phosphate', 'Nitrate', 'Iron', 'Manganese',
    'Colour', 'Smell', 'Taste'
]

# Vérification des normes
def verifier_parametres_entres(valeurs: dict):
    alertes = []
    for param, valeur in valeurs.items():
        if param in normes:
            seuil = normes[param]
            if ("min" in seuil and valeur < seuil["min"]) or ("max" in seuil and valeur > seuil["max"]):
                min_val = seuil.get("min", "-")
                max_val = seuil.get("max", "-")
                conseil = seuil.get("conseil", "")
                alertes.append(f"⚠️ **{param} = {valeur:.2f}** est hors norme ({min_val} - {max_val}). 💡 {conseil}")
    return alertes

# Fonction export Excel
def to_excel(df_to_export):
    try:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df_to_export.to_excel(writer, index=False)
        output.seek(0)
        return output.read()
    except Exception as e:
        st.error(f"Erreur Excel : {e}")
        return b""

# Image
if os.path.exists("eau.jpg"):
    st.image("eau.jpg", use_column_width=True, caption="Analyse de la qualité de l'eau")

# Titre principal
st.markdown("<h1 style='text-align: center; color: teal;'>💧 Application d'analyse de la qualité de l'eau potable</h1>", unsafe_allow_html=True)

# Menu de navigation
tabs = st.tabs(["🔍 Prédiction", "🧪 Classification", "📋 Gestion", "📊 Visualisation", "📤 Export"])

# ==========================
# 🧠 MODE PRÉDICTION
# ==========================
with tabs[0]:
    st.subheader("🔍 Prédiction d’un paramètre manquant")
    parametre_cible = st.selectbox("Quel paramètre veux-tu prédire ?", parametres)
    model_filename = f"modele_{parametre_cible.replace(' ', '_')}.pkl"

    if os.path.exists(model_filename):
        model = joblib.load(model_filename)
        valeurs = {}
        for param in parametres:
            if param != parametre_cible:
                valeurs[param] = st.number_input(param, value=0.0, format="%.4f", key=f"pred_{param}")

        if st.button("Prédire", key="btn_predire"):
            X = np.array([valeurs[p] for p in parametres if p != parametre_cible]).reshape(1, -1)
            prediction = model.predict(X)[0]
            st.success(f"🔎 Valeur prédite pour **{parametre_cible}** : `{prediction:.4f}`")
            for a in verifier_parametres_entres({parametre_cible: prediction}):
                st.warning(a)
    else:
        st.error(f"Modèle {model_filename} introuvable.")

# ==========================
# 🧪 MODE CLASSIFICATION
# ==========================
with tabs[1]:
    st.subheader("🧪 Classification de la qualité de l’eau")
    classes = {3: "Très bonne", 0: "Bonne", 2: "Moyenne", 1: "Mauvaise", 4: "Très mauvaise"}
    valeurs = {}
    for param in parametres:
        valeurs[param] = st.number_input(param, value=0.0, format="%.4f", key=f"class_{param}")
    if st.button("Prédire la classe", key="btn_class"):
        try:
            model = joblib.load("modele_Classification.pkl")
            X = np.array([valeurs[p] for p in parametres]).reshape(1, -1)
            prediction = model.predict(X)[0]
            st.success(f"✅ Classe prédite : **{classes.get(prediction, 'Inconnue')}**")
            for a in verifier_parametres_entres(valeurs):
                st.warning(a)
        except Exception as e:
            st.error(f"Erreur : {e}")

# ==========================
# 📋 GESTION DES PRÉLÈVEMENTS
# ==========================
with tabs[2]:
    st.subheader("📋 Gestion journalière des prélèvements")

    if 'df_prelèvements' not in st.session_state:
        try:
            st.session_state.df_prelèvements = pd.read_pickle("prelevements_sauvegarde.pkl")
        except:
            st.session_state.df_prelèvements = pd.DataFrame()

    if 'parametres_dynamiques' not in st.session_state:
        st.session_state.parametres_dynamiques = {}

    # Ajouter paramètre personnalisé
    st.markdown("### ➕ Paramètre personnalisé")
    new_param = st.text_input("Nom du paramètre", key="new_param")
    new_val = st.number_input("Valeur", key="new_val")
    if st.button("Ajouter ce paramètre"):
        if new_param.strip():
            st.session_state.parametres_dynamiques[new_param.strip()] = new_val
            st.success(f"Ajouté : {new_param}")

    with st.form("form_prelevement"):
        col1, col2 = st.columns(2)
        with col1:
            date = st.date_input("Date", value=datetime.today())
            entreprise = st.text_input("Entreprise")
            code = st.text_input("Code")
            preleveur = st.text_input("Préleveur")
        with col2:
            heure = st.time_input("Heure")
            localisation = st.text_input("Localisation")
            analyste = st.text_input("Analyste")

        resultats = {}
        for p in parametres:
            resultats[p] = st.number_input(p, value=0.0, format="%.4f", key=f"gestion_{p}")
        for p in st.session_state.parametres_dynamiques:
            resultats[p] = st.number_input(p, value=0.0, format="%.4f", key=f"dyn_{p}")

        submitted = st.form_submit_button("Ajouter")
        if submitted:
            new_data = {
                "Date": date, "Heure": heure, "Entreprise": entreprise,
                "Localisation": localisation, "Code": code,
                "Préleveur": preleveur, "Analyste": analyste
            }
            new_data.update(resultats)
            st.session_state.df_prelèvements = pd.concat(
                [st.session_state.df_prelèvements, pd.DataFrame([new_data])],
                ignore_index=True
            )
            st.session_state.df_prelèvements.to_pickle("prelevements_sauvegarde.pkl")
            st.success("✅ Prélèvement ajouté")
            for a in verifier_parametres_entres(resultats):
                st.warning(a)

# ==========================
# 📊 VISUALISATION
# ==========================
with tabs[3]:
    st.subheader("📊 Visualisation des paramètres")

    df = st.session_state.df_prelèvements.copy()
    if not df.empty:
        param_to_plot = st.selectbox("Sélectionner un paramètre à visualiser", parametres)
        mode = st.radio("Comparer par :", ["Date", "Entreprise", "Préleveur"])

        df['Date'] = pd.to_datetime(df['Date'])
        try:
            if mode == "Date":
                chart = alt.Chart(df).mark_line(point=True).encode(
                    x='Date:T',
                    y=param_to_plot,
                    tooltip=['Date', param_to_plot]
                ).properties(
                    title=f"Évolution de {param_to_plot} dans le temps"
                )
            else:
                chart = alt.Chart(df).mark_bar().encode(
                    x=alt.X(mode, sort='-y'),
                    y=param_to_plot,
                    tooltip=[mode, param_to_plot],
                    color=alt.condition(
                        alt.datum[param_to_plot] > normes.get(param_to_plot, {}).get("max", 999),
                        alt.value('red'),
                        alt.value('green')
                    ) if param_to_plot in normes else alt.value("steelblue")
                ).properties(
                    title=f"{param_to_plot} par {mode}"
                )

            st.altair_chart(chart, use_container_width=True)
        except Exception as e:
            st.error(f"Erreur de visualisation : {e}")
    else:
        st.info("Aucune donnée disponible pour afficher un graphique.")

# ==========================
# 📤 EXPORT
# ==========================
with tabs[4]:
    st.subheader("📤 Export des données")

    df = st.session_state.df_prelèvements.copy()
    if not df.empty:
        excel_data = to_excel(df)
        if excel_data:
            st.download_button(
                label="📥 Télécharger le fichier Excel",
                data=excel_data,
                file_name="prelevements_qualite_eau.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="download_excel"
            )
        else:
            st.warning("Erreur lors de la génération du fichier.")
    else:
        st.info("Aucune donnée à exporter.")
