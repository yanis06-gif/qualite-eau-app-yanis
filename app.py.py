
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import io
from PIL import Image
from datetime import datetime
import matplotlib.pyplot as plt

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

# Image
if os.path.exists("eau.jpg"):
    st.image("eau.jpg", use_column_width=True, caption="Analyse de la qualité de l'eau")

# Menu de navigation
tabs = st.tabs(["🔍 Prédiction", "🧪 Classification", "📋 Gestion"])

# ==========================
# 🧠 MODE PRÉDICTION
# ==========================
with tabs[0]:
    st.header("🔍 Prédiction d’un paramètre manquant")

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

            alertes = verifier_parametres_entres({parametre_cible: prediction})
            for a in alertes:
                st.warning(a)
    else:
        st.error(f"Modèle {model_filename} introuvable.")

# ==========================
# 🧪 MODE CLASSIFICATION
# ==========================
with tabs[1]:
    st.header("🧪 Classification de la qualité de l’eau")

    classes = {3: "Très bonne", 0: "Bonne", 2: "Moyenne", 1: "Mauvaise", 4: "Très mauvaise"}

    valeurs = {}
    for param in parametres:
        valeurs[param] = st.number_input(param, value=0.0, format="%.4f", key=f"class_{param}")

    if st.button("Prédire la classe", key="btn_class"):
        try:
            model = joblib.load("modele_Classification.pkl")
            X = np.array([valeurs[p] for p in parametres]).reshape(1, -1)
            prediction = model.predict(X)[0]
            st.success(f"Classe prédite : **{classes.get(prediction, 'Inconnue')}**")
            alertes = verifier_parametres_entres(valeurs)
            for a in alertes:
                st.warning(a)
        except Exception as e:
            st.error(f"Erreur : {e}")

# ==========================
# 📋 MODE GESTION
# ==========================
with tabs[2]:
    st.header("📋 Gestion des prélèvements")

    if 'df_prelèvements' not in st.session_state:
        try:
            st.session_state.df_prelèvements = pd.read_pickle("prelevements_sauvegarde.pkl")
        except:
            st.session_state.df_prelèvements = pd.DataFrame()

    if 'parametres_dynamiques' not in st.session_state:
        st.session_state.parametres_dynamiques = {}

    # Ajout paramètre personnalisé
    st.markdown("### ➕ Ajouter un paramètre personnalisé")
    nouveau_param = st.text_input("Nom du paramètre", key="new_param")
    valeur_param = st.number_input("Valeur", value=0.0, key="val_param")
    if st.button("Ajouter ce paramètre"):
        if nouveau_param.strip():
            st.session_state.parametres_dynamiques[nouveau_param.strip()] = valeur_param
            st.success(f"Ajouté : {nouveau_param}")

    # Formulaire
    with st.form(key="form_prelevement"):
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
            st.success("✅ Prélèvement enregistré")
            for a in verifier_parametres_entres(resultats):
                st.warning(a)

    # Affichage
    df = st.session_state.df_prelèvements.copy()
    if not df.empty:
        st.markdown("### 📊 Données enregistrées")
        st.dataframe(df, use_container_width=True)

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

        excel_data = to_excel(df)
        if excel_data:
            st.download_button("📥 Télécharger (Excel)", data=excel_data,
                               file_name="prelevements.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    else:
        st.info("Aucun prélèvement encore saisi.")
