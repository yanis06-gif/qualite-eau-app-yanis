
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
# ==========================
# 🧭 Navigation : Entrée & Sortie de l'application
# ==========================

# Initialisation de la session pour la navigation
if "page_active" not in st.session_state:
    st.session_state.page_active = "accueil"

# Bouton de retour dans la barre latérale (visible après entrée)
if st.session_state.page_active != "accueil":
    with st.sidebar:
        if st.button("🔙 Retour à l'accueil"):
            st.session_state.page_active = "accueil"
            st.stop()

# ==========================
# 🎉 PAGE D'ACCUEIL INTERACTIVE
# ==========================
if st.session_state.page_active == "accueil":
    col1, col2 = st.columns([1, 2])
    with col1:
        if os.path.exists("logo.png"):
            st.image("logo.png", width=150)
    with col2:
        st.markdown("### 🌍 Projet IA - Qualité de l’Eau Potable en Algérie")
        st.markdown("Une plateforme intelligente de surveillance, d'analyse et de recommandation.")
        st.markdown("🧪 Basée sur des modèles d’intelligence artificielle appliqués à des données physico-chimiques et bactériologiques.")

    st.markdown("---")
    # ✅ BOUTON AU DÉBUT
    if st.button("🚀 Entrer dans l'application"):
        st.session_state.page_active = "application"
        st.stop()

    # Présentation (après le bouton)
    st.markdown("---")
    st.markdown("### 🧪 À propos du projet")
    st.markdown("Cette application utilise l’intelligence artificielle pour analyser et surveiller la qualité de l’eau potable en Algérie, selon la norme NA 6361-2016.")

    st.markdown("### ⚙️ Fonctionnalités principales :")
    st.markdown("- 🔍 **Prédiction** d’un paramètre manquant")
    st.markdown("- 🧪 **Classification** de la qualité de l’eau")
    st.markdown("- 📋 **Gestion des prélèvements** (saisie, suivi, export)")
    st.markdown("- 📊 **Visualisation** interactive")
    st.markdown("- 📘 **Conseils personnalisés** selon les normes algériennes")

    st.markdown("### 💡 Mode d’emploi :")
    st.markdown("1. Saisissez ou importez vos résultats d’analyse")
    st.markdown("2. Utilisez les outils de prédiction ou classification")
    st.markdown("3. Téléchargez vos rapports et consultez les recommandations")

    st.markdown("---")
    st.markdown("### 🧑‍🔬 Équipe projet")
    st.markdown("**Équipe IA & Eau Potable – Algérie 🇩🇿**")
    st.markdown("- Étudiant(e)s : FETOUH Yanis, AYAD Lewisa Lysa")
    st.markdown("- Encadrant : BOUCHRAKI Faiza ")
    st.markdown("- Structure : Université / Laboratoire partenaire")

    st.stop()  # On bloque le reste tant qu'on est sur l'accueil



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
col1, col2 = st.columns(2)

with col1:
    st.image("eau.jpg", caption="Qualité de l'eau", use_container_width=True)

with col2:
    st.image("eau1.png", caption="Analyse bactériologique", use_container_width=True)


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

# ==========================
# 📘 ÉTHIQUE ET TRANSPARENCE
# ==========================
with st.sidebar.expander("ℹ️ À propos de cette application"):
    st.markdown("""
    **Développée par :** Votre Équipe IA 🧠  
    **Objectif :** Améliorer la surveillance de la qualité de l'eau potable en Algérie.  
    **Modèles utilisés :** Random Forest Regressor & Classifier.  
    """)

with st.tabs(["🔍 Prédiction", "🧪 Classification", "📋 Gestion", "📊 Visualisation", "📤 Export", "📘 Éthique et transparence"])[5]:
    st.header("📘 Éthique et transparence")

    st.markdown("""
    ### 🤖 Utilisation de l'intelligence artificielle
    Cette application utilise des modèles de machine learning pour aider à prédire et classifier la qualité de l'eau potable, à partir de données physico-chimiques et bactériologiques.

    - Les modèles ont été entraînés à partir de données historiques, validées.
    - Les résultats sont fournis à titre indicatif, et **ne remplacent pas une expertise humaine ou un avis technique**.

    ### 🔐 Respect de la vie privée et des données
    - Les données enregistrées localement (sur votre machine ou serveur personnel) ne sont **ni partagées, ni transmises** à un serveur externe.
    - Aucun traitement automatique de données à caractère personnel n’est effectué.

    ### ⚠️ Limitations des prédictions
    - La précision dépend fortement de la qualité et de la complétude des données saisies.
    - En cas de doute, une analyse en laboratoire reste **indispensable**.

    ### ✅ Engagement qualité
    Cette application respecte les normes algériennes de qualité de l’eau potable (NA 6361-2016).  
    Elle est conçue pour être **transparente, pédagogique et évolutive**.

    ### 📩 Contact
    Pour améliorer ou contribuer à ce projet, contactez-nous à : `contact@equipe-eau-dz.dz`
    """)

# ==========================
# 🌟 Personnalisation visuelle & charte
# ==========================

# Affichage du logo en haut à gauche
col_logo, col_title = st.columns([1, 5])
with col_logo:
    if os.path.exists("logo.png"):
        st.image("logo.png", width=100)
with col_title:
    st.markdown("## 🚰 Projet IA Qualité de l'Eau – Algérie")

# Charte graphique CSS (pour titres & mise en page)
st.markdown("""
<style>
h1, h2, h3 {
    color: #0a9396;
}
.stButton > button {
    background-color: #0a9396;
    color: white;
    font-weight: bold;
    border-radius: 5px;
}
.stDownloadButton > button {
    background-color: #005f73;
    color: white;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# Signature en bas de page
st.markdown("""
<hr>
<p style='text-align: center; font-size: 0.9em; color: grey;'>
Développé avec ❤️ par l'équipe IA & Eau Potable – Algérie 🇩🇿<br>
© 2025 | Tous droits réservés
</p>
""", unsafe_allow_html=True)
