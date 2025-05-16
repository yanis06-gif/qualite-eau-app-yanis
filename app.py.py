import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
from PIL import Image
import io
from datetime import datetime
import matplotlib.pyplot as plt

# Configuration de la page
st.set_page_config(page_title="Qualité de l'eau potable", page_icon="💧", layout="centered")

# Titre
st.title("💧 Application IA - Prédiction et Classification de la qualité de l'eau")


image = Image.open("eau.jpg")
st.image(image, caption="Analyse de la qualité de l'eau", use_column_width=True)


# Liste des paramètres
parametres = [
    'Total Coliform', 'Escherichia Coli', 'Faecal Streptococci', 'Turbidity', 'pH', 'Temperature',
    'Free Chlorine', 'Chlorates', 'Sulfate', 'Magnesium', 'Calcium', 'Conductivity', 'Dry Residue',
    'Complete Alkaline Title', 'Nitrite', 'Ammonium', 'Phosphate', 'Nitrate', 'Iron', 'Manganese',
    'Colour', 'Smell', 'Taste'
]

# Menu dans la barre latérale
st.sidebar.title("🔧 Menu")
mode = st.sidebar.radio("Choisir une action :", ("🔍 Prédire un paramètre manquant", "🧪 Classifier la qualité de l’eau"))

# === Mode 1 : Prédiction d'un paramètre manquant ===
if mode == "🔍 Prédire un paramètre manquant":
    st.subheader("🔍 Prédiction d’un paramètre manquant")

    parametre_cible = st.selectbox("Quel paramètre veux-tu prédire ?", parametres)
    model_filename = f"modele_{parametre_cible.replace(' ', '_')}.pkl"

    if not os.path.exists(model_filename):
        st.error(f"❌ Le modèle '{model_filename}' est introuvable.")
        st.stop()

    model = joblib.load(model_filename)
    valeurs = []

    for param in parametres:
        if param != parametre_cible:
            val = st.number_input(param, value=0.0, format="%.4f")
            valeurs.append(val)

    if st.button("Prédire le paramètre manquant"):
        X_input = np.array(valeurs).reshape(1, -1)
        prediction = model.predict(X_input)
        st.success(f"🔍 Prédiction pour **{parametre_cible}** : `{prediction[0]:.4f}`")

# === Mode 2 : Classification de la qualité de l'eau ===

elif mode == "🧪 Classifier la qualité de l’eau":
    classes = {
        3: "Très bonne",
        0: "Bonne",
        2: "Moyenne",
        1: "Mauvaise",
        4: "Très mauvaise"
    }

    parametres = [
        'Total Coliform', 'Escherichia Coli', 'Faecal Streptococci', 'Turbidity', 'pH', 'Temperature',
        'Free Chlorine', 'Chlorates', 'Sulfate', 'Magnesium', 'Calcium', 'Conductivity', 'Dry Residue',
        'Complete Alkaline Title', 'Nitrite', 'Ammonium', 'Phosphate', 'Nitrate', 'Iron', 'Manganese',
        'Colour', 'Smell', 'Taste'
    ]

    st.subheader("🧪 Classification de la qualité de l’eau")

    valeurs = []
    for param in parametres:
        val = st.number_input(param, value=0.0, format="%.4f")
        valeurs.append(val)

    if st.button("Prédire la classe de qualité"):
        try:
            model = joblib.load("modele_Classification.pkl")
            X_input = np.array(valeurs).reshape(1, -1)
            prediction = model.predict(X_input)
            classe = classes.get(prediction[0], "Inconnue")
            st.success(f"✅ Classe prédite : **{classe}**")
        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {e}")

    with st.expander("📘 Voir les correspondances des classes encodées"):
        for code, label in classes.items():
            st.write(f"**{code}** → {label}")


st.set_page_config(page_title="Gestion des Prélèvements", layout="wide")
st.title("💧 Application de Suivi et de Gestion des Prélèvements Journaliers")

# === Initialisation session state ===
if 'df_prelèvements' not in st.session_state:
    try:
        st.session_state.df_prelèvements = pd.read_pickle("prelevements_sauvegarde.pkl")
    except:
        st.session_state.df_prelèvements = pd.DataFrame()

# === Liste des paramètres ===
parametres = [
    'Total Coliform', 'Escherichia Coli', 'Faecal Streptococci', 'Turbidity', 'pH', 'Temperature',
    'Free Chlorine', 'Chlorates', 'Sulfate', 'Magnesium', 'Calcium', 'Conductivity', 'Dry Residue',
    'Complete Alkaline Title', 'Nitrite', 'Ammonium', 'Phosphate', 'Nitrate', 'Iron', 'Manganese',
    'Colour', 'Smell', 'Taste'
]

# === Saisie de données ===
st.markdown("### 📝 Saisie d’un nouveau prélèvement")
with st.form(key="saisie_prelevement"):
    col1, col2 = st.columns(2)
    with col1:
        date = st.date_input("Date du prélèvement", value=datetime.today())
        entreprise = st.text_input("Nom de l’entreprise")
        code = st.text_input("Code de l’échantillon")
        preleveur = st.text_input("Nom du préleveur")
    with col2:
        heure = st.time_input("Heure du prélèvement")
        localisation = st.text_input("Localisation")
        analyste = st.text_input("Nom de l’analyste")

    st.markdown("### 🔬 Résultats des analyses")
    resultats = {}
    for param in parametres:
        resultats[param] = st.number_input(param, value=0.0, format="%.4f", key=f"saisie_{param}")

    submitted = st.form_submit_button("Ajouter le prélèvement")

    if submitted:
        new_data = {
            "Date": date,
            "Heure": heure,
            "Entreprise": entreprise,
            "Localisation": localisation,
            "Code": code,
            "Préleveur": preleveur,
            "Analyste": analyste,
        }
        new_data.update(resultats)
        new_df = pd.DataFrame([new_data])
        st.session_state.df_prelèvements = pd.concat([st.session_state.df_prelèvements, new_df], ignore_index=True)
        st.session_state.df_prelèvements.to_pickle("prelevements_sauvegarde.pkl")
        st.success("✅ Prélèvement ajouté avec succès")

# === Filtrage ===
st.markdown("### 🔍 Filtrer les prélèvements")
df = st.session_state.df_prelèvements.copy()
if not df.empty:
    with st.expander("🗂️ Filtres avancés"):
        entreprises = df['Entreprise'].dropna().unique().tolist()
        selected_entreprise = st.selectbox("Entreprise", ["Toutes"] + entreprises)

        dates = df['Date'].astype(str).dropna().unique().tolist()
        selected_date = st.selectbox("Date du prélèvement", ["Toutes"] + sorted(dates))

        if selected_entreprise != "Toutes":
            df = df[df['Entreprise'] == selected_entreprise]
        if selected_date != "Toutes":
            df = df[df['Date'].astype(str) == selected_date]

# === Affichage tableau ===
st.markdown("### 📋 Tableau des prélèvements")
if not df.empty:
    st.dataframe(df)

    def to_excel(df):
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Prélèvements')
        return output.getvalue()

    excel_data = to_excel(df)
    st.download_button("📥 Télécharger (Excel)", data=excel_data, file_name="prelevements.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
else:
    st.info("Aucun prélèvement à afficher.")

# === Graphiques ===
st.markdown("### 📊 Visualisation des paramètres")
if not df.empty:
    parametre_graph = st.selectbox("Sélectionner un paramètre à visualiser :", parametres)
    if parametre_graph:
        fig, ax = plt.subplots()
        try:
            df_sorted = df.sort_values("Date")
            ax.plot(df_sorted["Date"], df_sorted[parametre_graph], marker='o')
            ax.set_title(f"Évolution de {parametre_graph} dans le temps")
            ax.set_xlabel("Date")
            ax.set_ylabel(parametre_graph)
            ax.grid(True)
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Impossible de tracer ce paramètre : {e}")

# === Importation fichier ===
st.markdown("### 📁 Importer un fichier Excel ou CSV")
uploaded_file = st.file_uploader("Choisissez un fichier", type=["xlsx", "csv"])
if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            imported_df = pd.read_csv(uploaded_file)
        else:
            imported_df = pd.read_excel(uploaded_file)

        st.write("✅ Données importées :")
        st.dataframe(imported_df)
    except Exception as e:
        st.error(f"Erreur lors de l'importation : {e}")
