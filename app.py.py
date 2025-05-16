import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
from PIL import Image


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

           import streamlit as st
import pandas as pd
import io
from datetime import datetime

st.subheader("📊 Gestion des prélèvements journaliers")

# === Initialisation session state ===
if 'df_prelèvements' not in st.session_state:
    st.session_state.df_prelèvements = pd.DataFrame()

# === Formulaire de saisie ===
st.markdown("### 📝 Saisie d’un nouveau prélèvement")

with st.form(key="saisie_prelevement"):
    date = st.date_input("Date du prélèvement", value=datetime.today())
    heure = st.time_input("Heure du prélèvement")
    entreprise = st.text_input("Nom de l’entreprise")
    localisation = st.text_input("Localisation")
    code = st.text_input("Code de l’échantillon")
    preleveur = st.text_input("Nom du préleveur")
    analyste = st.text_input("Nom de l’analyste")

    # Quelques paramètres d'exemple
    pH = st.number_input("pH", value=7.0)
    temperature = st.number_input("Température (°C)", value=20.0)
    chlore = st.number_input("Chlore libre (mg/L)", value=0.5)
    turbidite = st.number_input("Turbidité (NTU)", value=0.3)

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
            "pH": pH,
            "Température": temperature,
            "Chlore libre": chlore,
            "Turbidité": turbidite
        }
        new_df = pd.DataFrame([new_data])
        st.session_state.df_prelèvements = pd.concat([st.session_state.df_prelèvements, new_df], ignore_index=True)
        st.success("✅ Prélèvement ajouté avec succès")

# === Affichage du tableau ===
st.markdown("### 📋 Prélèvements enregistrés")
if not st.session_state.df_prelèvements.empty:
    st.dataframe(st.session_state.df_prelèvements)

    # === Export Excel ===
    def to_excel(df):
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Prélèvements')
        return output.getvalue()

    excel_data = to_excel(st.session_state.df_prelèvements)

    st.download_button(
        label="📥 Télécharger les prélèvements (Excel)",
        data=excel_data,
        file_name="prelevements_journaliers.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

else:
    st.info("Aucun prélèvement enregistré pour le moment.")

# === Importation de fichier ===
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
