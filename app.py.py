import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from fpdf import FPDF
import io
from tensorflow.keras.models import load_model

# Configuration de la page
st.set_page_config(
    page_title="Analyse de la Qualité de l'Eau",
    page_icon="💧",
    layout="wide"
)

# Style CSS pour améliorer l'esthétique
st.markdown("""
<style>
body {
    background-color: #f0f4f8;
}
h1, h2, h3 {
    color: #0077b6;
    font-family: 'Segoe UI', sans-serif;
}
.stButton > button {
    background-color: #0077b6;
    color: white;
    border: none;
    padding: 0.5em 1.3em;
    border-radius: 8px;
    font-weight: 600;
    transition: background-color 0.3s ease;
}
.stButton > button:hover {
    background-color: #005f8a;
}
.stSelectbox, .stTextInput, .stNumberInput {
    border-radius: 6px;
    border: 1px solid #d1d5db;
    padding: 0.4em 0.7em;
}
.stDownloadButton > button {
    background-color: #00b4d8;
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    transition: background-color 0.3s ease;
}
.stDownloadButton > button:hover {
    background-color: #0096c7;
}
.header {
    text-align: center;
    margin-bottom: 20px;
}
.image-container {
    display: flex;
    justify-content: center;
    margin: 20px 0;
}
.image-container img {
    width: 300px;
    margin: 0 10px;
}
</style>
""", unsafe_allow_html=True)

# Fonction pour afficher la page d'accueil
def show_home_page():
    st.title("💧 Application d'Analyse de la Qualité de l'Eau")
    st.markdown("""
    <div class="header">
        <h2>Bienvenue dans notre application d'analyse de la qualité de l'eau</h2>
        <p>Cette application permet de surveiller, analyser et prédire la qualité de l'eau selon les normes algériennes (NA 6361-2016).</p>
    </div>
    """, unsafe_allow_html=True)

    # Ajout d'images d'illustration
    st.markdown("<div class='image-container'>", unsafe_allow_html=True)
    st.markdown("<img src='https://example.com/image1.jpg' alt='Image 1'>", unsafe_allow_html=True)  # Remplacez par vos images
    st.markdown("<img src='https://example.com/image2.jpg' alt='Image 2'>", unsafe_allow_html=True)  # Remplacez par vos images
    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("🚀 Accéder à l'application"):
        st.session_state.page = "main"

# Fonction pour afficher le menu de navigation
def show_navigation():
    st.sidebar.title("🌐 Navigation")
    page = st.sidebar.radio("Aller vers :", [
        "Accueil",
        "Gestion des Prélèvements",
        "Classification",
        "Prédiction d'un Paramètre",
        "Détection de Pollution",
        "Assistant IA",
        "Exportation"
    ])
    st.session_state.page = page

# Fonction pour afficher la page principale
def show_main_page():
    st.title("💧 Analyse de la Qualité de l'Eau")
    st.markdown("Cette application permet de surveiller, analyser et prédire la qualité de l'eau selon les normes algériennes (NA 6361-2016).")

    # Liste des 23 paramètres
    parametres = [
        'Total Coliform', 'Escherichia Coli', 'Faecal Streptococci', 'Turbidity', 'pH', 'Temperature',
        'Free Chlorine', 'Chlorates', 'Sulfate', 'Magnesium', 'Calcium', 'Conductivity', 'Dry Residue',
        'Complete Alkaline Title', 'Nitrite', 'Ammonium', 'Phosphate', 'Nitrate', 'Iron', 'Manganese',
        'Colour', 'Smell', 'Taste'
    ]

    # Gestion des prélèvements
    st.header("📋 Gestion des Prélèvements Journaliers")
    if "df_prelèvements" not in st.session_state:
        st.session_state.df_prelèvements = pd.DataFrame(columns=["Date", "Heure", "Entreprise", "Code", "Préleveur", "Localisation", "Analyste"] + parametres)

    with st.form("form_prelevement"):
        st.subheader("Saisir un nouveau prélèvement")
        date = st.date_input("Date", value=datetime.today())
        heure = st.time_input("Heure")
        entreprise = st.text_input("Entreprise")
        code = st.text_input("Code")
        preleveur = st.text_input("Préleveur")
        localisation = st.text_input("Localisation")
        analyste = st.text_input("Analyste")

        st.markdown("### 🔬 Résultats des paramètres")
        resultats = {}
        for param in parametres:
            resultats[param] = st.number_input(param, value=0.0, format="%.4f", key=f"val_{param}")

        submitted = st.form_submit_button("💾 Enregistrer le prélèvement")
        if submitted:
            new_data = {
                "Date": date,
                "Heure": heure,
                "Entreprise": entreprise,
                "Code": code,
                "Préleveur": preleveur,
                "Localisation": localisation,
                "Analyste": analyste,
            }
            new_data.update(resultats)
            st.session_state.df_prelèvements = pd.concat([
                st.session_state.df_prelèvements,
                pd.DataFrame([new_data])
            ], ignore_index=True)
            st.session_state.df_prelèvements.to_pickle("prelevements_sauvegarde.pkl")
            st.success("✅ Prélèvement enregistré avec succès.")

    # Affichage du tableau des prélèvements
    st.markdown("### 📊 Tableau des Prélèvements")
    if not st.session_state.df_prelèvements.empty:
        st.dataframe(st.session_state.df_prelèvements)
    else:
        st.info("Aucune donnée enregistrée pour le moment.")

    # Chargement du fichier utilisateur
    uploaded_file = st.file_uploader("📤 Charger un fichier contenant les paramètres d’analyse", type=["xlsx", "csv"])

    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                df_uploaded = pd.read_csv(uploaded_file)
            else:
                df_uploaded = pd.read_excel(uploaded_file)

            st.success("✅ Fichier importé avec succès !")
            st.dataframe(df_uploaded)

            # Vérification des colonnes nécessaires
            if not all(param in df_uploaded.columns for param in parametres):
                st.error("❌ Le fichier importé ne contient pas toutes les colonnes nécessaires.")
            else:
                # Choix d’action
                action = st.selectbox("Sélectionnez une action à appliquer :", ["Aucune", "Classification", "Prédiction d'un paramètre", "Détection du type de pollution"])

                if action == "Classification":
                    model_class = joblib.load("modele_Classification.pk1")
                    X = df_uploaded[parametres]
                    y_pred = model_class.predict(X)
                    classes = {0: "Bonne", 1: "Mauvaise", 2: "Moyenne", 3: "Très bonne", 4: "Très mauvaise"}
                    df_uploaded["Classe Prédite"] = [classes.get(i, "Inconnue") for i in y_pred]
                    st.success("✅ Classification effectuée.")
                    st.dataframe(df_uploaded)

                    # Conseils après classification
                    st.markdown("### 📝 Conseils après Classification")
                    for index, row in df_uploaded.iterrows():
                        classe = row["Classe Prédite"]
                        if classe == "Mauvaise":
                            st.warning(f"⚠️ Prélèvement {index + 1}: La qualité de l'eau est **Mauvaise**. Recommandation: Vérifiez la source et effectuez des tests supplémentaires.")
                        elif classe == "Moyenne":
                            st.info(f"ℹ️ Prélèvement {index + 1}: La qualité de l'eau est **Moyenne**. Recommandation: Surveillez régulièrement la qualité.")
                        elif classe == "Bonne":
                            st.success(f"✅ Prélèvement {index + 1}: La qualité de l'eau est **Bonne**. Continuez à surveiller.")

                elif action == "Prédiction d'un paramètre":
                    param_to_predict = st.selectbox("Quel paramètre voulez-vous prédire ?", parametres)
                    model_file = f"modele_{param_to_predict.replace(' ', '_')}.pk1"
                    if os.path.exists(model_file):
                        model = joblib.load(model_file)
                        input_cols = [p for p in parametres if p != param_to_predict]
                        pred = model.predict(df_uploaded[input_cols])
                        df_uploaded[f"{param_to_predict}_Prédit"] = pred
                        st.success(f"✅ Prédiction du paramètre {param_to_predict} terminée.")
                        st.dataframe(df_uploaded)
                    else:
                        st.warning(f"❌ Modèle non trouvé pour {param_to_predict}")

                elif action == "Détection du type de pollution":
                    def detect_pollution(row):
                        types = []
                        if row["Escherichia Coli"] > 0 or row["Total Coliform"] > 0:
                            types.append("biologique")
                        if row["Nitrate"] > 50 or row["Chlorates"] > 0.7:
                            types.append("chimique")
                        if row["Ammonium"] > 0.5 or row["Turbidity"] > 5:
                            types.append("organique")
                        if row["Iron"] > 0.3 or row["Manganese"] > 0.1:
                            types.append("métallique")
                        return "multiple" if len(types) > 1 else (types[0] if types else "aucune")

                    df_uploaded["Type de pollution"] = df_uploaded.apply(detect_pollution, axis=1)
                    st.success("✅ Type de pollution détecté.")
                    st.dataframe(df_uploaded)

                # Exporter les résultats en Excel
                def to_excel(dataframe):
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        dataframe.to_excel(writer, index=False, sheet_name="Résultats")
                    output.seek(0)
                    return output.read()

                excel_data = to_excel(df_uploaded)

                st.download_button(
                    label="📥 Télécharger les résultats (Excel)",
                    data=excel_data,
                    file_name="resultats_analyse_eau.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

                # Exporter les résultats en PDF
                def to_pdf(df):
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_font("Arial", size=10)
                    pdf.cell(200, 10, txt="Rapport des résultats d’analyse de la qualité de l’eau", ln=True, align='C')
                    pdf.ln(10)

                    cols = list(df.columns)
                    col_width = 180 / len(cols)

                    pdf.set_font("Arial", 'B', 7)
                    for col in cols:
                        pdf.cell(col_width, 8, col[:15], border=1)
                    pdf.ln()

                    pdf.set_font("Arial", size=7)
                    for i, row in df.iterrows():
                        for col in cols:
                            pdf.cell(col_width, 8, str(row[col])[:15], border=1)
                        pdf.ln()
                        if i >= 30:
                            pdf.cell(200, 10, "… (résultats tronqués)", ln=True, align='C')
                            break

                    output = io.BytesIO()
                    pdf.output(output)
                    return output.getvalue()

                pdf_bytes = to_pdf(df_uploaded)

                st.download_button(
                    label="📄 Télécharger le rapport PDF",
                    data=pdf_bytes,
                    file_name="rapport_resultats_eau.pdf",
                    mime="application/pdf"
                )

# Gestion de la navigation entre les pages
if "page" not in st.session_state:
    st.session_state.page = "home"

if st.session_state.page == "home":
    show_home_page()
else:
    show_main_page()

# Menu de navigation
show_navigation()

# Bouton de sortie
if st.button("🚪 Quitter l'application"):
    st.session_state.page = "home"
