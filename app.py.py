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

# Fonction pour la classification
def classify_samples():
    st.header("🧪 Classification de la Qualité de l'Eau")
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

        except Exception as e:
            st.error(f"❌ Erreur de traitement : {e}")

# Fonction pour prédire un paramètre
def predict_parameter():
    st.header("🔍 Prédiction d’un Paramètre Manquant")
    parametres = [
        'Total Coliform', 'Escherichia Coli', 'Faecal Streptococci', 'Turbidity', 'pH', 'Temperature',
        'Free Chlorine', 'Chlorates', 'Sulfate', 'Magnesium', 'Calcium', 'Conductivity', 'Dry Residue',
        'Complete Alkaline Title', 'Nitrite', 'Ammonium', 'Phosphate', 'Nitrate', 'Iron', 'Manganese',
        'Colour', 'Smell', 'Taste'
    ]

    # Choix du paramètre cible
    parametre_cible = st.selectbox("Choisir le paramètre à prédire :", parametres)

    # Saisie des autres paramètres
    valeurs_pred = {}
    st.markdown("### ✏️ Saisie des autres paramètres :")
    for param in parametres:
        if param != parametre_cible:
            valeurs_pred[param] = st.number_input(param, value=0.0, format="%.4f", key=f"pred_{param}")

    X_input = np.array([valeurs_pred[p] for p in valeurs_pred]).reshape(1, -1)

    # Prédiction avec Random Forest
    model_rf_file = f"modele_{parametre_cible.replace(' ', '_')}.pk1"
    if os.path.exists(model_rf_file):
        model_rf = joblib.load(model_rf_file)
        if st.button("📈 Prédire avec Random Forest"):
            try:
                pred_rf = model_rf.predict(X_input)[0]
                st.success(f"📊 Valeur prédite (RF) pour **{parametre_cible}** : `{pred_rf:.4f}`")
            except Exception as e:
                st.error(f"Erreur avec Random Forest : {e}")
    else:
        st.warning(f"Modèle RF non trouvé : {model_rf_file}")

    # Prédiction avec Deep Learning
    model_dnn_file = f"modele_dnn_{parametre_cible.replace(' ', '_')}.h5"
    if os.path.exists(model_dnn_file):
        model_dnn = load_model(model_dnn_file)
        if st.button("🤖 Prédire avec Deep Learning"):
            try:
                pred_dnn = model_dnn.predict(X_input)[0][0]
                st.success(f"🤖 Valeur prédite (DNN) pour **{parametre_cible}** : `{pred_dnn:.4f}`")
            except Exception as e:
                st.error(f"Erreur avec Deep Learning : {e}")
    else:
        st.warning(f"Modèle DNN non trouvé : {model_dnn_file}")

# Fonction pour détecter le type de pollution
def detect_pollution():
    st.header("🧪 Détection du Type de Pollution")

    valeurs_pollution = {}
    for param in parametres:
        valeurs_pollution[param] = st.number_input(f"{param}", value=0.0, format="%.4f", key=f"poll_{param}")

    def detecter_type_pollution(valeurs):
        types = set()

        if valeurs["Escherichia Coli"] > 0 or valeurs["Total Coliform"] > 0 or valeurs["Faecal Streptococci"] > 0:
            types.add("biologique")
        if valeurs["Nitrate"] > 50 or valeurs["Nitrite"] > 0.5 or valeurs["Chlorates"] > 0.7 or valeurs["Phosphate"] > 5:
            types.add("chimique")
        if valeurs["Ammonium"] > 0.5 or valeurs["Turbidity"] > 5 or valeurs["Temperature"] > 25:
            types.add("organique")
        if valeurs["Iron"] > 0.3 or valeurs["Manganese"] > 0.1:
            types.add("métallique")
        if valeurs["Calcium"] > 200 or valeurs["Magnesium"] > 50 or valeurs["Sulfate"] > 250 or valeurs["Dry Residue"] > 1500:
            types.add("minéralogique")
        if valeurs["Smell"] > 0 or valeurs["Taste"] > 0 or valeurs["Colour"] > 0:
            types.add("sensorielle")

        if not types:
            return ["aucune"], []

        # Recommandations automatiques
        conseils = []
        for t in types:
            if t == "biologique":
                conseils.append("🔬 Pollution biologique détectée : désinfecter le réseau, vérifier la source.")
            elif t == "chimique":
                conseils.append("🧪 Pollution chimique détectée : contrôler les intrants agricoles ou industriels.")
            elif t == "organique":
                conseils.append("🧫 Pollution organique détectée : renforcer la filtration et l’assainissement.")
            elif t == "métallique":
                conseils.append("⚙️ Pollution métallique détectée : utiliser des filtres spécifiques (Fe/Mn).")
            elif t == "minéralogique":
                conseils.append("🧱 Pollution minéralogique détectée : vérifier la source et réduire les minéraux dissous.")
            elif t == "sensorielle":
                conseils.append("👃 Pollution sensorielle détectée : analyser les composés organoleptiques.")

        return list(types), conseils

    if st.button("🧠 Détecter le type de pollution", key="btn_detect_pollution"):
        types_detectés, conseils = detecter_type_pollution(valeurs_pollution)
        
        if "aucune" in types_detectés:
            st.success("✅ Aucune pollution détectée selon les normes.")
        else:
            st.error(f"⚠️ Types de pollution détectés : {', '.join(types_detectés).capitalize()}")
            for c in conseils:
                st.info(c)

# Fonction pour l'assistant IA
def assistant_ia():
    st.title("🤖 Assistant IA – Aide et support intelligent")

    # Historique de conversation
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Bonjour 👋, je suis l’assistant IA. Posez-moi vos questions sur l’application ou la qualité de l’eau."}]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Zone de saisie utilisateur
    if prompt := st.chat_input("Posez votre question ici..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Appel à l’API OpenAI
        try:
            with st.chat_message("assistant"):
                with st.spinner("Réflexion..."):
                    completion = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=st.session_state.messages
                    )
                    response = completion.choices[0].message.content
                    st.markdown(response)

            st.session_state.messages.append({"role": "assistant", "content": response})

        except Exception as e:
            st.error(f"Erreur d’appel à l’API OpenAI : {e}")

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
