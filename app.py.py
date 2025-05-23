# app.py - Point d'entrée principal de l'application

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from PIL import Image
import matplotlib.pyplot as plt
import altair as alt
from tensorflow.keras.models import load_model


# Configuration globale
st.set_page_config(
    page_title="Qualité de l'eau - Algérie 🇩🇿",
    layout="wide",
    page_icon="💧"
)
st.markdown("""
<style>
/* ✅ Arrière-plan élégant */
body {
    background-color: #f9f9fb;
}

/* ✅ Titres */
h1, h2, h3, h4 {
    color: #003566;
    font-family: 'Segoe UI', sans-serif;
    font-weight: 700;
}

/* ✅ Texte */
html, body, [class*="css"]  {
    font-family: 'Segoe UI', sans-serif;
    font-size: 15px;
    color: #333;
}

/* ✅ Boutons principaux */
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

/* ✅ Boutons de téléchargement */
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

/* ✅ Input text & number */
input[type="text"], input[type="number"] {
    border-radius: 6px;
    border: 1px solid #d1d5db;
    padding: 0.4em 0.7em;
}

/* ✅ Select, radio, file uploader */
.stSelectbox, .stRadio, .stFileUploader {
    background-color: #f0f4f8;
    border-radius: 8px;
    padding: 1em;
}

/* ✅ Cards et contenants */
[data-testid="stExpander"] {
    background-color: #ffffff;
    border: 1px solid #e0e0e0;
    border-radius: 10px;
    box-shadow: 0px 2px 10px rgba(0,0,0,0.04);
    padding: 10px;
}

/* ✅ Expander header */
[data-testid="stExpander"] summary {
    font-weight: bold;
    color: #023e8a;
}

/* ✅ Colonnes */
.stColumn {
    padding: 0.5rem;
}

/* ✅ Barre latérale */
[data-testid="stSidebar"] {
    background-color: #edf6f9;
}

/* ✅ Tableau */
thead th {
    background-color: #0077b6;
    color: white;
}

/* ✅ Graphiques */
.vega-embed {
    background-color: #ffffff !important;
    border-radius: 10px;
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)


# Initialisation de la session
if "page" not in st.session_state:
    st.session_state.page = "accueil"


# Design en colonnes
col1, col2 = st.columns([1, 3])
with col1:
    if os.path.exists("logo.png"):
        st.image("logo.png", width=200)

with col2:
    st.markdown("""
        <h1 style='color:#0a9396;'>💧 Analyse intelligente de la qualité de l’eau potable</h1>
        <p style='font-size:18px;'>Un système intelligent pour la surveillance, la prédiction et la classification de la qualité de l’eau selon les normes algériennes (NA 6361-2016).</p>
    """, unsafe_allow_html=True)

st.markdown("---")

# Images d'illustration
img1, img2 = st.columns(2)
with img1:
    if os.path.exists("eau.jpg"):
        st.image("eau.jpg", caption="Analyse physico-chimique", use_container_width=True)
with img2:
    if os.path.exists("eau1.png"):
        st.image("eau1.png", caption="Analyse bactériologique", use_container_width=True)

st.markdown("---")

# Initialisation de la page active
if "page_active" not in st.session_state:
    st.session_state.page_active = "accueil"

# Page d'accueil
if st.session_state.page_active == "accueil":
    st.title("💧 Bienvenue sur l'application Qualité de l'eau")

    if st.button("🚀 Accéder à l’application", use_container_width=True):
        st.session_state.page_active = "application"

# Page principale (après avoir cliqué sur "Accéder")
if st.session_state.page_active == "application":
    st.sidebar.markdown("### Navigation")
    if st.sidebar.button("🔙 Retour à l'accueil"):
        st.session_state.page_active = "accueil"

    # ... ici tu mets les autres blocs de l'app, prédiction, classification, etc.
    st.subheader("📋 Interface principale")


# Équipe projet
st.markdown("""
### 👥 Équipe projet
- 🎓 Étudiants : **FETOUH Yanis**, **AYAD Lewisa Lysa**
- 👩‍🏫 Encadrante : **BOUCHRAKI Faiza**
- 🏛️ Université / Laboratoire partenaire : **[à personnaliser]**

### 💼 Fonctionnalités disponibles :
- 🔍 Prédiction d’un paramètre manquant
- 🧪 Classification IA de la qualité de l’eau
- 📂 Import/Export Excel & PDF
- 📊 Visualisation dynamique des analyses
- 🤖 Assistant IA intégré (chatbot)
- 📘 Recommandations selon les normes algériennes

---
<p style='text-align:center;'>Développée avec ❤️ par l'équipe IA & Eau Potable – Algérie 🇩🇿</p>
""", unsafe_allow_html=True)

# ============ CHARGEMENT DES DONNÉES ============
if "page_active" not in st.session_state:
    st.session_state.page_active = "accueil"

# ============ MENU LATÉRAL DE NAVIGATION ============
menu = st.sidebar.radio("Navigation", [
    "🏠 Accueil",
    "📋 Base de données",
    "🔍 Prédiction paramètre",
    "🧪 Classification",
    "☣️ Type de pollution",
    "📊 Visualisation",
    "📤 Export",
    "💬 Assistant IA",
])

# ============ REDIRECTION PAGE ============
st.session_state.page_active = menu

# ========= IMPORT DES MODULES ==========
# Les prochains blocs s'ajouteront ici dans la suite du code (base de données, prediction, classification, etc.)

# Titre en haut permanent
st.markdown("""
    <style>
        .title {
            font-size: 40px;
            font-weight: 700;
            color: #0a9396;
            text-align: center;
            margin-bottom: 25px;
        }
    </style>
    <div class="title">💧 Application d'analyse intelligente de la qualité de l'eau</div>
""", unsafe_allow_html=True)

# ============ AFFICHAGE ACCUEIL ============
if st.session_state.page_active == "🏠 Accueil":
    col1, col2 = st.columns([1, 2])
    with col1:
        if os.path.exists("logo.png"):
            st.image("logo.png", width=180)
    with col2:
        st.markdown("### 🌍 Projet IA - Surveillance de l'eau potable en Algérie")
        st.markdown("Cette plateforme permet de surveiller, analyser et prédire la qualité de l'eau selon la norme NA 6361-2016.")
        st.markdown("""
        **Fonctionnalités incluses :**
        - Prédiction par Random Forest & Deep Learning
        - Classification de la qualité (5 classes)
        - Détection automatique du type de pollution
        - Interface professionnelle d'analyse et export
        - Assistant IA pour aider les utilisateurs
        """)
        st.markdown("**👩‍💻 Équipe :** FETOUH Yanis, AYAD Lewisa Lysa | Université algérienne 🇩🇿")

    st.markdown("---")
    st.success("Utilise le menu à gauche pour commencer ✅")

# ✅ Liste des 23 paramètres standards
parametres = [
    "Total Coliform", "Escherichia Coli", "Faecal Streptococci",
    "Turbidity", "pH", "Temperature", "Free Chlorine", "Chlorates",
    "Sulfate", "Magnesium", "Calcium", "Conductivity", "Dry Residue",
    "Complete Alkaline Title", "Nitrite", "Ammonium", "Phosphate",
    "Nitrate", "Iron", "Manganese", "Colour", "Smell", "Taste"
]


# ==============================
# 📋 GESTION DES PRÉLÈVEMENTS
# ==============================
st.markdown("<h1 class='main-title'>📋 Base de Données des Prélèvements</h1>", unsafe_allow_html=True)

# Initialisation
if "df_prelèvements" not in st.session_state:
    try:
        st.session_state.df_prelèvements = pd.read_pickle("prelevements_sauvegarde.pkl")
    except:
        st.session_state.df_prelèvements = pd.DataFrame()

if "parametres_dynamiques" not in st.session_state:
    st.session_state.parametres_dynamiques = []

parametres_principaux = [
    'Total Coliform', 'Escherichia Coli', 'Faecal Streptococci', 'Turbidity', 'pH', 'Temperature',
    'Free Chlorine', 'Chlorates', 'Sulfate', 'Magnesium', 'Calcium', 'Conductivity', 'Dry Residue',
    'Complete Alkaline Title', 'Nitrite', 'Ammonium', 'Phosphate', 'Nitrate', 'Iron', 'Manganese',
    'Colour', 'Smell', 'Taste'
]

if "parametres_actifs" not in st.session_state:
    st.session_state.parametres_actifs = parametres_principaux.copy()

# Modification des paramètres disponibles
st.markdown("### ⚙️ Gérer les Paramètres Mesurés")
col1, col2 = st.columns(2)
with col1:
    param_to_remove = st.selectbox("🔻 Supprimer un paramètre", st.session_state.parametres_actifs)
    if st.button("❌ Supprimer"):
        if param_to_remove in st.session_state.parametres_actifs:
            st.session_state.parametres_actifs.remove(param_to_remove)
            st.success(f"✅ '{param_to_remove}' supprimé des paramètres mesurés.")

with col2:
    param_to_add = st.text_input("➕ Ajouter un nouveau paramètre")
    if st.button("✅ Ajouter"):
        if param_to_add and param_to_add not in st.session_state.parametres_actifs:
            st.session_state.parametres_actifs.append(param_to_add.strip())
            st.success(f"✅ '{param_to_add}' ajouté aux paramètres mesurés.")

# Formulaire de saisie de prélèvement
st.markdown("### 🧾 Saisir un nouveau prélèvement")
with st.form("form_prelevement"):
    col1, col2 = st.columns(2)
    with col1:
        date = st.date_input("Date", value=datetime.today())
        heure = st.time_input("Heure")
        entreprise = st.text_input("Entreprise")
        code = st.text_input("Code")
    with col2:
        preleveur = st.text_input("Préleveur")
        localisation = st.text_input("Localisation")
        analyste = st.text_input("Analyste")

    st.markdown("### 🔬 Résultats des paramètres")
    resultats = {}
    for param in st.session_state.parametres_actifs:
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

# Affichage du tableau
st.markdown("### 📊 Tableau des Prélèvements")
if not st.session_state.df_prelèvements.empty:
    st.dataframe(st.session_state.df_prelèvements)
else:
    st.info("Aucune donnée enregistrée pour le moment.")
# ================================
# 🔍 PRÉDICTION D’UN PARAMÈTRE MANQUANT (VERSION COMPLÈTE)
# ================================
# ✅ Liste complète des 23 paramètres selon les normes algériennes
parametres = [
    "Total Coliform", "Escherichia Coli", "Faecal Streptococci",
    "Turbidity", "pH", "Temperature", "Free Chlorine", "Chlorates",
    "Sulfate", "Magnesium", "Calcium", "Conductivity", "Dry Residue",
    "Complete Alkaline Title", "Nitrite", "Ammonium", "Phosphate",
    "Nitrate", "Iron", "Manganese", "Colour", "Smell", "Taste"
]

# ✅ Sélection du paramètre à prédire (important à faire avant la saisie)
parametre_cible = st.selectbox("🧪 Quel paramètre veux-tu prédire ?", parametres)

st.header("🔍 Prédiction d’un paramètre manquant")

# 🔧 Normes algériennes des 23 paramètres et conseils
normes = {
    "Total Coliform": {"max": 0, "conseil": "Désinfecter le réseau et contrôler la source d’eau."},
    "Escherichia Coli": {"max": 0, "conseil": "Procéder à une chloration et vérifier les sources fécales."},
    "Faecal Streptococci": {"max": 0, "conseil": "Analyser les infiltrations et renforcer le traitement."},
    "Turbidity": {"max": 5, "conseil": "Utiliser un préfiltre ou une clarification plus poussée."},
    "pH": {"min": 6.5, "max": 8.5, "conseil": "Corriger avec des agents basifiants ou acidifiants."},
    "Temperature": {"max": 25, "conseil": "Protéger les réservoirs de la chaleur excessive."},
    "Free Chlorine": {"min": 0.2, "max": 0.5, "conseil": "Ajuster le dosage de chlore dans l’eau."},
    "Chlorates": {"max": 0.7, "conseil": "Réduire les sous-produits de désinfection."},
    "Sulfate": {"max": 250, "conseil": "Filtrer avec des résines échangeuses d’ions si excès."},
    "Magnesium": {"max": 50, "conseil": "Utiliser un adoucisseur si besoin."},
    "Calcium": {"max": 200, "conseil": "Réguler pour éviter l’entartrage."},
    "Conductivity": {"max": 2800, "conseil": "Vérifier les sels dissous totaux."},
    "Dry Residue": {"max": 1500, "conseil": "Effectuer une osmose inverse si excès."},
    "Complete Alkaline Title": {"min": 100, "max": 300, "conseil": "Ajuster pour la stabilité de l’eau."},
    "Nitrite": {"max": 0.5, "conseil": "Vérifier la dégradation de la matière organique."},
    "Ammonium": {"max": 0.5, "conseil": "Contrôler les contaminations fécales et organiques."},
    "Phosphate": {"max": 5, "conseil": "Réduire les rejets domestiques ou agricoles."},
    "Nitrate": {"max": 50, "conseil": "Limiter l’usage des engrais et assainir les sources."},
    "Iron": {"max": 0.3, "conseil": "Filtrer à l’aide d’oxydation préalable."},
    "Manganese": {"max": 0.1, "conseil": "Utiliser un filtre catalytique."},
    "Colour": {"max": 0, "conseil": "Identifier les composés organiques ou ferreux."},
    "Smell": {"max": 0, "conseil": "Chercher les sources de contamination ou stagnation."},
    "Taste": {"max": 0, "conseil": "Analyser les composés désinfectants ou organiques."}
}

# ✅ Fonction de vérification des normes
def verifier_parametres_entres(valeurs: dict):
    alertes = []
    for param, valeur in valeurs.items():
        if param in normes:
            règle = normes[param]
            if ("min" in règle and valeur < règle["min"]) or ("max" in règle and valeur > règle["max"]):
                min_val = règle.get("min", "-")
                max_val = règle.get("max", "-")
                conseil = règle.get("conseil", "")
                alertes.append(f"⚠️ **{param} = {valeur:.2f}** est hors norme ({min_val} - {max_val}). 💡 {conseil}")
    return alertes

st.subheader("🔍 Prédiction d’un paramètre manquant (IA)")

# Saisie des autres paramètres
valeurs_pred = {}
st.markdown("### ✏️ Saisissez les valeurs mesurées :")
for param in parametres:
    if param != parametre_cible:
        valeurs_pred[param] = st.number_input(param, value=0.0, format="%.4f", key=f"pred_{param}")

# Conversion en tableau
X_input = np.array([valeurs_pred[p] for p in valeurs_pred]).reshape(1, -1)

# ============================
# 🌲 Prédiction avec Random Forest
# ============================
model_rf_filename = f"modele_{parametre_cible.replace(' ', '_')}.pk1"
if os.path.exists(model_rf_filename):
    model_rf = joblib.load(model_rf_filename)
    if st.button("📈 Prédire avec Random Forest"):
        pred_rf = model_rf.predict(X_input)[0]
        st.success(f"🌲 Résultat RF : **{parametre_cible} = {pred_rf:.4f}**")
        for alerte in verifier_parametres_entres({parametre_cible: pred_rf}):
            st.warning(alerte)
else:
    st.warning(f"Modèle Random Forest indisponible pour {parametre_cible}")

# ============================
# 🤖 Prédiction avec Deep Learning
# ============================
model_dnn_filename = f"modele_dnn_{parametre_cible.replace(' ', '_')}.h5"
if os.path.exists(model_dnn_filename):
    try:
        model_dnn = load_model(model_dnn_filename, compile=False)
        if st.button("🤖 Prédire avec Deep Learning"):
            pred_dnn = model_dnn.predict(X_input)[0][0]
            st.success(f"🤖 Résultat DNN : **{parametre_cible} = {pred_dnn:.4f}**")
            for alerte in verifier_parametres_entres({parametre_cible: pred_dnn}):
                st.warning(alerte)
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle DNN : {e}")
else:
    st.warning(f"Modèle DNN indisponible pour {parametre_cible}")

# Saisie des paramètres
valeurs_class = {}
st.markdown("### 🔢 Veuillez saisir les 23 paramètres pour la classification :")
for param in parametres:
    valeurs_class[param] = st.number_input(param, value=0.0, format="%.4f", key=f"class_{param}")

X_input = np.array([valeurs_class[p] for p in parametres]).reshape(1, -1)

# === Prédiction avec Random Forest
if os.path.exists("modele_Classification.pk1"):
    model_class_rf = joblib.load("modele_Classification.pk1")
    # Définir le dictionnaire des classes
    # Cette ligne manquait probablement

classes = {
    0: "Bonne",
    1: "Mauvaise",
    2: "Moyenne",
    3: "Très bonne",
    4: "Très mauvaise"
}

# Chargement du modèle
try:
    model_class_rf = joblib.load("modele_Classification.pk1")
    prediction_rf = model_class_rf.predict(X_input)[0]
    label_rf = classes.get(prediction_rf, "Inconnue")
    st.success(f"🌲 Classification Random Forest : **{label_rf}**")
except Exception as e:
    st.error(f"❌ Erreur lors de la prédiction avec Random Forest : {e}")


# Explication des classes
with st.expander("ℹ️ Voir la signification des classes de qualité d’eau"):
    for code, label in classes.items():
        st.write(f"**{code}** → {label}")
from fpdf import FPDF

def generer_rapport_pdf(classe_predite, valeurs, alertes, methode="Random Forest"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    pdf.set_text_color(0, 102, 204)
    pdf.cell(200, 10, txt="Rapport de Classification de la Qualité de l’Eau", ln=True, align='C')
    
    pdf.set_text_color(0, 0, 0)
    pdf.ln(10)
    pdf.set_font("Arial", style="B", size=11)
    pdf.cell(200, 10, txt=f"🧠 Méthode : {methode}", ln=True)
    pdf.cell(200, 10, txt=f"✅ Classe prédite : {classe_predite}", ln=True)

    pdf.ln(5)
    pdf.set_font("Arial", style="B", size=10)
    pdf.cell(200, 10, txt="📌 Paramètres analysés :", ln=True)
    
    pdf.set_font("Arial", size=9)
    for k, v in valeurs.items():
        pdf.cell(95, 8, f"{k}: {v:.2f}", ln=False)
        if (list(valeurs.keys()).index(k) + 1) % 2 == 0:
            pdf.ln()

    pdf.ln(10)
    if alertes:
        pdf.set_font("Arial", style="B", size=10)
        pdf.set_text_color(255, 0, 0)
        pdf.cell(200, 10, txt="⚠️ Paramètres hors normes :", ln=True)
        pdf.set_font("Arial", size=9)
        pdf.set_text_color(0, 0, 0)
        for alerte in alertes:
            pdf.multi_cell(0, 8, alerte)
    else:
        pdf.set_text_color(0, 150, 0)
        pdf.cell(200, 10, txt="Tous les paramètres sont dans les normes ✅", ln=True)

    buffer = io.BytesIO()
    pdf.output(buffer)
    buffer.seek(0)
    return buffer
# === Bloc : Détection automatique du type de pollution ===
st.header("🧪 Détection du type de pollution")

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
if st.session_state.page_active == "assistant":
    st.title("🤖 Assistant IA – Aide et support intelligent")

    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")

    if not openai.api_key:
        st.error("⚠️ Clé API OpenAI manquante. Vérifie ton fichier `.env`")
    else:
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

            try:
                with st.chat_message("assistant"):
                    with st.spinner("Réflexion..."):
                        response = openai.ChatCompletion.create(
                            model="gpt-3.5-turbo",
                            messages=st.session_state.messages
                        ).choices[0].message.content
                        st.markdown(response)

                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Erreur d’appel OpenAI : {e}")

import base64
from fpdf import FPDF

st.header("📂 Importer / Exporter un fichier Excel ou CSV")

# === Import du fichier utilisateur
uploaded_file = st.file_uploader("📤 Charger un fichier contenant les paramètres d’analyse", type=["xlsx", "csv"])

if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df_uploaded = pd.read_csv(uploaded_file)
        else:
            df_uploaded = pd.read_excel(uploaded_file)

        st.success("✅ Fichier importé avec succès !")
        st.dataframe(df_uploaded)

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

        # === Bouton Export Excel
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

        # === Bouton Export PDF
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

    except Exception as e:
        st.error(f"❌ Erreur de traitement : {e}")
