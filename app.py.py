# ==========================================
# 💧 APPLICATION QUALITÉ DE L'EAU - STREAMLIT
# ==========================================
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
from tensorflow.keras.models import load_model

# Configuration de la page
st.set_page_config(page_title="Qualité de l'eau potable", page_icon="💧", layout="wide")

# Initialisation de la navigation
if "page_active" not in st.session_state:
    st.session_state.page_active = "accueil"

if st.session_state.page_active != "accueil":
    with st.sidebar:
        if st.button("🔙 Retour à l'accueil"):
            st.session_state.page_active = "accueil"
            st.stop()

# Page d’accueil
if st.session_state.page_active == "accueil":
    col1, col2 = st.columns([1, 2])
    with col1:
        if os.path.exists("logo.png"):
            st.image("logo.png", width=150)
    with col2:
        st.markdown("### 🌍 Projet IA - Qualité de l’Eau Potable en Algérie")
        st.markdown("🧪 Analyse et prédiction avec intelligence artificielle")
    st.markdown("---")
    if st.button("🚀 Entrer dans l'application"):
        st.session_state.page_active = "application"
        st.stop()
    st.markdown("---")
    st.stop()

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

# Liste des paramètres (extraite automatiquement des normes)
parametres = list(normes.keys())

# 🧠 Fonction de vérification des normes
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

# 🧾 Fonction d’export Excel
def to_excel(df_to_export):
    try:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df_to_export.to_excel(writer, index=False)
        output.seek(0)
        return output.read()
    except:
        return b""
# ================================
# 📋 GESTION JOURNALIÈRE DES PRÉLÈVEMENTS
# ================================
st.header("📋 Base de données des prélèvements journaliers")

# Initialisation du DataFrame
if "df_prelèvements" not in st.session_state:
    try:
        st.session_state.df_prelèvements = pd.read_pickle("prelevements_sauvegarde.pkl")
    except:
        st.session_state.df_prelèvements = pd.DataFrame()

# Initialisation des paramètres dynamiques
if "parametres_dynamiques" not in st.session_state:
    st.session_state.parametres_dynamiques = []

# === Saisie d’un nouveau prélèvement ===
st.subheader("🧾 Ajouter un nouveau prélèvement")

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

    st.markdown("### 🔬 Résultats d’analyse")
    resultats = {}

    for param in parametres:
        resultats[param] = st.number_input(param, value=0.0, format="%.4f", key=f"gestion_{param}")

    for param in st.session_state.parametres_dynamiques:
        resultats[param] = st.number_input(param, value=0.0, format="%.4f", key=f"gestion_dyn_{param}")

    # Validation
    submitted = st.form_submit_button("✅ Enregistrer le prélèvement")
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
        st.success("✅ Prélèvement ajouté avec succès")
        # Affichage alertes
        for a in verifier_parametres_entres(resultats):
            st.warning(a)

# === Paramètres personnalisés ===
st.subheader("⚙️ Ajouter ou supprimer un paramètre personnalisé")

col_add, col_del = st.columns(2)
with col_add:
    new_param = st.text_input("Nouveau paramètre")
    if st.button("➕ Ajouter"):
        if new_param.strip() and new_param not in st.session_state.parametres_dynamiques:
            st.session_state.parametres_dynamiques.append(new_param.strip())
            st.success(f"Ajouté : {new_param}")

with col_del:
    if st.session_state.parametres_dynamiques:
        to_delete = st.selectbox("Paramètre à supprimer", st.session_state.parametres_dynamiques)
        if st.button("❌ Supprimer"):
            st.session_state.parametres_dynamiques.remove(to_delete)
            st.success(f"Supprimé : {to_delete}")
# ================================
# 🔍 PRÉDICTION D’UN PARAMÈTRE MANQUANT
# ================================
st.header("🔍 Prédiction d’un paramètre manquant")

parametre_cible = st.selectbox("Choisir le paramètre à prédire :", parametres)
valeurs = {}

st.markdown("### ✏️ Saisie des autres paramètres :")
for param in parametres:
    if param != parametre_cible:
        valeurs[param] = st.number_input(param, value=0.0, format="%.4f", key=f"pred_{param}")

# Format d’entrée
X = np.array([valeurs[p] for p in valeurs]).reshape(1, -1)

# Bloc RF
model_rf_filename = f"modele_{parametre_cible.replace(' ', '_')}.pkl"
if os.path.exists(model_rf_filename):
    model_rf = joblib.load(model_rf_filename)
    if st.button(f"📈 Prédire avec Random Forest", key="btn_rf"):
        pred_rf = model_rf.predict(X)[0]
        st.success(f"📊 Prédiction avec Random Forest : **{pred_rf:.4f}**")
        for a in verifier_parametres_entres({parametre_cible: pred_rf}):
            st.warning(a)

# Bloc DNN
model_dnn_filename = f"modele_dnn_{parametre_cible.replace(' ', '_')}.h5"
if os.path.exists(model_dnn_filename):
    model_dnn = load_model(model_dnn_filename)
    if st.button(f"🤖 Prédire avec Deep Learning", key="btn_dnn"):
        pred_dnn = model_dnn.predict(X)[0][0]
        st.success(f"🤖 Prédiction avec Deep Learning : **{pred_dnn:.4f}**")
        for a in verifier_parametres_entres({parametre_cible: pred_dnn}):
            st.warning(a)
# ================================
# 🧪 CLASSIFICATION DE LA QUALITÉ DE L’EAU
# ================================
st.header("🧪 Classification de la qualité de l’eau")

# Encodage des classes (doit être identique à l’entraînement)
classes = {
    0: "Bonne",
    1: "Mauvaise",
    2: "Moyenne",
    3: "Très bonne",
    4: "Très mauvaise"
}

# Saisie des paramètres
valeurs_class = {}
for param in parametres:
    valeurs_class[param] = st.number_input(param, value=0.0, format="%.4f", key=f"class_{param}")

X_input = np.array([valeurs_class[p] for p in parametres]).reshape(1, -1)

# === Modèle Random Forest
if os.path.exists("modele_Classification.pkl"):
    model_class_rf = joblib.load("modele_Classification.pkl")
    if st.button("📈 Classifier avec Random Forest", key="btn_class_rf"):
        y_pred_rf = model_class_rf.predict(X_input)[0]
        classe = classes.get(y_pred_rf, "Inconnue")
        st.success(f"✅ Classe prédite (RF) : **{classe}**")
        for a in verifier_parametres_entres(valeurs_class):
            st.warning(a)

# === Modèle Deep Learning
if os.path.exists("modele_classification_dnn.h5"):
    model_class_dnn = load_model("modele_classification_dnn.h5")
    if st.button("🤖 Classifier avec Deep Learning", key="btn_class_dnn"):
        y_pred_dl = model_class_dnn.predict(X_input)
        classe_dl = np.argmax(y_pred_dl, axis=1)[0]
        label = classes.get(classe_dl, "Inconnue")
        st.success(f"🤖 Classe prédite (DNN) : **{label}**")
        for a in verifier_parametres_entres(valeurs_class):
            st.warning(a)

# Affichage des classes disponibles
with st.expander("📘 Voir les correspondances des classes encodées"):
    for code, label in classes.items():
        st.write(f"**{code}** → {label}")
# ================================
# 📊 VISUALISATION & 📤 EXPORT
# ================================
st.header("📊 Visualisation des prélèvements")

df = st.session_state.df_prelèvements.copy()

if not df.empty:
    st.markdown("### 📈 Choisissez un paramètre à visualiser")

    param_to_plot = st.selectbox("Paramètre :", parametres)
    group_by = st.radio("Comparer selon :", ["Date", "Entreprise", "Préleveur"])

    df["Date"] = pd.to_datetime(df["Date"])

    try:
        if group_by == "Date":
            chart = alt.Chart(df).mark_line(point=True).encode(
                x='Date:T',
                y=param_to_plot,
                tooltip=['Date', param_to_plot]
            ).properties(title=f"Évolution de {param_to_plot}")
        else:
            chart = alt.Chart(df).mark_bar().encode(
                x=alt.X(group_by, sort='-y'),
                y=param_to_plot,
                tooltip=[group_by, param_to_plot],
                color=alt.condition(
                    alt.datum[param_to_plot] > normes.get(param_to_plot, {}).get("max", 999),
                    alt.value('red'),
                    alt.value('green')
                ) if param_to_plot in normes else alt.value("steelblue")
            ).properties(title=f"{param_to_plot} par {group_by}")

        st.altair_chart(chart, use_container_width=True)
    except Exception as e:
        st.warning(f"Erreur de graphique : {e}")
else:
    st.info("Aucun prélèvement à visualiser.")

# ================================
# 📤 EXPORT DES DONNÉES
# ================================
st.header("📤 Exporter les données")

if not df.empty:

    # Export Excel
    def to_excel(dataframe):
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            dataframe.to_excel(writer, sheet_name='Prelevements', index=False)
        output.seek(0)
        return output.read()

    excel_file = to_excel(df)

    st.download_button(
        label="📥 Télécharger en Excel",
        data=excel_file,
        file_name="prelevements_eau.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="download_excel"
    )

    # Export PDF
    from fpdf import FPDF

    def to_pdf(dataframe):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=10)
        pdf.cell(200, 10, txt="Rapport des prélèvements d'eau", ln=True, align='C')

        col_names = list(dataframe.columns)
        col_width = 180 / len(col_names)

        pdf.set_font("Arial", 'B', 8)
        for col in col_names:
            pdf.cell(col_width, 8, col[:15], border=1)
        pdf.ln()

        pdf.set_font("Arial", size=7)
        for i, row in dataframe.iterrows():
            for col in col_names:
                val = str(row[col])[:15]
                pdf.cell(col_width, 8, val, border=1)
            pdf.ln()
            if i == 20:  # Limiter à 20 lignes pour éviter surcharge
                pdf.cell(200, 10, "… (données tronquées)", ln=True, align='C')
                break

        output_pdf = io.BytesIO()
        pdf.output(output_pdf)
        return output_pdf.getvalue()

    try:
        pdf_data = to_pdf(df)
        st.download_button(
            label="📄 Télécharger en PDF",
            data=pdf_data,
            file_name="rapport_prelevements.pdf",
            mime="application/pdf",
            key="download_pdf"
        )
    except Exception as e:
        st.warning(f"Erreur PDF : {e}")
else:
    st.info("Aucune donnée à exporter.")


