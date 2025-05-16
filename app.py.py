import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
from PIL import Image
import io
from datetime import datetime
import matplotlib.pyplot as plt

# ======= CONFIGURATION PAGE =======
st.set_page_config(page_title="Qualité de l'eau potable", page_icon="💧", layout="wide")

# ======= DICTIONNAIRE DES NORMES ALGÉRIENNES ET CONSEILS =======
normes = {
    'pH': {"min": 6.5, "max": 8.5, "conseil": "Pour corriger le pH, ajustez avec des agents acidifiants ou basifiants adaptés."},
    'Turbidity': {"max": 5, "conseil": "Utilisez une filtration adaptée pour réduire la turbidité."},
    'Free Chlorine': {"min": 0.2, "max": 0.5, "conseil": "Ajustez le dosage de chlore libre pour respecter les normes."},
    'Nitrate': {"max": 50, "conseil": "Évitez les sources de pollution agricole et industrielle."},
    'Temperature': {"max": 30, "conseil": "Stockez l’eau à une température adéquate pour limiter la prolifération bactérienne."},
    # Compléter avec les autres paramètres et leurs seuils si besoin
}

# ======= PARAMÈTRES DE BASE =======
parametres = [
    'Total Coliform', 'Escherichia Coli', 'Faecal Streptococci', 'Turbidity', 'pH', 'Temperature',
    'Free Chlorine', 'Chlorates', 'Sulfate', 'Magnesium', 'Calcium', 'Conductivity', 'Dry Residue',
    'Complete Alkaline Title', 'Nitrite', 'Ammonium', 'Phosphate', 'Nitrate', 'Iron', 'Manganese',
    'Colour', 'Smell', 'Taste'
]

# ======= FONCTION POUR VÉRIFIER LES NORMES ET FOURNIR DES CONSEILS =======
def verifier_parametres_entres(valeurs: dict):
    alertes = []
    for param, valeur in valeurs.items():
        if param in normes:
            seuil = normes[param]
            if ("min" in seuil and valeur < seuil["min"]) or ("max" in seuil and valeur > seuil["max"]):
                min_text = f"{seuil['min']}" if "min" in seuil else "-"
                max_text = f"{seuil['max']}" if "max" in seuil else "-"
                message = (f"⚠️ **{param} = {valeur:.2f}** est hors norme "
                           f"(norme : {min_text} - {max_text}).\n"
                           f"💡 Conseil : {seuil['conseil']}")
                alertes.append(message)
    return alertes

# ======= TITRE PRINCIPAL =======
st.title("💧 Application IA - Prédiction, Classification et Gestion de la qualité de l'eau")

# ======= IMAGE =======
if os.path.exists("eau.jpg"):
    image = Image.open("eau.jpg")
    st.image(image, caption="Analyse de la qualité de l'eau", use_column_width=True)

# ======= MENU ONGLET =======
tabs = st.tabs(["🔍 Prédiction", "🧪 Classification", "📋 Gestion des prélèvements"])

# ======= ONGLET 1 : PRÉDICTION D’UN PARAMÈTRE MANQUANT =======
with tabs[0]:
    st.header("🔍 Prédiction d’un paramètre manquant")

    parametre_cible = st.selectbox("Quel paramètre veux-tu prédire ?", parametres)
    model_filename = f"modele_{parametre_cible.replace(' ', '_')}.pkl"

    if not os.path.exists(model_filename):
        st.error(f"❌ Le modèle '{model_filename}' est introuvable.")
    else:
        model = joblib.load(model_filename)
        valeurs = {}
        for param in parametres:
            if param != parametre_cible:
                valeurs[param] = st.number_input(param, value=0.0, format="%.4f")

        if st.button("Prédire le paramètre manquant"):
            X_input = np.array([valeurs[param] for param in parametres if param != parametre_cible]).reshape(1, -1)
            prediction = model.predict(X_input)
            st.success(f"🔍 Prédiction pour **{parametre_cible}** : `{prediction[0]:.4f}`")

            # Vérification normes + conseils
            alertes = verifier_parametres_entres({parametre_cible: prediction[0]})
            if alertes:
                for msg in alertes:
                    st.warning(msg)
            else:
                st.success("✅ Le paramètre prédit est conforme aux normes.")

# ======= ONGLET 2 : CLASSIFICATION DE LA QUALITÉ =======
with tabs[1]:
    st.header("🧪 Classification de la qualité de l’eau")
    classes = {3: "Très bonne", 0: "Bonne", 2: "Moyenne", 1: "Mauvaise", 4: "Très mauvaise"}

    valeurs = {}
    for param in parametres:
        valeurs[param] = st.number_input(param, value=0.0, format="%.4f")

    if st.button("Prédire la classe de qualité"):
        try:
            model = joblib.load("modele_Classification.pkl")
            X_input = np.array([valeurs[param] for param in parametres]).reshape(1, -1)
            prediction = model.predict(X_input)
            classe = classes.get(prediction[0], "Inconnue")
            st.success(f"✅ Classe prédite : **{classe}**")

            # Vérification normes + conseils
            alertes = verifier_parametres_entres(valeurs)
            if alertes:
                for msg in alertes:
                    st.warning(msg)
            else:
                st.success("✅ Tous les paramètres respectent les normes.")
        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {e}")

    with st.expander("📘 Voir les correspondances des classes encodées"):
        for code, label in classes.items():
            st.write(f"**{code}** → {label}")

# ======= ONGLET 3 : GESTION DES PRÉLÈVEMENTS =======
with tabs[2]:
    st.header("📋 Gestion des prélèvements journaliers")

    # Initialisation ou chargement des données sauvegardées
    if 'df_prelèvements' not in st.session_state:
        try:
            st.session_state.df_prelèvements = pd.read_pickle("prelevements_sauvegarde.pkl")
        except:
            st.session_state.df_prelèvements = pd.DataFrame()

    # Paramètres dynamiques ajoutés par l’utilisateur
    if 'parametres_dynamiques' not in st.session_state:
        st.session_state.parametres_dynamiques = {}

    # Formulaire saisie nouveau prélèvement
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

        # Affichage paramètres dynamiques existants
        if st.session_state.parametres_dynamiques:
            st.markdown("### ⚙️ Paramètres personnalisés ajoutés")
            for p, v in st.session_state.parametres_dynamiques.items():
                resultats[p] = st.number_input(p, value=float(v), format="%.4f", key=f"saisie_dyn_{p}")

        # Section ajout paramètre personnalisé
        with st.expander("➕ Ajouter un paramètre personnalisé"):
            nouveau_param = st.text_input("Nom du paramètre")
            valeur_param = st.number_input("Valeur", value=0.0, format="%.4f")
            if st.button("Ajouter ce paramètre"):
                if nouveau_param.strip() != "":
                    st.session_state.parametres_dynamiques[nouveau_param.strip()] = valeur_param
                    st.success(f"✅ Paramètre '{nouveau_param.strip()}' ajouté.")

        submitted = st.form_submit_button("Ajouter le prélèvement")
        if submitted:
            new_data = {
                "Date": date, "Heure": heure, "Entreprise": entreprise,
                "Localisation": localisation, "Code": code,
                "Préleveur": preleveur, "Analyste": analyste
            }
            new_data.update(resultats)

            st.session_state.df_prelèvements = pd.concat([st.session_state.df_prelèvements, pd.DataFrame([new_data])], ignore_index=True)
            st.session_state.df_prelèvements.to_pickle("prelevements_sauvegarde.pkl")
            st.success("✅ Prélèvement ajouté avec succès")

            # Afficher alertes normes
            alertes = verifier_parametres_entres(new_data)
            if alertes:
                for msg in alertes:
                    st.warning(msg)
            else:
                st.success("✅ Tous les paramètres respectent les normes.")

    # Filtrage des prélèvements
    st.markdown("### 🔍 Filtrer les prél
