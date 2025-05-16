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
            model = joblib.load("modele_classification.pkl")
            X_input = np.array(valeurs).reshape(1, -1)
            prediction = model.predict(X_input)
            classe = classes.get(prediction[0], "Inconnue")
            st.success(f"✅ Classe prédite : **{classe}**")
        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {e}")

    with st.expander("📘 Voir les correspondances des classes encodées"):
        for code, label in classes.items():
            st.write(f"**{code}** → {label}")



