import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# Liste des 23 paramètres
parametres = [
    'Total Coliform', 'Escherichia Coli', 'Faecal Streptococci', 'Turbidity', 'pH', 'Temperature',
    'Free Chlorine', 'Chlorates', 'Sulfate', 'Magnesium', 'Calcium', 'Conductivity', 'Dry Residue',
    'Complete Alkaline Title', 'Nitrite', 'Ammonium', 'Phosphate', 'Nitrate', 'Iron', 'Manganese',
    'Colour', 'Smell', 'Taste'
]

st.title("💧 Prédiction d’un paramètre manquant")

# Sélection du paramètre à prédire
parametre_cible = st.selectbox("Quel paramètre veux-tu prédire ?", parametres)

# Chargement du bon modèle
model_filename = f"modele_{parametre_cible.replace(' ', '_')}.pkl"

if not os.path.exists(model_filename):
    st.error(f"❌ Le modèle '{model_filename}' n'existe pas. Vérifie son nom ou entraîne-le.")
    st.stop()

model = joblib.load(model_filename)

st.info(f"✅ Modèle chargé : {model_filename}")

# Interface de saisie ou chargement d’un fichier
st.markdown("### 🔢 Saisie des 22 autres paramètres :")

valeurs = []
for param in parametres:
    if param != parametre_cible:
        val = st.number_input(param, value=0.0, format="%.4f")
        valeurs.append(val)

# Prédiction
if st.button("Prédire le paramètre manquant"):
    X_input = np.array(valeurs, dtype=np.float64).reshape(1, -1)
    prediction = model.predict(X_input)
    st.success(f"🔍 Prédiction pour **{parametre_cible}** : `{prediction[0]:.4f}`")
