import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
from PIL import Image


st.set_page_config(page_title="Qualité de l'eau potable", page_icon="💧", layout="centered")

st.title("💧 Application IA pour la prédiction des paramètres de l'eau potable")

image = Image.open("eau.jpg")
st.image(image, caption="Analyse de la qualité de l'eau", use_column_width=True)





st.markdown("""
Bienvenue dans cette application développée dans le cadre d'un projet scientifique 👩‍🔬🧑‍🔬  
Elle permet de **prédire un paramètre manquant** parmi 23 mesures de qualité d'eau.

---

### ✨ Comment l'utiliser :
1. Sélectionne le **paramètre à prédire**
2. Renseigne les **22 autres valeurs mesurées**
3. Clique sur **"Prédire le paramètre manquant"**
4. Consulte la valeur estimée

---

### 📘 Paramètres disponibles :
- Physico-chimiques (pH, turbidité, température, etc.)
- Bactériologiques (Coliformes, E. coli, etc.)
- Minéraux (Calcium, Fer, Nitrates…)

---

> Développé avec ❤️ par Yanis  
> Basé sur un modèle Random Forest Regressor
""")
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

st.sidebar.title("🔧 Menu")
mode = st.sidebar.radio("Choisir une action :", ("🔍 Prédire un paramètre manquant", "🧪 Classifier la qualité de l’eau"))

if mode == "🔍 Prédire un paramètre manquant":
    # Ton code actuel ici (avec la liste des paramètres, les inputs, et la prédiction)

elif mode == "🧪 Classifier la qualité de l’eau":
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
            st.success(f"✅ Classe prédite : **{prediction[0]}**")
        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {e}")

