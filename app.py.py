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

            import pandas as pd
import io

st.subheader("📊 Gestion des prélèvements journaliers")

# Liste étendue de paramètres (exemple, tu pourras l'ajuster)
parametres_etendus = [
    'Date', 'Heure', 'Nom de l’entreprise', 'Localisation', 'Technicien', 
    'Total Coliform', 'Escherichia Coli', 'Faecal Streptococci', 'Turbidity', 
    'pH', 'Temperature', 'Free Chlorine', 'Chlorates', 'Sulfate', 'Magnesium', 
    'Calcium', 'Conductivity', 'Dry Residue', 'Complete Alkaline Title', 
    'Nitrite', 'Ammonium', 'Phosphate', 'Nitrate', 'Iron', 'Manganese', 
    'Colour', 'Smell', 'Taste'
]

# Saisie manuelle via formulaire
st.write("📝 Saisie manuelle des prélèvements")

donnees = {}
for i, param in enumerate(parametres_etendus):
    if param == 'Date':
        donnees[param] = st.date_input(param, key=f"{param}_{i}")
    elif param == 'Heure':
        donnees[param] = st.time_input(param, key=f"{param}_{i}")
    elif param in ['Nom de l’entreprise', 'Localisation', 'Technicien']:
        donnees[param] = st.text_input(param, key=f"{param}_{i}")
    else:
        donnees[param] = st.number_input(param, value=0.0, format="%.4f", key=f"{param}_{i}")


# Button pour ajouter la ligne dans un dataframe stocké dans la session
if 'df_prelèvements' not in st.session_state:
    st.session_state.df_prelèvements = pd.DataFrame(columns=parametres_etendus)

if st.button("Ajouter ce prélèvement"):
    nouvelle_ligne = pd.DataFrame([donnees])
    st.session_state.df_prelèvements = pd.concat([st.session_state.df_prelèvements, nouvelle_ligne], ignore_index=True)
    st.success("✅ Prélèvement ajouté !")

# Affiche le tableau des prélèvements ajoutés
st.write("📋 Tableau des prélèvements enregistrés :")
st.dataframe(st.session_state.df_prelèvements)

# Fonction pour exporter en Excel
def to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Prélèvements')
        writer.save()
    return output.getvalue()

# Bouton pour télécharger le fichier Excel
def to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Prélèvements')
    return output.getvalue()


# Upload d'un fichier Excel ou CSV
uploaded_file = st.file_uploader("📁 Importer un fichier Excel", type=["xlsx"])

if uploaded_file:
    try:
        xls = pd.ExcelFile(uploaded_file)
        st.write("📄 Feuilles disponibles :", xls.sheet_names)
        selected_sheet = st.selectbox("🧾 Sélectionne la feuille :", xls.sheet_names)

        st.markdown("### 🔍 Aperçu brut du contenu (aucun skip, aucune colonne)")
        raw_df = pd.read_excel(xls, sheet_name=selected_sheet, header=None)
        st.dataframe(raw_df)

    except Exception as e:
        st.error(f"❌ Erreur de lecture : {e}")

        # Tentative de lecture avec skiprows
        df_import = pd.read_excel(xls, sheet_name=selected_sheet, skiprows=1000)

        if df_import.empty:
            st.warning("⚠️ Données toujours vides. Essaie d’augmenter la valeur de `skiprows`.")
        else:
            st.success("✅ Données importées avec succès !")
            st.dataframe(df_import)
    except Exception as e:
        st.error(f"❌ Erreur : {e}")

# Tu peux ensuite choisir de fusionner avec st.session_state.df_prelèvements ou autre traitement




