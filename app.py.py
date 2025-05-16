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

    parametre_cible = st.selectbox("Quel paramètre veux-tu prédire ?", parametres, key="pred_cible")
    model_filename = f"modele_{parametre_cible.replace(' ', '_')}.pkl"

    if not os.path.exists(model_filename):
        st.error(f"❌ Le modèle '{model_filename}' est introuvable.")
    else:
        model = joblib.load(model_filename)
        valeurs = {}
        for param in parametres:
            if param != parametre_cible:
                valeurs[param] = st.number_input(param, value=0.0, format="%.4f", key=f"pred_{param}")

        if st.button("Prédire le paramètre manquant", key="pred_button"):
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
        valeurs[param] = st.number_input(param, value=0.0, format="%.4f", key=f"classif_{param}")

    if st.button("Prédire la classe de qualité", key="classif_button"):
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
            date = st.date_input("Date du prélèvement", value=datetime.today(), key="gestion_date")
            entreprise = st.text_input("Nom de l’entreprise", key="gestion_entreprise")
            code = st.text_input("Code de l’échantillon", key="gestion_code")
            preleveur = st.text_input("Nom du préleveur", key="gestion_preleveur")
        with col2:
            heure = st.time_input("Heure du prélèvement", key="gestion_heure")
            localisation = st.text_input("Localisation", key="gestion_localisation")
            analyste = st.text_input("Nom de l’analyste", key="gestion_analyste")

        st.markdown("### 🔬 Résultats des analyses")
        resultats = {}
        for param in parametres:
            resultats[param] = st.number_input(param, value=0.0, format="%.4f", key=f"gestion_{param}")

        # Affichage paramètres dynamiques existants
        if st.session_state.parametres_dynamiques:
            st.markdown("### ⚙️ Paramètres personnalisés ajoutés")
            for p, v in st.session_state.parametres_dynamiques.items():
                resultats[p] = st.number_input(p, value=float(v), format="%.4f", key=f"gestion_dyn_{p}")

        # Section ajout paramètre personnalisé
        with st.expander("➕ Ajouter un paramètre personnalisé"):
            nouveau_param = st.text_input("Nom du paramètre", key="new_param_name")
            valeur_param = st.number_input("Valeur", value=0.0, format="%.4f", key="new_param_value")
            if st.button("Ajouter ce paramètre", key="add_param_button"):
                if nouveau_param.strip() != "":
                    st.session_state.parametres_dynamiques[nouveau_param.strip()] = valeur_param
                    st.success(f"✅ Paramètre '{nouveau_param.strip()}' ajouté.")

        submitted = st.form_submit_button("Ajouter le prélèvement", key="submit_prelevement")
        if submitted:
            new_data = {
                "Date": date, "Heure": heure, "Entreprise": entreprise,
                "Localisation": localisation, "Code": code,
                "Préleveur": preleveur, "Analyste": analyste
            }
            new_data.update(resultats)
            
             st.session_state.df_prelèvements = pd.concat([st.session_state.df_prelèvements, pd.DataFrame([new_data])], ignore_index=True)
            st.session_state.df_prelèvements.to

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
    st.markdown("### 🔍 Filtrer les prélèvements")
    df = st.session_state.df_prelèvements.copy()
    if not df.empty:
        with st.expander("🗂️ Filtres avancés"):
            entreprises = df['Entreprise'].dropna().unique().tolist()
            selected_entreprise = st.selectbox("Entreprise", ["Toutes"] + entreprises, key="filtre_entreprise")
            dates = df['Date'].astype(str).dropna().unique().tolist()
            selected_date = st.selectbox("Date du prélèvement", ["Toutes"] + sorted(dates), key="filtre_date")

            if selected_entreprise != "Toutes":
                df = df[df['Entreprise'] == selected_entreprise]
            if selected_date != "Toutes":
                df = df[df['Date'].astype(str) == selected_date]

        # Choix de la période pour la visualisation
        periode = st.selectbox(
            "Choisir la période de comparaison",
            ["Journalière", "Mensuelle", "Annuelle"],
            key="periode_comparaison"
        )

        # Agrégation selon la période choisie
        df['Date'] = pd.to_datetime(df['Date'])
        if periode == "Mensuelle":
            df_grouped = df.groupby(df['Date'].dt.to_period("M")).mean()
            df_grouped.index = df_grouped.index.to_timestamp()
        elif periode == "Annuelle":
            df_grouped = df.groupby(df['Date'].dt.to_period("Y")).mean()
            df_grouped.index = df_grouped.index.to_timestamp()
        else:
            df_grouped = df.set_index('Date')

        st.markdown("### 📋 Tableau des prélèvements filtrés")
        st.dataframe(df)

        def to_excel(df_to_export):
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df_to_export.to_excel(writer, index=False, sheet_name='Prélèvements')
            return output.getvalue()

        excel_data = to_excel(df)
        st.download_button("📥 Télécharger (Excel)", data=excel_data,
                           file_name="prelevements.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        # Visualisation multi-paramètres avec codes couleurs selon normes
        st.markdown("### 📊 Visualisation des paramètres")

        selected_params = st.multiselect(
            "Sélectionne un ou plusieurs paramètres à tracer :",
            parametres + list(st.session_state.parametres_dynamiques.keys()) if st.session_state.parametres_dynamiques else parametres,
            default=["pH", "Temperature"],
            key="select_params_graph"
        )

        if selected_params:
            fig, ax = plt.subplots(figsize=(10, 5))
            for param in selected_params:
                if param in df_grouped.columns:
                    y = df_grouped[param]
                    x = df_grouped.index

                    # Gestion des couleurs selon normes si seuil défini
                    if param in normes:
                        seuil = normes[param]
                        min_val = seuil.get("min", -float('inf'))
                        max_val = seuil.get("max", float('inf'))
                        in_range = (y >= min_val) & (y <= max_val)
                        out_range = ~in_range

                        ax.plot(x[in_range], y[in_range], 'go-', label=f"{param} (OK)")
                        ax.plot(x[out_range], y[out_range], 'ro-', label=f"{param} (Hors norme)")
                    else:
                        ax.plot(x, y, marker='o', label=param)

            ax.set_title(f"Évolution {periode.lower()} des paramètres sélectionnés")
            ax.set_xlabel("Date")
            ax.set_ylabel("Valeur")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
    else:
        st.info("Aucun prélèvement à afficher.")

    # Importation fichier Excel ou CSV
    st.markdown("### 📁 Importer un fichier Excel ou CSV")
    uploaded_file = st.file_uploader("Choisissez un fichier", type=["xlsx", "csv"], key="upload_file")
    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                imported_df = pd.read_csv(uploaded_file)
            else:
                imported_df = pd.read_excel(uploaded_file)

            st.success("✅ Données importées :")
            st.dataframe(imported_df)
        except Exception as e:
            st.error(f"Erreur lors de l'importation : {e}")

