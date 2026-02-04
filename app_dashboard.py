import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

# 1. Configuration
st.set_page_config(page_title="IA Maintenance", layout="wide")
st.title(" Dashboard de Maintenance Prédictive")

# 2. Chargement du modèle
@st.cache_resource
def load_model():
    try:
        return joblib.load('maintenance_model.joblib')
    except:
        return None

model = load_model()

# 3. Interface
st.sidebar.header("Données")
uploaded_file = st.sidebar.file_uploader("Charger train_FD001.csv", type=["csv", "txt"])

if uploaded_file is not None and model is not None:
    try:
        # LECTURE MANUELLE LIGNE PAR LIGNE
        raw_bytes = uploaded_file.getvalue().decode("utf-8")
        lines = raw_bytes.splitlines()
        
        data_rows = []
        for line in lines:
            # On nettoie la ligne et on sépare par n'importe quel espace OU virgule
            clean_line = line.replace(',', ' ').strip()
            parts = clean_line.split()
            
            # On ne garde que les lignes qui contiennent uniquement des chiffres
            # (Cela permet d'ignorer automatiquement la ligne d'en-tête engine_id, cycle...)
            try:
                float_parts = [float(p) for p in parts]
                if len(float_parts) >= 3:
                    data_rows.append(float_parts)
            except ValueError:
                continue # On ignore les lignes de texte (en-tête)

        # Création du tableau final
        df = pd.DataFrame(data_rows)

        if not df.empty and df.shape[1] >= 10:
            st.success(f" Succès ! {df.shape[0]} moteurs détectés avec {df.shape[1]} capteurs.")

            # Prédiction : X = tout sauf les colonnes 0 et 1 (ID et Cycle)
            X = df.iloc[:, 2:] 
            predictions = model.predict(X)
            
            # Affichage des colonnes
            col1, col2 = st.columns([1, 1.2])
            with col1:
                st.subheader(" État de la Flotte")
                results = pd.DataFrame({
                    'ID Moteur': df.iloc[:, 0].astype(int),
                    'Cycle': df.iloc[:, 1].astype(int),
                    'RUL Prédit': predictions.round(1)
                })
                st.dataframe(results.head(100), use_container_width=True)

            with col2:
                st.subheader(" Courbe de RUL")
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(predictions[:150], color='red', linewidth=2)
                ax.set_ylabel("Cycles restants")
                st.pyplot(fig)
        else:
            st.error("Le fichier ne contient pas assez de données numériques valides.")

    except Exception as e:
        st.error(f"Erreur lors du traitement : {e}")

elif model is None:

    st.error("Modèle introuvable. Vérifiez que 'maintenance_model.joblib' est bien dans le dossier.")
