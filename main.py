from src.data_ingestion import DataIngestor
from src.preprocessing import DataPreprocessor
from src.model_training import ModelTrainer

def main():
    # 1. Ingestion des données
    ingestor = DataIngestor(r'data/train_FD001.csv')
    df = ingestor.load_data()
    
    if df is not None:
        print(f" Données chargées : {df.shape[0]} lignes, {df.shape[1]} colonnes.")

        # 2. Preprocessing (Calcul du RUL)
        preprocessor = DataPreprocessor()
        df = preprocessor.calculate_rul(df)
        
        # 3. Entraînement de l'IA
        # Note : La sauvegarde se fait automatiquement à l'intérieur de la méthode train()
        trainer = ModelTrainer()
        model = trainer.train(df)
        
        # 4. Affichage du graphique de performance
        # Cette méthode va ouvrir une fenêtre avec les courbes Bleue vs Rouge
        trainer.plot_results(df)

        # 5. Petit test de prédiction sur la toute première ligne
        print("\n--- Test de prédiction final ---")
        
        # On identifie dynamiquement les colonnes à exclure pour la prédiction
        cols_id_cycle = [df.columns[0], df.columns[1], 'RUL']
        sample_data = df.drop(cols_id_cycle, axis=1).iloc[[0]]
        
        # Prédiction
        prediction = model.predict(sample_data)
        
        print(f" Prédiction IA : {prediction[0]:.1f} cycles restants.")
        print(f" Valeur réelle (RUL) : {df['RUL'].iloc[0]} cycles.")

# Point d'entrée du programme
if __name__ == "__main__":
    main()