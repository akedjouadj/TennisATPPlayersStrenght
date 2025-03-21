from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib
import numpy as np

class TennisPostGameScoreModel():

    def __init__(self):
        
        self.model = None
        self.scaler = StandardScaler()

        self.features_general = ['surface_Hard', 'surface_Clay', 'surface_Grass', 'tourney_level_G', 'tourney_level_M', 'tourney_level_A', 'tourney_level_F', 'best_of_3', 'best_of_5']
        self.features_player_1 = [
            'p1_ace', 'p1_df', 'p1_svpt', 'p1_1stIn', 'p1_1stWon', 'p1_2ndWon',
            'p1_bpSaved', 'p1_bpFaced', 'p1_1stServePct', 'p1_1stServeWinPct', 'p1_2ndServeWinPct',
        ]
        self.features_player_2 = [
            'p2_ace', 'p2_df', 'p2_svpt', 'p2_1stIn', 'p2_1stWon', 'p2_2ndWon',
            'p2_bpSaved', 'p2_bpFaced', 'p2_1stServePct', 'p2_1stServeWinPct', 'p2_2ndServeWinPct',
        ]

    def train_logistic_model(self, X, Y):
        """
        Entraîne un modèle de régression logistique pour prédire la probabilité de victoire.
        """
        
        # Standardisation
        X_scaled = self.scaler.fit_transform(X)
        
        # Entraînement du modèle
        self.model = LogisticRegression(max_iter=1000)
        self.model.fit(X_scaled, Y)
        
    def evaluate_model(self, X, Y):

        # Standardisation
        X_scaled = self.scaler.transform(X)
        
        # Évaluation
        y_pred_proba = self.model.predict_proba(X_scaled)[:, 1]
        acc = accuracy_score(Y, (y_pred_proba > 0.5).astype(int))

        print(f"Accuracy: {acc:.4f}")

        # Sauvegarde du modèle et du scaler
        joblib.dump(self.model, 'outputs/tennis_logistic_model.pkl')
        joblib.dump(self.scaler, 'outputs/tennis_scaler.pkl')

        return {'accuracy': acc, 'probs': y_pred_proba}

    def calculate_actual_score(self, row):
        """
        Calcule le score prédit basé sur le modèle de régression logistique.
        """
        # Charger modèle et scaler
        if self.model is None:
            self.model = joblib.load('outputs/tennis_logistic_model.pkl')
            self.scaler = joblib.load('outputs/tennis_scaler.pkl')

        # Création de la feature pour un match donné
        
        features = row[self.features_general+ self.features_player_1+ self.features_player_2]
        
        # Standardiser et prédire
        features_scaled = self.scaler.transform(features)
        proba = self.model.predict_proba(features_scaled)[0, 1]

        return round(proba, 4)