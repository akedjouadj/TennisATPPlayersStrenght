from tqdm import tqdm
import numpy as np
import pandas as pd
import random

from DataProcessorForTennisStrenghtModel import process_dates

def evaluate_accuracy_on_next_match_prediction(df_matches, ratings_story):
    """
    Évalue l'accuracy du modèle en prédiction du vainqueur du prochain match.
    :param df_matches: DataFrame contenant les matchs avec les colonnes 'tourney_date',
                        'winner_name', 'loser_name', 'round', 'tourney_name'.
    :return: Accuracy (float)
    """
    correct_predictions = 0
    total_predictions = 0
    ignored_matches = 0
    ignored_players = []

    # preprocessing et tri des matchs par date et round
    X = process_dates(df_matches.copy())
    
    for _, match in tqdm(X.iterrows()):
        winner_name = match["winner_name"]
        loser_name = match["loser_name"]
        
        # Assurer que les joueurs ont une force initiale
        if winner_name not in ratings_story or loser_name not in ratings_story:
            ignored_matches += 1
            continue

        if winner_name in ignored_players or loser_name in ignored_players:
            ignored_matches += 1
            continue

        winner_previous_rating = ratings_story[winner_name]["ratings"][ratings_story[winner_name]["slider"]]
        loser_previous_rating = ratings_story[loser_name]["ratings"][ratings_story[loser_name]["slider"]]
        if len(ratings_story[winner_name]["ratings"])>ratings_story[winner_name]["slider"]+1:
            ratings_story[winner_name]["slider"] += 1
        else:
            if winner_name not in ignored_players:
                ignored_players.append(winner_name)
        if len(ratings_story[loser_name]["ratings"])>ratings_story[loser_name]["slider"]+1:
            ratings_story[loser_name]["slider"] += 1
        else:
            if loser_name not in ignored_players:
                ignored_players.append(loser_name)

        # Prédiction : le joueur avec la plus grande force est considéré comme gagnant
        if winner_previous_rating > loser_previous_rating:
            predicted_winner = winner_name
        elif winner_previous_rating < loser_previous_rating:
            predicted_winner = loser_name
        else: 
            predicted_winner = random.choice([winner_name, loser_name])

        # Vérifier si la prédiction est correcte
        if predicted_winner == winner_name:
            correct_predictions += 1

        total_predictions += 1

    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    print(f"Accuracy: {accuracy}")
    print(f"Ignored matches: {ignored_matches}")

    return ignored_players


class TennisStrenghtModel:
    def __init__(self, initial_ratings, tournament_points):
        """
        :param initial_ratings: dict {player_id: initial_strength}
        :param tournament_points: dict {tournament_name: {round: points}}
        """
        # copy initial ratings
        self.ratings = initial_ratings.copy()
        self.ratings_story = {player_name: {"slider": 0, "match_key": ["date_0"], "ratings": [initial_ratings[player_name]]} for player_name in initial_ratings.keys()}  # Historique des forces des joueurs
        self.tournament_points = tournament_points  # Points ATP selon le tournoi et le round
    
    def normalize_expected_score(self, score_A, score_B):
        """
        Calcule l'expected score avec une transformation Softmax-like
        afin que expA + expB = 1.

        Parameters:
        - score_A (float): Moyenne des scores précédents du joueur A
        - score_B (float): Moyenne des scores précédents du joueur B

        Returns:
        - tuple: (expected_score_A, expected_score_B)
        """
        # Applique la transformation Softmax
        expA = np.exp(score_A)
        expB = np.exp(score_B)
        
        sum_exp = expA + expB
        expected_score_A = expA / sum_exp
        expected_score_B = expB / sum_exp

        return expected_score_A, expected_score_B

    def get_expected_score(self, match_history, tournament, playerA, playerB):
        """
        Calcule le score attendu basé sur les précédents matchs du tournoi ou du tournoi précédent.
        :param match_history: dict contenant l'historique des matchs
        :param tournament: nom du tournoi actuel
        :param player1: ID du joueur 1
        :param player2: ID du joueur 2
        :return: expected score (float)
        """
        scores_A = []
        scores_B = []

        # Récupérer les scores des matchs précédents dans le même tournoi
        if tournament in match_history:
            for match in match_history[tournament]:
                if match["player1"] == playerA:
                    scores_A.append(match["actual_score"])
                elif match["player2"] == playerA:
                    scores_A.append(1 - match["actual_score"])
                elif match["player1"] == playerB:
                    scores_B.append(match["actual_score"])
                elif match["player2"] == playerB:
                    scores_B.append(1 - match["actual_score"])

        # Si aucun historique dans ce tournoi, prendre les scores du tournoi précédent
        if len(scores_A) == 0:
            prev_tournament = self.get_previous_tournament(match_history, tournament)
            if prev_tournament:
                for match in match_history[prev_tournament]:
                    if match["player1"] == playerA:
                        scores_A.append(match["actual_score"])
                    elif match["player2"] == playerA:
                        scores_A.append(1 - match["actual_score"])
                    elif match["player1"] == playerB:
                        scores_B.append(match["actual_score"])
                    elif match["player2"] == playerB:
                        scores_B.append(1 - match["actual_score"])
        
        score_A = sum(scores_A) / len(scores_A) if len(scores_A)>0 else 0.5
        score_B = sum(scores_B) / len(scores_B) if len(scores_B)>0 else 0.5

        score_A, score_B = self.normalize_expected_score(score_A, score_B)

        return score_A, score_B

    def get_previous_tournament(self, match_history, current_tournament):
        """
        Trouve le tournoi précédent basé sur les données disponibles.
        :param match_history: dict contenant l'historique des matchs
        :param current_tournament: tournoi actuel
        :return: nom du tournoi précédent (str) ou None
        """
        tournaments = list(match_history.keys()) # Assumer que les clés sont ordonnées chronologiquement
        if current_tournament in tournaments:
            idx = tournaments.index(current_tournament)
            return tournaments[idx - 1] if idx > 0 else None
        return None

    def update_ratings(self, matches):
        """
        Met à jour les forces des joueurs après chaque match.
        :param matches: Liste des matchs sous forme de dictionnaires
        :return: Dictionnaire des forces des joueurs mises à jour
        """
        match_history = {}  # Stocker les résultats pour le calcul du score attendu

        print("Updating ratings...")
        for match in tqdm(matches):
            tournament = match["tournament"]
            date = match["date"]
            round_name = match["round"]
            player1, player2 = match["player1"], match["player2"]
            actual_score = match["actual_score"]  # Probabilité de victoire du joueur 1 donnée par M_p
            target = match["target"]  # 1 si joueur 1 gagne, 0 sinon
            
            # Vérifier si les joueurs ont une force initiale, sinon les initialiser
            if player1 not in self.ratings:
                self.ratings[player1] = 0
            if player2 not in self.ratings:
                self.ratings[player2] = 0
            
            if player1 not in self.ratings_story:
                self.ratings_story[player1] = {"slider": 0, "match_key": ["date_0"], "ratings": [0]}
            if player2 not in self.ratings_story:
                self.ratings_story[player2] = {"slider": 0, "match_key": ["date_0"], "ratings": [0]}

            # Calculer le score attendu
            expected_score_1, expected_score_2 = self.get_expected_score(match_history, tournament, player1, player2)

            # Mettre à jour les ratings avec la formule Elo modifiée

            K_1 = self.tournament_points[tournament][round_name]["W"] if target == 1 else self.tournament_points[tournament][round_name]["L"]
            K_2 = self.tournament_points[tournament][round_name]["W"] if target == 0 else self.tournament_points[tournament][round_name]["L"]
            #K = self.tournament_points.get(tournament, {}).get(round_name, 32)  # to robustify the model

            #self.ratings[player1] += K_1 * abs(actual_score - expected_score_1)
            #self.ratings[player2] += K_2 * abs((1 - actual_score) - expected_score_2)
            self.ratings[player1] += K_1 * actual_score
            self.ratings[player2] += K_2 * (1 - actual_score)

            # Stocker l'historique des ratings
            self.ratings_story[player1]["match_key"].append(f"date_{date}_tournament_{tournament}_round_{round_name}")
            self.ratings_story[player1]["ratings"].append(self.ratings[player1])

            self.ratings_story[player2]["match_key"].append(f"date_{date}_tournament_{tournament}_round_{round_name}")
            self.ratings_story[player2]["ratings"].append(self.ratings[player2])

            # Stocker le match pour l'historique
            if tournament not in match_history:
                match_history[tournament] = []
            match_history[tournament].append(match)

        print("Done!")
