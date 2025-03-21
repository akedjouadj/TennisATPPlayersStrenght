import pandas as pd
import numpy as np
from tqdm import tqdm

def process_dates(df):
    """
    Convertit les dates au format datetime.
    """

    df['tourney_date'] = pd.to_datetime(df['tourney_date'], format='%Y%m%d')
    
    df = df[~(df.tourney_name.isin(["Paris Olympics", "United Cup", "Laver Cup", "Next Gen Finals"]))]
    # drop all davis cup matches, tourney_name starts with "Davis Cup"
    df = df[~df.tourney_name.str.contains("Davis Cup")]
    df = df[~df["round"].isin(["BR"])]

    round_number_map = {
        'R128': 1,
        'R64': 2,
        'R32': 3,
        'R16': 4,
        'QF': 5,
        'RR': 6, # after match group stage, there's only SF and F
        'SF': 7,
        'F': 8
        }
    
    df['round_number'] = df['round'].map(round_number_map)
    df = df.sort_values(by=['tourney_date', 'round_number'])
    df = df.reset_index(drop=True)

    return df

class DataProcessorForTennisStrenghtModel:
    def __init__(self, df):
        """
        Initialise le DataProcessor avec le DataFrame brut.
        """
        self.df = df.copy()
        self.df = process_dates(self.df)
        self.tournament_points = self.create_tournament_points()
        self.model = None

    def create_tournament_points(self):
        """
        Crée le dictionnaire des points ATP par tournoi et par round.
        """
        points = {
            # Grand Chelem
            "Australian Open": self._grand_slam_points(),
            "Roland Garros": self._grand_slam_points(),
            "Wimbledon": self._grand_slam_points(),
            "Us Open": self._grand_slam_points(),

            # Masters 1000
            "Indian Wells Masters": self._masters_points(),
            "Miami Masters": self._masters_points(),
            "Monte Carlo Masters": self._masters_points(),
            "Madrid Masters": self._masters_points(),
            "Rome Masters": self._masters_points(),
            "Canada Masters": self._masters_points(),
            "Cincinnati Masters": self._masters_points(),
            "Shanghai Masters": self._masters_points(),
            "Paris Masters": self._masters_points(),

            # ATP 500
            "Rotterdam": self._atp_500_points(),
            "Dubai": self._atp_500_points(),
            "Acapulco": self._atp_500_points(),
            "Barcelona": self._atp_500_points(),
            "Hamburg": self._atp_500_points(),
            "Washington": self._atp_500_points(),
            "Tokyo": self._atp_500_points(),
            "Vienna": self._atp_500_points(),
            "Basel": self._atp_500_points(),
            "Beijing": self._atp_500_points(),
            "Halle": self._atp_500_points(),
            "Queen's Club": self._atp_500_points(),
            "Rio De Janeiro": self._atp_500_points(),

            # ATP 250
            "Brisbane": self._atp_250_points(),
            "Hong Kong": self._atp_250_points(),
            "Adelaide": self._atp_250_points(),
            "Auckland": self._atp_250_points(),
            "Montpellier": self._atp_250_points(),
            "Cordoba": self._atp_250_points(),
            "Dallas": self._atp_250_points(),
            "Marseille": self._atp_250_points(),
            "Delray Beach": self._atp_250_points(),
            "Buenos Aires": self._atp_250_points(),
            "Santiago": self._atp_250_points(),
            "Estoril": self._atp_250_points(),
            "Houston": self._atp_250_points(),
            "Marrakech": self._atp_250_points(),
            "Munich": self._atp_250_points(),
            "Bucharest": self._atp_250_points(),
            "Geneva": self._atp_250_points(),
            "Lyon": self._atp_250_points(),
            "s Hertogenbosch": self._atp_250_points(),
            "Stuttgart": self._atp_250_points(),
            "Eastbourne": self._atp_250_points(),
            "Mallorca": self._atp_250_points(),
            "Atlanta": self._atp_250_points(),
            "Kitzbuhel": self._atp_250_points(),
            "Umag": self._atp_250_points(),
            "Newport": self._atp_250_points(),
            "Chengdu": self._atp_250_points(),
            "Hangzhou": self._atp_250_points(),
            "Stockholm": self._atp_250_points(),
            "Antwerp": self._atp_250_points(),
            "Belgrade": self._atp_250_points(),
            "Metz": self._atp_250_points(),
            "Almaty": self._atp_250_points(),
            "Doha": self._atp_250_points(),
            "Los Cabos": self._atp_250_points(),
            "Gstaad": self._atp_250_points(),
            "Bastad": self._atp_250_points(),
            "Winston-Salem": self._atp_250_points(),
            "Astana": self._atp_250_points(),
            "Banja Luka": self._atp_250_points(),
            "Belgrade 2": self._atp_250_points(),
            "Cagliari": self._atp_250_points(),
            "Florence": self._atp_250_points(),
            "Gijon": self._atp_250_points(),
            "Moscow": self._atp_250_points(),
            "Naples": self._atp_250_points(),
            "Parma": self._atp_250_points(),
            "Pune": self._atp_250_points(),
            "San Diego": self._atp_250_points(),
            "Seoul": self._atp_250_points(),
            "Sofia": self._atp_250_points(),
            "St. Petersburg": self._atp_250_points(),
            "Tel Aviv": self._atp_250_points(),
            "Zhuhai": self._atp_250_points(),
            "Adelaide 1": self._atp_250_points(),
            "Adelaide 2": self._atp_250_points(),
            "Antalya": self._atp_250_points(),
            "Great Ocean Road Open": self._atp_250_points(),
            "Marbella": self._atp_250_points(),
            "Melbourne": self._atp_250_points(),
            "Murray River Open": self._atp_250_points(),
            "Singapore": self._atp_250_points(),
            "Sydney": self._atp_250_points(),

            # Autres événements
            #"United Cup": self._exhibition_points(),
            #"Paris Olympics": self._olympics_points(),
            #"Laver Cup": self._exhibition_points(),
            "Tour Finals": self._tour_finals_points(),
            #"Next Gen Finals": self._exhibition_points()
        }

        return points

    def _grand_slam_points(self):
        return {
            "F": {"W": 2000, "L": 1200},  # Finale
            "SF": {"W": 1200, "L": 720},  # Demi-finale
            "QF": {"W": 720, "L": 360},   # Quart de finale
            "R16": {"W": 360, "L": 180},  # 4e tour (Round of 16)
            "R32": {"W": 180, "L": 90},   # 3e tour (Round of 32)
            "R64": {"W": 90, "L": 45},    # 2e tour (Round of 64)
            "R128": {"W": 10, "L": 0}     # 1er tour (Round of 128)
        }

    def _masters_points(self):
        return {
            "F": {"W": 1000, "L": 600},
            "SF": {"W": 600, "L": 360},
            "QF": {"W": 360, "L": 180},
            "R16": {"W": 180, "L": 90},
            "R32": {"W": 90, "L": 45},
            "R64": {"W": 45, "L": 10},
            "R128": {"W": 10, "L": 0}
        }

    def _atp_500_points(self):
        return {
            "F": {"W": 500, "L": 300},
            "SF": {"W": 300, "L": 180},
            "QF": {"W": 180, "L": 90},
            "R16": {"W": 90, "L": 45},
            "R32": {"W": 45, "L": 20},
            "R64": {"W": 20, "L": 0}
        }

    def _atp_250_points(self):
        return {
            "F": {"W": 250, "L": 150},
            "SF": {"W": 150, "L": 90},
            "QF": {"W": 90, "L": 45},
            "R16": {"W": 45, "L": 20},
            "R32": {"W": 20, "L": 10},
            "R64": {"W": 10, "L": 0}
        }

    def _olympic_points(self):
        return {
            "F": {"W": 750, "L": 450},    # Finale (Médaille d'or vs argent)
            "SF": {"W": 450, "L": 340},   # Demi-finale (Médaille d'argent vs bronze)
            "BR": {"W": 340, "L": 270},   # Match pour la médaille de bronze
            "QF": {"W": 270, "L": 135},   # Quart de finale
            "R16": {"W": 135, "L": 70},   # Huitième de finale (Round of 16)
            "R32": {"W": 70, "L": 35},    # Deuxième tour (Round of 32)
            "R64": {"W": 35, "L": 0}      # Premier tour (Round of 64)
        }

    def _tour_finals_points(self):
        return {
            "F": {"W": 1500, "L": 1000},
            "SF": {"W": 1000, "L": 400},
            "RR": {"W": 200, "L": 0}      # Round Robin
        }

    def _exhibition_points(self):
        return {
            "F": {"W": 0, "L": 0},
            "SF": {"W": 0, "L": 0},
            "RR": {"W": 0, "L": 0}
        }

    def feature_engineering(self):
        """
        Crée les features nécessaires pour le modèle de régression logistique.
        """

        # Se restreindre aux matchs de tourney_level 'A' 'G' 'M' 'F'
        tourney_level_of_interest = ['A', 'G', 'M', 'F']
        self.df = self.df[self.df['tourney_level'].isin(tourney_level_of_interest)].reset_index(drop=True)
        
        # Création de ratios
        self.df['w_1stServePct'] = self.df['w_1stIn'] / self.df['w_svpt']
        self.df['w_1stServeWinPct'] = self.df['w_1stWon'] / self.df['w_1stIn']
        self.df['w_2ndServeWinPct'] = self.df['w_2ndWon'] / (self.df['w_svpt'] - self.df['w_1stIn'])
        
        self.df['l_1stServePct'] = self.df['l_1stIn'] / self.df['l_svpt']
        self.df['l_1stServeWinPct'] = self.df['l_1stWon'] / self.df['l_1stIn']
        self.df['l_2ndServeWinPct'] = self.df['l_2ndWon'] / (self.df['l_svpt'] - self.df['l_1stIn'])

        # One hot encoding de la surface: columns surface_Hard, surface_Clay, surface_Grass
        surface_dummies_columns = ['surface_Hard', 'surface_Clay', 'surface_Grass']
        surface_dummies = pd.get_dummies(self.df['surface'], prefix='surface')
        for col in surface_dummies_columns:
            if col not in surface_dummies.columns:
                surface_dummies[col] = 0
        self.df = pd.concat([self.df, surface_dummies], axis=1)

        # One hot encoding du tourney_level: columns tourney_level_G, tourney_level_M, tourney_level_A, tourney_level_F 
        tourney_level_dummies_columns = ['tourney_level_G', 'tourney_level_M', 'tourney_level_A', 'tourney_level_F']
        tourney_level_dummies = pd.get_dummies(self.df['tourney_level'], prefix='tourney_level')
        for col in tourney_level_dummies_columns:
            if col not in tourney_level_dummies.columns:
                tourney_level_dummies[col] = 0
        self.df = pd.concat([self.df, tourney_level_dummies], axis=1)

        # One hot encoding for best_of: columns best_of_3, best_of_5
        best_of_dummies_columns = ['best_of_3', 'best_of_5']
        best_of_dummies = pd.get_dummies(self.df['best_of'], prefix='best_of')
        for col in best_of_dummies_columns:
            if col not in best_of_dummies.columns:
                best_of_dummies[col] = 0
        self.df = pd.concat([self.df, best_of_dummies], axis=1)

        features_winner = [
            'w_ace', 'w_df', 'w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon',
            'w_bpSaved', 'w_bpFaced', 'w_1stServePct', 'w_1stServeWinPct', 'w_2ndServeWinPct',
        ] 

        features_loser = [
            'l_ace', 'l_df', 'l_svpt', 'l_1stIn', 'l_1stWon', 'l_2ndWon',
            'l_bpSaved', 'l_bpFaced', 'l_1stServePct', 'l_1stServeWinPct', 'l_2ndServeWinPct',
        ]   
            
        features_match_infos = ['tourney_name', 'tourney_date', 'round', 'surface', 'tourney_level']
        player_names_columns = ['player1_name', 'player2_name']
        features_general = ['surface_Hard', 'surface_Clay', 'surface_Grass', 'tourney_level_G', 'tourney_level_M', 'tourney_level_A', 'tourney_level_F', 'best_of_3', 'best_of_5']
        features_player_1 = [
            'p1_ace', 'p1_df', 'p1_svpt', 'p1_1stIn', 'p1_1stWon', 'p1_2ndWon',
            'p1_bpSaved', 'p1_bpFaced', 'p1_1stServePct', 'p1_1stServeWinPct', 'p1_2ndServeWinPct',
        ]
        features_player_2 = [
            'p2_ace', 'p2_df', 'p2_svpt', 'p2_1stIn', 'p2_1stWon', 'p2_2ndWon',
            'p2_bpSaved', 'p2_bpFaced', 'p2_1stServePct', 'p2_1stServeWinPct', 'p2_2ndServeWinPct',
        ]

        self.X = pd.DataFrame([], columns=features_match_infos+player_names_columns+features_general+features_player_1+features_player_2)
        self.Y = pd.DataFrame([], columns=['target'])

        # fill the inpput data X and the target y
        print("preparing inputs and target...")
        for count in tqdm(range(len(self.df))):
            u = np.random.uniform(0, 1)
            if u < 0.5:
                features = features_match_infos+ ['winner_name', 'loser_name']+ features_general+ features_winner + features_loser
                target = 1
            else:
                features = features_match_infos+ ['loser_name', 'winner_name']+ features_general+ features_loser + features_winner
                target = 0
                
            self.X.loc[count] = self.df[features].loc[count].values
            self.Y.loc[count] = target  
        
        self.X['target'] = self.Y['target'].copy()
        self.X = self.X.dropna().reset_index(drop=True)
        self.Y = self.X['target'].copy()
        self.X.drop(['target'], axis=1, inplace=True)

        print("Done !") 
    
    def prepare_matches(self, post_game_model):
        """
        Transforme le DataFrame en une liste de dictionnaires pour la classe TennisStrengthModel.
        :return: Liste de matchs sous forme de dictionnaires
        """
        matches = []
        
        if self.X is None:
            raise ValueError("You must call feature_engineering() before prepare_matches()")
        
        X_with_target = pd.concat([self.X, self.Y], axis=1)
        
        print("preparing matches per tournament...")
        for tourney in tqdm(X_with_target.tourney_name.unique()):
            group = X_with_target[X_with_target.tourney_name == tourney]
            for idx in range(len(group)):
                row = group.iloc[idx:idx+1]
                actual_score = post_game_model.calculate_actual_score(row)

                player1 = row["player1_name"].values[0]
                player2 = row["player2_name"].values[0]

                match_info = {
                    "tournament": row["tourney_name"].values[0],
                    "round": row["round"].values[0],
                    "player1": player1,
                    "player2": player2,
                    "actual_score": actual_score,
                    "target": row["target"].values[0],
                    "date": row["tourney_date"].values[0]
                }
                matches.append(match_info)

        return matches

        

if __name__ == "__main__":
    # === Exemple d'utilisation ===
    # Chargement du dataset (à adapter selon ton chemin)
    df = pd.read_csv("data/atp_matches_2024.csv")

    # Instanciation et préparation
    data_processor = DataProcessorForTennisStrenghtModel(df)
    tournament_points = data_processor.tournament_points
    matches = data_processor.prepare_matches()

    print(tournament_points)
    print(matches[:5])

    data_processor.train_logistic_model()
    # Exemple d'utilisation :
    row = df.iloc[0]
    score = data_processor.calculate_actual_score(row)
    print(f"Proba victoire winner: {score}")
