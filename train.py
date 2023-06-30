import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as KNN

class nba_knn():
  def __init__(self):
    self.data = pd.read_csv("https://raw.githubusercontent.com/ssuni2/nba_data_csv/main/nba_advanced.csv")
    self.data.drop(['Unnamed: 0', 'Visitor_PTS', 'Home_PTS', 'Home_POSS', 'Visitor_POSS'], axis = 1, inplace = True)
    self.data.dropna(inplace=True)
    self.data.reset_index(drop=True, inplace=True)

    target = self.data['VisitorWin']
    input_columns = self.data.loc[:, ~self.data.columns.isin(['Visitor', 'Home', 'VisitorWin'])]


    k = 6
    self.my_KNN_model = KNN(n_neighbors=k, algorithm='ball_tree')
    self.my_KNN_model.fit(input_columns, target)

    

  def classify(self, VisitorTeamName, HomeTeamName):
    # data = pd.read_csv("https://raw.githubusercontent.com/ssuni2/nba_data_csv/main/nba_advanced.csv")
    # data = data.drop(['Unnamed: 0', 'Visitor_PTS', 'Home_PTS', 'Home_POSS', 'Visitor_POSS'], axis = 1)
    # data.dropna(inplace=True)
    # data.reset_index(drop=True, inplace=True)
    
    data = self.data
    
    vis_data = data.drop_duplicates('Visitor')
    home_data = data.drop_duplicates('Home')
    vis = vis_data.loc[vis_data['Visitor'] == VisitorTeamName]
    to_drop_1 = ['Home', 'VisitorWin', 'Home_GP', 'Home_W',
       'Home_L', 'Home_MIN', 'Home_OffRtg', 'Home_DefRtg', 'Home_NetRtg',
       'Home_AST%', 'Home_AST/TO', 'Home_AST', 'Home_OREB%', 'Home_DREB%',
       'Home_REB%', 'Home_TOV%', 'Home_eFG%', 'Home_TS%', 'Home_PACE', 'Home_PIE', 'Visitor']
    vis = vis.drop(to_drop_1, axis = 1)
    vis.reset_index(drop=True, inplace=True)
    home = home_data.loc[home_data['Home'] == HomeTeamName]
    to_drop_2 = ['Visitor', 'VisitorWin', 'Visitor_GP', 'Visitor_W', 'Visitor_L',
       'Visitor_MIN', 'Visitor_OffRtg', 'Visitor_DefRtg', 'Visitor_NetRtg',
       'Visitor_AST%', 'Visitor_AST/TO', 'Visitor_AST', 'Visitor_OREB%',
       'Visitor_DREB%', 'Visitor_REB%', 'Visitor_TOV%', 'Visitor_eFG%',
       'Visitor_TS%', 'Visitor_PACE', 'Visitor_PIE', 'Home']
    home = home.drop(to_drop_2, axis=1)
    home.reset_index(drop=True, inplace=True)

    combined = pd.concat([vis, home], axis = 1)
  
    prediction = self.my_KNN_model.predict(combined)
    return prediction
