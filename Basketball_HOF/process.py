#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 13:35:18 2020

@author: jdarius
"""
import pandas as pd
import numpy as np
import os
import scipy
from sklearn.pipeline import make_pipeline

PATH = "nba_data"
files = os.listdir(PATH)
data = {}
for file in files[:]:
    data[file] = pd.read_csv(PATH + "/" + file, encoding='latin')
    
bball_ref = pd.read_csv(PATH + "/" + "bball_ref_hof_pred.csv", encoding='latin').set_index("id")

class JoinMvp():
    def fit(self, df, y=None):
        return self
    
    def transform(self, df):
        data["mvp_counts.csv"].set_index("id", inplace=True)
        df =  df.join(data["mvp_counts.csv"][["League", "Count"]].add_suffix("_mvp"),
                             on="id", how="left")
    
        df["Count_mvp"].fillna("0", inplace=True)
    
        return df

class JoinDpoy():
    def fit(self, df, y=None):
        return self
    
    def transform(self, df):
        dpoy = data["dpoy_by_year.csv"].copy()

        dpoy_count = dpoy.groupby("id").agg("count").rename(columns={"Name": "Count_dpoy"})["Count_dpoy"]

        df = df.join(dpoy_count, on="id", how="left")
        df["Count_dpoy"].fillna("0", inplace=True)
        
        return df

class JoinAllStar():
    def fit(self, df, y=None):
        return self
    
    def transform(self, df):
        data["all_star_game_counts.csv"].rename(columns={"Games": "All_stars"},
                                       inplace=True)
        data["all_star_game_counts.csv"].set_index("id", inplace=True)

        df = df.join(data["all_star_game_counts.csv"][["All_stars"]],
                           on="id", how="left")
        df["All_stars"].fillna("0", inplace=True)
        
        return df
    
class JoinFinalsMvp():
    def fit(self, df, y=None):
        return self
    
    def transform(self, df):
        finals_mvp = data["finals_mvp_by_year.csv"].copy()
        finals_mvp_count = finals_mvp.groupby("id").agg("count")\
                            .rename(columns={"Name": "Count_finals_mvp"})["Count_finals_mvp"]

        df = df.join(finals_mvp_count, on="id", how="left")
        df["Count_finals_mvp"].fillna("0", inplace=True)
        
        return df

def get_team(team):
    try:
        if str(team) == "nan":
            return "None"
        else:
            team = team[5:]
            team = team.split("_")
            team_name = team[0][0] + team[0][1:].lower()
            
            for word in team[1:]:
                word = word[0] + word[1:].lower()
                team_name = team_name + " " + word
            return team_name
    except:
        print(team)


class JoinPlayoffs():
    def fit(self, df, y=None):
        return self
    
    def transform(self, df):
        playoff_totals = data["player_playoff_totals.csv"].copy()
        playoff_totals["team"] = playoff_totals["team"].apply(get_team)

        champs = data["nba_champions.csv"]
        champs["team_year"] = champs["Year"].astype("str") + " " + champs["Champion"]

        #
        playoff_totals["team_year"] = playoff_totals["year"].astype(int)\
                                    .astype("str") + " " + playoff_totals["team"]
        playoff_totals["Championships"] = playoff_totals["team_year"]\
                                           .apply(lambda x: x in list(champs["team_year"]))
        
        playoff_totals = playoff_totals.groupby("slug").agg(np.sum)
        playoff_totals.drop(["year", "age"], axis=1, inplace=True)
        playoff_totals = playoff_totals.add_suffix("_playoffs")
        
        df = df.join(playoff_totals, on="id", how="left")
        
        return df
    
class JoinRegularSeason():
    def fit(self, df, y=None):
        return self
    
    def transform(self, df):
        season_totals = data["player_season_totals.csv"].copy()
        season_totals.drop(["age", "team", "year"], axis=1, inplace=True)
        season_totals = season_totals.groupby("slug").agg(np.sum)
        
        df = df.join(season_totals, on="id", how="left")
        
        return df

class JoinAdvanced():
    def fit(self, df, y=None):
        return self
    
    def transform(self, df):
        advanced_stats = data["player_season_advanced.csv"].copy()
        advanced_stats = advanced_stats[advanced_stats["year"] >= 1952]
        advanced_stats = advanced_stats.drop(["year", "age", "name", "positions", "team"], axis=1)
        
        #Get Peak Win Shares and VORP
        ws = advanced_stats.groupby("slug").agg(np.max)["win_shares"]
        vorp = advanced_stats.groupby("slug").agg(np.max)["value_over_replacement_player"]
        
        
        ids = advanced_stats["slug"]
        total_minutes = advanced_stats["minutes_played"] * advanced_stats["games_played"]
        total_minutes.index = ids
        advanced_stats.drop(["slug", "minutes_played", "games_played"], axis=1, inplace=True)
        
        #Get weighted average of advanced stats
        advanced_arr = np.array(advanced_stats) * np.array(total_minutes).reshape((-1, 1))
        advanced_stats_mult = pd.DataFrame(advanced_arr, columns=advanced_stats.columns)
        advanced_stats_mult.index = ids
        
        total_minutes = total_minutes.groupby("slug").agg(np.sum)
        
        advanced_stats_sum = advanced_stats_mult.groupby("slug").agg(np.sum)
        advanced_stats_avg = np.array(advanced_stats_sum) / np.array(total_minutes).reshape((-1, 1))
        
        advanced_stats = pd.DataFrame(advanced_stats_avg, columns = advanced_stats.columns, index=advanced_stats_sum.index)
        
        df = df.join(advanced_stats, on="id", how="left")
        df["peak_win_shares"] = ws
        df["peak_value_over_replacement_player"] = vorp
        
        return df
    
class JoinAllLeague():
    def fit(self, df, y=None):
        return self
    
    def transform(self, df):
        all_league = data["all_league_counts.csv"]
        all_league.set_index("id", inplace=True)
        
        df = df.join(all_league.drop("Name", axis=1), on="id", how="left")
              
        return df

def get_most_common_teams():
    """
    Get each player's most common team
    """
    
    seasons = data["player_season_totals.csv"]
    seasons.team = seasons.team.apply(get_team)
    seasons = seasons[["slug", "team"]]
    
    most_common_teams = seasons.groupby("slug").agg(scipy.stats.mode)
    most_common_teams.team = most_common_teams.team.apply(lambda x: x[0][0])
    
    return most_common_teams

if __name__ == "__main__":
    bball_ref = pd.read_csv(PATH + "/" + "bball_ref_hof_pred.csv", 
                            encoding='latin').set_index("id")
    
    pipe = make_pipeline(JoinMvp(),
                         JoinDpoy(),
                         JoinAllStar(),
                         JoinFinalsMvp(),
                         JoinPlayoffs(),
                         JoinRegularSeason(),
                         JoinAdvanced())
    
    merged_df = pipe.fit_transform(bball_ref)
    merged_df.to_csv(PATH + "/merged.csv")
    
    
    get_most_common_teams().to_csv(PATH + "/most_common_teams.csv")
    