#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 09:58:54 2020

@author: jdarius
"""
import pandas as pd
import numpy as np
import os
from sklearn.pipeline import make_pipeline

#all_nba = pd.read_csv("nba_data/all_nba_by_year.csv")
#all_star_old = pd.read_csv("data/basketball_player_allstar.csv")
#mvp = pd.read_csv("nba_data/mvp_by_year.csv")
#dpoy = pd.read_csv("nba_data/dpoy_by_year.csv")
#advanced = pd.read_csv("nba_data/player_season_advanced.csv")
#finals_mvp = pd.read_csv("nba_data/finals_mvp_by_year.csv")
#champions = pd.read_csv("nba_data/nba_champions.csv")
#all_star = pd.read_csv("nba_data/all_star_by_year.csv")

PATH = "nba_data"
files = os.listdir(PATH)
data = {}
for file in files[:]:
    data[file] = pd.read_csv(PATH + "/" + file, encoding='latin')

def fix_season(season):
    return int(season[:4]) +1

class JoinMvp():
    def fit(self, df, y=None):
        return self
    
    def transform(self, df):
        mvp = data["mvp_by_year.csv"]
        
        mvp = mvp[["Season", "Player"]]
        mvp["Season"] = mvp["Season"].apply(fix_season)
        
        mvp["Name"] = mvp["Player"].apply(lambda x: x.split("\\")[0])
        mvp["Id"] = mvp["Player"].apply(lambda x: x.split("\\")[1])
        mvp.drop("Player", axis=1, inplace=True)
        
        mvp["Count_mvp"] = 1
        mvp.set_index(["Id", "Season"], inplace=True)
        
        df = df.join(mvp[["Count_mvp"]], on=["slug", "year"], how="left")
        df["Count_mvp"] = df["Count_mvp"].fillna(0)
        
        return df

def all_nba_names(name):
    if name[-2:] in [" C", " F", " G"]:
        return name[:-2]
    else:
        return name
    
def fix_names(x):
    if x[-1] == "*":
        return x[:-1]
    else:
        return x
    
class JoinAllNBA():
    def fit(self, df, y=None):
        return self
    
    def transform(self, df):
        all_nba = data["all_nba_by_year.csv"]
        
        all_nba.rename(columns={"0":"Player"}, inplace=True)
        all_nba.Player = all_nba.Player.apply(all_nba_names)
        
        all_nba["First_Team_All_NBA"] = (all_nba["Team"] == "1st").astype(int)
        all_nba["Second_Team_All_NBA"] = (all_nba["Team"] == "2nd").astype(int)
        
        all_nba.set_index(["Player", "Season"], inplace=True)
        
        df.name = df.name.apply(fix_names)
        
        df = df.join(all_nba[["First_Team_All_NBA", "Second_Team_All_NBA"]],
                     on=["name", "year"], how="left").fillna(0)
        
        return df
    
class JoinDpoy():
    def fit(self, df, y=None):
        return self
    
    def transform(self, df):
        dpoy = data["dpoy_by_year.csv"]
        
        dpoy.Year = dpoy.Year.apply(fix_season)
        dpoy["Count_dpoy"] = 1
        dpoy.set_index(["id", "Year"], inplace=True)
        
        df = df.join(dpoy[["Count_dpoy"]], on=["slug", "year"],
                            how="left").fillna(0)
        
        return df

class JoinFinalsMvp():
    def fit(self, df, y=None):
        return self
    
    def transform(self, df):
        finals_mvp = data["finals_mvp_by_year.csv"]
        
        finals_mvp.Year = finals_mvp.Year.apply(fix_season)
        
        finals_mvp["Count_finals_mvp"] = 1
        finals_mvp.set_index(["id", "Year"], inplace=True)
        
        df = df.join(finals_mvp[["Count_finals_mvp"]],
                            on=["slug", "year"], how="left").fillna(0)
        
        return df
    
class JoinAllStar():
    def fit(self, df, y=None):
        return self
    
    def transform(self, df):
        all_star_old = data["basketball_player_allstar.csv"]
        all_star = data["all_star_by_year.csv"]
        
        all_star_old = all_star_old[all_star_old["league_id"]=="NBA"]
        all_star_old = all_star_old[["player_id", "season_id", "games_played"]]
        
        all_star_old = all_star_old.rename(
                            columns={"player_id":"Id",
                            "season_id":"Season",
                            "games_played":"All_stars"}
                            )
        
        all_star = all_star.rename(columns={"Unnamed: 1":"Season"}).dropna()
        all_star["Name"] = all_star["Starters"].apply(lambda x: x.split("\\")[0])
        all_star["Id"] = all_star["Starters"].apply(lambda x: x.split("\\")[1])
        all_star["All_stars"] = 1

        all_star = all_star.append(all_star_old, ignore_index=True)
        
        all_star.drop("Starters", axis=1, inplace=True)
        all_star.to_csv(PATH + "/all_stars_by_year.csv", index=False)
        
        all_star.set_index(["Id", "Season"], inplace=True)
        df = df.join(all_star["All_stars"], on=["slug", "year"], how="left").fillna(0)
        
        return df

def get_team(team):
    try:
        if str(team) == "nan":
            return "None"
        else:
            team = team[5:]
            team = team.split("_")
            team[0] = team[0][0] + team[0][1:].lower()
            team_name = team[0]
            for word in team[1:]:
                word = word[0] + word[1:].lower()
                team_name = team_name + " " + word
            return team_name
    except:
        print(team) 
        
        
class JoinChampionships():
    def fit(self, df, y=None):
        return self
    
    def transform(self, df):
        champions = data["nba_champions.csv"]
        playoffs = data["player_playoff_totals.csv"]
        
        champions["team_year"] = (champions["Year"].astype("str") + " " +
                              champions["Champion"])
        
        playoffs.team = playoffs.team.apply(get_team)
        playoffs["team_year"] = (playoffs["year"].astype(int).astype(
                                'str')) + " " + playoffs["team"]
        
        playoffs["Championships"] = playoffs["team_year"]\
            .apply(lambda x: x in list(champions["team_year"])).astype("int8")
            
        playoffs.year = playoffs.year.astype('int32')
        playoffs = playoffs[["slug", "year", "Championships"]]
        
        playoffs.set_index(["slug", "year"], inplace=True)
        df = df.join(playoffs, on=["slug", "year"], how="left").fillna(0)
        
        return df
    
class GetQualifiedPlayers():
    def fit(self, df, y=None):
        return self
    
    def transform(self, df):
        ids = data["bball_ref_hof_pred.csv"]["id"]
        
        df["qualified"] = df.slug.apply(lambda x: x in list(ids))
        df = df[df["qualified"]==True]
        
        return df
    
class GetCumulativeAwards():
    def fit(self, df, y=None):
        return self
    
    def transform(self, df):
        player_ids = list(df.slug.unique())
        master_df = df
        df = pd.DataFrame()
        for player_id in player_ids:
            player_df = master_df[master_df["slug"] == player_id].sort_values("year")
    
            slug = player_df["slug"].iloc[0]
            name = player_df["name"].iloc[0]
    
            for i in range(len(player_df)):
                temp_series = player_df[:i+1]
                year = temp_series["year"].iloc[-1]
                temp_series = temp_series[["value_over_replacement_player",
                                           "Count_mvp", "First_Team_All_NBA",
                                           "Second_Team_All_NBA", "Count_dpoy",
                                           "Count_finals_mvp", "All_stars", 
                                           "Championships"]].sum()
                temp_series["slug"] = slug
                temp_series["name"] = name
                temp_series["year"] = year
        
                df = df.append(temp_series, ignore_index=True)
            
        df.year = df.year.astype(int)
        
        return df
        
    
if __name__ == "__main__":
    
    df =  data["player_season_advanced.csv"].groupby(["slug", "name", "year"],
                                                     as_index=False).agg(np.sum)
    
    df = df[["slug", "name", 
                   "value_over_replacement_player", "year"]]
    
    pipe = make_pipeline(JoinMvp(),
                          JoinAllNBA(),
                          JoinDpoy(),
                          JoinFinalsMvp(),
                          JoinAllStar(),
                          JoinChampionships(),
                          GetQualifiedPlayers(),
                          GetCumulativeAwards())
    
    df = pipe.fit_transform(df)
    df.to_csv(PATH + "/players_by_year.csv")