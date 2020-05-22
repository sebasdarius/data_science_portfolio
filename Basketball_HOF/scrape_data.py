from basketball_reference_web_scraper import client
import requests
from bs4 import BeautifulSoup
import pandas as pd

def get_season_totals():
    """
    Scrapes season total stats for all seasons since 1950
    """
    df = pd.DataFrame()
    for year in range(1950, 2020):
        players = client.players_season_totals(season_end_year=year)
        for player in players:
            player["year"] = year
            try:
                player["team"] = player["team"].value
            except:
                None
            df = df.append(player, ignore_index=True)
    return df

def get_season_advanced():
    """
    Scrapes advanced stats for all seasons since 1950
    """
    df = pd.DataFrame()
    for year in range(1950, 2020):
        players = client.players_advanced_season_totals(season_end_year=year)
        for player in players:
            player["year"] = year
            
            df = df.append(player, ignore_index=True)
    return df

def get_playoff_totals():
    """
    Scrapes playoff total stats for all seasons since 1950
    """
    df = pd.DataFrame()
    for year in range(1950, 2020):
        players = client.players_playoff_totals(season_end_year=year)
        for player in players:
            player["year"] = year
            
            df = df.append(player, ignore_index=True)
    return df
    
def write_season_df(df):
    df.to_csv("nba_data/player_season_totals.csv")
    
def get_mvps():
    """
    """
    df = pd.DataFrame()
    
    url = "https://www.basketball-reference.com/awards/mvp.html"
    page = requests.get(url).content
    soup = BeautifulSoup(page, features="lxml")
    
    table = soup.find("table", id="mvp_summary")
    rows = table.findAll("tr")
    
    player_ids = []
    names = []
    lgs = []
    counts = []
    
    for row in rows[1:]:
        data = row.findAll()
        if len(data) != 0:
            player_tag = data[1]
            lgs.append(data[2].text)
            counts.append(data[3].text)
            names.append(player_tag.text)
            player_id = get_player_id(player_tag)
            
            player_ids.append(player_id)
            
    df = pd.DataFrame({"id":player_ids,
                      "Name":names,
                      "League":lgs,
                      "Count":counts
                      })
    
    return df

def get_all_star_career():
    df = pd.DataFrame()
    
    url = "https://www.basketball-reference.com/allstar/NBA-allstar-career-stats.html"
    page = requests.get(url).content
    soup = BeautifulSoup(page, features="lxml")
    
    table = soup.find("table", id="career-stats")
    rows = table.findAll("tr")
    
    player_ids = []
    names = []
    games = []
    
    for row in rows[1:]:
        if len(row.findAll("td")) != 0:
            data = row.findAll()
            player_tag = data[1]
            games.append(data[2].text)
            names.append(player_tag.text)
            player_id = get_player_id(player_tag)
            
            player_ids.append(player_id)
            
            
    df = pd.DataFrame({"id":player_ids,
                      "Name":names,
                      "Games":games,
                      })
    
    return df
    
def get_all_nba_selections():
    url = "https://www.basketball-reference.com/awards/all_league_by_player.html"
    page = requests.get(url).content
    soup = BeautifulSoup(page, features="lxml")
    
    table = soup.find("table", id="all_league_by_player")
    rows = table.findAll("tr")
    
    player_ids = []
    names = []
    total = []
    nba_1st = []
    nba_2nd = []
    nba_3rd = []
    aba_1st = []
    aba_2nd = []
    for row in rows[1:]:
        if len(row.findAll("td")) != 0:
            data = row.findAll()
            player_tag = data[2]
            names.append(player_tag.text)
            player_id = get_player_id(player_tag)
            player_ids.append(player_id)
            total.append(data[3].text)
            nba_1st.append(data[4].text)
            nba_2nd.append(data[5].text)
            nba_3rd.append(data[6].text)
            aba_1st.append(data[8].text)
            aba_2nd.append(data[9].text)
            
    df = pd.DataFrame({"id":player_ids,
                      "Name":names,
                      "AllNbaSelections":total,
                      "First_Team_All_NBA":nba_1st,
                      "Second_Team_All_NBA":nba_2nd,
                      "Third_Team_All_NBA":nba_3rd,
                      "First_Team_All_ABA":aba_1st,
                      "Second_Team_All_ABA":aba_2nd,
                      })
    return df
    
    
def get_dpoys():
    url = "https://www.basketball-reference.com/awards/dpoy.html"
    page = requests.get(url).content
    soup = BeautifulSoup(page, features="lxml")
    
    table = soup.find("table", id="dpoy_NBA")
    body = table.find("tbody")
    rows = body.findAll("tr")
    
    player_ids = []
    names = []
    year = []
    for row in rows[:]:
        if len(row.findAll("td")) != 0:
            data = row.findAll()
            year.append(data[0].text)
            player_tag = data[5]
            names.append(player_tag.text)
            player_id = get_player_id(player_tag)
            player_ids.append(player_id)
            
    df =  pd.DataFrame({"id":player_ids,
                         "Name":names,
                         "Year":year
                         })
    return df
    
def get_finals_mvp():
    url = "https://www.basketball-reference.com/awards/finals_mvp.html"
    page = requests.get(url).content
    soup = BeautifulSoup(page, features="lxml")
    
    table = soup.find("table", id="finals_mvp_NBA")
    body = table.find("tbody")
    rows = body.findAll("tr")
    
    player_ids = []
    names = []
    year = []
    for row in rows[:]:
        if len(row.findAll("td")) != 0:
            data = row.findAll()
            year.append(data[0].text)
            player_tag = data[5]
            names.append(player_tag.text)
            player_id = get_player_id(player_tag)
            player_ids.append(player_id)
            
            
    df = pd.DataFrame({"id":player_ids,
                       "Name":names,
                       "Year":year
                       })
    
    return df
            
            
def get_player_id(player_tag):
    """Returns the player ids for an event"""
    try:
        tag1 = player_tag
        player1 = str(tag1).split('.html')[0].split('/')[-1]
    except IndexError:
        player1 = ''

    return player1

if __name__ == "__main__":
    PATH = "nba_data"
    get_season_totals().to_csv(PATH + "/player_season_totals.csv", index=False)
    get_season_advanced().to_csv(PATH + "/player_season_advanced.csv", index=False)
    get_playoff_totals().to_csv(PATH + "/player_playoff_totals.csv", index=False)
    get_mvps().to_csv(PATH + "/mvp_counts.csv", index=False)
    get_all_star_career().to_csv(PATH + "/all_star_by_year.csv", index=False)
    get_all_nba_selections().to_csv(PATH + "all_league_counts.csv", index=False)
    get_dpoys().to_csv(PATH + "dpoy_by_year")
    get_finals_mvp().to_csv(PATH + "/finals_mvp_by_year.csv")
    
    
