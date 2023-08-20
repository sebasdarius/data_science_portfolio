#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 12:45:41 2023

@author: sebastiendarius
"""

import pandas as pd
import numpy as np
import time
import seaborn as sns

PATH = "data"

plays = pd.read_csv(f"{PATH}/plays.csv")
players = pd.read_csv(f"{PATH}/players.csv")
pff_scouting = pd.read_csv(f"{PATH}/pffScoutingData.csv")

#Read Tracking Data
weeks = np.arange(1, 9, 1)

df_tracking = pd.DataFrame()

for w in weeks:
    #temporary data frame for reading given week
    df_tracking_temp = pd.read_csv(f"{PATH}/week{w}.csv")
    
    df_tracking = pd.concat([df_tracking, df_tracking_temp])
    
#Make it so that offense is always facing right
df_tracking['x'] = np.where(df_tracking['playDirection'] == 'left', 120 - df_tracking['x'], df_tracking['x'])
df_tracking['y'] = np.where(df_tracking['playDirection'] == 'left', 160/3 - df_tracking['y'], df_tracking['y'])

def add_side_of_ball(df):
    df_temp = df.copy()
    #value of 0 for offensive players, 1 for ball, 2 for defense
    df_temp = df_temp.merge(plays, how='left', on=['gameId', 'playId'])
    side_of_ball = (df_temp['team'] == df_temp['possessionTeam'])*2
    side_of_ball = side_of_ball + (df_temp['team'] == 'football')
    df_temp['side_of_ball'] = side_of_ball
    
    return df_temp

def merge_with_scouting(df):
    #merges with the scouting data and get only needed features
    df_temp = df.copy()
    
    df_temp = df_temp[['gameId', 'playId', 'nflId', 'frameId','jerseyNumber', 'team', 'playDirection', 'x', 'y', 's', 'a', 'dis', 'o', 'dir', 'side_of_ball', 'event']]
    scouting = pff_scouting[['gameId', 'playId', 'pff_role', 'nflId', 'pff_hit', 'pff_hurry', 'pff_sack', 'pff_positionLinedUp']]
    df_temp = df_temp.merge(scouting, how='left', on=['gameId', 'playId', 'nflId'])
    return df_temp
    
def get_specific_frame(df, frame):
    df_temp = df.copy()
    df_temp = df_temp[df_temp['frameId'] == frame]
    return df_temp

def get_rushers(df):
    #Adds a column that identifies whether or not that player was a pass rusher on that play
    df_temp = df.copy()
    df_temp = df_temp.sort_values(['gameId', 'playId', 'side_of_ball', 'y'])
    df_temp['isRusher'] = (df_temp['pff_role'] == "Pass Rush").astype(int)
    
    return df_temp

def add_game_play_player(df):
    #Adds a unique identifier for the player during that play
    df_temp = df.copy()
    df_temp['nflId'] = np.where(df_temp['nflId'].isna(), 0, df_temp['nflId'])
    game_play_player_temp = df_temp['gameId'].astype(str) + df_temp['playId'].astype(str) + df_temp['nflId'].astype(int).astype(str)
    game_play_temp = df_temp['gameId'].astype(str) + df_temp['playId'].astype(str)
    df_temp.insert(4, 'game_play', game_play_temp, True)
    df_temp.insert(5, 'game_play_player', game_play_player_temp, True)
    
    return df_temp

def add_ball_location(df):
    df_temp = df.copy()
    df_temp_ball = tracking_merged[tracking_merged['team'] == 'football'].copy()
    df_temp_ball.rename({'x': 'ball_x', 'y':'ball_y'}, axis=1, inplace=True);
    df_temp_ball = df_temp_ball[['playframe', 'ball_x', 'ball_y']]
    df_temp = df_temp.merge(df_temp_ball, how='left', on=['playframe'])
    
    return df_temp

def add_qb_location(df):
    df_temp = df.copy()
    df_temp['playframe'] = df_temp['playframe'].astype(str)
    
    df_temp_qb = tracking_merged[tracking_merged['pff_role'] == 'Pass']
    df_temp_qb.rename({'x': 'qb_x', 'y':'qb_y'}, axis=1, inplace=True)
    df_temp_qb = df_temp_qb[['playframe', 'qb_x', 'qb_y']]
    df_temp = df_temp.merge(df_temp_qb, how='left', on=['playframe'])

    return df_temp
    


def get_rushers_blockers_after_snap(tracking_merged):
    #Figure out which frame the snap occurs
    snap = tracking_merged[tracking_merged['event'] == 'ball_snap'].groupby(['gameId', 'playId']).agg(np.min)
    snap.rename({'frameId' : 'snap_frame'}, axis=1, inplace=True)

    #Only include blockers and pass rushers
    rushers_blockers = tracking_merged[(tracking_merged['pff_role'] == 'Pass Block') | (tracking_merged['pff_role'] == 'Pass Rush')]
    rushers_blockers.insert(0, 'playframe', rushers_blockers['gameId'].astype(str) + rushers_blockers['playId'].astype(str) + rushers_blockers['frameId'].astype(str), True)

    #Only include frames at or after the snap
    rushers_blockers = rushers_blockers.merge(snap[['snap_frame']], how='inner', left_on=['gameId', 'playId'], right_index=True)
    rushers_blockers = rushers_blockers[rushers_blockers['frameId'] >= rushers_blockers['snap_frame']]
    
    return rushers_blockers
    
def get_ambiguous_rushers():
    #Add the game Id, play Id and NFL Id of the blocked player
    pff_scouting_blockers = pff_scouting[pff_scouting['pff_role'] == 'Pass Block']

    pff_scouting_blockers = pff_scouting_blockers[pff_scouting_blockers['pff_nflIdBlockedPlayer'].notna()]

    index = pff_scouting_blockers['gameId'].astype(str) + pff_scouting_blockers['playId'].astype(str) + pff_scouting_blockers['pff_nflIdBlockedPlayer'].astype(int).astype(str)
    pff_scouting_blockers.insert(3, 'blocked_players', index, True)
    
    #Adds whether or not the pass rusher was initially blocked on the play
    pff_scouting_rushers = pff_scouting[pff_scouting['pff_role'] == 'Pass Rush']
    index = pff_scouting_rushers['gameId'].astype(str) + pff_scouting_rushers['playId'].astype(str) + pff_scouting_rushers['nflId'].astype(str)
    pff_scouting_rushers.insert(3, 'game_play_player', index, True)

    #Finds the pass rushers for whom PFF does not specify whether or not 
    was_blocked_init = pff_scouting_rushers['game_play_player'].isin(pff_scouting_blockers['blocked_players'])
    pff_scouting_rushers.insert(5, 'was_blocked_init', was_blocked_init, True)
    scouting_rushers_no_data = pff_scouting_rushers[pff_scouting_rushers['was_blocked_init'] == False]
    scouting_rushers_no_data
    
    return scouting_rushers_no_data

def calculate_nearest_blocker(playframe, nflId):
    """Takes the playframe and nflId of the defender to find the distance of the nearest blocker on the play"""
    #Get Blockers On Play
    df_play_temp = blockers[blockers['playframe'] == playframe]
    rusher_row = rushers_blockers_temp.loc[(playframe, nflId)]
    rusher_loc = rusher_row['x'][0], rusher_row['y'][0]

    #Calculate all distances
    distances = df_play_temp.apply(lambda blocker: np.linalg.norm(np.array(rusher_loc) - np.array([blocker.x, blocker.y])), axis=1)
    
    return distances.min()

def calculate_avg_dist_from_blocker(rushers_no_data):
    start = time.time()

    dist_from_blocker = rushers_no_data.iloc[::5, :].apply(lambda x: calculate_nearest_blocker(x['playframe'],x['nflId']), axis=1)
    rushers_no_data.insert(7, 'dist_from_blocker', dist_from_blocker, True)
    rushers_no_data.to_csv('rushers_no_data.csv')

    end = time.time()

    #Subtract Start Time from The End Time
    total_time = end - start
    print("\nCalculating Average Distance:"+ str(total_time))
    
    return rushers_no_data

def add_displacement(df):
    df_temp = df.copy()
    x_diff = df_temp['qb_x'] - df_temp['x']
    y_diff = df_temp['qb_y'] - df_temp['y']
    
    df_temp['disp_x'] = x_diff
    df_temp['disp_y'] = y_diff
    
    return df_temp

def add_velocity(df):
    df_temp = df.copy()
    angle = 450 - df_temp['dir'] #make direction counter-clockwise with positive x-axis as 0 degrees
    df_temp['angle'] = angle * np.pi / 180
    vel_x = np.cos(angle) * df_temp['s']
    vel_y = np.sin(angle) * df_temp['s']
    
    df_temp['vel_x'] = vel_x
    df_temp['vel_y'] = vel_y
    
    return df_temp
    

def get_speed_towards_qb(df):
    df_temp = df.copy()
    
    df_temp['speed_to_qb'] = df_temp['disp_x'] * df_temp['vel_x'] + df_temp['disp_y'] * df_temp['vel_y']
    
    return df_temp

def get_averages(df):
    #Gets average speed and distance from nearest blocker on the play
    df_temp = df.copy()[['playframe', 'nflId', 'game_play_player', 'gameId', 'playId', 'dist_from_blocker', 's', 'speed_to_qb']]
    df_avg = df_temp.groupby('game_play_player').agg(np.mean)
    
    return df_avg
    
if __name__=="__main__":
    
    tracking_merged = add_side_of_ball(df_tracking)
    tracking_merged = merge_with_scouting(tracking_merged)
    tracking_merged = get_rushers(tracking_merged)
    tracking_merged = add_game_play_player(tracking_merged)
    tracking_merged['playframe'] = tracking_merged['game_play'].astype(str) + tracking_merged['frameId'].astype(str)

    
    rushers_blockers = get_rushers_blockers_after_snap(tracking_merged)
    scouting_rushers_no_data = get_ambiguous_rushers()
    
    #Remove certain columns in order to speed up the calculations
    rushers_blockers_temp = rushers_blockers[['game_play_player', 'playframe', 'gameId', 
                                              'playId', 'nflId', 'x', 'y', 's', 'dir', 'pff_role']]
    blockers = rushers_blockers_temp[rushers_blockers_temp['pff_role'] == "Pass Block"].drop(['pff_role', 'gameId', 'playId'], axis=1)
    rushers = rushers_blockers_temp[rushers_blockers_temp['pff_role'] == "Pass Rush"].drop(['pff_role'], axis=1)
    #only include players that were not initially blocked on the play. These are the players for 
    #whom we need to determine whether or not they were free rushers

    rushers_no_data = rushers[rushers['game_play_player'].isin(scouting_rushers_no_data['game_play_player'])]
    
    rushers_no_data = calculate_avg_dist_from_blocker(rushers_no_data)
    
    tracking_merged.to_csv('tracking_merged.csv', index=False)
    rushers_no_data = pd.read_csv('rushers_no_data.csv')
    rushers_no_data = add_qb_location(rushers_no_data)
    rushers_no_data = add_displacement(rushers_no_data)
    rushers_no_data = add_velocity(rushers_no_data)
    rushers_no_data = get_speed_towards_qb(rushers_no_data)
    
    df_avg = get_averages(rushers_no_data)
    df_avg.to_csv('df_avg.csv')
    
    
    
