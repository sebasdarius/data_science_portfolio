#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 11:38:21 2023

@author: sebastiendarius
"""

import pandas as pd
import numpy as np
import time
PATH = "oline_data"

def get_specific_frame(df, frame):
    df_temp = df.copy()
    df_temp = df_temp[df_temp['event'] == 'ball_snap']
    return df_temp

def get_data_at_frame(play, features, feature_by_pos):
    values = play[features].to_numpy().T.flatten()
    values = pd.Series(values, index=feature_by_pos)
    return values

#new
def flatten_tracking_data(tracking_single_frame):
    #give a number to each player and the ball
    player_pos = np.arange(1, 24, 1)
    features = ['x', 'y', 's', 'a', 'dis', 'isRusher', 'pff_hurry', 'pff_sack', 'nflId']

    feature_by_pos = []

    for feature in features:
        for p in player_pos:
            feature_by_pos.append(f'{feature}_{p}')

    #add values for features by play
    df_flattened = tracking_single_frame.groupby(['gameId', 'playId']).apply(get_data_at_frame, features=features, feature_by_pos=feature_by_pos)

    return df_flattened

def update_locations(df_flattened):
    #Update x and y to be relative to the placement of the ball
    df_flattened_temp = df_flattened.copy()
    df_flattened_temp['x_ball'] = df_flattened_temp['x_12']
    df_flattened_temp['y_ball'] = df_flattened_temp['y_12']

    for i in range(1, 24, 1):
        df_flattened_temp[f'x_{i}'] = df_flattened_temp[f'x_{i}'] - df_flattened_temp['x_ball']
        df_flattened_temp[f'y_{i}'] = df_flattened_temp[f'y_{i}'] - df_flattened_temp['y_ball']

        
    return df_flattened_temp
    

tracking_merged = pd.read_csv(f"{PATH}/tracking_merged.csv")
tracking_first_frame = get_specific_frame(tracking_merged, 1)
df_flattened = flatten_tracking_data(tracking_first_frame)
df_flattened = update_locations(df_flattened)


#df_flattened.to_csv('df_flattened.csv')



