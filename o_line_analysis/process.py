#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 11:38:21 2023

@author: sebastiendarius
"""

import pandas as pd
import numpy as np
import time
import seaborn as sns
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def get_specific_frame(df, frame):
    df_temp = df.copy()
    df_temp = df_temp[df_temp['event'] == 'ball_snap']
    return df_temp

def flatten_tracking_data(tracking_single_frame):
    #give a number to each player and the ball
    player_pos = np.arange(1, 24, 1)
    features = ['x', 'y', 's', 'a', 'dis', 'isRusher', 'pff_hurry', 'pff_sack', 'nflId']

    feature_by_pos = []

    for feature in features:
        for p in player_pos:
            feature_by_pos.append(f'{feature}_{p}')
        


    df_all_pos = pd.DataFrame(columns=feature_by_pos)

    #add values for features by play
    tracking_first_frame

    for (game_id, play_id) in tracking_single_frame.index.unique():
        #add values for features by play
        values_by_pos = []

        for feature in features: 
            values_by_pos = values_by_pos + list(tracking_first_frame.loc[game_id, play_id][feature])
    
        values_by_pos = pd.Series(values_by_pos, index=feature_by_pos)
        values_by_pos = pd.DataFrame([values_by_pos], index=[[game_id], [play_id]])
    
    
        df_all_pos = pd.concat([df_all_pos, values_by_pos])

    return df_all_pos

def update_locations(df_all_pos):
    #Update x and y to be relative to the placement of the ball
    df_eng = df_all_pos.copy()
    df_eng['x_ball'] = df_eng['x_12']
    df_eng['y_ball'] = df_eng['y_12']

    for i in range(1, 24, 1):
        df_eng[f'x_{i}'] = df_eng[f'x_{i}'] - df_eng['x_ball']
        df_eng[f'y_{i}'] = df_eng[f'y_{i}'] - df_eng['y_ball']
        
        df_eng['gameId'] = pd.Series(df_eng.index, index=df_eng.index).apply(lambda x: x[0])
        df_eng['playId'] = pd.Series(df_eng.index, index=df_eng.index).apply(lambda x: x[1])
    
    df_eng.reset_index(inplace=True, drop=True)
    df_eng.set_index(['gameId', 'playId'], inplace=True)
        
    return df_eng
    
    
tracking_merged = pd.read_csv("tracking_merged.csv").set_index(['gameId', 'playId'], drop=False)
tracking_first_frame = get_specific_frame(tracking_merged, 1)
df_flattened = flatten_tracking_data(tracking_first_frame)
df_flattened = update_locations(df_flattened)


df_flattened.to_csv('df_flattened.csv')

