#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 20:22:49 2023

@author: sebastiendarius
"""

import pandas as pd
import numpy as np
import time
import seaborn as sns
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

df_avg = pd.read_csv('df_avg.csv')

# Select the columns to use for clustering
X = df_avg[["dist_from_blocker", "s"]]

# Create a KMeans model with 3 clusters
kmeans = KMeans(n_clusters=2)

# Fit the model to the data
kmeans.fit(X)

# Predict the clusters for each data point
predicted_clusters = kmeans.predict(X)

# Add the predicted clusters back to the original dataframe
df_avg['isBlocked'] = predicted_clusters

# View the resulting clusters
# Create a scatter plot with the first column on the x-axis and the second column on the y-axis
sns.scatterplot(x='dist_from_blocker', y='s', data=df_avg, hue='isBlocked')

# Add a legend
plt.legend()

# Show the plot
plt.show()

df_avg.to_csv("df_avg_w_clusters.csv")
