# NFL Pass Rusher Deception Metric

This project aims to develop a metric to evaluate the effectiveness of an NFL defense in disguising their pass rushers. By analyzing player tracking data and scouting data, the project involves clustering unblocked pass rushers, building a predictive model, and calculating the deception metric to assess the defense's performance.

## Overview

The project is divided into three main parts:

1. Clustering unblocked pass rushers
2. Building a predictive model for pass rushers
3. Calculating the deception metric and comparing it against the number of free rushers on a play

## Data

The data used in this project includes:

* Player tracking data: Contains the players' location, speed, and other attributes during each play.
* PFF Scouting data: Provides additional information about the players, such as their roles and performance in specific plays.
* Plays: Information about each play such as down, distance to go, and offensive and defensive teams.

## Methodology
### 1. Clustering unblocked pass rushers

The first part of the project involves preprocessing the tracking data and applying a clustering algorithm to identify unblocked pass rushers. The steps include:

    Preprocessing the tracking data to extract pass rushers and their attributes for each play
    Applying a K-means clustering algorithm to group pass rushers into clusters representing unblocked and blocked players

### 2. Building a predictive model for pass rushers

The second part of the project involves building a machine learning model to predict which players will rush the passer. The steps include:

* Preparing a dataset containing features and target labels (whether the player is a pass rusher or not)
* Splitting the dataset into training and test sets
* Training a 2D Convolutional Neural Network on the training set
* Evaluating the model's performance on the test set

### 3. Calculating the deception metric

The final part of the project involves calculating the deception metric, which assesses how well a defense disguises its pass rushers. The steps include:

* Using the predictive model to estimate the probability of each player being a pass rusher
* Calculating the deception metric based on these probabilities
* Comparing the deception metric against the actual number of free rushers on a play to evaluate the metrics's performance