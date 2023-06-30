#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 20:47:15 2023

@author: sebastiendarius
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation, Input


# Load and process the dataset
def load_data():
    df_flattened = pd.read_csv("df_flattened.csv").set_index(['gameId', 'playId'])
    
    return df_flattened

def process_data(df_flattened):

    num_samples = len(df_flattened)
    num_players = 23
    num_defensive_players = 11
    num_offensive_players = 11
    num_features = 5  # x, y, speed, acceleration, direction

    # Extract features and labels from the dataframe
    X = df_flattened.iloc[:, :num_players * num_features]
    y = df_flattened.iloc[:, num_players * 5:num_players * 5 + 11]

    # Reshape features into the desired format
    X_reshaped = np.zeros((num_samples, 2, 11, num_features))

    for i in range(num_samples):
        for j in range(num_defensive_players):
            X_reshaped[i, 0, j] = X.iloc[i, j*num_features:(j+1)*num_features].values
        for j in range(num_defensive_players, num_players-1):  # Exclude the ball
            X_reshaped[i, 1, j-num_defensive_players] = X.iloc[i, j*num_features:(j+1)*num_features].values

    return X_reshaped, y

# Define the 2D CNN model
def create_2d_cnn(input_shape, num_outputs):
    inputs = Input(input_shape)
    x = Conv2D(filters=32, kernel_size=(2, 3), activation='relu')(inputs)
    x = Conv2D(filters=16, kernel_size=(1, 3), activation='relu')(x)
    x = Flatten()(x)
    outputs = [Dense(1, activation='sigmoid')(x) for _ in range(num_outputs)]

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss=['binary_crossentropy'] * num_outputs, metrics=[['accuracy']]*11)

    return model

# Plot the training loss
def plot_training_loss(history):
    plt.plot(history.history['loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

# Plot the training accuracy for each output
def plot_training_accuracy(history, num_outputs):
    output_accuracy_keys = [key for key in history.history.keys() if 'accuracy' in key]

    plt.figure()
    for i, key in enumerate(output_accuracy_keys):
        plt.plot(history.history[key])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend([f'Output {i+1}' for i in range(num_outputs)], loc='upper left')
    plt.show()

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    binary_predictions = [np.round(pred) for pred in predictions]

    binary_predictions_stacked = np.hstack(np.hstack(binary_predictions))
    y_test_stacked = np.hstack(y_test.values)

    overall_binary_accuracy = accuracy_score(y_test_stacked, binary_predictions_stacked)
    cm = confusion_matrix(y_test_stacked, binary_predictions_stacked)
    
    true_negatives, false_positives, false_negatives, true_positives = cm.ravel()

    return overall_binary_accuracy, false_negatives, false_positives

def get_four_closest(play):
    x, y = np.reshape(np.array(play), (2, 11))
    distances = np.sqrt(x**2 + y**2)
    four_closest = np.argsort(distances)[:4]
    
    predictions = np.zeros(11)
    predictions.put(four_closest, 1)

    return predictions
    
def print_baseline_accuracy(df_flattened):
    df_flattened_loc = df_flattened[[f"x_{i + 1}" for i in range(11)] + [f"y_{i + 1}" for i in range(11)]]
    
    y_total = df_flattened[[f"isRusher_{i+1}" for i in range(11)]]
    y_true = np.hstack(y_total.values)
    
    baseline_predictions = np.hstack(df_flattened_loc.apply(get_four_closest, axis=1))
    baseline_accuracy = accuracy_score(y_true, baseline_predictions)
    
    print(f"Baseline Accuracy: {baseline_accuracy}")
    
# Main function to run the entire pipeline
if __name__ == "__main__":
    
    df_flattened = load_data()
    df_flattened_loc = df_flattened[[f"x_{i + 1}" for i in range(11)] + [f"y_{i + 1}" for i in range(11)]]
    
    print_baseline_accuracy(df_flattened)
    # Load and process data
    X, y = process_data(df_flattened)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the model
    input_shape = X.shape[1:]
    num_outputs = 11
    model = create_2d_cnn(input_shape, num_outputs)
    
    # Convert each column of y_train to a separate NumPy array and put all these arrays into a list
    y_train_list = [y_train.values[:, i] for i in range(num_outputs)]
    # Train the 2D convolutional network
    history = model.fit(X_train, y_train_list, epochs=10, batch_size=32, verbose=0)

    # Plot training loss and accuracy
    plot_training_loss(history)
    plot_training_accuracy(history, num_outputs)

    # Evaluate the model
    overall_binary_accuracy, false_negatives, false_positives = evaluate_model(model, X_test, y_test)

    print(f"Overall Binary Accuracy: {overall_binary_accuracy}")
    print(f"False Negatives: {false_negatives}")
    print(f"False Positives: {false_positives}")
    
    predictions = model.predict(X)
    
    for i in range(11):
        df_flattened[f"prediction_{i+1}"] = predictions[i].reshape((-1, ))
    
    df_flattened.to_csv("df_flattened_w_preds.csv")