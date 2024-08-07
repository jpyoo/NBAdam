import numpy as np
import pickle
import csv
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import os
from scipy.stats import entropy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def load_images(file_path):
    """Load images from the MNIST dataset file."""
    with open(file_path, 'rb') as file:
        file.read(16)  # Skip the header
        data = np.frombuffer(file.read(), dtype=np.uint8)
    num_images = data.size // (28 * 28)
    return data.reshape(num_images, 28, 28)

def load_labels(file_path):
    """Load labels from the MNIST dataset file."""
    with open(file_path, 'rb') as file:
        file.read(8)  # Skip the header
        data = np.frombuffer(file.read(), dtype=np.uint8)
    return data

def visualize_samples(images):
    """Visualize a grid of the first 9 images."""
    fig, axes = plt.subplots(3, 3, figsize=(3, 3))
    axes = axes.ravel()
    for i in range(9):
        axes[i].imshow(images[i], cmap='gray')
        axes[i].set_title(f'Image {i+1}')
        axes[i].axis('off')
    plt.subplots_adjust(hspace=0.5)
    plt.show()

# visualize_samples(train_images)

def flatten_and_normalize(X):
    """Flatten and normalize the image data."""
    flattened = X.reshape(X.shape[0], -1)
    mean = np.mean(flattened)
    std = np.std(flattened)
    return (flattened - mean) / std

def one_hot_encode(labels):
    """One-hot encode the labels."""
    num_classes = len(np.unique(labels))
    return np.eye(num_classes)[labels]

def split_data(X, ratio=0.8):
    """Split the data into training and validation sets."""
    split_size = int(len(X) * ratio)
    return X[:split_size], X[split_size:]



def calculate_absolute_sum(model_layers):
    total_sum = 0.0
    for layer in model_layers:
        total_sum += np.sum(np.abs(layer[0]))
    return total_sum

def compute_accuracy(predictions, y_true):
    """Compute accuracy of the predictions."""
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_true, axis=1)
    return np.mean(predicted_classes == true_classes) * 100

def params_hist(loaded_params):
  # Assuming each item in layers is a dictionary with keys 'weights' and 'biases'
  all_weights = []
  all_biases = []

  for layer in loaded_params:
      # Flatten the weights and biases and add them to the lists
      all_weights.extend(np.ravel(layer[0]))  # Flatten and append weights
      all_biases.extend(np.ravel(layer[1]))  # Flatten and append biases

  # Plotting the histograms
  fig, ax = plt.subplots(2, 1, figsize=(10, 8))

  # Histogram for weights
  ax[0].hist(all_weights, bins=50, color='blue', edgecolor='black')
  ax[0].set_title('Histogram of Weights')
  ax[0].set_xlabel('Weights')
  ax[0].set_ylabel('Frequency')

  # Histogram for biases
  ax[1].hist(all_biases, bins=50, color='red', edgecolor='black')
  ax[1].set_title('Histogram of Biases')
  ax[1].set_xlabel('Biases')
  ax[1].set_ylabel('Frequency')

  plt.tight_layout()
  plt.show()