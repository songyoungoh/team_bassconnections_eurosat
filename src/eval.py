# Import necessary libraries and modules for deep learning, data visualization, and data processing
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import models
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import transforms, models, datasets
from sklearn.metrics import confusion_matrix
import seaborn as sns
import random
from torch.utils.data import random_split
from collections import Counter
import pandas as pd



# Function to evaluate a model and get the confusion matrix
def eval_confusion(model, dataloaders):
    # Set the device to CUDA if available, otherwise CPU.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Set the model to evaluation mode
    model.eval()

    # Lists to store the true labels and model's predictions
    true_labels = []
    predictions = []

    # Store the indices of the correct/incorrect predictions and their probabilities
    incorrect_indices = []
    incorrect_probs = []
    correct_indices = []
    correct_probs = []
    total_probs = []

    # Loop through the validation data
    for idx, (inputs, labels) in enumerate(dataloaders['valid']):
        # Move the inputs and labels to the appropriate device (GPU or CPU)
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Ensure no gradients are calculated as we are only evaluating, not training
        with torch.no_grad():
            # Forward pass to get the model's predictions
            outputs = model(inputs)
            # Get the index of the predicted class
            _, preds = torch.max(outputs, 1)

        # Extend the true labels and predictions lists
        true_labels.extend(labels.cpu().numpy())
        predictions.extend(preds.cpu().numpy())

        # Store the incorrectly and correctly classified images and their probabilities
        for i, (pred, label) in enumerate(zip(preds, labels)):
            total_probs.append(outputs[i].cpu().numpy())
            if pred != label:
                incorrect_indices.append(idx * dataloaders['valid'].batch_size + i)
                incorrect_probs.append(outputs[i].cpu().numpy())
            else:
                correct_indices.append(idx * dataloaders['valid'].batch_size + i)
                correct_probs.append(outputs[i].cpu().numpy())

    # Compute the test accuracy
    test_acc = np.sum(np.array(true_labels) == np.array(predictions)) / len(true_labels)
    print('')
    print(f'Test Accuracy: {test_acc * 100:.2f}%')
    print(f'Number of Correct Predictions: {np.sum(np.array(true_labels) == np.array(predictions))} / 5400')
    print(f'Number of Incorrect Predictions: {len(true_labels) - np.sum(np.array(true_labels) == np.array(predictions))} / 5400')

    # Get the string representations of class labels
    class_labels = dataloaders['valid'].dataset.dataset.classes

    # Map integer class indices to their corresponding string class labels
    true_labels_str = [class_labels[i] for i in true_labels]
    predictions_str = [class_labels[i] for i in predictions]
    
    return true_labels_str, predictions_str


    # Map integer values with string values
    true_labels_str = [class_labels[i] for i in true_labels]
    predictions_str = [class_labels[i] for i in predictions]
    return true_labels_str, predictions_str
